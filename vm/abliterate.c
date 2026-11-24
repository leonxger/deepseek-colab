// In-place rank-2 directional ablation (W' = W - s * Q * (Q^T * W)) on GGUF tensors via libggml-base.
// Uses original block scales + unbiased stochastic rounding for Q6_K, Q5_K, and Q8_0 so small low-rank edits
// are not erased by round-to-nearest quantization while preserving Unsloth's imatrix-calibrated block scales.
// Falls back to ggml_quantize_chunk for floating-point or other supported ggml types.
// Usage: abliterate Q_BIN PLAN_TXT
#define _FILE_OFFSET_BITS 64
#ifndef ABLITERATE_STANDALONE
#include "ggml.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ABLITERATE_STANDALONE
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
};

static inline float ggml_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t) (h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline int ggml_quantize_requires_imatrix(enum ggml_type type) {
    return !(type == GGML_TYPE_F32 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K);
}

static inline size_t ggml_row_size(enum ggml_type type, int64_t ne0) {
    switch (type) {
        case GGML_TYPE_F32:  return (size_t) ne0 * sizeof(float);
        case GGML_TYPE_Q8_0: return (size_t) (ne0 / 32)  * 34;
        case GGML_TYPE_Q5_K: return (size_t) (ne0 / 256) * 176;
        case GGML_TYPE_Q6_K: return (size_t) (ne0 / 256) * 210;
        default: return 0;
    }
}

static inline const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        default: return "unknown";
    }
}
#endif

static inline uint64_t splitmix64(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Uniform double in [0, 1) with exact 0 exclusion of 1.0 so exact integers stay unchanged under floor(x + u).
static inline double rand_u01(uint64_t * state) {
    return (double) (splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int stoch_round_clamp(double x, int lo, int hi, uint64_t * rng) {
    double fl = floor(x);
    double frac = x - fl;
    int q = (int) fl + (frac > 0.0 && rand_u01(rng) < frac ? 1 : 0);
    if (q < lo) q = lo;
    if (q > hi) q = hi;
    return q;
}

static inline uint16_t read_u16(const uint8_t * p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

#ifdef ABLITERATE_STANDALONE
static void dequant_row_q8_0(const uint8_t * src, float * dst, int64_t ne0) {
    const int64_t nb = ne0 / 32;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t * blk = src + (size_t) i * 34;
        const float d = ggml_fp16_to_fp32(read_u16(blk));
        const int8_t * qs = (const int8_t *) (blk + 2);
        float * y = dst + (size_t) i * 32;
        for (int j = 0; j < 32; j++) y[j] = (float) qs[j] * d;
    }
}

static void dequant_row_q5_K(const uint8_t * src, float * dst, int64_t ne0) {
    const int64_t nb = ne0 / 256;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t * blk = src + (size_t) i * 176;
        const float d    = ggml_fp16_to_fp32(read_u16(blk + 0));
        const float dmin = ggml_fp16_to_fp32(read_u16(blk + 2));
        const uint8_t * scales = blk + 4;
        const uint8_t * qh     = blk + 16;
        const uint8_t * ql     = blk + 48;
        float * y = dst + (size_t) i * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; l++) y[l +  0] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) y[l + 32] = d2 * ((ql[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2;
            y += 64; ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequant_row_q6_K(const uint8_t * src, float * dst, int64_t ne0) {
    const int64_t nb = ne0 / 256;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t * blk = src + (size_t) i * 210;
        const uint8_t * ql  = blk + 0;
        const uint8_t * qh  = blk + 128;
        const int8_t  * sc  = (const int8_t *) (blk + 192);
        const float d       = ggml_fp16_to_fp32(read_u16(blk + 208));
        float * y = dst + (size_t) i * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int8_t q1 = (int8_t) ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t) ((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t) ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}
#endif

// Fixed-scale stochastic requantization for one row of Q8_0 (34 bytes per 32 elements: fp16 d, int8 qs[32]).
static void requant_row_q8_0_stoch(const float * src, uint8_t * dst, int64_t ne0, uint64_t * rng) {
    const int64_t nb = ne0 / 32;
    for (int64_t i = 0; i < nb; i++) {
        uint8_t * blk = dst + (size_t) i * 34;
        const double d = (double) ggml_fp16_to_fp32(read_u16(blk));
        if (d == 0.0) continue;
        const double inv_d = 1.0 / d;
        int8_t * qs = (int8_t *) (blk + 2);
        const float * x = src + (size_t) i * 32;
        for (int j = 0; j < 32; j++) {
            qs[j] = (int8_t) stoch_round_clamp((double) x[j] * inv_d, -127, 127, rng);
        }
    }
}

// Fixed-scale stochastic requantization for one row of Q5_K (176 bytes per 256 elements:
// fp16 d [0..1], fp16 dmin [2..3], uint8 scales[12] [4..15], uint8 qh[32] [16..47], uint8 qs[128] [48..175]).
static void requant_row_q5_K_stoch(const float * src, uint8_t * dst, int64_t ne0, uint64_t * rng) {
    const int64_t nb = ne0 / 256;
    for (int64_t i = 0; i < nb; i++) {
        uint8_t * blk = dst + (size_t) i * 176;
        const double d    = (double) ggml_fp16_to_fp32(read_u16(blk + 0));
        const double dmin = (double) ggml_fp16_to_fp32(read_u16(blk + 2));
        const uint8_t * scales = blk + 4;
        uint8_t * qh = blk + 16;
        uint8_t * ql = blk + 48;
        memset(qh, 0, 32);

        const float * x = src + (size_t) i * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1_q, sc2, m2_q;
            get_scale_min_k4(is + 0, scales, &sc1, &m1_q);
            get_scale_min_k4(is + 1, scales, &sc2, &m2_q);
            const double d1 = d * (double) sc1, m1 = dmin * (double) m1_q;
            const double d2 = d * (double) sc2, m2 = dmin * (double) m2_q;
            const double inv_d1 = d1 != 0.0 ? 1.0 / d1 : 0.0;
            const double inv_d2 = d2 != 0.0 ? 1.0 / d2 : 0.0;
            for (int l = 0; l < 32; l++) {
                int q1 = d1 != 0.0 ? stoch_round_clamp(((double) x[l +  0] + m1) * inv_d1, 0, 31, rng) : 0;
                int q2 = d2 != 0.0 ? stoch_round_clamp(((double) x[l + 32] + m2) * inv_d2, 0, 31, rng) : 0;
                ql[l] = (uint8_t) ((q1 & 0xF) | ((q2 & 0xF) << 4));
                if (q1 & 16) qh[l] |= u1;
                if (q2 & 16) qh[l] |= u2;
            }
            x += 64;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// Fixed-scale stochastic requantization for one row of Q6_K (210 bytes per 256 elements:
// uint8 ql[128] [0..127], uint8 qh[64] [128..191], int8 scales[16] [192..207], fp16 d [208..209]).
static void requant_row_q6_K_stoch(const float * src, uint8_t * dst, int64_t ne0, uint64_t * rng) {
    const int64_t nb = ne0 / 256;
    for (int64_t i = 0; i < nb; i++) {
        uint8_t * blk = dst + (size_t) i * 210;
        uint8_t * ql = blk + 0;
        uint8_t * qh = blk + 128;
        const int8_t * sc = (const int8_t *) (blk + 192);
        const double d = (double) ggml_fp16_to_fp32(read_u16(blk + 208));

        const float * x = src + (size_t) i * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const double s0 = d * (double) sc[is + 0];
                const double s2 = d * (double) sc[is + 2];
                const double s4 = d * (double) sc[is + 4];
                const double s6 = d * (double) sc[is + 6];
                const int q1 = (s0 != 0.0 ? stoch_round_clamp((double) x[l +  0] / s0, -32, 31, rng) : 0) + 32;
                const int q2 = (s2 != 0.0 ? stoch_round_clamp((double) x[l + 32] / s2, -32, 31, rng) : 0) + 32;
                const int q3 = (s4 != 0.0 ? stoch_round_clamp((double) x[l + 64] / s4, -32, 31, rng) : 0) + 32;
                const int q4 = (s6 != 0.0 ? stoch_round_clamp((double) x[l + 96] / s6, -32, 31, rng) : 0) + 32;
                ql[l +  0] = (uint8_t) ((q1 & 0xF) | ((q3 & 0xF) << 4));
                ql[l + 32] = (uint8_t) ((q2 & 0xF) | ((q4 & 0xF) << 4));
                qh[l] = (uint8_t) (((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) |
                                   (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6));
            }
            x  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void dequant_matrix(enum ggml_type type, const uint8_t * raw, float * W, int64_t ne0, int64_t ne1, size_t row_bytes) {
    if (type == GGML_TYPE_F32) {
        memcpy(W, raw, (size_t) ne1 * row_bytes);
        return;
    }
#ifdef ABLITERATE_STANDALONE
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < ne1; r++) {
        const uint8_t * src = raw + (size_t) r * row_bytes;
        float * dst = W + (size_t) r * (size_t) ne0;
        if (type == GGML_TYPE_Q8_0) dequant_row_q8_0(src, dst, ne0);
        else if (type == GGML_TYPE_Q5_K) dequant_row_q5_K(src, dst, ne0);
        else if (type == GGML_TYPE_Q6_K) dequant_row_q6_K(src, dst, ne0);
        else { fprintf(stderr, "abliterate: unsupported standalone type %d\n", (int) type); exit(1); }
    }
#else
    const struct ggml_type_traits * tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) {
        fprintf(stderr, "abliterate: no to_float for ggml_type %d\n", (int) type);
        exit(1);
    }
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < ne1; r++) {
        tr->to_float(raw + (size_t) r * row_bytes, W + (size_t) r * (size_t) ne0, ne0);
    }
#endif
}

static double project_qt_w(const double * q0, const double * q1, const float * W, int64_t ne0, int64_t ne1, double * P0, double * P1) {
    memset(P0, 0, (size_t) ne0 * sizeof(double));
    memset(P1, 0, (size_t) ne0 * sizeof(double));
    #pragma omp parallel
    {
        double * l0 = (double *) calloc((size_t) ne0, sizeof(double));
        double * l1 = (double *) calloc((size_t) ne0, sizeof(double));
        if (!l0 || !l1) {
            fprintf(stderr, "abliterate: out of memory\n");
            exit(1);
        }
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < ne1; r++) {
            const double a0 = q0[r];
            const double a1 = q1[r];
            const float * row = W + (size_t) r * (size_t) ne0;
            for (int64_t c = 0; c < ne0; c++) {
                const double w = (double) row[c];
                l0[c] += a0 * w;
                l1[c] += a1 * w;
            }
        }
        #pragma omp critical
        {
            for (int64_t c = 0; c < ne0; c++) {
                P0[c] += l0[c];
                P1[c] += l1[c];
            }
        }
        free(l0);
        free(l1);
    }
    double norm2 = 0.0;
    for (int64_t c = 0; c < ne0; c++) {
        norm2 += P0[c] * P0[c] + P1[c] * P1[c];
    }
    return sqrt(norm2);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s Q_BIN PLAN_TXT\n", argv[0]);
        return 2;
    }

    FILE * fq = fopen(argv[1], "rb");
    if (!fq) {
        perror(argv[1]);
        return 1;
    }
    int64_t dim = 0;
    if (fread(&dim, sizeof(dim), 1, fq) != 1 || dim <= 0 || dim > 1048576) {
        fprintf(stderr, "abliterate: invalid Q_BIN header\n");
        fclose(fq);
        return 1;
    }
    double * q0 = (double *) malloc((size_t) dim * sizeof(double));
    double * q1 = (double *) malloc((size_t) dim * sizeof(double));
    if (!q0 || !q1 || fread(q0, sizeof(double), (size_t) dim, fq) != (size_t) dim ||
        fread(q1, sizeof(double), (size_t) dim, fq) != (size_t) dim) {
        fprintf(stderr, "abliterate: failed to read Q vectors\n");
        fclose(fq);
        return 1;
    }
    fclose(fq);

    FILE * fp = fopen(argv[2], "r");
    if (!fp) {
        perror(argv[2]);
        return 1;
    }

    char line[4096];
    int count = 0;
    size_t total_bytes = 0;
    double max_err = 0.0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
            continue;
        }
        char path[2048], name[256];
        int gtype = 0;
        long long ne0 = 0, ne1 = 0, off = 0, nbytes = 0;
        double scale = 0.0;
        if (sscanf(line, "%2047[^\t]\t%255[^\t]\t%d\t%lld\t%lld\t%lld\t%lld\t%lf",
                   path, name, &gtype, &ne0, &ne1, &off, &nbytes, &scale) != 8) {
            fprintf(stderr, "abliterate: malformed plan line: %s", line);
            return 1;
        }
        if (ne1 != dim) {
            fprintf(stderr, "abliterate: %s ne1=%lld != Q dim=%lld\n", name, ne1, (long long) dim);
            return 1;
        }
        enum ggml_type type = (enum ggml_type) gtype;
        if (ggml_quantize_requires_imatrix(type)) {
            fprintf(stderr, "abliterate: %s uses type %d which requires an imatrix\n", name, gtype);
            return 1;
        }
        size_t row_bytes = ggml_row_size(type, ne0);
        if ((size_t) ne1 * row_bytes != (size_t) nbytes) {
            fprintf(stderr, "abliterate: %s size mismatch: ne1*row_bytes=%zu != nbytes=%lld\n",
                    name, (size_t) ne1 * row_bytes, nbytes);
            return 1;
        }

        uint8_t * raw_in  = (uint8_t *) malloc((size_t) nbytes);
        uint8_t * raw_out = (uint8_t *) malloc((size_t) nbytes);
        float   * W       = (float   *) malloc((size_t) ne0 * (size_t) ne1 * sizeof(float));
        double  * P0      = (double  *) malloc((size_t) ne0 * sizeof(double));
        double  * P1      = (double  *) malloc((size_t) ne0 * sizeof(double));
        if (!raw_in || !raw_out || !W || !P0 || !P1) {
            fprintf(stderr, "abliterate: out of memory for %s\n", name);
            return 1;
        }

        FILE * fs = fopen(path, "r+b");
        if (!fs) {
            perror(path);
            return 1;
        }
        if (fseeko(fs, (off_t) off, SEEK_SET) != 0 || fread(raw_in, 1, (size_t) nbytes, fs) != (size_t) nbytes) {
            fprintf(stderr, "abliterate: read failed for %s at %lld\n", name, off);
            fclose(fs);
            return 1;
        }

        dequant_matrix(type, raw_in, W, ne0, ne1, row_bytes);
        const double norm_before = project_qt_w(q0, q1, W, ne0, ne1, P0, P1);

        #pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < ne1; r++) {
            const double a0 = scale * q0[r];
            const double a1 = scale * q1[r];
            float * row = W + (size_t) r * (size_t) ne0;
            for (int64_t c = 0; c < ne0; c++) {
                row[c] = (float) ((double) row[c] - a0 * P0[c] - a1 * P1[c]);
            }
        }

        const uint64_t tensor_seed = 0x243f6a8885a308d3ULL ^ ((uint64_t) (count + 1) * 0x9e3779b97f4a7c15ULL);
        if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K) {
            memcpy(raw_out, raw_in, (size_t) nbytes);
            #pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < ne1; r++) {
                uint64_t rng = tensor_seed ^ ((uint64_t) r * 0xbf58476d1ce4e5b9ULL);
                const float * row_w = W + (size_t) r * (size_t) ne0;
                uint8_t * row_out = raw_out + (size_t) r * row_bytes;
                if (type == GGML_TYPE_Q8_0) {
                    requant_row_q8_0_stoch(row_w, row_out, ne0, &rng);
                } else if (type == GGML_TYPE_Q5_K) {
                    requant_row_q5_K_stoch(row_w, row_out, ne0, &rng);
                } else {
                    requant_row_q6_K_stoch(row_w, row_out, ne0, &rng);
                }
            }
        } else if (type == GGML_TYPE_F32) {
            memcpy(raw_out, W, (size_t) nbytes);
        } else {
#ifdef ABLITERATE_STANDALONE
            fprintf(stderr, "abliterate: unsupported standalone type %d on %s\n", (int) type, name);
            fclose(fs);
            return 1;
#else
            ggml_quantize_init(type);
            int qerr = 0;
            #pragma omp parallel for schedule(static) reduction(|:qerr)
            for (int64_t r = 0; r < ne1; r++) {
                size_t got = ggml_quantize_chunk(type, W, raw_out, r * ne0, 1, ne0, NULL);
                if (got != row_bytes) {
                    qerr = 1;
                }
            }
            if (qerr) {
                fprintf(stderr, "abliterate: ggml_quantize_chunk failed on %s\n", name);
                fclose(fs);
                return 1;
            }
#endif
        }

        if (fseeko(fs, (off_t) off, SEEK_SET) != 0 || fwrite(raw_out, 1, (size_t) nbytes, fs) != (size_t) nbytes || fflush(fs) != 0) {
            fprintf(stderr, "abliterate: write failed for %s at %lld\n", name, off);
            fclose(fs);
            return 1;
        }
        fclose(fs);

        dequant_matrix(type, raw_out, W, ne0, ne1, row_bytes);
        const double norm_after = project_qt_w(q0, q1, W, ne0, ne1, P0, P1);
        const double target = fabs(1.0 - scale);
        const double actual = norm_before > 0.0 ? norm_after / norm_before : 0.0;
        const double err = fabs(actual - target);
        if (err > max_err) {
            max_err = err;
        }
        printf("abliterated %-34s type=%-5s scale=%.4f target=%.4f actual=%.4f\n",
               name, ggml_type_name(type), scale, target, actual);
        fflush(stdout);

        free(raw_in);
        free(raw_out);
        free(W);
        free(P0);
        free(P1);
        count++;
        total_bytes += (size_t) nbytes;
    }

    fclose(fp);
#ifndef ABLITERATE_STANDALONE
    ggml_quantize_free();
#endif
    free(q0);
    free(q1);
    printf("abliterate done: %d tensors, %zu bytes, max_ratio_err=%.4f\n", count, total_bytes, max_err);
    return count > 0 ? 0 : 1;
}
