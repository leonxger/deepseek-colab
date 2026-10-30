"""One DeepSeek-V4.1 hyper-connection in two CUDA kernels for batches of 1-8 tokens (GGML_CUDA_HCFUSE=1). usage: python3 patch_hcfuse.py LLAMA_SRC_DIR"""
import os, sys

src = sys.argv[1]


def patch(rel, old, new):
    p = os.path.join(src, rel)
    s = open(p).read()
    if new in s:
        print('already patched', rel, new.strip()[:50])
        return
    assert s.count(old) == 1, (rel, old)
    open(p, 'w').write(s.replace(old, new))
    print('patched', rel, new.strip()[:50])


# ---- declarations (dsv4-hc.cuh) ----
H = 'ggml/src/ggml-cuda/dsv4-hc.cuh'
A = '''bool ggml_cuda_dsv4_hc_post_add_supported(const ggml_tensor * add, const ggml_tensor * post);
void ggml_cuda_op_dsv4_hc_post_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * dst);
'''
patch(H, A, A + '''// HCFUSE: false when GGML_CUDA_HCFUSE=0
bool ggml_cuda_hcfuse_enabled();
static constexpr int DSV4_HCF_MAX_TOKENS = 8;
// HCFUSE: the nodes of one hc instance, [ADD ->] DSV4_HC_POST -> RMS_NORM -> MUL_MAT -> post gate -> DSV4_HC_COMB -> pre gate -> DSV4_HC_PRE -> RMS_NORM -> MUL
struct ggml_cuda_dsv4_hcf {
    const ggml_tensor * xa   = nullptr; // hc_post input, or the first summand of the fused ADD
    const ggml_tensor * xb   = nullptr; // second summand, nullptr without ADD
    ggml_tensor       * post = nullptr; // DSV4_HC_POST: the new stream
    const ggml_tensor * norm = nullptr; // RMS_NORM of the flat stream (not written)
    ggml_tensor       * mm   = nullptr; // MUL_MAT: the mixes
    const ggml_tensor * px   = nullptr; // post gate: view of the mixes, scalar, bias
    const ggml_tensor * ps   = nullptr;
    const ggml_tensor * pb   = nullptr;
    ggml_tensor       * pg   = nullptr; // post gate SCALE: the post weights
    ggml_tensor       * comb = nullptr; // DSV4_HC_COMB
    const ggml_tensor * gx   = nullptr; // pre gate: view of the previous mixes, scalar, bias
    const ggml_tensor * gs   = nullptr;
    const ggml_tensor * gb   = nullptr;
    const ggml_tensor * qg   = nullptr; // pre gate SCALE (not written)
    const ggml_tensor * pre  = nullptr; // DSV4_HC_PRE (not written)
    const ggml_tensor * rms  = nullptr; // RMS_NORM of the hc_pre output (not written)
    ggml_tensor       * out  = nullptr; // MUL: the sublayer input
};
// HCFUSE: shapes, types and strides the two kernels handle
bool ggml_cuda_dsv4_hcf_supported(const ggml_cuda_dsv4_hcf & m);
void ggml_cuda_op_dsv4_hcf(ggml_backend_cuda_context & ctx, const ggml_cuda_dsv4_hcf & m);
''')

# ---- kernels (dsv4-hc.cu) ----
C = 'ggml/src/ggml-cuda/dsv4-hc.cu'
s = open(os.path.join(src, C)).read()
b0 = s.find('    const float scale_comb = scale[2*ss0];\n')
b1 = s.find('\ntemplate <bool gated>\nstatic __global__ void dsv4_hc_pre_f32(')
assert b0 > 0 and b1 > b0, 'dsv4_hc_comb_f32 body not found'
COMB_BODY = s[b0:b1]  # verbatim body of dsv4_hc_comb_f32, closing brace included
assert 'return' not in COMB_BODY, 'dsv4_hc_comb_f32 body changed'

A = '''            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
'''
patch(C, A, A + '''
bool ggml_cuda_hcfuse_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_HCFUSE");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

static constexpr int DSV4_HCF_BLOCK  = 128;                   // threads per block = stream elements per block, both kernels
static constexpr int DSV4_HCF_NMIX   = (2 + DSV4_HC)*DSV4_HC; // mixes per token: pre, post, comb
static constexpr int DSV4_HCF_MAXBLK = 64;                    // partials per mix the last block sums: n_embd <= 8192

// HCFUSE: hc_pre value of one stream element, the arithmetic of dsv4_hc_pre_norm_f32
static __device__ __forceinline__ float dsv4_hcf_pre_val(const float * x, const float * wh, const float pre_scale) {
    float sum = 0.0f;
#pragma unroll
    for (int ih = 0; ih < DSV4_HC; ++ih) {
        sum += x[ih] * wh[ih];
    }
    return pre_scale * sum;
}

// HCFUSE: dsv4_hc_comb_f32 for one token (body copied verbatim)
static __device__ __forceinline__ void dsv4_hcf_comb_token(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        const int64_t it,
        const int64_t sm0,
        const int64_t sm1,
        const int64_t ss0,
        const int64_t sb0,
        const int64_t sd0,
        const int64_t sd1,
        const int64_t sd2,
        const float eps,
        const int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

''' + COMB_BODY + '''
// HCFUSE: kernel 1, one thread per stream element and token: hc_post, then per-block sums of squares of the stream and of its hc_pre value
template <bool has_xb>
static __global__ void __launch_bounds__(DSV4_HCF_BLOCK) dsv4_hcf_post_f32(
        const float * xa,
        const float * xb,
        const float * residual,
        const float * post,
        const float * comb,
        const float * gx,
        const float * gs,
        const float * gb,
        float * dst,
        float * part,
        float * wh_out,
        unsigned int * counter,
        const int n_embd,
        const int64_t sa1,
        const int64_t sb1,
        const int64_t sr1,
        const int64_t sr2,
        const int64_t sp0,
        const int64_t sp1,
        const int64_t sc0,
        const int64_t sc1,
        const int64_t sc2,
        const int64_t sgx1,
        const float gscale,
        const float gbias,
        const float pre_scale) {
    constexpr int nwarps = DSV4_HCF_BLOCK/WARP_SIZE;

    ggml_cuda_pdl_lc();
    const int     tid  = threadIdx.x;
    const int     lane = tid % WARP_SIZE;
    const int     warp = tid / WARP_SIZE;
    const int     it   = blockIdx.y;
    const int64_t i0   = (int64_t) blockIdx.x*DSV4_HCF_BLOCK + tid;

    __shared__ float s_ss[nwarps];
    __shared__ float s_ps[nwarps];

    ggml_cuda_pdl_sync();

    // pre weights of the previous mixes, as dsv4_hc_pre_norm_f32 computes them
    float wh[DSV4_HC];
#pragma unroll
    for (int ih = 0; ih < DSV4_HC; ++ih) {
        const float g = __fadd_rn(__fmul_rn(gx[ih + it*sgx1], gs[0]), gb[ih]);
        wh[ih] = gscale * (1.0f / (1.0f + expf(-g))) + gbias;
    }

    float xv;
    if constexpr (has_xb) {
        xv = __fadd_rn(xa[i0 + it*sa1], xb[i0 + it*sb1]);
    } else {
        xv = xa[i0 + it*sa1];
    }

    float res[DSV4_HC];
#pragma unroll
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        res[isrc] = residual[i0 + isrc*sr1 + it*sr2];
    }

    // x*p rounded first, then one fma per residual stream: the order of the loop in dsv4_hc_post_f32
    float xo[DSV4_HC];
    float ss = 0.0f;
#pragma unroll
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = __fmul_rn(xv, post[idst*sp0 + it*sp1]);
#pragma unroll
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum = __fmaf_rn(res[isrc], comb[idst*sc0 + isrc*sc1 + it*sc2], sum);
        }
        xo[idst] = sum;
        dst[((int64_t) it*DSV4_HC + idst)*n_embd + i0] = sum;
        ss += sum*sum;
    }
    const float pv = dsv4_hcf_pre_val(xo, wh, pre_scale);
    float ps = pv*pv;

    ss = warp_reduce_sum<WARP_SIZE>(ss);
    ps = warp_reduce_sum<WARP_SIZE>(ps);
    if (lane == 0) {
        s_ss[warp] = ss;
        s_ps[warp] = ps;
    }
    __syncthreads();

    if (tid == 0) {
        float a = s_ss[0];
        float b = s_ps[0];
#pragma unroll
        for (int w = 1; w < nwarps; ++w) {
            a += s_ss[w];
            b += s_ps[w];
        }
        const int64_t ip = (int64_t) it*gridDim.x + blockIdx.x;
        part[2*ip + 0] = a;
        part[2*ip + 1] = b;
        if (blockIdx.x == 0) {
#pragma unroll
            for (int ih = 0; ih < DSV4_HC; ++ih) {
                wh_out[it*DSV4_HC + ih] = wh[ih];
            }
            if (it == 0) {
                *counter = 0;
            }
        }
    }
}

// HCFUSE: kernel 2, one thread per stream element: hc_fn dots of the normalized stream and the normalized sublayer input; the last block finishes the mixes
template <int ncols>
static __global__ void __launch_bounds__(DSV4_HCF_BLOCK) dsv4_hcf_mix_f32(
        const float * x,
        const float * fn,
        const float * nw,
        const float * part1,
        const float * wh,
        float * part2,
        unsigned int * counter,
        float * mixes,
        float * dst,
        float * pdst,
        float * cdst,
        const float * ps,
        const float * pb,
        const float * cscale,
        const float * cbase,
        const int n_embd,
        const int ncols_flat,
        const int64_t sfn1,
        const float eps_flat,
        const float eps_pre,
        const float pre_scale,
        const int goff,
        const float pscale,
        const float pbias,
        const int64_t ss0,
        const int64_t sb0,
        const int64_t sd0,
        const int64_t sd1,
        const int64_t sd2,
        const float ceps,
        const int32_t n_iter) {
    constexpr int nmix   = DSV4_HCF_NMIX;
    constexpr int nwarps = DSV4_HCF_BLOCK/WARP_SIZE;

    ggml_cuda_pdl_lc();
    const int     tid  = threadIdx.x;
    const int     lane = tid % WARP_SIZE;
    const int     warp = tid / WARP_SIZE;
    const int     nblk = gridDim.x;
    const int64_t i0   = (int64_t) blockIdx.x*DSV4_HCF_BLOCK + tid;

    __shared__ float s_r[ncols];
    __shared__ float s_rp[ncols];
    __shared__ float s_wh[ncols*DSV4_HC];
    __shared__ float s_red[nwarps][ncols*nmix];
    __shared__ bool  s_last;

    // hc_fn and the norm weight are graph leaves: load them while kernel 1 still runs
    float w[nmix][DSV4_HC];
#pragma unroll
    for (int o = 0; o < nmix; ++o) {
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            w[o][s] = fn[o*sfn1 + s*n_embd + i0];
        }
    }
    const float wn = nw[i0];

    ggml_cuda_pdl_sync();

    float xs[ncols][DSV4_HC];
#pragma unroll
    for (int t = 0; t < ncols; ++t) {
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            xs[t][s] = x[((int64_t) t*DSV4_HC + s)*n_embd + i0];
        }
    }

    // rms_norm scales of the flat stream and of the hc_pre output, kernel 1 partials summed in a fixed order
    for (int t = warp; t < ncols; t += nwarps) {
        float a  = 0.0f;
        float ap = 0.0f;
        for (int b = lane; b < nblk; b += WARP_SIZE) {
            a  += part1[2*(t*nblk + b) + 0];
            ap += part1[2*(t*nblk + b) + 1];
        }
        a  = warp_reduce_sum<WARP_SIZE>(a);
        ap = warp_reduce_sum<WARP_SIZE>(ap);
        if (lane == 0) {
            s_r[t]  = rsqrtf(a / ncols_flat + eps_flat);
            s_rp[t] = rsqrtf(ap / n_embd + eps_pre);
        }
    }
    if (tid < ncols*DSV4_HC) {
        s_wh[tid] = wh[tid];
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < ncols; ++t) {
        float y[DSV4_HC];
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            y[s] = s_r[t]*xs[t][s];
        }
#pragma unroll
        for (int o = 0; o < nmix; ++o) {
            float a = 0.0f;
#pragma unroll
            for (int s = 0; s < DSV4_HC; ++s) {
                a += w[o][s]*y[s];
            }
            a = warp_reduce_sum<WARP_SIZE>(a);
            if (lane == 0) {
                s_red[warp][t*nmix + o] = a;
            }
        }
        float r = s_rp[t]*dsv4_hcf_pre_val(xs[t], s_wh + t*DSV4_HC, pre_scale);
        r *= wn;
        dst[(int64_t) t*n_embd + i0] = r;
    }
    __syncthreads();

    for (int v = tid; v < ncols*nmix; v += DSV4_HCF_BLOCK) {
        float a = s_red[0][v];
#pragma unroll
        for (int k = 1; k < nwarps; ++k) {
            a += s_red[k][v];
        }
        part2[(int64_t) blockIdx.x*ncols*nmix + v] = a;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counter, 1u) == (unsigned int) (nblk - 1);
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // last block: the mixes summed in block order, then the post weights and the comb of each token
    float * s_mix = s_red[0];
    for (int v = tid; v < ncols*nmix; v += DSV4_HCF_BLOCK) {
        float pv[DSV4_HCF_MAXBLK];
#pragma unroll
        for (int b = 0; b < DSV4_HCF_MAXBLK; ++b) {
            pv[b] = b < nblk ? __ldcg(part2 + (int64_t) b*ncols*nmix + v) : 0.0f;
        }
        float a = 0.0f;
#pragma unroll
        for (int b = 0; b < DSV4_HCF_MAXBLK; ++b) {
            if (b < nblk) {
                a += pv[b];
            }
        }
        s_mix[v] = a;
        mixes[v] = a;
    }
    __syncthreads();

    if (tid < ncols*DSV4_HC) {
        const int t = tid / DSV4_HC;
        const int d = tid % DSV4_HC;
        const float g = __fadd_rn(__fmul_rn(s_mix[t*nmix + goff + d], ps[0]), pb[d]);
        pdst[tid] = pscale * (1.0f / (1.0f + expf(-g))) + pbias;
    }
    if (tid >= WARP_SIZE && tid < WARP_SIZE + ncols) {
        dsv4_hcf_comb_token(s_mix, cscale, cbase, cdst, tid - WARP_SIZE, 1, nmix, ss0, sb0, sd0, sd1, sd2, ceps, n_iter);
    }
}

bool ggml_cuda_dsv4_hcf_supported(const ggml_cuda_dsv4_hcf & m) {
    const ggml_tensor * hc   = m.post;
    const ggml_tensor * res  = hc->src[1];
    const ggml_tensor * pw   = hc->src[2];
    const ggml_tensor * cw   = hc->src[3];
    const ggml_tensor * flat = m.norm->src[0];
    const ggml_tensor * fn   = m.mm->src[0];
    const ggml_tensor * nw   = m.out->src[0] == m.rms ? m.out->src[1] : m.out->src[0];
    const ggml_tensor * cs   = m.comb->src[1];
    const ggml_tensor * cb   = m.comb->src[2];

    const int64_t n_embd = hc->ne[0];
    const int64_t nt     = hc->ne[2];

    const auto f32 = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] % sizeof(float) == 0 && t->nb[1] % sizeof(float) == 0 &&
            t->nb[2] % sizeof(float) == 0 && t->nb[3] % sizeof(float) == 0;
    };
    // the kernels index dim 0 of these directly
    const auto dense = [&](const ggml_tensor * t) {
        return f32(t) && t->nb[0] == sizeof(float);
    };

    return dense(hc) && ggml_is_contiguous(hc) && hc->ne[1] == DSV4_HC && hc->ne[3] == 1 &&
        nt >= 1 && nt <= DSV4_HCF_MAX_TOKENS && n_embd % DSV4_HCF_BLOCK == 0 && n_embd/DSV4_HCF_BLOCK <= DSV4_HCF_MAXBLK &&
        dense(m.xa) && m.xa->ne[0] == n_embd && m.xa->ne[1] == nt && m.xa->ne[2] == 1 && m.xa->ne[3] == 1 &&
        (m.xb == nullptr || (dense(m.xb) && ggml_are_same_shape(m.xa, m.xb))) &&
        dense(res) && res->ne[0] == n_embd && res->ne[1] == DSV4_HC && res->ne[2] == nt && res->ne[3] == 1 &&
        f32(pw) && pw->ne[0] == DSV4_HC && pw->ne[1] == nt && pw->ne[2] == 1 && pw->ne[3] == 1 &&
        f32(cw) && cw->ne[0] == DSV4_HC && cw->ne[1] == DSV4_HC && cw->ne[2] == nt && cw->ne[3] == 1 &&
        flat->view_src == hc && flat->view_offs == 0 && ggml_is_contiguous(flat) && flat->ne[0] == DSV4_HC*n_embd && flat->ne[1] == nt &&
        m.norm->type == GGML_TYPE_F32 &&
        dense(fn) && fn->op == GGML_OP_NONE && fn->ne[0] == DSV4_HC*n_embd && fn->ne[1] == DSV4_HCF_NMIX && fn->ne[2] == 1 && fn->ne[3] == 1 &&
        dense(m.mm) && ggml_is_contiguous(m.mm) && m.mm->ne[0] == DSV4_HCF_NMIX && m.mm->ne[1] == nt && m.mm->ne[2] == 1 && m.mm->ne[3] == 1 &&
        m.px->view_src == m.mm && m.px->ne[0] == DSV4_HC && m.px->ne[1] == nt && m.px->nb[0] == sizeof(float) && m.px->nb[1] == m.mm->nb[1] &&
        m.px->view_offs % sizeof(float) == 0 && m.px->view_offs/sizeof(float) + DSV4_HC <= DSV4_HCF_NMIX &&
        dense(m.pg) && ggml_is_contiguous(m.pg) && m.pg->ne[0] == DSV4_HC && m.pg->ne[1] == nt &&
        f32(m.comb) && m.comb->ne[0] == DSV4_HC && m.comb->ne[1] == DSV4_HC && m.comb->ne[2] == nt &&
        f32(cs) && cs->ne[0] >= 3 && f32(cb) && cb->ne[0] == DSV4_HCF_NMIX &&
        m.gx->ne[0] == DSV4_HC && m.gx->ne[1] == nt &&
        dense(m.out) && ggml_is_contiguous(m.out) && m.out->ne[0] == n_embd && m.out->ne[1] == nt &&
        dense(nw) && nw->op == GGML_OP_NONE && nw->ne[0] == n_embd;
}

void ggml_cuda_op_dsv4_hcf(ggml_backend_cuda_context & ctx, const ggml_cuda_dsv4_hcf & m) {
    GGML_ASSERT(ggml_cuda_dsv4_hcf_supported(m));

    const ggml_tensor * hc  = m.post;
    const ggml_tensor * res = hc->src[1];
    const ggml_tensor * pw  = hc->src[2];
    const ggml_tensor * cw  = hc->src[3];
    const ggml_tensor * fn  = m.mm->src[0];
    const ggml_tensor * nw  = m.out->src[0] == m.rms ? m.out->src[1] : m.out->src[0];
    const ggml_tensor * cs  = m.comb->src[1];
    const ggml_tensor * cb  = m.comb->src[2];

    const int n_embd = (int) hc->ne[0];
    const int ncols  = (int) hc->ne[2];
    const int nblk   = n_embd / DSV4_HCF_BLOCK;

    float eps_flat;
    float eps_pre;
    memcpy(&eps_flat, m.norm->op_params, sizeof(float));
    memcpy(&eps_pre,  m.rms->op_params,  sizeof(float));
    const float pre_scale = ggml_get_op_params_f32(m.pre, 0);

    // kernel 1 sums of squares, the pre weights, kernel 2 mix partials, the last-block counter
    const size_t n_part1 = (size_t) 2*ncols*nblk;
    const size_t n_wh    = (size_t) ncols*DSV4_HC;
    const size_t n_part2 = (size_t) nblk*ncols*DSV4_HCF_NMIX;
    ggml_cuda_pool_alloc<float> buf(ctx.pool(), n_part1 + n_wh + n_part2 + 1);
    float        * part1   = buf.get();
    float        * wh      = part1 + n_part1;
    float        * part2   = wh + n_wh;
    unsigned int * counter = (unsigned int *) (part2 + n_part2);

    cudaStream_t stream = ctx.stream();

    const ggml_cuda_kernel_launch_params params1 = ggml_cuda_kernel_launch_params(dim3(nblk, ncols, 1), dim3(DSV4_HCF_BLOCK, 1, 1), 0, stream);
    auto post_kernel = m.xb ? dsv4_hcf_post_f32<true> : dsv4_hcf_post_f32<false>;
    ggml_cuda_kernel_launch(post_kernel, params1,
            (const float *) m.xa->data, m.xb ? (const float *) m.xb->data : (const float *) nullptr,
            (const float *) res->data, (const float *) pw->data, (const float *) cw->data,
            (const float *) m.gx->data, (const float *) m.gs->data, (const float *) m.gb->data,
            (float *) hc->data, part1, wh, counter, n_embd,
            (int64_t) (m.xa->nb[1] / sizeof(float)), (int64_t) (m.xb ? m.xb->nb[1] / sizeof(float) : 0),
            (int64_t) (res->nb[1] / sizeof(float)), (int64_t) (res->nb[2] / sizeof(float)),
            (int64_t) (pw->nb[0] / sizeof(float)), (int64_t) (pw->nb[1] / sizeof(float)),
            (int64_t) (cw->nb[0] / sizeof(float)), (int64_t) (cw->nb[1] / sizeof(float)), (int64_t) (cw->nb[2] / sizeof(float)),
            (int64_t) (m.gx->nb[1] / sizeof(float)),
            ggml_get_op_params_f32(m.qg, 0), ggml_get_op_params_f32(m.qg, 1), pre_scale);

    const ggml_cuda_kernel_launch_params params2 = ggml_cuda_kernel_launch_params(dim3(nblk, 1, 1), dim3(DSV4_HCF_BLOCK, 1, 1), 0, stream);
    auto launch_mix = [&](auto mix_kernel) {
        ggml_cuda_kernel_launch(mix_kernel, params2,
                (const float *) hc->data, (const float *) fn->data, (const float *) nw->data, (const float *) part1, (const float *) wh,
                part2, counter, (float *) m.mm->data, (float *) m.out->data, (float *) m.pg->data, (float *) m.comb->data,
                (const float *) m.ps->data, (const float *) m.pb->data, (const float *) cs->data, (const float *) cb->data,
                n_embd, DSV4_HC*n_embd, (int64_t) (fn->nb[1] / sizeof(float)), eps_flat, eps_pre, pre_scale,
                (int) (m.px->view_offs / sizeof(float)), ggml_get_op_params_f32(m.pg, 0), ggml_get_op_params_f32(m.pg, 1),
                (int64_t) (cs->nb[0] / sizeof(float)), (int64_t) (cb->nb[0] / sizeof(float)),
                (int64_t) (m.comb->nb[0] / sizeof(float)), (int64_t) (m.comb->nb[1] / sizeof(float)), (int64_t) (m.comb->nb[2] / sizeof(float)),
                ggml_get_op_params_f32(m.comb, 0), ggml_get_op_params_i32(m.comb, 1));
    };
    switch (ncols) {
        case 1: launch_mix(dsv4_hcf_mix_f32<1>); break;
        case 2: launch_mix(dsv4_hcf_mix_f32<2>); break;
        case 3: launch_mix(dsv4_hcf_mix_f32<3>); break;
        case 4: launch_mix(dsv4_hcf_mix_f32<4>); break;
        case 5: launch_mix(dsv4_hcf_mix_f32<5>); break;
        case 6: launch_mix(dsv4_hcf_mix_f32<6>); break;
        case 7: launch_mix(dsv4_hcf_mix_f32<7>); break;
        case 8: launch_mix(dsv4_hcf_mix_f32<8>); break;
        default:
            GGML_ABORT("fatal error");
    }
}
''')

# ---- matcher, dispatch, fused-count log and allocation deps (ggml-cuda.cu) ----
C = 'ggml/src/ggml-cuda/ggml-cuda.cu'
assert 'ggml_cuda_match_dsv4_hc_pre_norm' in open(os.path.join(src, C)).read(), 'apply patch_fuse.py and patch_fuse2.py first'
A = '''// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
'''
patch(C, A, '''// HCFUSE: next node after i that is not a view or no-op, n_nodes if none
static int ggml_cuda_hcf_next(const ggml_cgraph * cgraph, int i) {
    do {
        ++i;
    } while (i < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[i]));
    return i;
}

// HCFUSE: one hc instance starting at node i ([ADD ->] DSV4_HC_POST); returns the index of its last node (the norm MUL), 0 if no match
static int ggml_cuda_match_dsv4_hcf(const ggml_cgraph * cgraph, int i, ggml_cuda_dsv4_hcf & m) {
    m = ggml_cuda_dsv4_hcf();

    int ip = i;
    if (cgraph->nodes[i]->op == GGML_OP_ADD) {
        if (i + 1 >= cgraph->n_nodes || !ggml_node_has_n_uses(cgraph, i, 1) ||
                !ggml_cuda_dsv4_hc_post_add_supported(cgraph->nodes[i], cgraph->nodes[i + 1])) {
            return 0;
        }
        m.xa = cgraph->nodes[i]->src[0];
        m.xb = cgraph->nodes[i]->src[1];
        ip   = i + 1;
    }
    m.post = cgraph->nodes[ip];
    if (m.post->op != GGML_OP_DSV4_HC_POST || (m.post->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || m.post->src[3] == nullptr) {
        return 0;
    }
    if (m.xa == nullptr) {
        m.xa = m.post->src[0];
    }

    // only views may sit between the parts: a stream tap, Engram or the output head ends the match
    int k = ggml_cuda_hcf_next(cgraph, ip);
    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_RMS_NORM || !ggml_node_has_n_uses(cgraph, k, 1)) {
        return 0;
    }
    m.norm = cgraph->nodes[k];

    k = ggml_cuda_hcf_next(cgraph, k);
    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_MUL_MAT || cgraph->nodes[k]->src[1] != m.norm) {
        return 0;
    }
    m.mm = cgraph->nodes[k];

    k = ggml_cuda_hcf_next(cgraph, k);
    const int n_post = k < cgraph->n_nodes ? ggml_cuda_match_dsv4_hc_gate(cgraph, k, &m.px, &m.ps, &m.pb) : 0;
    if (n_post == 0) {
        return 0;
    }
    m.pg = cgraph->nodes[k + n_post];

    k = ggml_cuda_hcf_next(cgraph, k + n_post);
    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_DSV4_HC_COMB || cgraph->nodes[k]->src[0] != m.mm ||
            !ggml_cuda_dsv4_hc_comb_gate_supported(m.pg, cgraph->nodes[k])) {
        return 0;
    }
    m.comb = cgraph->nodes[k];

    k = ggml_cuda_hcf_next(cgraph, k);
    const int last = k < cgraph->n_nodes ? ggml_cuda_match_dsv4_hc_pre_norm(cgraph, k, &m.gx, &m.gs, &m.gb) : 0;
    if (last == 0) {
        return 0;
    }
    m.qg  = cgraph->nodes[last - 3];
    m.pre = cgraph->nodes[last - 2];
    m.rms = cgraph->nodes[last - 1];
    m.out = cgraph->nodes[last];

    for (const ggml_tensor * t : { m.norm, (const ggml_tensor *) m.mm }) {
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
    }
    return m.pre->src[0] == m.post && ggml_cuda_dsv4_hcf_supported(m) ? last : 0;
}

// HCFUSE: kernel 1 must not write the stream over what it reads, kernel 2 not its outputs over the stream, the weights or each other
static bool ggml_cuda_dsv4_hcf_disjoint(const ggml_cuda_dsv4_hcf & m) {
    const ggml_tensor * nw     = m.out->src[0] == m.rms ? m.out->src[1] : m.out->src[0];
    const ggml_tensor * in1[]  = { m.xa, m.xb, m.post->src[1], m.post->src[2], m.post->src[3], m.gx, m.gs, m.gb };
    const ggml_tensor * in2[]  = { m.post, m.mm->src[0], nw, m.ps, m.pb, m.comb->src[1], m.comb->src[2] };
    const ggml_tensor * out2[] = { m.mm, m.pg, m.comb, m.out };
    for (const ggml_tensor * t : in1) {
        if (t != nullptr && ggml_cuda_dsv4_overlap(m.post, t)) {
            return false;
        }
    }
    for (int a = 0; a < 4; ++a) {
        for (const ggml_tensor * t : in2) {
            if (ggml_cuda_dsv4_overlap(out2[a], t)) {
                return false;
            }
        }
        for (int b = a + 1; b < 4; ++b) {
            if (ggml_cuda_dsv4_overlap(out2[a], out2[b])) {
                return false;
            }
        }
    }
    return true;
}

// HCFUSE: fused hc instances in the graph evaluation in progress
static thread_local int ggml_cuda_hcfuse_count = 0;

// HCFUSE: GGML_CUDA_HCFUSE_LOG=1 prints the fused count of each decode graph (split) once per distinct result (CUDA graph replays do not get here)
static void ggml_cuda_hcfuse_report(const ggml_cgraph * cgraph) {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_HCFUSE_LOG");
    if (!enabled) {
        return;
    }
    int                 n_post   = 0;
    int64_t             n_tokens = 0;
    const ggml_tensor * first    = nullptr;
    const ggml_tensor * last     = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op == GGML_OP_DSV4_HC_POST) {
            n_post++;
            n_tokens = cgraph->nodes[i]->ne[2];
            first    = first ? first : cgraph->nodes[i];
            last     = cgraph->nodes[i];
        }
    }
    if (n_post == 0 || n_tokens > DSV4_HCF_MAX_TOKENS) {
        return;
    }
    static std::mutex                                                   mutex;
    static std::vector<std::pair<std::string, std::array<int64_t, 3>>> seen;
    const std::pair<std::string, std::array<int64_t, 3>> key = { first->name, { n_tokens, n_post, ggml_cuda_hcfuse_count } };
    std::lock_guard<std::mutex> lock(mutex);
    if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
        seen.push_back(key);
        GGML_LOG_INFO("%s: HCFUSE n_tokens %d, graph %s .. %s: %d of %d DSV4_HC_POST nodes start a fused hc instance\\n",
                __func__, (int) n_tokens, first->name, last->name, ggml_cuda_hcfuse_count, n_post);
    }
}

''' + A)

patch(C, '''    ggml_tensor * node = cgraph->nodes[i];

    if (ggml_cuda_fuse_dsv4_2_enabled()) {
''', '''    ggml_tensor * node = cgraph->nodes[i];

    if (ggml_cuda_hcfuse_enabled() && (node->op == GGML_OP_ADD || node->op == GGML_OP_DSV4_HC_POST)) {
        // HCFUSE: one hc instance in two kernels, decode and verify batches only
        ggml_cuda_dsv4_hcf hcf;
        const int last = ggml_cuda_match_dsv4_hcf(cgraph, i, hcf);
        if (last > 0 && ggml_cuda_dsv4_hcf_disjoint(hcf)) {
            ggml_cuda_op_dsv4_hcf(*cuda_ctx, hcf);
            ggml_cuda_hcfuse_count++;
            return last - i;
        }
    }

    if (ggml_cuda_fuse_dsv4_2_enabled()) {
''')

patch(C, '''            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // fused nodes are skipped together, so a cut never splits a fusion
''', '''            ggml_cuda_hcfuse_count = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // fused nodes are skipped together, so a cut never splits a fusion
''')

patch(C, '''                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
''', '''                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
            ggml_cuda_hcfuse_report(cgraph);
''')

A = '''#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
'''
patch(C, A, '''    if (!disable_fusion && ggml_cuda_hcfuse_enabled()) {
        // HCFUSE: kernel 1 reads both summands of the experts ADD; the DSV4_FUSE2 deps, batch-size independent
        for (int i = 0; i + 1 < cgraph->n_nodes; ++i) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_ADD && ggml_cuda_dsv4_hc_post_add_supported(node, cgraph->nodes[i + 1])) {
                params->add_alloc_dep(params->user_data, ggml_cuda_dsv4_view_root(node->src[0]), cgraph->nodes[i + 1]);
                params->add_alloc_dep(params->user_data, ggml_cuda_dsv4_view_root(node->src[1]), cgraph->nodes[i + 1]);
            }
        }
    }

''' + A)

# ---- graph order: attention mixes and both norms right after the previous hc_post (deepseek41.cpp) ----
D = 'src/models/deepseek41.cpp'
patch(D, '''        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
''', '''        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        // HCFUSE: visit the attention mixes before the attention, for every ubatch size (ggml-alloc reuses its plan by node index)
        static const bool hcfuse = dsv41_env_on("GGML_CUDA_HCFUSE");
        if (hcfuse) {
            ggml_build_forward_expand(gf, attn_mix.post);
            ggml_build_forward_expand(gf, attn_mix.comb);
        }
''')
patch(D, '''        cb(cur, "attn_norm", il);

        cur = build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il);
''', '''        cb(cur, "attn_norm", il);
        if (hcfuse) {
            ggml_build_forward_expand(gf, cur); // HCFUSE: the norm right after the comb
        }

        cur = build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il);
''')
patch(D, '''        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
''', '''        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        if (hcfuse) {
            ggml_build_forward_expand(gf, cur); // HCFUSE: the norm right after the comb
        }
''')

# ---- tests (test-backend-ops.cpp) ----
T = 'tests/test-backend-ops.cpp'
A = '''// DSV4_FUSE2: GET_ROWS (F16) -> RESHAPE -> cast to F16, the dsv4 gather of the selected compressed rows (mask: of their kq mask)
struct test_dsv4_fuse2_get_rows_cast : public test_case {
'''
patch(T, A, '''// HCFUSE: one hc instance in the node order of the dsv4 verify graph: [ADD ->] hc_post -> flat rms_norm -> hc_fn -> post gate -> comb -> pre gate -> hc_pre -> rms_norm -> mul
struct test_dsv4_hcfuse : public test_dsv4_hc {
    const int64_t n_embd;
    const int64_t n_tokens;
    const bool    add; // the hc_post input is the sum of the routed and the shared experts
    const bool    tap; // a reader of the new stream between hc_post and the mixes, as a DSpark tap: stays unfused

    ggml_tensor * hc_new = nullptr;
    ggml_tensor * mixes  = nullptr;
    ggml_tensor * post   = nullptr;
    ggml_tensor * comb   = nullptr;
    ggml_tensor * tapped = nullptr;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_HCFUSE";
    }

    bool run_whole_graph() override { return true; }

    std::vector<ggml_tensor *> fusion_test_nodes() override {
        std::vector<ggml_tensor *> nodes = { hc_new, mixes, post, comb, out };
        if (tapped) {
            nodes.push_back(tapped);
        }
        return nodes;
    }

    std::string vars() override {
        return VARS_TO_STR4(n_embd, n_tokens, add, tap);
    }

    test_dsv4_hcfuse(int64_t n_embd = 5120, int64_t n_tokens = 3, bool add = false, bool tap = false)
        : n_embd(n_embd), n_tokens(n_tokens), add(add), tap(tap) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t nmix = (2 + hc)*hc;

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(x, "x");
        if (add) {
            ggml_tensor * xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_name(xb, "xb");
            x = ggml_add(ctx, x, xb);
        }
        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(residual, "residual");
        ggml_tensor * post_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(post_prev, "post");
        ggml_tensor * comb_prev = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);
        ggml_set_name(comb_prev, "comb");
        ggml_tensor * mixes_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nmix, n_tokens);
        ggml_set_name(mixes_prev, "mixes");
        ggml_tensor * scale_prev = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(scale_prev, "scale_prev");
        ggml_tensor * base_prev = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nmix);
        ggml_set_name(base_prev, "base_prev");
        ggml_tensor * fn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc*n_embd, nmix);
        ggml_set_name(fn, "hc_fn");
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(scale, "scale");
        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nmix);
        ggml_set_name(base, "base");
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_set_name(w, "w");

        hc_new = ggml_dsv4_hc_post(ctx, x, residual, post_prev, comb_prev);
        ggml_set_name(hc_new, "hc_new");
        if (tap) {
            tapped = ggml_add(ctx, ggml_view_2d(ctx, hc_new, n_embd, n_tokens, hc_new->nb[2], 0),
                    ggml_view_2d(ctx, hc_new, n_embd, n_tokens, hc_new->nb[2], hc_new->nb[1]));
            tapped = ggml_scale(ctx, tapped, 0.5f);
            ggml_set_name(tapped, "tapped");
            if (mode == MODE_TEST) {
                ggml_build_forward_expand(gf, tapped);
            }
        }

        mixes = ggml_mul_mat(ctx, fn, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, hc_new, hc*n_embd, n_tokens), 1e-6f));
        ggml_set_name(mixes, "mixes_new");

        post = ggml_view_2d(ctx, mixes, hc, n_tokens, mixes->nb[1], hc*sizeof(float));
        post = ggml_add(ctx, ggml_mul(ctx, post, ggml_view_1d(ctx, scale, 1, sizeof(float))), ggml_view_1d(ctx, base, hc, hc*sizeof(float)));
        post = ggml_scale(ctx, ggml_sigmoid(ctx, post), 2.0f);
        ggml_set_name(post, "post_new");

        comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, 1e-6f, 20);
        ggml_set_name(comb, "comb_new");

        ggml_tensor * pre = ggml_view_2d(ctx, mixes_prev, hc, n_tokens, mixes_prev->nb[1], 0);
        pre = ggml_add(ctx, ggml_mul(ctx, pre, ggml_view_1d(ctx, scale_prev, 1, 0)), ggml_view_1d(ctx, base_prev, hc, 0));
        pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, pre), 1.0f, 1e-6f);

        out = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_dsv4_hc_pre(ctx, hc_new, pre), 1e-6f), w);
        ggml_set_name(out, "out");

        if (mode == MODE_TEST) {
            // deepseek41.cpp visits the post weights and the comb before the norm
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);
        }
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) {
                continue;
            }
            std::string name = ggml_get_name(t);
            if (name.size() > 5 && name.compare(name.size() - 5, 5, "_prev") == 0) {
                name.resize(name.size() - 5);
            }
            float lo = -1.0f;
            float hi =  1.0f;
            if (name == "hc_fn") {
                // mixes of order 1, so the gates and the softmax do not saturate
                lo = -0.02f;
                hi =  0.02f;
            } else if (!tensor_range(name, lo, hi)) {
                init_tensor_uniform(t);
                continue;
            }
            std::mt19937 rng(tensor_seed(t));
            std::uniform_real_distribution<float> dist(lo, hi);
            std::vector<float> data(ggml_nelements(t));
            for (float & v : data) {
                v = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(float));
        }
    }
};

''' + A)

patch(T, '''    test_cases.emplace_back(new test_dsv4_fuse2_get_rows_cast(  1,   1, 300, 64, 3, true));
''', '''    test_cases.emplace_back(new test_dsv4_fuse2_get_rows_cast(  1,   1, 300, 64, 3, true));

    // HCFUSE: decode and verify shapes fuse; a stream tap, more than 8 tokens and n_embd % 128 != 0 take the unfused path
    for (int64_t n_tokens : {1, 2, 3, 4, 8}) {
        test_cases.emplace_back(new test_dsv4_hcfuse(5120, n_tokens, false, false));
        test_cases.emplace_back(new test_dsv4_hcfuse(5120, n_tokens, true,  false));
    }
    test_cases.emplace_back(new test_dsv4_hcfuse(4224,  3, true,  false));
    test_cases.emplace_back(new test_dsv4_hcfuse(5120,  3, true,  true));
    test_cases.emplace_back(new test_dsv4_hcfuse(5120,  9, true,  false));
    test_cases.emplace_back(new test_dsv4_hcfuse(5120, 64, false, false));
    test_cases.emplace_back(new test_dsv4_hcfuse(4160,  3, false, false));
''')
