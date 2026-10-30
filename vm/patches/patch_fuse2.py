"""Second round of DeepSeek-V4.1 verify-step fusions (hc pre, post and comb, expert sum, F16 get_rows); GGML_CUDA_FUSE_DSV4_2=0 disables them. usage: python3 patch_fuse2.py LLAMA_SRC_DIR"""
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
A = '''void ggml_cuda_op_dsv4_hc_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * s,
        const ggml_tensor * b, ggml_tensor * dst);
'''
patch(H, A, A + '''// DSV4_FUSE2: false when GGML_CUDA_FUSE_DSV4_2=0 or GGML_CUDA_FUSE_DSV4=0
bool ggml_cuda_fuse_dsv4_2_enabled();
// DSV4_FUSE2: gate (SCALE node) -> DSV4_HC_PRE -> RMS_NORM -> MUL; gx, gs, gb are the inputs of the gate chain
bool ggml_cuda_dsv4_hc_pre_norm_supported(const ggml_tensor * gate, const ggml_tensor * pre, const ggml_tensor * norm, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_pre_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * gx, const ggml_tensor * gs, const ggml_tensor * gb,
        const ggml_tensor * gate, const ggml_tensor * pre, const ggml_tensor * norm, ggml_tensor * mul);
// DSV4_FUSE2: gate (SCALE node) and DSV4_HC_COMB in one kernel
bool ggml_cuda_dsv4_hc_comb_gate_supported(const ggml_tensor * gate, const ggml_tensor * comb);
void ggml_cuda_op_dsv4_hc_comb_gate(ggml_backend_cuda_context & ctx, ggml_tensor * comb, const ggml_tensor * gx, const ggml_tensor * gs,
        const ggml_tensor * gb, ggml_tensor * gate);
// DSV4_FUSE2: DSV4_HC_POST whose x is an ADD of two same-shape tensors
bool ggml_cuda_dsv4_hc_post_add_supported(const ggml_tensor * add, const ggml_tensor * post);
void ggml_cuda_op_dsv4_hc_post_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * dst);
''')

# ---- kernels (dsv4-hc.cu) ----
C = 'ggml/src/ggml-cuda/dsv4-hc.cu'
s = open(os.path.join(src, C)).read()
b0 = s.find('    const float scale_comb = scale[2*ss0];\n')
b1 = s.find('\ntemplate <bool gated>\nstatic __global__ void dsv4_hc_pre_f32(')
assert b0 > 0 and b1 > b0, 'dsv4_hc_comb_f32 body not found'
COMB_BODY = s[b0:b1]  # verbatim body of dsv4_hc_comb_f32, closing brace included

A = '''            ne0, n, (int64_t) (x->nb[1] / sizeof(float)), scale, bias);
}
'''
patch(C, A, A + '''
bool ggml_cuda_fuse_dsv4_2_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FUSE_DSV4_2");
        return ggml_cuda_fuse_dsv4_enabled() && (env == nullptr || atoi(env) != 0);
    }();
    return enabled;
}

// DSV4_FUSE2: one block per token; the gate, hc_pre and float4 rms_norm+mul steps repeat the arithmetic of their unfused kernels
template <int block_size, int vpt>
static __global__ void dsv4_hc_pre_norm_f32(
        const float * x,
        const float * gx,
        const float * gs,
        const float * gb,
        const float * w,
        float * dst,
        int ncols,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sgx1,
        float gscale,
        float gbias,
        float pre_scale,
        float eps) {
    ggml_cuda_pdl_lc();
    extern __shared__ float s_sum[];

    const int it  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    float wh[DSV4_HC];
    for (int ih = 0; ih < DSV4_HC; ++ih) {
        const float g = __fadd_rn(__fmul_rn(gx[ih + it*sgx1], gs[0]), gb[ih]);
        wh[ih] = gscale * (1.0f / (1.0f + expf(-g))) + gbias;
    }

    const int ncols4 = ncols / 4;
    float4 xs[vpt];
    float  tmp = 0.0f;

#pragma unroll
    for (int k = 0; k < vpt; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            float v[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int64_t i0 = 4*i + c;
                float sum = 0.0f;
                for (int ih = 0; ih < DSV4_HC; ++ih) {
                    sum += x[i0*sx0 + ih*sx1 + it*sx2] * wh[ih];
                }
                v[c] = pre_scale * sum;
            }
            xs[k] = make_float4(v[0], v[1], v[2], v[3]);
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

    const float4 * w4   = (const float4 *) w;
    float4       * dst4 = (float4 *) (dst + (int64_t) it*ncols);

#pragma unroll
    for (int k = 0; k < vpt; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
            const float4 m = w4[i];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            dst4[i] = r;
        }
    }
}

bool ggml_cuda_dsv4_hc_pre_norm_supported(const ggml_tensor * gate, const ggml_tensor * pre, const ggml_tensor * norm, const ggml_tensor * mul) {
    if (pre->op != GGML_OP_DSV4_HC_PRE || pre->src[1] != gate || ggml_get_op_params_i32(pre, 1) != 0 ||
            norm->op != GGML_OP_RMS_NORM || norm->src[0] != pre || mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm)) {
        return false;
    }
    const ggml_tensor * x = pre->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const int64_t n_embd  = x->ne[0];

    // only rows the unfused rms_norm runs as float4 with 4 values per thread (block 1024), so the sum order is the same
    return x->type == GGML_TYPE_F32 && pre->type == GGML_TYPE_F32 && norm->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        w->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
        x->ne[1] == DSV4_HC && x->ne[3] == 1 && gate->ne[0] == DSV4_HC && gate->ne[1] == x->ne[2] && ggml_is_contiguous(gate) &&
        n_embd % 4 == 0 && n_embd/4 > 1024 && n_embd/4 <= 4096 &&
        ggml_is_contiguous(pre) && ggml_are_same_shape(mul, norm) && ggml_is_contiguous(mul) &&
        w->ne[0] == n_embd && ggml_nrows(w) == 1 && w->nb[0] == sizeof(float) &&
        ((uintptr_t) pre->data) % 16 == 0 && ((uintptr_t) mul->data) % 16 == 0 && ((uintptr_t) w->data) % 16 == 0;
}

void ggml_cuda_op_dsv4_hc_pre_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * gx, const ggml_tensor * gs, const ggml_tensor * gb,
        const ggml_tensor * gate, const ggml_tensor * pre, const ggml_tensor * norm, ggml_tensor * mul) {
    GGML_ASSERT(ggml_cuda_dsv4_hc_pre_norm_supported(gate, pre, norm, mul));
    GGML_ASSERT(gx->type == GGML_TYPE_F32 && gs->type == GGML_TYPE_F32 && gb->type == GGML_TYPE_F32);
    GGML_ASSERT(gx->nb[0] == sizeof(float) && gx->nb[1] % sizeof(float) == 0 && ggml_nelements(gs) == 1 && gb->nb[0] == sizeof(float));

    const ggml_tensor * x = pre->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));

    constexpr int block_size = 1024;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims(x->ne[2], 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 32*sizeof(float), ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_norm_f32<block_size, 4>, launch_params,
            (const float *) x->data, (const float *) gx->data, (const float *) gs->data, (const float *) gb->data,
            (const float *) w->data, (float *) mul->data, (int) x->ne[0],
            (int64_t) (x->nb[0] / sizeof(float)), (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)),
            (int64_t) (gx->nb[1] / sizeof(float)),
            ggml_get_op_params_f32(gate, 0), ggml_get_op_params_f32(gate, 1), ggml_get_op_params_f32(pre, 0), eps);
}

// DSV4_FUSE2: dsv4_hc_comb_f32 (body copied verbatim) that first writes the post weights of its token as dsv4_hc_gate_f32 does
static __global__ void dsv4_hc_comb_gate_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter,
        const float * gx,
        const float * gs,
        const float * gb,
        float * gdst,
        int64_t gne0,
        int64_t sgx1,
        float gscale,
        float gbias) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    for (int64_t j = 0; j < gne0; ++j) {
        const float g = __fadd_rn(__fmul_rn(gx[j + it*sgx1], gs[0]), gb[j]);
        gdst[j + it*gne0] = gscale * (1.0f / (1.0f + expf(-g))) + gbias;
    }

''' + COMB_BODY + '''
bool ggml_cuda_dsv4_hc_comb_gate_supported(const ggml_tensor * gate, const ggml_tensor * comb) {
    if (comb->op != GGML_OP_DSV4_HC_COMB || (comb->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            comb->src[0] == gate || comb->src[1] == gate || comb->src[2] == gate) {
        return false;
    }
    return comb->type == GGML_TYPE_F32 && comb->src[0]->type == GGML_TYPE_F32 && comb->src[1]->type == GGML_TYPE_F32 &&
        comb->src[2]->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) &&
        gate->ne[2] == 1 && gate->ne[3] == 1 && gate->ne[1] == comb->src[0]->ne[1];
}

void ggml_cuda_op_dsv4_hc_comb_gate(ggml_backend_cuda_context & ctx, ggml_tensor * comb, const ggml_tensor * gx, const ggml_tensor * gs,
        const ggml_tensor * gb, ggml_tensor * gate) {
    const ggml_tensor * mixes = comb->src[0];
    const ggml_tensor * scale = comb->src[1];
    const ggml_tensor * base  = comb->src[2];

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(ggml_cuda_dsv4_hc_comb_gate_supported(gate, comb));
    GGML_ASSERT(mixes->ne[0] == hc_mix_dim && comb->ne[0] == DSV4_HC && comb->ne[1] == DSV4_HC && comb->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3 && base->ne[0] == hc_mix_dim);
    GGML_ASSERT(gx->type == GGML_TYPE_F32 && gs->type == GGML_TYPE_F32 && gb->type == GGML_TYPE_F32);
    GGML_ASSERT(gx->nb[0] == sizeof(float) && gx->nb[1] % sizeof(float) == 0 && ggml_nelements(gs) == 1 && gb->nb[0] == sizeof(float));

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, comb,  nb);

    const int64_t n_tokens = mixes->ne[1];
    const float   eps      = ggml_get_op_params_f32(comb, 0);
    const int32_t n_iter   = ggml_get_op_params_i32(comb, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_gate_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) comb->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter,
            (const float *) gx->data, (const float *) gs->data, (const float *) gb->data, (float *) gate->data,
            gate->ne[0], (int64_t) (gx->nb[1] / sizeof(float)), ggml_get_op_params_f32(gate, 0), ggml_get_op_params_f32(gate, 1));
}

// DSV4_FUSE2: dsv4_hc_post_f32<true, false> with x = a + b rounded as the unfused ADD
static __global__ void dsv4_hc_post_add_f32(
        const float * xa,
        const float * xb,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sa0,
        int64_t sa1,
        int64_t sb0,
        int64_t sb1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    const float p = post[idst*sp0 + it*sp1];

    float sum = __fadd_rn(xa[i0*sa0 + it*sa1], xb[i0*sb0 + it*sb1]) * p;
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

bool ggml_cuda_dsv4_hc_post_add_supported(const ggml_tensor * add, const ggml_tensor * post) {
    if (add->op != GGML_OP_ADD || post->op != GGML_OP_DSV4_HC_POST || (post->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            post->src[0] != add || post->src[3] == nullptr || post->src[1] == add || post->src[2] == add || post->src[3] == add) {
        return false;
    }
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    return add->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(a, add) && ggml_are_same_shape(b, add) && a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) &&
        post->type == GGML_TYPE_F32 && post->src[1]->type == GGML_TYPE_F32 && post->src[2]->type == GGML_TYPE_F32 &&
        post->src[3]->type == GGML_TYPE_F32;
}

void ggml_cuda_op_dsv4_hc_post_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv4_hc_post_add_supported(add, dst));

    const ggml_tensor * a        = add->src[0];
    const ggml_tensor * b        = add->src[1];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_TENSOR_LOCALS(size_t, nba, a,        nb);
    GGML_TENSOR_LOCALS(size_t, nbb, b,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_add_f32, launch_params,
            (const float *) a->data, (const float *) b->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nba0 / sizeof(float), nba1 / sizeof(float),
            nbb0 / sizeof(float), nbb1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
''')

# ---- matchers, dispatch and allocation deps (ggml-cuda.cu) ----
C = 'ggml/src/ggml-cuda/ggml-cuda.cu'
A = '''// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
'''
patch(C, A, '''// DSV4_FUSE2: GET_ROWS (F16 -> F32) -> RESHAPE -> ggml_cast to F16, done as one exact F16 -> F16 get_rows
static bool ggml_cuda_dsv4_get_rows_cast_supported(const ggml_tensor * get_rows, const ggml_tensor * reshape, const ggml_tensor * cpy) {
    if (get_rows->op != GGML_OP_GET_ROWS || reshape->op != GGML_OP_RESHAPE || cpy->op != GGML_OP_CPY ||
            reshape->src[0] != get_rows || cpy->src[0] != reshape || cpy->src[1] != cpy) {
        return false;
    }
    const ggml_tensor * src0 = get_rows->src[0];
    const ggml_tensor * src1 = get_rows->src[1];
    return src0->type == GGML_TYPE_F16 && get_rows->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && cpy->type == GGML_TYPE_F16 &&
        src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) && src1->ne[3] == 1 &&
        ggml_is_contiguous(get_rows) && ggml_is_contiguous(cpy) && ggml_nelements(cpy) == ggml_nelements(get_rows);
}

static void ggml_cuda_op_dsv4_get_rows_cast(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, ggml_tensor * cpy) {
    const ggml_tensor * src0 = get_rows->src[0];
    const ggml_tensor * src1 = get_rows->src[1];

    // dst strides of the get_rows shape in the F16 output, which has the same contiguous element order
    const size_t nb1 = get_rows->ne[0]*ggml_type_size(cpy->type);
    const size_t nb2 = get_rows->ne[1]*nb1;
    const size_t nb3 = get_rows->ne[2]*nb2;

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, cpy->data, cpy->type,
        src0->ne[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->nb[0], src1->nb[1], src1->nb[2],
        nb1, nb2, nb3, ctx.stream());
}

static bool ggml_cuda_dsv4_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return a0 < b1 && b0 < a1;
}

static ggml_tensor * ggml_cuda_dsv4_view_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return const_cast<ggml_tensor *>(t);
}

// DSV4_FUSE2: the gate chain at i followed by DSV4_HC_PRE -> RMS_NORM -> MUL; returns the index of the MUL, 0 if no match
static int ggml_cuda_match_dsv4_hc_pre_norm(const ggml_cgraph * cgraph, int i, const ggml_tensor ** x, const ggml_tensor ** s, const ggml_tensor ** b) {
    const int n_gate = ggml_cuda_match_dsv4_hc_gate(cgraph, i, x, s, b);
    const int i_gate = i + n_gate;
    if (n_gate == 0 || i_gate + 3 >= cgraph->n_nodes) {
        return 0;
    }
    for (int k = 1; k <= 3; ++k) {
        if ((cgraph->nodes[i_gate + k]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
    }
    if (!ggml_node_has_n_uses(cgraph, i_gate, 1) || !ggml_node_has_n_uses(cgraph, i_gate + 1, 1) || !ggml_node_has_n_uses(cgraph, i_gate + 2, 1)) {
        return 0;
    }
    if (!ggml_cuda_dsv4_hc_pre_norm_supported(cgraph->nodes[i_gate], cgraph->nodes[i_gate + 1], cgraph->nodes[i_gate + 2], cgraph->nodes[i_gate + 3])) {
        return 0;
    }
    return i_gate + 3;
}

''' + A)

patch(C, '''    ggml_tensor * node = cgraph->nodes[i];

    if (ggml_cuda_fuse_dsv4_enabled()) {
''', '''    ggml_tensor * node = cgraph->nodes[i];

    if (ggml_cuda_fuse_dsv4_2_enabled()) {
        // DSV4_FUSE2: gate -> hc_pre -> rms_norm -> mul, gate + hc_comb, add -> hc_post, get_rows -> reshape -> cast
        if (node->op == GGML_OP_MUL) {
            const ggml_tensor * x = nullptr;
            const ggml_tensor * s = nullptr;
            const ggml_tensor * b = nullptr;
            const int out_idx = ggml_cuda_match_dsv4_hc_pre_norm(cgraph, i, &x, &s, &b);
            if (out_idx > 0 && ggml_cuda_check_fusion_memory_ranges(cgraph, i, out_idx - i + 1, &out_idx, 1)) {
                ggml_cuda_op_dsv4_hc_pre_norm(*cuda_ctx, x, s, b, cgraph->nodes[out_idx - 3], cgraph->nodes[out_idx - 2],
                        cgraph->nodes[out_idx - 1], cgraph->nodes[out_idx]);
                return out_idx - i;
            }
            const int i_gate = i + ggml_cuda_match_dsv4_hc_gate(cgraph, i, &x, &s, &b);
            if (i_gate > i && i_gate + 1 < cgraph->n_nodes && ggml_cuda_dsv4_hc_comb_gate_supported(cgraph->nodes[i_gate], cgraph->nodes[i_gate + 1])) {
                const int outs[2] = { i_gate, i_gate + 1 };
                if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, i_gate - i + 2, outs, 2)) {
                    ggml_cuda_op_dsv4_hc_comb_gate(*cuda_ctx, cgraph->nodes[i_gate + 1], x, s, b, cgraph->nodes[i_gate]);
                    return i_gate - i + 1;
                }
            }
        }
        if (node->op == GGML_OP_ADD && i + 1 < cgraph->n_nodes && ggml_node_has_n_uses(cgraph, i, 1) &&
                ggml_cuda_dsv4_hc_post_add_supported(node, cgraph->nodes[i + 1])) {
            const int out_idx = i + 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, &out_idx, 1)) {
                ggml_cuda_op_dsv4_hc_post_add(*cuda_ctx, node, cgraph->nodes[out_idx]);
                return 1;
            }
        }
        if (node->op == GGML_OP_GET_ROWS && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_CPY }, { i + 2 }) &&
                ggml_cuda_dsv4_get_rows_cast_supported(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) &&
                !ggml_cuda_dsv4_overlap(cgraph->nodes[i + 2], node->src[0]) && !ggml_cuda_dsv4_overlap(cgraph->nodes[i + 2], node->src[1])) {
            ggml_cuda_op_dsv4_get_rows_cast(*cuda_ctx, node, cgraph->nodes[i + 2]);
            return 2;
        }
    }

    if (ggml_cuda_fuse_dsv4_enabled()) {
''')

A = '''#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
'''
patch(C, A, '''    if (!disable_fusion && ggml_cuda_fuse_dsv4_2_enabled()) {
        // DSV4_FUSE2: the fused kernels read these inputs after the node where ggml-alloc would free them
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_MUL) {
                const ggml_tensor * x = nullptr;
                const ggml_tensor * s = nullptr;
                const ggml_tensor * b = nullptr;
                const int out_idx = ggml_cuda_match_dsv4_hc_pre_norm(cgraph, i, &x, &s, &b);
                if (out_idx > 0) {
                    params->add_alloc_dep(params->user_data, ggml_cuda_dsv4_view_root(x), cgraph->nodes[out_idx]);
                }
            } else if (node->op == GGML_OP_ADD && i + 1 < cgraph->n_nodes && ggml_cuda_dsv4_hc_post_add_supported(node, cgraph->nodes[i + 1])) {
                params->add_alloc_dep(params->user_data, ggml_cuda_dsv4_view_root(node->src[0]), cgraph->nodes[i + 1]);
                params->add_alloc_dep(params->user_data, ggml_cuda_dsv4_view_root(node->src[1]), cgraph->nodes[i + 1]);
            } else if (node->op == GGML_OP_GET_ROWS && i + 2 < cgraph->n_nodes &&
                    ggml_cuda_dsv4_get_rows_cast_supported(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
                params->add_alloc_dep(params->user_data, ggml_cuda_dsv4_view_root(node->src[1]), cgraph->nodes[i + 2]);
            }
        }
    }

''' + A)

# ---- tests (test-backend-ops.cpp) ----
T = 'tests/test-backend-ops.cpp'
A = '''        ggml_tensor * out = ggml_concat(ctx, rep, b, dim);
        ggml_set_name(out, "out");
        return out;
    }
};
'''
patch(T, A, A + '''
// DSV4_FUSE2: hc pre gate -> DSV4_HC_PRE -> RMS_NORM -> MUL, the attention and ffn input of a dsv4 layer
struct test_dsv4_fuse2_hc_pre_norm : public test_dsv4_hc {
    const int64_t n_embd;
    const int64_t n_tokens;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_FUSE2_HC_PRE_NORM";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(n_embd, n_tokens);
    }

    test_dsv4_fuse2_hc_pre_norm(int64_t n_embd = 5120, int64_t n_tokens = 3)
        : n_embd(n_embd), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc)*hc, n_tokens);
        ggml_set_name(mixes, "mixes");
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(scale, "scale");
        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc)*hc);
        ggml_set_name(base, "base");
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(x, "x");
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_set_name(w, "w");

        ggml_tensor * pre = ggml_view_2d(ctx, mixes, hc, n_tokens, mixes->nb[1], 0);
        pre = ggml_add(ctx, ggml_mul(ctx, pre, ggml_view_1d(ctx, scale, 1, 0)), ggml_view_1d(ctx, base, hc, 0));
        pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, pre), 1.0f, 1e-6f);

        out = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_dsv4_hc_pre(ctx, x, pre), 1e-6f), w);
        ggml_set_name(out, "out");
        return out;
    }
};

// DSV4_FUSE2: hc post gate and DSV4_HC_COMB feeding DSV4_HC_POST, the attention output of a dsv4 layer
struct test_dsv4_fuse2_hc_comb_gate : public test_dsv4_hc {
    const int64_t n_embd;
    const int64_t n_tokens;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_FUSE2_HC_COMB_GATE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(n_embd, n_tokens);
    }

    test_dsv4_fuse2_hc_comb_gate(int64_t n_embd = 5120, int64_t n_tokens = 3)
        : n_embd(n_embd), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc)*hc, n_tokens);
        ggml_set_name(mixes, "mixes");
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(scale, "scale");
        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc)*hc);
        ggml_set_name(base, "base");
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(x, "x");
        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(residual, "residual");

        ggml_tensor * post = ggml_view_2d(ctx, mixes, hc, n_tokens, mixes->nb[1], hc*sizeof(float));
        post = ggml_add(ctx, ggml_mul(ctx, post, ggml_view_1d(ctx, scale, 1, sizeof(float))), ggml_view_1d(ctx, base, hc, hc*sizeof(float)));
        post = ggml_scale(ctx, ggml_sigmoid(ctx, post), 2.0f);

        ggml_tensor * comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, 1e-6f, 20);

        out = ggml_dsv4_hc_post(ctx, x, residual, post, comb);
        ggml_set_name(out, "out");
        return out;
    }
};

// DSV4_FUSE2: ADD -> DSV4_HC_POST, the ffn output of a dsv4 layer (routed + shared experts)
struct test_dsv4_fuse2_hc_post_add : public test_dsv4_hc {
    const int64_t n_embd;
    const int64_t n_tokens;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_FUSE2_HC_POST_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(n_embd, n_tokens);
    }

    test_dsv4_fuse2_hc_post_add(int64_t n_embd = 5120, int64_t n_tokens = 3)
        : n_embd(n_embd), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(x, "x");
        ggml_tensor * xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(xb, "xb");
        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(residual, "residual");
        ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(post, "post");
        ggml_tensor * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);
        ggml_set_name(comb, "comb");

        out = ggml_dsv4_hc_post(ctx, ggml_add(ctx, x, xb), residual, post, comb);
        ggml_set_name(out, "out");
        return out;
    }
};

// DSV4_FUSE2: GET_ROWS (F16) -> RESHAPE -> cast to F16, the dsv4 gather of the selected compressed rows (mask: of their kq mask)
struct test_dsv4_fuse2_get_rows_cast : public test_case {
    const int64_t ne0;
    const int64_t row_len; // row stride of the source in elements, > ne0 for a strided view
    const int64_t n_rows;
    const int64_t k;
    const int64_t nt;
    const bool    mask;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_FUSE2_GET_ROWS_CAST";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR6(ne0, row_len, n_rows, k, nt, mask);
    }

    test_dsv4_fuse2_get_rows_cast(int64_t ne0 = 512, int64_t row_len = 640, int64_t n_rows = 300, int64_t k = 64, int64_t nt = 3, bool mask = false)
        : ne0(ne0), row_len(row_len), n_rows(n_rows), k(k), nt(nt), mask(mask) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out;
        if (mask) {
            ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, n_rows, nt);
            ggml_set_name(a, "a");
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, nt);
            ggml_set_name(ids, "ids");
            out = ggml_cast(ctx, ggml_reshape_4d(ctx, ggml_get_rows(ctx, a, ids), k, 1, 1, nt), GGML_TYPE_F16);
        } else {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, row_len, n_rows);
            ggml_set_name(a, "a");
            if (row_len != ne0) {
                a = ggml_view_2d(ctx, a, ne0, n_rows, a->nb[1], 0);
                ggml_set_name(a, "view of a");
            }
            ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, k*nt);
            ggml_set_name(ids, "ids");
            out = ggml_cast(ctx, ggml_reshape_4d(ctx, ggml_get_rows(ctx, a, ids), ne0, 1, k, nt), GGML_TYPE_F16);
        }
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op)) {
                continue;
            }
            if (t->type == GGML_TYPE_I32) {
                std::vector<int> data(ggml_nelements(t));
                for (size_t i = 0; i < data.size(); i++) {
                    data[i] = rand() % n_rows;
                }
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};
''')

patch(T, '''    test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {20480, 3, 2, 1}, 1e-6f, false, false, false, false, true));
''', '''    test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {20480, 3, 2, 1}, 1e-6f, false, false, false, false, true));

    // DSV4_FUSE2: fused shapes, then rows the hc_pre_norm matcher must reject (4096: the unfused rms_norm uses another float4 split)
    for (int64_t n_tokens : {1, 3, 17}) {
        test_cases.emplace_back(new test_dsv4_fuse2_hc_pre_norm(5120, n_tokens));
        test_cases.emplace_back(new test_dsv4_fuse2_hc_comb_gate(5120, n_tokens));
        test_cases.emplace_back(new test_dsv4_fuse2_hc_post_add(5120, n_tokens));
    }
    test_cases.emplace_back(new test_dsv4_fuse2_hc_pre_norm(4100, 3));
    test_cases.emplace_back(new test_dsv4_fuse2_hc_pre_norm(5120, 3072));
    test_cases.emplace_back(new test_dsv4_fuse2_hc_pre_norm(4096, 3));
    test_cases.emplace_back(new test_dsv4_fuse2_get_rows_cast(512, 640, 300, 64, 3, false));
    test_cases.emplace_back(new test_dsv4_fuse2_get_rows_cast(512, 512,  37,  9, 1, false));
    test_cases.emplace_back(new test_dsv4_fuse2_get_rows_cast(  1,   1, 300, 64, 3, true));
''')
