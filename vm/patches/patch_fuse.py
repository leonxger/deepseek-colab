"""CUDA fusions for the DeepSeek-V4.1 verify step (hc weights, gather-path keys, wide rms_norm); GGML_CUDA_FUSE_DSV4=0 disables them. usage: python3 patch_fuse.py LLAMA_SRC_DIR"""
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


# ---- kill switch and the hc gate kernel (dsv4-hc.cu) ----
H = 'ggml/src/ggml-cuda/dsv4-hc.cuh'
A = '''void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);
'''
patch(H, A, A + '''// DSV4_FUSE: false when GGML_CUDA_FUSE_DSV4=0
bool ggml_cuda_fuse_dsv4_enabled();
// DSV4_FUSE: dst = SCALE(SIGMOID(x*s + b)) with s a scalar and b a row broadcast over the rows of x
void ggml_cuda_op_dsv4_hc_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * s,
        const ggml_tensor * b, ggml_tensor * dst);
''')

C = 'ggml/src/ggml-cuda/dsv4-hc.cu'
A = '''            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
'''
patch(C, A, A + '''
bool ggml_cuda_fuse_dsv4_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FUSE_DSV4");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// DSV4_FUSE: __fmul_rn/__fadd_rn keep the two roundings of the unfused MUL and ADD (no FMA contraction)
static __global__ void dsv4_hc_gate_f32(
        const float * x,
        const float * s,
        const float * b,
        float * dst,
        int64_t ne0,
        int64_t n,
        int64_t sx1,
        float scale,
        float bias) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = i % ne0;
    const int64_t i1 = i / ne0;

    const float v = __fadd_rn(__fmul_rn(x[i0 + i1*sx1], s[0]), b[i0]);
    dst[i] = scale * (1.0f / (1.0f + expf(-v))) + bias;
}

void ggml_cuda_op_dsv4_hc_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * s,
        const ggml_tensor * b, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && x->nb[1] % sizeof(float) == 0 && b->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_are_same_shape(x, dst) && ggml_is_contiguous(dst) && ggml_nelements(s) == 1 && b->ne[0] == dst->ne[0]);

    const int64_t ne0 = dst->ne[0];
    const int64_t n   = ggml_nelements(dst);

    const float scale = ggml_get_op_params_f32(dst, 0);
    const float bias  = ggml_get_op_params_f32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_gate_f32, launch_params,
            (const float *) x->data, (const float *) s->data, (const float *) b->data, (float *) dst->data,
            ne0, n, (int64_t) (x->nb[1] / sizeof(float)), scale, bias);
}
''')

# ---- REPEAT -> CONCAT (concat.cu) ----
H = 'ggml/src/ggml-cuda/concat.cuh'
A = 'void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);\n'
patch(H, A, A + '''
// DSV4_FUSE: concat(repeat(a), b, dim) in one pass for dim 1..3 when a is not repeated along dim 0
bool ggml_cuda_repeat_concat_supported(const ggml_tensor * repeat, const ggml_tensor * concat);
void ggml_cuda_op_repeat_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * repeat, ggml_tensor * dst);
''')

C = 'ggml/src/ggml-cuda/concat.cu'
A = '''                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
        }
    }
}
'''
patch(C, A, A + '''
// DSV4_FUSE: one dst row per threadIdx.y; rows before split along dim come from the repeated src0, the rest from src1
template <typename T>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) repeat_concat_rows(
        const char *   src0,
        const char *   src1,
              char *   dst,
        const int      dim,
        const uint32_t split,
        const uint32_t nrows,
        const int      nchunks,
        const uint3    ne1,
        const uint3    ne2,
        const uint3    ne01,
        const uint3    ne02,
        const uint3    ne03,
        const size_t   nb01,
        const size_t   nb02,
        const size_t   nb03,
        const size_t   nb11,
        const size_t   nb12,
        const size_t   nb13,
        const size_t   nb1) {
    ggml_cuda_pdl_lc();
    const uint32_t row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    ggml_cuda_pdl_sync();

    const uint2 d1 = fast_div_modulo(row,  ne1); // <i2 + i3*ne2, i1>
    const uint2 d2 = fast_div_modulo(d1.x, ne2); // <i3, i2>

    const uint32_t i1 = d1.y;
    const uint32_t i2 = d2.y;
    const uint32_t i3 = d2.x;
    const uint32_t ic = dim == 1 ? i1 : (dim == 2 ? i2 : i3);

    const T * x;
    if (ic < split) {
        x = (const T *) (src0 + fastmodulo(i1, ne01)*nb01 + fastmodulo(i2, ne02)*nb02 + fastmodulo(i3, ne03)*nb03);
    } else {
        x = (const T *) (src1 + (i1 - (dim == 1 ? split : 0))*nb11 + (i2 - (dim == 2 ? split : 0))*nb12 + (i3 - (dim == 3 ? split : 0))*nb13);
    }
    T * y = (T *) (dst + row*nb1);

    for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
        y[c] = x[c];
    }
}

bool ggml_cuda_repeat_concat_supported(const ggml_tensor * repeat, const ggml_tensor * concat) {
    if (repeat->op != GGML_OP_REPEAT || concat->op != GGML_OP_CONCAT || concat->src[0] != repeat || concat->src[1] == repeat) {
        return false;
    }
    const ggml_tensor * src0 = repeat->src[0];
    const ggml_tensor * src1 = concat->src[1];
    const int32_t       dim  = ((const int32_t *) concat->op_params)[0];
    const size_t        ts   = ggml_type_size(concat->type);

    return dim >= 1 && dim <= 3 && ggml_blck_size(concat->type) == 1 &&
        src0->type == concat->type && repeat->type == concat->type && src1->type == concat->type &&
        src0->ne[0] == concat->ne[0] && src0->nb[0] == ts && src1->nb[0] == ts &&
        ggml_is_contiguous(concat) && ggml_nrows(concat) <= INT32_MAX;
}

void ggml_cuda_op_repeat_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * repeat, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_repeat_concat_supported(repeat, dst));

    const ggml_tensor * src0 = repeat->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int32_t       dim  = ((const int32_t *) dst->op_params)[0];

    const size_t   row_bytes = ggml_row_size(dst->type, dst->ne[0]);
    const uint32_t nrows     = (uint32_t) ggml_nrows(dst);

    // widest copy unit that divides the row size and every base pointer and stride
    const size_t align = row_bytes | (uintptr_t) src0->data | (uintptr_t) src1->data | (uintptr_t) dst->data |
        src0->nb[1] | src0->nb[2] | src0->nb[3] | src1->nb[1] | src1->nb[2] | src1->nb[3];
    const size_t width   = align % 16 == 0 ? 16 : (align % 4 == 0 ? 4 : (align % 2 == 0 ? 2 : 1));
    const int    nchunks = (int) (row_bytes / width);

    const int  bx = nchunks >= CUDA_CONCAT_BLOCK_SIZE ? CUDA_CONCAT_BLOCK_SIZE : std::max(WARP_SIZE, (nchunks + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 block_dims(bx, CUDA_CONCAT_BLOCK_SIZE / bx, 1);
    const dim3 grid_dims((nrows + block_dims.y - 1) / block_dims.y, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    const uint3 ne1  = init_fastdiv_values(dst->ne[1]);
    const uint3 ne2  = init_fastdiv_values(dst->ne[2]);
    const uint3 ne01 = init_fastdiv_values(src0->ne[1]);
    const uint3 ne02 = init_fastdiv_values(src0->ne[2]);
    const uint3 ne03 = init_fastdiv_values(src0->ne[3]);

    auto launch = [&](auto unit) {
        using T = decltype(unit);
        ggml_cuda_kernel_launch(repeat_concat_rows<T>, launch_params,
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                (int) dim, (uint32_t) repeat->ne[dim], nrows, nchunks,
                ne1, ne2, ne01, ne02, ne03,
                src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3], dst->nb[1]);
    };
    switch (width) {
        case 16:
            launch(int4{});
            break;
        case 4:
            launch(int{});
            break;
        case 2:
            launch(uint16_t{});
            break;
        default:
            launch(uint8_t{});
            break;
    }
}
''')

# ---- rms_norm: float4 path for 4096 < ncols/4 <= 5120 (norm.cu) ----
C = 'ggml/src/ggml-cuda/norm.cu'
patch(C, '#include "norm.cuh"\n', '#include "norm.cuh"\n#include "dsv4-hc.cuh" // DSV4_FUSE: ggml_cuda_fuse_dsv4_enabled\n')
patch(C, '''        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
''', '''        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else if (n4 <= 5120 && ggml_cuda_fuse_dsv4_enabled()) {
            RMS_NORM_LAUNCH(1024, 5, true); // DSV4_FUSE: the hc mixes norm over 4*5120 columns
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
''')

# ---- matcher and dispatch (ggml-cuda.cu) ----
C = 'ggml/src/ggml-cuda/ggml-cuda.cu'
A = '''// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
'''
patch(C, A, '''// DSV4_FUSE: MUL(x, scalar) -> [VIEW of the bias] -> ADD(row bias) -> SIGMOID -> SCALE; returns the nodes to skip, 0 if no match
static int ggml_cuda_match_dsv4_hc_gate(const ggml_cgraph * cgraph, int i, const ggml_tensor ** x, const ggml_tensor ** s, const ggml_tensor ** b) {
    const int j = (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_VIEW) ? i + 2 : i + 1;
    if (j + 2 >= cgraph->n_nodes) {
        return 0;
    }

    const ggml_tensor * mul = cgraph->nodes[i];
    const ggml_tensor * add = cgraph->nodes[j];
    const ggml_tensor * sig = cgraph->nodes[j + 1];
    const ggml_tensor * scl = cgraph->nodes[j + 2];

    if (mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD || sig->op != GGML_OP_UNARY || scl->op != GGML_OP_SCALE ||
            ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sig->src[0] != add || scl->src[0] != sig ||
            (add->src[0] != mul && add->src[1] != mul)) {
        return 0;
    }
    for (const ggml_tensor * t : { mul, add, sig, scl }) {
        if (t->type != GGML_TYPE_F32 || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
    }
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, j, 1) || !ggml_node_has_n_uses(cgraph, j + 1, 1)) {
        return 0;
    }

    // a skipped VIEW must be the bias operand: views are no-ops, skipping one changes nothing else
    const ggml_tensor * bias = add->src[0] == mul ? add->src[1] : add->src[0];
    if (j == i + 2 && cgraph->nodes[i + 1] != bias) {
        return 0;
    }
    const ggml_tensor * xt = ggml_are_same_shape(mul->src[0], mul) ? mul->src[0] : mul->src[1];
    const ggml_tensor * st = xt == mul->src[0] ? mul->src[1] : mul->src[0];

    const bool ok = mul->ne[2] == 1 && mul->ne[3] == 1 && ggml_are_same_shape(mul, add) && ggml_are_same_shape(mul, scl) &&
        ggml_is_contiguous(scl) && ggml_are_same_shape(xt, mul) && xt->type == GGML_TYPE_F32 &&
        xt->nb[0] == sizeof(float) && xt->nb[1] % sizeof(float) == 0 &&
        st->type == GGML_TYPE_F32 && ggml_nelements(st) == 1 &&
        bias->type == GGML_TYPE_F32 && bias->ne[0] == mul->ne[0] && ggml_nrows(bias) == 1 && bias->nb[0] == sizeof(float);
    if (!ok) {
        return 0;
    }

    *x = xt;
    *s = st;
    *b = bias;
    return j + 2 - i;
}

''' + A)

patch(C, '''    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
''', '''    ggml_tensor * node = cgraph->nodes[i];

    if (ggml_cuda_fuse_dsv4_enabled()) {
        // DSV4_FUSE: hc pre/post weight chain, and the gather-path window REPEAT feeding the key CONCAT
        if (node->op == GGML_OP_MUL) {
            const ggml_tensor * x = nullptr;
            const ggml_tensor * s = nullptr;
            const ggml_tensor * b = nullptr;
            const int n_skip  = ggml_cuda_match_dsv4_hc_gate(cgraph, i, &x, &s, &b);
            const int out_idx = i + n_skip;
            if (n_skip > 0 && ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_skip + 1, &out_idx, 1)) {
                ggml_cuda_op_dsv4_hc_gate(*cuda_ctx, x, s, b, cgraph->nodes[out_idx]);
                return n_skip;
            }
        }
        if (node->op == GGML_OP_REPEAT && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_REPEAT, GGML_OP_CONCAT }, { i + 1 }) &&
                ggml_cuda_repeat_concat_supported(node, cgraph->nodes[i + 1])) {
            const int out_idx = i + 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, &out_idx, 1)) {
                ggml_cuda_op_repeat_concat(*cuda_ctx, node, cgraph->nodes[out_idx]);
                return 1;
            }
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
''')

# ---- graph order: comp_k before the window REPEAT (deepseek41.cpp) ----
patch('src/models/deepseek41.cpp', '''                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type); // exact: rows are raw_k->type values
                }
                cb(comp_k, "comp_k_sel", il);
            }
''', '''                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type); // exact: rows are raw_k->type values
                }
                cb(comp_k, "comp_k_sel", il);
                // DSV4_FUSE: visit comp_k before the window REPEAT so the CUDA REPEAT -> CONCAT fusion finds both nodes adjacent
                static const bool fuse_dsv4 = dsv41_env_on("GGML_CUDA_FUSE_DSV4");
                if (fuse_dsv4) {
                    ggml_build_forward_expand(gf, comp_k);
                }
            }
''')

# ---- tests (test-backend-ops.cpp) ----
T = 'tests/test-backend-ops.cpp'
A = '''        out = ggml_dsv4_hc_post(ctx, x, residual, post, comb);
        ggml_set_name(out, "out");
        return out;
    }
};
'''
patch(T, A, A + '''
// DSV4_FUSE: MUL -> VIEW -> ADD -> SIGMOID -> SCALE as build_hc_mixes builds the hc pre (sigmoid + eps) and post (2*sigmoid) weights
struct test_dsv4_fuse_hc_gate : public test_dsv4_hc {
    const int64_t n_tokens;
    const bool    post;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_FUSE_HC_GATE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(n_tokens, post);
    }

    test_dsv4_fuse_hc_gate(int64_t n_tokens = 3, bool post = false)
        : n_tokens(n_tokens), post(post) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc)*hc, n_tokens);
        ggml_set_name(mixes, "mixes");

        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(scale, "scale");

        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc)*hc);
        ggml_set_name(base, "base");

        const int64_t off = post ? hc : 0;
        ggml_tensor * x = ggml_view_2d(ctx, mixes, hc, n_tokens, mixes->nb[1], off*sizeof(float));
        ggml_tensor * s = ggml_view_1d(ctx, scale, 1, (post ? 1 : 0)*sizeof(float));
        ggml_tensor * b = ggml_view_1d(ctx, base, hc, off*sizeof(float));

        out = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul(ctx, x, s), b));
        out = post ? ggml_scale(ctx, out, 2.0f) : ggml_scale_bias(ctx, out, 1.0f, 1e-6f);
        ggml_set_name(out, "out");
        return out;
    }
};

// DSV4_FUSE: REPEAT -> CONCAT as the dsv4 gather path builds its keys (window repeated per query, then the selected rows)
struct test_dsv4_fuse_repeat_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> nr;
    const int64_t ne_b; // src1 extent along dim
    const int dim;
    const bool v; // whether a is a strided view, as the KV cache provides it

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_FUSE_REPEAT_CONCAT";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR6(type, ne_a, nr, ne_b, dim, v);
    }

    test_dsv4_fuse_repeat_concat(ggml_type type = GGML_TYPE_F16,
            std::array<int64_t, 4> ne_a = {512, 1, 37, 1},
            std::array<int64_t, 4> nr = {1, 1, 1, 3},
            int64_t ne_b = 9, int dim = 2, bool v = true)
        : type(type), ne_a(ne_a), nr(nr), ne_b(ne_b), dim(dim), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne_a[0], ne_a[1], ne_a[2] + (v ? 2 : 0), ne_a[3]);
        ggml_set_name(a, "a");
        if (v) {
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], a->nb[2]);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * rep = ggml_repeat_4d(ctx, a, ne_a[0]*nr[0], ne_a[1]*nr[1], ne_a[2]*nr[2], ne_a[3]*nr[3]);

        std::array<int64_t, 4> ne1 = { rep->ne[0], rep->ne[1], rep->ne[2], rep->ne[3] };
        ne1[dim] = ne_b;
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne1.data());
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_concat(ctx, rep, b, dim);
        ggml_set_name(out, "out");
        return out;
    }
};
''')

patch(T, '''    test_cases.emplace_back(new test_dsv4_hc_post(31, 17, false, true));
''', '''    test_cases.emplace_back(new test_dsv4_hc_post(31, 17, false, true));

    // DSV4_FUSE: fused shapes first, then shapes the CUDA matcher must reject (dim 0, repeat along dim 0)
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F16, {512, 1,   37, 1}, {1, 1, 1, 3},   9, 2, true));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F16, {512, 1, 2304, 1}, {1, 1, 1, 3}, 512, 2, true));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F16, {  7, 1,    5, 1}, {1, 1, 1, 3},   4, 2, true));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F32, { 10, 2,    3, 2}, {1, 2, 2, 1},   5, 1, false));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F32, { 10, 2,    3, 2}, {1, 2, 2, 1},   5, 2, false));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F32, { 10, 2,    3, 2}, {1, 2, 2, 2},   5, 3, false));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F32, { 10, 2,    3, 1}, {1, 1, 1, 2},   4, 0, false));
    test_cases.emplace_back(new test_dsv4_fuse_repeat_concat(GGML_TYPE_F32, {  4, 2,    3, 1}, {2, 1, 1, 2},   5, 2, false));
    for (int64_t n_tokens : {1, 3, 17, 3072}) {
        test_cases.emplace_back(new test_dsv4_fuse_hc_gate(n_tokens, false));
        test_cases.emplace_back(new test_dsv4_fuse_hc_gate(n_tokens, true));
    }
    test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {20480, 3, 1, 1}));
    test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {16388, 3, 2, 1}));
    test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {20480, 3, 2, 1}, 1e-6f, false, false, false, false, true));
''')
