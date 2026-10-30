"""Split-K mat-vec for few long float rows (DeepSeek-V4 hc_*_fn 20480x24) in llama.cpp b11963: python3 patch_mmvf.py LLAMA_SRC_DIR (idempotent)."""
import os, sys

src = sys.argv[1]


def patch(rel, anchor, text, before=True):
    p = os.path.join(src, rel)
    s = open(p).read()
    if text.strip() in s:
        print('already patched', rel, text.strip()[:50])
        return
    assert s.count(anchor) == 1, (rel, anchor)
    s = s.replace(anchor, text + anchor if before else anchor + text)
    open(p, 'w').write(s)
    print('patched', rel, text.strip()[:50])


SPLITK = r'''// MMVF_SPLITK_PATCH: few long rows leave most SMs idle, so K is split over blocks and the partials are summed in a fixed order
static __global__ void mul_mat_vec_f_splitk_reduce(
        const float * partials, float * dst, const int nsplit, const int nrows, const int nout, const int stride_col_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x;
    const int i    = blockIdx.x*blockDim.y + threadIdx.y;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (i >= nout) {
        return;
    }

    float sum = lane < nsplit ? partials[lane*nout + i] : 0.0f;
    sum = warp_reduce_sum<warp_size>(sum);

    if (lane == 0) {
        const int col = i / nrows;
        dst[col*stride_col_dst + i - col*nrows] = sum;
    }
}

template <typename T, typename type_acc, int ncols_dst>
static void mul_mat_vec_f_splitk_partials(
        const T * x, const float * y, float * partials, const int64_t kslice, const int64_t nrows, const int64_t nsplit,
        const int64_t stride_row, const int64_t stride_col_y, cudaStream_t stream) {
    // K slice s is channel s of x and y, its partial sums land in partials[s][col][row]
    launch_mul_mat_vec_f_cuda<T, type_acc, ncols_dst>
        (x, y, nullptr, ggml_cuda_mm_fusion_args_device{}, partials, kslice, nrows, stride_row, stride_col_y, nrows,
         nsplit, nsplit, nsplit, kslice, kslice, ncols_dst*nrows, 1, 1, 0, 0, 0, 1, 0, stream);
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_splitk_cuda(
        const T * x, const float * y, float * partials, float * dst, const int64_t ncols_dst, const int64_t kslice, const int64_t nrows,
        const int64_t nsplit, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst, const int warp_size,
        cudaStream_t stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_f_splitk_partials<T, type_acc, 1>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 2: mul_mat_vec_f_splitk_partials<T, type_acc, 2>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 3: mul_mat_vec_f_splitk_partials<T, type_acc, 3>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 4: mul_mat_vec_f_splitk_partials<T, type_acc, 4>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 5: mul_mat_vec_f_splitk_partials<T, type_acc, 5>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 6: mul_mat_vec_f_splitk_partials<T, type_acc, 6>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 7: mul_mat_vec_f_splitk_partials<T, type_acc, 7>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        case 8: mul_mat_vec_f_splitk_partials<T, type_acc, 8>(x, y, partials, kslice, nrows, nsplit, stride_row, stride_col_y, stream); break;
        default:
            GGML_ABORT("fatal error");
    }

    const int     nwarps = 8;
    const int64_t nout   = ncols_dst*nrows;
    const dim3 block_nums((nout + nwarps - 1)/nwarps, 1, 1);
    const dim3 block_dims(warp_size, nwarps, 1);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(mul_mat_vec_f_splitk_reduce, launch_params,
        (const float *) partials, dst, (int) nsplit, (int) nrows, (int) nout, (int) stride_col_dst);
}

static bool ggml_cuda_mul_mat_vec_f_splitk(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const float * y, float * dst, const int64_t ncols, const int64_t nrows,
        const int64_t ncols_dst, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst, const enum ggml_prec prec) {
    // GGML_CUDA_MMVF_SPLITK: 0 = off, unset or 1 = auto, N >= 4 = at most N slices
    static const int env_splitk = []() {
        const char * env = getenv("GGML_CUDA_MMVF_SPLITK");
        return env == nullptr ? 1 : atoi(env);
    }();

    const auto & info = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (env_splitk <= 0 || !GGML_CUDA_CC_IS_NVIDIA(info.cc) || ncols_dst > MMVF_MAX_BATCH_SIZE || 2*nrows > info.nsm || ncols < 4096) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }

    // largest power of 2 that reaches ~2 blocks per SM while every slice stays even and at least 512 long
    const int64_t nsplit_max = env_splitk >= 2 ? std::min(env_splitk, info.warp_size) : info.warp_size;
    int64_t nsplit = 1;
    while (2*nsplit <= nsplit_max && (env_splitk >= 2 || nsplit*nrows < 2*info.nsm) && ncols % (4*nsplit) == 0 && ncols/(2*nsplit) >= 512) {
        nsplit *= 2;
    }
    if (nsplit < 4) {
        return false;
    }
    const int64_t kslice = ncols/nsplit;

    ggml_cuda_pool_alloc<float> partials(ctx.pool(), nsplit*ncols_dst*nrows);
    cudaStream_t stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32:
            mul_mat_vec_f_splitk_cuda<float, float>((const float *) src0->data, y, partials.get(), dst, ncols_dst, kslice, nrows, nsplit,
                stride_row, stride_col_y, stride_col_dst, info.warp_size, stream);
            break;
        case GGML_TYPE_F16:
            if (prec == GGML_PREC_DEFAULT) {
                mul_mat_vec_f_splitk_cuda<half, half>((const half *) src0->data, y, partials.get(), dst, ncols_dst, kslice, nrows, nsplit,
                    stride_row, stride_col_y, stride_col_dst, info.warp_size, stream);
            } else {
                mul_mat_vec_f_splitk_cuda<half, float>((const half *) src0->data, y, partials.get(), dst, ncols_dst, kslice, nrows, nsplit,
                    stride_row, stride_col_y, stride_col_dst, info.warp_size, stream);
            }
            break;
        default:
            mul_mat_vec_f_splitk_cuda<nv_bfloat16, float>((const nv_bfloat16 *) src0->data, y, partials.get(), dst, ncols_dst, kslice, nrows,
                nsplit, stride_row, stride_col_y, stride_col_dst, info.warp_size, stream);
            break;
    }
    return true;
}

'''

CALL = r'''    // MMVF_SPLITK_PATCH: plain 2D mat-vec without fusion, the gate on shape and device is in ggml_cuda_mul_mat_vec_f_splitk
    if (!ids && !fusion_local.gate && !fusion_local.x_bias && !fusion_local.gate_bias && ne02 == 1 && ne03 == 1 && ne2 == 1 && ne3 == 1 &&
            ggml_cuda_mul_mat_vec_f_splitk(ctx, src0, src1_d, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, prec)) {
        return;
    }

'''

EVAL = r'''    // MMVF_SPLITK_PATCH: few long rows (DeepSeek-V4 hc_*_fn, hc_head_fn, engram_gate, indexer.proj), row gate edges for 188 SMs, fallbacks
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (int n : {1, 2, 3}) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 24, n, 20480, {1, 1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32,  4, n, 20480, {1, 1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 94, n,  4096, {1, 1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 95, n,  4096, {1, 1}, {1, 1}));
        }
    }
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 24, 1, 20480, {1, 1}, {1, 1}, {0, 1, 2, 3}, 20480 + 64));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 24, 2, 20480, {1, 1}, {1, 1}, {0, 1, 2, 3}, 20480 + 64));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 24, 1,  4098, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32,  1, 16, 20480, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32,  GGML_TYPE_F32,  4, 1, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 32, 1, 5120, {1, 1}, {1, 1}));

'''

PERF = r'''
    // MMVF_SPLITK_PATCH: DeepSeek-V4 hyper-connection projections (hc_attn_fn/hc_ffn_fn 20480x24, hc_head_fn 20480x4)
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (int n : {1, 2}) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 24, n, 20480, {1, 1}, {1, 1}));
        }
    }
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32,  GGML_TYPE_F32,  4, 1, 20480, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32,  GGML_TYPE_F32,  4, 1,  5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 32, 1,  5120, {1, 1}, {1, 1}));
'''

M = 'ggml/src/ggml-cuda/mmvf.cu'
patch(M, 'void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, '
         'const ggml_tensor * ids, ggml_tensor * dst,\n', SPLITK)
patch(M, '    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;\n\n', CALL, before=False)

T = 'tests/test-backend-ops.cpp'
patch(T, '    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 31, 509, 2051, {1, 1}, {1, 1}));\n', EVAL)
patch(T, 'static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n',
      PERF, before=False)
