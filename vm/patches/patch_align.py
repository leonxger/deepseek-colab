"""Miss copies read whole 128-byte lines of the host source (ALIGN_PATCH): usage python3 patch_align.py LLAMA_SRC_DIR, runtime env GGML_CUDA_MOE_COPY_ALIGN=1."""
import os, sys

src = sys.argv[1]


def replace(rel, old, new):
    p = os.path.join(src, rel)
    s = open(p).read()
    if new in s:
        print('already replaced', rel, new[:50])
        return
    assert s.count(old) == 1, (rel, old)
    open(p, 'w').write(s.replace(old, new))
    print('replaced', rel, new[:50])


ENV = '''
// ALIGN_PATCH: GGML_CUDA_MOE_COPY_ALIGN=1 lets every warp of a copy from mapped host memory start on a 128-byte line of the source
static int ggml_cuda_moe_copy_align() {
    static const int v = getenv("GGML_CUDA_MOE_COPY_ALIGN") ? atoi(getenv("GGML_CUDA_MOE_COPY_ALIGN")) : 0;
    return v;
}
'''

MC = 'ggml/src/ggml-cuda/moe-cache.cu'
replace(MC, '#include "moe-cache.cuh"\n', '#include "moe-cache.cuh"\n\n#include <cstdlib>\n' + ENV)
replace(MC, '''        int4 * b0, int4 * b1, int4 * b2, const int64_t n0, const int64_t n1, const int64_t n2) {
    const int nm = miss[0];''', '''        int4 * b0, int4 * b1, int4 * b2, const int64_t n0, const int64_t n1, const int64_t n2, const int align) {
    const int nm = miss[0];''')
replace(MC, '''        for (int64_t i0 = (int64_t) blockIdx.x*blockDim.x*MC_COPY_U + threadIdx.x; i0 < n; i0 += step) {
            int4 v[MC_COPY_U];
#pragma unroll
            for (int u = 0; u < MC_COPY_U; ++u) {
                const int64_t i = i0 + (int64_t) u*blockDim.x;
                v[u] = i < n ? x[i] : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < MC_COPY_U; ++u) {
                const int64_t i = i0 + (int64_t) u*blockDim.x;
                if (i < n) {''', '''        const int64_t k = align ? (int64_t) (((uintptr_t) x >> 4) & 7) : 0;
        for (int64_t i0 = (int64_t) blockIdx.x*blockDim.x*MC_COPY_U + threadIdx.x - k; i0 < n; i0 += step) {
            int4 v[MC_COPY_U];
#pragma unroll
            for (int u = 0; u < MC_COPY_U; ++u) {
                const int64_t i = i0 + (int64_t) u*blockDim.x;
                v[u] = i >= 0 && i < n ? x[i] : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < MC_COPY_U; ++u) {
                const int64_t i = i0 + (int64_t) u*blockDim.x;
                if (i >= 0 && i < n) {''')
replace(MC, '''    k_moe_cache_copy<<<dim3(128, n_banks), 256, 0, stream>>>(miss, h[0], h[1], h[2], b[0], b[1], b[2], n[0], n[1], n[2]);''',
        '''    k_moe_cache_copy<<<dim3(128, n_banks), 256, 0, stream>>>(miss, h[0], h[1], h[2], b[0], b[1], b[2], n[0], n[1], n[2], ggml_cuda_moe_copy_align());''')

CB = 'ggml/src/ggml-cuda/copy-batch.cu'
replace(CB, '#include "copy-batch.cuh"\n', '#include "copy-batch.cuh"\n\n#include <cstdlib>\n' + ENV)
replace(CB, 'static __global__ void k_copy_batch(const uint64_t * __restrict__ d, const int n_desc, char * __restrict__ dst) {',
        'static __global__ void k_copy_batch(const uint64_t * __restrict__ d, const int n_desc, char * __restrict__ dst, const int align) {')
replace(CB, '''        for (int64_t i0 = threadIdx.x; i0 < n4; i0 += U*GGML_CUDA_COPY_BATCH_BLOCK) {
            int4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u*GGML_CUDA_COPY_BATCH_BLOCK;
                v[u] = src != 0 && i < n4 ? x4[i] : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u*GGML_CUDA_COPY_BATCH_BLOCK;
                if (i < n4) {''', '''        const int64_t k = align ? (int64_t) (((src + c0) >> 4) & 7) : 0;
        for (int64_t i0 = (int64_t) threadIdx.x - k; i0 < n4; i0 += U*GGML_CUDA_COPY_BATCH_BLOCK) {
            int4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u*GGML_CUDA_COPY_BATCH_BLOCK;
                v[u] = src != 0 && i >= 0 && i < n4 ? x4[i] : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u*GGML_CUDA_COPY_BATCH_BLOCK;
                if (i >= 0 && i < n4) {''')
replace(CB, '    k_copy_batch<<<(unsigned int) n_chunks, GGML_CUDA_COPY_BATCH_BLOCK, 0, stream>>>(d_dev.get(), n_desc, (char *) dst->data);',
        '    k_copy_batch<<<(unsigned int) n_chunks, GGML_CUDA_COPY_BATCH_BLOCK, 0, stream>>>(d_dev.get(), n_desc, (char *) dst->data, ggml_cuda_moe_copy_align());')
