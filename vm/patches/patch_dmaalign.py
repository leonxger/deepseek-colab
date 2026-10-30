"""DMA misses from 128-byte aligned sources (DMAALIGN_PATCH): usage python3 patch_dmaalign.py LLAMA_SRC_DIR (after patch_dma.py and patch_align.py), on with LLAMA_MOE_DEVCACHE_DMA=1 and GGML_CUDA_MOE_COPY_ALIGN=1."""
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


MC = 'ggml/src/ggml-cuda/moe-cache.cu'
replace(MC, '#define MC_HQ_STRIDE   (2 + 2*MC_DMA_MAX_IDS)\n', '''#define MC_HQ_STRIDE   (2 + 2*MC_DMA_MAX_IDS)

// DMAALIGN_PATCH: the plan kernel copies the bytes before the first 128-byte line of each miss source, the proxy's DMA the aligned rest
struct mc_heads {
    const int4 * h[3];
    int4       * b[3];
    int64_t      n[3];
    int          n_banks;
    int          on;
};

static mc_heads mc_heads_of(const int4 * const * h, int4 * const * b, const int64_t * n, const int n_banks) {
    mc_heads r;
    for (int i = 0; i < 3; ++i) {
        r.h[i] = h[i];
        r.b[i] = b[i];
        r.n[i] = n[i];
    }
    r.n_banks = n_banks;
    r.on      = ggml_cuda_moe_copy_align();
    return r;
}

static size_t mc_head_bytes(const void * x, const size_t n) {
    const size_t h = (128 - ((uintptr_t) x & 127)) & 127;
    return h < n ? h : n;
}
''')
replace(MC, '''        int32_t * hq, uint32_t * dma_seq, const uint32_t * dma_done, unsigned long long * dma_stat) {''',
        '''        int32_t * hq, uint32_t * dma_seq, const uint32_t * dma_done, unsigned long long * dma_stat, const mc_heads hd) {''')
replace(MC, '''    if (dseq != 0) {
        mc_dma_spin(dma_done, dseq, cnt + 3, dma_stat); // DMA_PATCH: gate and up of every miss are in VRAM before the next kernel starts
    }''', '''    if (hd.on && hq != nullptr) { // DMAALIGN_PATCH: heads of the misses, whole int4s because the sources are 16-byte aligned
        for (int j = tid; j < nm*hd.n_banks*8; j += MC_BLOCK) {
            const int m = j/(8*hd.n_banks);
            const int b = (j/8) % hd.n_banks;
            const int w = j % 8;
            const int e = miss[1 + 2*m];
            const int s = miss[2 + 2*m];
            if (e < 0 || s < 0) {
                continue;
            }
            const int4 * x = hd.h[b] + (int64_t) e*hd.n[b];
            const int    k = (int) (((uintptr_t) x >> 4) & 7);
            if (k != 0 && w < 8 - k && w < hd.n[b]) {
                hd.b[b][(int64_t) s*hd.n[b] + w] = x[w];
            }
        }
    }
    if (dseq != 0) {
        mc_dma_spin(dma_done, dseq, cnt + 3, dma_stat); // DMA_PATCH: gate and up of every miss are in VRAM before the next kernel starts
    }''')
replace(MC, '''        dma ? dma->hq_dev + (size_t) il*MC_HQ_STRIDE : nullptr, dma ? dma->seq + il : nullptr, dma ? dma->done + 2*il : nullptr, dma ? dma->stat : nullptr);''',
        '''        dma ? dma->hq_dev + (size_t) il*MC_HQ_STRIDE : nullptr, dma ? dma->seq + il : nullptr, dma ? dma->done + 2*il : nullptr, dma ? dma->stat : nullptr,
        mc_heads_of(h, b, n, n_banks));''')
replace(MC, '''    void run() {
        cudaSetDevice(device);
''', '''    void run() {
        cudaSetDevice(device);
        const bool align = ggml_cuda_moe_copy_align() != 0; // DMAALIGN_PATCH
        if (align) {
            fprintf(stderr, "moe-cache DMA: the plan kernel copies the bytes before the first 128-byte line of each source, the DMA the aligned rest\\n");
        }
''')
replace(MC, '''                        for (int b = b0; b < b1; ++b) {
                            n_fail  += cudaMemcpyAsync(d.b[b] + (size_t) sl*d.n[b], d.h[b] + (size_t) e*d.n[b], d.n[b], cudaMemcpyHostToDevice, stream) != cudaSuccess;
                            n_copy  += 1;
                            n_bytes += d.n[b];
                        }''', '''                        for (int b = b0; b < b1; ++b) {
                            const char * x    = d.h[b] + (size_t) e*d.n[b];
                            const size_t head = align ? mc_head_bytes(x, d.n[b]) : 0; // DMAALIGN_PATCH
                            n_fail  += cudaMemcpyAsync(d.b[b] + (size_t) sl*d.n[b] + head, x + head, d.n[b] - head, cudaMemcpyHostToDevice, stream) != cudaSuccess;
                            n_copy  += 1;
                            n_bytes += d.n[b];
                        }''')
