"""MoE cache misses copied by the DMA engine (DMA_PATCH, after patch_devcache.py): usage python3 patch_dma.py LLAMA_SRC_DIR, runtime env LLAMA_MOE_DEVCACHE_DMA=1."""
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

patch(MC, '#include "moe-cache.cuh"\n', '''
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
''', before=False)

patch(MC, '#define MC_COPY_U   4\n', '''
// DMA_PATCH: lookups of up to 24 expert ids (4 tokens) take their misses from the DMA engine, driven by a host proxy thread
#define MC_DMA_MAX_IDS 24
#define MC_DMA_LAYERS  512
#define MC_HQ_STRIDE   (2 + 2*MC_DMA_MAX_IDS)

static __device__ __forceinline__ uint32_t mc_ld_acquire(const uint32_t * p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ uint64_t mc_now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// spins until *done == want and adds the wait to stat[0..1]; after 2 s it counts an error and traps, so a late copy never feeds a matvec
static __device__ void mc_dma_spin(const uint32_t * done, const uint32_t want, uint32_t * err, unsigned long long * stat) {
    if (mc_ld_acquire(done) == want) {
        return;
    }
    const uint64_t t0 = mc_now_ns();
    uint64_t t = t0;
    while (mc_ld_acquire(done) != want) {
        t = mc_now_ns();
        if (t - t0 > 2000000000ull) {
            atomicAdd(err, 1u);
            __threadfence_system();
            __trap();
        }
        __nanosleep(100);
    }
    atomicAdd(stat + 0, (unsigned long long) (t - t0));
    atomicAdd(stat + 1, 1ull);
}
''', before=False)

replace(MC, '''        int32_t * __restrict__ state, const int n_slots, const int n_dyn, int32_t * __restrict__ out) {
    int32_t       * key_of = state;''', '''        int32_t * __restrict__ state, const int n_slots, const int n_dyn, int32_t * __restrict__ out,
        int32_t * hq, uint32_t * dma_seq, const uint32_t * dma_done, unsigned long long * dma_stat) {
    int32_t       * key_of = state;''')

replace(MC, '''    if (tid == 0) {
        miss[0] = nm;
        cnt[0] += n_h;
        cnt[1] += nm;
        cnt[2]  = now;
    }
    __syncthreads();
    for (int i = tid; i < n_ids; i += MC_BLOCK) {
        const int e = ids[i];
        out[i] = e >= 0 && e < n_expert && lmap[e] >= 0 ? lmap[e] : 0;
    }
}
''', '''    uint32_t dseq = 0;
    if (tid == 0) {
        miss[0] = nm;
        cnt[0] += n_h;
        cnt[1] += nm;
        cnt[2]  = now;
        if (hq != nullptr && nm > 0) { // DMA_PATCH: publish the misses to the proxy thread
            dseq = *dma_seq + 1;
            *dma_seq = dseq;
            for (int m = 0; m < nm; ++m) {
                hq[2 + 2*m] = miss[1 + 2*m];
                hq[3 + 2*m] = miss[2 + 2*m];
            }
            hq[1] = nm;
            __threadfence_system();
            *(volatile int32_t *) hq = (int32_t) dseq;
        }
    }
    __syncthreads();
    for (int i = tid; i < n_ids; i += MC_BLOCK) {
        const int e = ids[i];
        out[i] = e >= 0 && e < n_expert && lmap[e] >= 0 ? lmap[e] : 0;
    }
    if (dseq != 0) {
        mc_dma_spin(dma_done, dseq, cnt + 3, dma_stat); // DMA_PATCH: gate and up of every miss are in VRAM before the next kernel starts
    }
}
''')

patch(MC, 'void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n', '''// DMA_PATCH: launched before the down matvec as a PDL dependent, so it also waits for the previous kernel like any other node
static __global__ void k_moe_dma_wait(const uint32_t * dma_seq, const uint32_t * dma_done, uint32_t * err, unsigned long long * dma_stat) {
    ggml_cuda_pdl_lc();
    if (threadIdx.x == 0) {
        mc_dma_spin(dma_done, *(const volatile uint32_t *) dma_seq, err, dma_stat);
    }
    ggml_cuda_pdl_sync();
}

struct mc_dma_desc {
    std::atomic<int> ready{0};
    const char * h[3] = { nullptr, nullptr, nullptr };
    char       * b[3] = { nullptr, nullptr, nullptr };
    size_t       n[3] = { 0, 0, 0 };
    int          n_banks = 0;
};

// per layer: queue hq[seq, count, (expert, slot) x count] in mapped host memory, seq[il] on the device, done[2*il] gate+up and done[2*il + 1] down copied; stat = wait ns and count for gate+up, down
struct mc_dma_state {
    int               device = -1;
    int32_t         * hq     = nullptr;
    int32_t         * hq_dev = nullptr;
    uint32_t        * seq    = nullptr;
    uint32_t        * done   = nullptr;
    uint32_t        * val    = nullptr;
    unsigned long long * stat = nullptr;
    cudaStream_t      stream = nullptr;
    mc_dma_desc       desc[MC_DMA_LAYERS];
    std::atomic<int>  n_q{0};
    std::atomic<bool> stop{false};
    std::thread       th;
    uint64_t          n_pub = 0, n_copy = 0, n_bytes = 0, n_fail = 0;
    bool              fail_logged = false;

    void run() {
        cudaSetDevice(device);
        std::vector<uint32_t> seen(MC_DMA_LAYERS, 0);
        auto last = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            bool busy = false;
            const int nq = n_q.load(std::memory_order_acquire);
            for (int q = 0; q < nq; ++q) {
                const mc_dma_desc & d = desc[q];
                const volatile int32_t * p = hq + (size_t) q*MC_HQ_STRIDE;
                const uint32_t s = (uint32_t) p[0];
                if (s == seen[q] || !d.ready.load(std::memory_order_acquire)) {
                    continue;
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                seen[q] = s;
                busy    = true;
                n_pub++;
                const int nm    = std::min((int) p[1], MC_DMA_MAX_IDS);
                const auto t_seen = std::chrono::steady_clock::now();
                const int nb_gu = d.n_banks > 1 ? d.n_banks - 1 : d.n_banks; // gate and up (or gate_up) first, down last
                for (int pass = 0; pass < 2; ++pass) {
                    const int b0 = pass == 0 ? 0 : nb_gu;
                    const int b1 = pass == 0 ? nb_gu : d.n_banks;
                    for (int m = 0; m < nm; ++m) {
                        const int e  = p[2 + 2*m];
                        const int sl = p[3 + 2*m];
                        if (e < 0 || sl < 0) {
                            continue;
                        }
                        for (int b = b0; b < b1; ++b) {
                            n_fail  += cudaMemcpyAsync(d.b[b] + (size_t) sl*d.n[b], d.h[b] + (size_t) e*d.n[b], d.n[b], cudaMemcpyHostToDevice, stream) != cudaSuccess;
                            n_copy  += 1;
                            n_bytes += d.n[b];
                        }
                    }
                    val[2*q + pass] = s;
                    n_fail += cudaMemcpyAsync(done + 2*q + pass, val + 2*q + pass, sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess;
                }
                if (n_fail > 0 && !fail_logged) {
                    fail_logged = true;
                    fprintf(stderr, "moe-cache DMA: a copy call failed: %s\\n", cudaGetErrorString(cudaGetLastError()));
                }
                if (n_pub <= 3) {
                    fprintf(stderr, "moe-cache DMA: layer %d seq %u: %d misses, calls issued in %.1f us\\n", q, s, nm,
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_seen).count());
                }
            }
            const auto t = std::chrono::steady_clock::now();
            if (busy) {
                last = t;
            } else if (t - last > std::chrono::milliseconds(100)) {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }
    }

    ~mc_dma_state() {
        if (th.joinable()) {
            stop = true;
            th.join();
            unsigned long long w[4] = { 0, 0, 0, 0 };
            cudaMemcpy(w, stat, sizeof(w), cudaMemcpyDeviceToHost);
            fprintf(stderr, "moe-cache DMA: %llu lookups with misses, %llu copies, %.2f MiB, %llu failed calls; waits gate+up %llu (%.2f ms), down %llu (%.2f ms)\\n",
                (unsigned long long) n_pub, (unsigned long long) n_copy, n_bytes/1048576.0, (unsigned long long) n_fail, w[1], w[0]*1e-6, w[3], w[2]*1e-6);
        }
    }
};

static bool mc_dma_env() {
    static const bool on = getenv("LLAMA_MOE_DEVCACHE_DMA") != nullptr && atoi(getenv("LLAMA_MOE_DEVCACHE_DMA")) != 0;
    return on;
}

// only lookups captured into a CUDA graph take the DMA path: eagerly launched kernels can fill the launch queue while a plan kernel waits for the proxy
static bool mc_dma_capturing(ggml_backend_cuda_context & ctx) {
    if (!mc_dma_env()) {
        return false;
    }
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cs));
    static std::atomic<bool> logged{false};
    if (cs != cudaStreamCaptureStatusActive && !logged.exchange(true)) {
        GGML_LOG_INFO("%s: MoE cache lookup outside a CUDA graph capture: misses take the copy kernel\\n", __func__);
    }
    return cs == cudaStreamCaptureStatusActive;
}

static mc_dma_state * mc_dma_get() {
    static mc_dma_state * st = []() -> mc_dma_state * {
        const char * v = getenv("LLAMA_MOE_DEVCACHE_DMA");
        if (v == nullptr || atoi(v) == 0) {
            return nullptr;
        }
        static mc_dma_state s; // a function-local static joins its thread before the CUDA runtime shuts down
        CUDA_CHECK(cudaGetDevice(&s.device));
        int prio_lo = 0, prio_hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s.stream, cudaStreamNonBlocking, prio_hi));
        CUDA_CHECK(cudaHostAlloc((void **) &s.hq, sizeof(int32_t)*MC_DMA_LAYERS*MC_HQ_STRIDE, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(s.hq, 0, sizeof(int32_t)*MC_DMA_LAYERS*MC_HQ_STRIDE);
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &s.hq_dev, s.hq, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &s.val, sizeof(uint32_t)*2*MC_DMA_LAYERS, cudaHostAllocPortable));
        CUDA_CHECK(cudaMalloc((void **) &s.seq, sizeof(uint32_t)*3*MC_DMA_LAYERS + 4*sizeof(unsigned long long)));
        s.done = s.seq + MC_DMA_LAYERS;
        s.stat = (unsigned long long *) (s.seq + 3*MC_DMA_LAYERS);
        CUDA_CHECK(cudaMemsetAsync(s.seq, 0, sizeof(uint32_t)*3*MC_DMA_LAYERS + 4*sizeof(unsigned long long), s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
        s.th = std::thread([] { s.run(); });
        GGML_LOG_INFO("%s: MoE cache misses of lookups with up to %d expert ids are copied by the DMA engine (proxy thread)\\n", __func__, MC_DMA_MAX_IDS);
        return &s;
    }();
    return st;
}

static void mc_dma_register(mc_dma_state & s, const int il, const int4 * const * h, int4 * const * b, const int64_t * n, const int n_banks) {
    mc_dma_desc & d = s.desc[il];
    if (d.ready.load(std::memory_order_acquire)) {
        GGML_ASSERT(d.n_banks == n_banks && d.h[0] == (const char *) h[0] && d.b[0] == (char *) b[0]);
        return;
    }
    for (int i = 0; i < 3; ++i) {
        d.h[i] = (const char *) h[i];
        d.b[i] = (char *) b[i];
        d.n[i] = (size_t) n[i]*16;
    }
    d.n_banks = n_banks;
    d.ready.store(1, std::memory_order_release);
    int nq = s.n_q.load();
    while (nq < il + 1 && !s.n_q.compare_exchange_weak(nq, il + 1)) {
    }
}

// the lookup behind the ids of a mul_mat_id (a reshape of it), if it took the DMA path (the same rule as in the lookup op)
static const ggml_tensor * mc_dma_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * ids) {
    for (const ggml_tensor * t = ids; t != nullptr; t = t->view_src) {
        if (t->op == GGML_OP_MOE_CACHE_LOOKUP) {
            return ggml_nelements(t->src[0]) <= MC_DMA_MAX_IDS && mc_dma_env() && mc_dma_capturing(ctx) ? t : nullptr;
        }
    }
    return nullptr;
}

bool ggml_cuda_moe_dma_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids) {
    return ids != nullptr && mc_dma_lookup(ctx, ids) != nullptr;
}

void ggml_cuda_moe_dma_wait(ggml_backend_cuda_context & ctx, const ggml_tensor * ids, const ggml_tensor * src0) {
    const ggml_tensor * lk = ids != nullptr ? mc_dma_lookup(ctx, ids) : nullptr;
    if (lk == nullptr) {
        return;
    }
    const int32_t * p = (const int32_t *) lk->op_params;
    const int il      = p[0];
    const int n_slots = p[1];
    const int n_dyn   = p[2];
    const int n_banks = p[3];
    const ggml_tensor * down = n_banks > 1 ? lk->src[3 + n_banks - 1] : nullptr;
    if (down == nullptr || src0->data != down->data) {
        return; // gate and up were waited for in the plan kernel
    }
    mc_dma_state * s = mc_dma_get();
    uint32_t * err = (uint32_t *) lk->src[2]->data + 2*n_slots + n_dyn + 1 + 2*MC_MAX_MISS + 3;
    const ggml_cuda_kernel_launch_params launch_params(dim3(1), dim3(32), 0, ctx.stream());
    ggml_cuda_kernel_launch(k_moe_dma_wait, launch_params, (const uint32_t *) (s->seq + il), (const uint32_t *) (s->done + 2*il + 1), err, s->stat + 2);
}

''')

replace(MC, '''    cudaStream_t stream = ctx.stream();
    const int n_ids    = (int) ggml_nelements(ids);
    const int n_expert = (int) map->ne[0];
    k_moe_cache_plan<<<1, MC_BLOCK, 0, stream>>>((const int32_t *) ids->data, n_ids, (int32_t *) map->data, il, n_expert,
        (int32_t *) state->data, n_slots, n_dyn, (int32_t *) dst->data);
    CUDA_CHECK(cudaGetLastError());

    const int4 * h[3] = { nullptr, nullptr, nullptr };
    int4       * b[3] = { nullptr, nullptr, nullptr };
    int64_t      n[3] = { 0, 0, 0 };
    for (int i = 0; i < n_banks; ++i) {
        const ggml_tensor * bank = dst->src[3 + i];
        GGML_ASSERT(bank != nullptr && bank->nb[2] % 16 == 0 && ((uintptr_t) bank->data & 15) == 0 && ((uintptr_t) host[i] & 15) == 0);
        h[i] = (const int4 *) host[i];
        b[i] = (int4 *) bank->data;
        n[i] = (int64_t) (bank->nb[2]/16);
    }
    const int32_t * miss''', '''    cudaStream_t stream = ctx.stream();
    const int n_ids    = (int) ggml_nelements(ids);
    const int n_expert = (int) map->ne[0];

    const int4 * h[3] = { nullptr, nullptr, nullptr };
    int4       * b[3] = { nullptr, nullptr, nullptr };
    int64_t      n[3] = { 0, 0, 0 };
    for (int i = 0; i < n_banks; ++i) {
        const ggml_tensor * bank = dst->src[3 + i];
        GGML_ASSERT(bank != nullptr && bank->nb[2] % 16 == 0 && ((uintptr_t) bank->data & 15) == 0 && ((uintptr_t) host[i] & 15) == 0);
        h[i] = (const int4 *) host[i];
        b[i] = (int4 *) bank->data;
        n[i] = (int64_t) (bank->nb[2]/16);
    }

    mc_dma_state * dma = n_ids <= MC_DMA_MAX_IDS && mc_dma_capturing(ctx) ? mc_dma_get() : nullptr; // DMA_PATCH
    if (dma != nullptr) {
        GGML_ASSERT(il >= 0 && il < MC_DMA_LAYERS);
        mc_dma_register(*dma, il, h, b, n, n_banks);
    }
    k_moe_cache_plan<<<1, MC_BLOCK, 0, stream>>>((const int32_t *) ids->data, n_ids, (int32_t *) map->data, il, n_expert,
        (int32_t *) state->data, n_slots, n_dyn, (int32_t *) dst->data,
        dma ? dma->hq_dev + (size_t) il*MC_HQ_STRIDE : nullptr, dma ? dma->seq + il : nullptr, dma ? dma->done + 2*il : nullptr, dma ? dma->stat : nullptr);
    CUDA_CHECK(cudaGetLastError());
    if (dma != nullptr) {
        return;
    }

    const int32_t * miss''')

patch('ggml/src/ggml-cuda/moe-cache.cuh', 'void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);\n', '''
// DMA_PATCH: true if ids come from a lookup whose misses the DMA engine copies; the down matvec of such a lookup waits for them
bool ggml_cuda_moe_dma_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids);
void ggml_cuda_moe_dma_wait(ggml_backend_cuda_context & ctx, const ggml_tensor * ids, const ggml_tensor * src0);
''', before=False)

patch('ggml/src/ggml-cuda/mmvq.cu', '#include "vecdotq.cuh"\n', '#include "moe-cache.cuh" // DMA_PATCH\n', before=False)
patch('ggml/src/ggml-cuda/mmvq.cu', '''        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
''', '''    ggml_cuda_moe_dma_wait(ctx, ids, src0); // DMA_PATCH
''', before=False)

patch('ggml/src/ggml-cuda/ggml-cuda.cu', '''
    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
''', '''    GGML_ASSERT(!ggml_cuda_moe_dma_ids(ctx, ids) || (ggml_is_quantized(src0->type) && ne2 <= get_mmvq_mmid_max_batch(src0->type, cc))); // DMA_PATCH: only mmvq waits for DMA misses
''')
