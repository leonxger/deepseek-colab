"""One gather kernel per expert copy of a large batch, from cache slots and mapped pinned host memory (LLAMA_MOE_COPY_BATCH=0 keeps the old path). usage: python3 patch_copybatch.py LLAMA_SRC_DIR"""
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


def write(rel, text):
    p = os.path.join(src, rel)
    if os.path.exists(p) and open(p).read() == text:
        print('already written', rel)
        return
    assert not os.path.exists(p), ('exists with other content', rel)
    open(p, 'w').write(text)
    print('written', rel)


def has(rel, text):
    return text in open(os.path.join(src, rel)).read()


C = 'src/llama-context.cpp'
PREFETCH = has(C, 'moe_prefetch_begin(backend, src); // PREFETCH_PATCH')
assert has(C, 'PADZERO_PATCH'), 'needs patch_e.py (PADZERO_PATCH)'

# ---- CUDA: gather kernel and the host pointer check ----
write('ggml/src/ggml-cuda/copy-batch.cuh', '''#pragma once

#include "common.cuh"

// COPYBATCH_PATCH: desc = n_desc x {src address (device or mapped host, 0 = zeros), dst offset, bytes}, one kernel on the stream of backend
bool ggml_backend_cuda_copy_batch(ggml_backend_t backend, ggml_tensor * dst, const uint64_t * desc, int n_desc);

// COPYBATCH_PATCH: device address of the mapped pinned host range [ptr, ptr + size), false if a sampled address is not mapped
bool ggml_backend_cuda_host_device_ptr(ggml_backend_t backend, const void * ptr, size_t size, uint64_t * dev);
''')

write('ggml/src/ggml-cuda/copy-batch.cu', r'''#include "copy-batch.cuh"
#include "ggml-backend-impl.h"

#include <vector>

#define GGML_CUDA_COPY_BATCH_CHUNK (64*1024)
#define GGML_CUDA_COPY_BATCH_BLOCK 256

// one block per 64 KiB chunk, d[4*i] = {src, dst offset, bytes, first chunk of descriptor i}
static __global__ void k_copy_batch(const uint64_t * __restrict__ d, const int n_desc, char * __restrict__ dst) {
    const uint64_t chunk = blockIdx.x;

    int lo = 0;
    int hi = n_desc - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1)/2;
        if (d[4*mid + 3] <= chunk) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    const uint64_t c0   = (chunk - d[4*lo + 3])*GGML_CUDA_COPY_BATCH_CHUNK;
    const uint64_t src  = d[4*lo + 0];
    const uint64_t rest = d[4*lo + 2] - c0;
    const int64_t  n    = rest < GGML_CUDA_COPY_BATCH_CHUNK ? (int64_t) rest : GGML_CUDA_COPY_BATCH_CHUNK;
    char       * y = dst + d[4*lo + 1] + c0;
    const char * x = (const char *) (src + c0);

    int64_t n4 = 0;
    if ((((src + c0) | (uint64_t) (uintptr_t) y) & 15) == 0) {
        n4 = n/16;
        const int4 * x4 = (const int4 *) x;
        int4       * y4 = (int4 *) y;
        constexpr int U = 4;
        // all loads of a thread are issued before its stores, so many reads over PCIe are in flight
        for (int64_t i0 = threadIdx.x; i0 < n4; i0 += U*GGML_CUDA_COPY_BATCH_BLOCK) {
            int4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u*GGML_CUDA_COPY_BATCH_BLOCK;
                v[u] = src != 0 && i < n4 ? x4[i] : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u*GGML_CUDA_COPY_BATCH_BLOCK;
                if (i < n4) {
                    y4[i] = v[u];
                }
            }
        }
    }
    for (int64_t i = 16*n4 + threadIdx.x; i < n; i += GGML_CUDA_COPY_BATCH_BLOCK) {
        y[i] = src != 0 ? x[i] : 0;
    }
}

bool ggml_backend_cuda_copy_batch(ggml_backend_t backend, ggml_tensor * dst, const uint64_t * desc, int n_desc) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend) || dst == nullptr || dst->data == nullptr || desc == nullptr || n_desc <= 0) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (buf == nullptr || ggml_backend_buffer_get_type(buf) != ggml_backend_cuda_buffer_type(ctx->device)) {
        return false;
    }

    std::vector<uint64_t> d(4*(size_t) n_desc);
    uint64_t n_chunks = 0;
    for (int i = 0; i < n_desc; ++i) {
        const uint64_t offs = desc[3*i + 1];
        const uint64_t size = desc[3*i + 2];
        GGML_ASSERT(size > 0 && offs + size <= ggml_nbytes(dst));
        d[4*i + 0] = desc[3*i + 0];
        d[4*i + 1] = offs;
        d[4*i + 2] = size;
        d[4*i + 3] = n_chunks;
        n_chunks += (size + GGML_CUDA_COPY_BATCH_CHUNK - 1)/GGML_CUDA_COPY_BATCH_CHUNK;
    }
    GGML_ASSERT(n_chunks <= (uint64_t) INT32_MAX);

    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();

    // d is pageable, so cudaMemcpyAsync has read it when it returns; the pool block is reused only by later work on this stream
    ggml_cuda_pool_alloc<uint64_t> d_dev(ctx->pool(), d.size());
    CUDA_CHECK(cudaMemcpyAsync(d_dev.get(), d.data(), d.size()*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    k_copy_batch<<<(unsigned int) n_chunks, GGML_CUDA_COPY_BATCH_BLOCK, 0, stream>>>(d_dev.get(), n_desc, (char *) dst->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_backend_cuda_host_device_ptr(ggml_backend_t backend, const void * ptr, size_t size, uint64_t * dev) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(backend, ptr, size, dev);
    return false;
#else
    if (backend == nullptr || !ggml_backend_is_cuda(backend) || ptr == nullptr || size == 0 || dev == nullptr) {
        return false;
    }
    ggml_cuda_set_device(((ggml_backend_cuda_context *) backend->context)->device);

    // registrations are whole runs of experts, so a range is accepted if every 256 KiB step and its last byte map linearly
    const char * base = (const char *) ptr;
    const char * dev0 = nullptr;
    for (size_t offs = 0; ; offs = std::min(offs + 256*1024, size - 1)) {
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, base + offs) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        if (attr.type != cudaMemoryTypeHost || attr.devicePointer == nullptr) {
            return false;
        }
        if (offs == 0) {
            dev0 = (const char *) attr.devicePointer;
        } else if ((const char *) attr.devicePointer != dev0 + offs) {
            return false;
        }
        if (offs == size - 1) {
            break;
        }
    }
    *dev = (uint64_t) (uintptr_t) dev0;
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}
''')

G = 'ggml/src/ggml-cuda/ggml-cuda.cu'
A = '#include "ggml-cuda/wkv.cuh"\n'
patch(G, A, A + '#include "ggml-cuda/copy-batch.cuh" // COPYBATCH_PATCH\n')
A = '''    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
'''
patch(G, A, A + '''    if (strcmp(name, "ggml_backend_cuda_copy_batch") == 0) {
        return (void *)ggml_backend_cuda_copy_batch; // COPYBATCH_PATCH
    }
    if (strcmp(name, "ggml_backend_cuda_host_device_ptr") == 0) {
        return (void *)ggml_backend_cuda_host_device_ptr; // COPYBATCH_PATCH
    }
''')

# ---- MoE cache: device address of a cached expert ----
H = 'src/llama-moe-cache.h'
A = '    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;\n'
patch(H, A, '''    // COPYBATCH_PATCH: device address of expert e of w if copy_experts would copy it from the cache for a copy on backend, nullptr otherwise
    const void * cached_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const;

''' + A)

MC = 'src/llama-moe-cache.cpp'
A = '''        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }
'''
patch(MC, A, A + '''
    // COPYBATCH_PATCH: the checks of copy_experts
    const void * cached_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        const int32_t * slots = g.slru.slot_map[b.il];
        if (backend != devices[g.id].backend || slots == nullptr || slots[e] >= g.n_slots) {
            return nullptr;
        }
        const ggml_tensor * bank = g.banks[b.ip];
        return (const char *) bank->data + (size_t) slots[e]*bank->nb[2];
    }
''')
A = 'std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {\n'
patch(MC, A, '''const void * llama_moe_cache::cached_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
    return pimpl->cached_expert(backend, w, e); // COPYBATCH_PATCH
}

''' + A)

# ---- context state ----
CH = 'src/llama-context.h'
A = '    void    moe_prefetch_log();\n' if PREFETCH else '    copy_experts_info copy_experts;\n'
patch(CH, A, A + '''
    // COPYBATCH_PATCH: LLAMA_MOE_COPY_BATCH (on unless 0), the used experts of a large batch are staged by one gather kernel
    struct moe_copy_batch_stats {
        size_t n_call     = 0;
        size_t n_desc     = 0;
        size_t n_bank     = 0; // experts read from cache bank slots
        size_t n_host     = 0; // experts read from mapped pinned host memory
        size_t n_fallback = 0; // experts uploaded as before
    };

    struct moe_copy_batch_info {
        ggml_backend_t backend = nullptr; // nullptr if the copy batch is off
        bool (*copy)(ggml_backend_t backend, ggml_tensor * dst, const uint64_t * desc, int n_desc) = nullptr;
        bool (*resolve)(ggml_backend_t backend, const void * ptr, size_t size, uint64_t * dev) = nullptr;
        bool pad_zero = false;

        std::map<const ggml_tensor *, std::vector<uint64_t>> host_dev; // device address of each host expert, 0 unknown, 1 not mapped
        std::vector<uint64_t> desc;                                    // {src, dst offset, bytes} per range
        std::vector<std::pair<size_t, size_t>> fallback;              // {dst offset, bytes} uploaded from host as before

        moe_copy_batch_stats graph;
        moe_copy_batch_stats total;

        ~moe_copy_batch_info() {
            if (backend != nullptr) {
                LLAMA_LOG_INFO("%s: MoE copy batch total: %zu copies, %zu descriptors, %zu experts from the cache, %zu from mapped host memory, %zu uploaded as before\\n",
                    __func__, total.n_call, total.n_desc, total.n_bank, total.n_host, total.n_fallback);
            }
        }
    };

    moe_copy_batch_info moe_batch;

    void moe_copy_batch_init();
    bool moe_copy_batch(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst);
    void moe_copy_batch_log();
''')

# ---- context code ----
A = ('        moe_prefetch_init(params.ctx_other == nullptr); // PREFETCH_PATCH\n' if PREFETCH else
     '''                moe_cache->prefill(prof, getenv("LLAMA_MOE_CACHE_RESIDENT")); // FROZEN_PATCH
            }
        }

        sched_reserve();
''')
patch(C, A, A + '        moe_copy_batch_init(); // COPYBATCH_PATCH\n')

A = ('    moe_prefetch_log(); // PREFETCH_PATCH\n' if PREFETCH else
     '''    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\\n", __func__, status);
    }
''')
patch(C, A, A + '    moe_copy_batch_log(); // COPYBATCH_PATCH\n')

A = ('    lctx->moe_prefetch_begin(backend, src); // PREFETCH_PATCH\n' if PREFETCH else
     '    // group consecutive experts and copy them together\n')
patch(C, A, '''    if (lctx->moe_copy_batch(backend, src, dst)) { // COPYBATCH_PATCH
        return true;
    }

''' + A)

DROP_PREFETCH = '''
    if (moe_prefetch.backend != nullptr) {
        // the copy batch replaces the copy loop that consumes the slots
        moe_prefetch.backend = nullptr;
        moe_prefetch.slots.clear();
        moe_prefetch.buf.reset();
        moe_prefetch.stream.reset();
        LLAMA_LOG_INFO("%s: MoE prefetch is off while the copy batch is on\\n", __func__);
    }
''' if PREFETCH else ''

A = ('// PREFETCH_PATCH: docs/prefetch.md\n' if PREFETCH else 'llm_graph_cb llama_context::graph_get_cb() const {\n')
patch(C, A, r'''// COPYBATCH_PATCH: docs/copybatch.md
void llama_context::moe_copy_batch_init() {
    const char * env = getenv("LLAMA_MOE_COPY_BATCH");
    if (model.no_alloc() || (env != nullptr && atoi(env) == 0)) {
        return;
    }
    auto & cb = moe_batch;

    // the scheduler offloads the matmuls of host experts to the first GPU
    ggml_backend_t backend = nullptr;
    for (ggml_backend_t b : backend_ptrs) {
        if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = b;
            break;
        }
    }
    if (backend == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    cb.copy    = (decltype(cb.copy))    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_copy_batch");
    cb.resolve = (decltype(cb.resolve)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_device_ptr");
    if (cb.copy == nullptr || cb.resolve == nullptr) {
        cb.copy    = nullptr;
        cb.resolve = nullptr;
        LLAMA_LOG_WARN("%s: MoE copy batch: %s has no batched copy, it is off\n", __func__, ggml_backend_name(backend));
        return;
    }
    cb.backend  = backend;
    cb.pad_zero = getenv("LLAMA_MOE_PAD_ZERO") != nullptr;
    LLAMA_LOG_INFO("%s: MoE copy batch: on for %s, guard after each run from %s\n", __func__, ggml_backend_name(backend),
        cb.pad_zero ? "zeros" : "the host bytes of the next expert");
''' + DROP_PREFETCH + r'''}

// stages the used experts of w into dst like the copy loop of sched_copy_experts, returns false (nothing issued) if that loop must run
bool llama_context::moe_copy_batch(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst) {
    auto & cb = moe_batch;
    if (cb.backend == nullptr || backend != cb.backend || !ggml_is_contiguous(w)) {
        return false;
    }
    const std::vector<bool> & used = copy_experts.used;
    const int64_t n_expert    = w->ne[2];
    const size_t  expert_size = w->nb[2];
    const size_t  padding     = std::min<size_t>(expert_size, 512);
    if ((int64_t) used.size() != n_expert) {
        return false;
    }
    std::vector<uint64_t> & dev = cb.host_dev[w];
    if ((int64_t) dev.size() != n_expert) {
        dev.assign(n_expert, 0);
    }

    auto host_addr = [&](int64_t e) -> uint64_t {
        if (dev[e] == 0) {
            uint64_t addr = 0;
            dev[e] = cb.resolve(backend, (const uint8_t *) w->data + e*expert_size, expert_size, &addr) && addr > 1 ? addr : 1;
        }
        return dev[e] > 1 ? dev[e] : 0;
    };
    auto add = [&](uint64_t addr, size_t offs, size_t size) {
        const size_t n = cb.desc.size();
        if (addr != 0 && n >= 3 && cb.desc[n - 3] != 0 && cb.desc[n - 3] + cb.desc[n - 1] == addr && cb.desc[n - 2] + cb.desc[n - 1] == offs) {
            cb.desc[n - 1] += size;
            return;
        }
        cb.desc.push_back(addr);
        cb.desc.push_back(offs);
        cb.desc.push_back(size);
    };
    auto upload = [&](size_t offs, size_t size) {
        if (!cb.fallback.empty() && cb.fallback.back().first + cb.fallback.back().second == offs) {
            cb.fallback.back().second += size;
            return;
        }
        cb.fallback.emplace_back(offs, size);
    };

    cb.desc.clear();
    cb.fallback.clear();
    moe_copy_batch_stats st;
    for (int64_t e = 0; e < n_expert; ++e) {
        if (!used[e]) {
            continue;
        }
        const void * cached = moe_cache ? moe_cache->cached_expert(backend, w, e) : nullptr;
        if (cached != nullptr) {
            add((uint64_t) (uintptr_t) cached, e*expert_size, expert_size);
            st.n_bank++;
        } else if (const uint64_t addr = host_addr(e); addr != 0) {
            add(addr, e*expert_size, expert_size);
            st.n_host++;
        } else {
            upload(e*expert_size, expert_size);
            st.n_fallback++;
        }

        // the guard after the last expert of a run: zeros with LLAMA_MOE_PAD_ZERO, else the host bytes of the next expert
        if (e + 1 < n_expert && !used[e + 1]) {
            const size_t offs = (e + 1)*expert_size;
            if (cb.pad_zero) {
                cb.desc.push_back(0);
                cb.desc.push_back(offs);
                cb.desc.push_back(padding);
            } else if (const uint64_t addr = host_addr(e + 1); addr != 0) {
                add(addr, offs, padding);
            } else {
                upload(offs, padding);
            }
        }
    }

    if (!cb.desc.empty() && !cb.copy(backend, dst, cb.desc.data(), (int) (cb.desc.size()/3))) {
        return false;
    }
    for (const auto & [offs, size] : cb.fallback) {
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) w->data + offs, offs, size);
    }

    cb.graph.n_call++;
    cb.graph.n_desc     += cb.desc.size()/3;
    cb.graph.n_bank     += st.n_bank;
    cb.graph.n_host     += st.n_host;
    cb.graph.n_fallback += st.n_fallback;
    return true;
}

void llama_context::moe_copy_batch_log() {
    auto & cb = moe_batch;
    moe_copy_batch_stats & g = cb.graph;
    if (g.n_call == 0) {
        return;
    }
    LLAMA_LOG_INFO("%s: MoE copy batch: %zu copies, %zu descriptors, %zu experts from the cache, %zu from mapped host memory, %zu uploaded as before\n", __func__,
        g.n_call, g.n_desc, g.n_bank, g.n_host, g.n_fallback);
    cb.total.n_call     += g.n_call;
    cb.total.n_desc     += g.n_desc;
    cb.total.n_bank     += g.n_bank;
    cb.total.n_host     += g.n_host;
    cb.total.n_fallback += g.n_fallback;
    g = {};
}

''' + A)
