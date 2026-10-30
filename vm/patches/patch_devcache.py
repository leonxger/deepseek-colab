"""Device-side MoE cache lookup (LLAMA_MOE_DEVCACHE=1): small ubatches find and copy their experts on the GPU, so a verify step is one CUDA graph. usage: python3 patch_devcache.py LLAMA_SRC_DIR"""
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


def write(rel, text):
    p = os.path.join(src, rel)
    if os.path.exists(p) and open(p).read() == text:
        print('already written', rel)
        return
    open(p, 'w').write(text)
    print('wrote', rel)


CU = r'''#include "moe-cache.cuh"

// state of a cache group, shared with src/llama-moe-cache.cpp: key_of[n_slots], stamp[n_slots], dyn[n_dyn], miss[1 + 2*MC_MAX_MISS], cnt[4] = hits, misses, clock, errors
#define MC_MAX_MISS 256
#define MC_BLOCK    256
#define MC_COPY_U   4

static __device__ __forceinline__ bool mc_better(uint32_t v, int s, uint32_t bv, int bs) {
    return s >= 0 && (bs < 0 || v < bv || (v == bv && s < bs));
}

// one block: hits refresh their stamp, each miss takes the dynamic slot with the oldest stamp not used by this lookup
static __global__ void k_moe_cache_plan(const int32_t * __restrict__ ids, const int n_ids, int32_t * __restrict__ map, const int il, const int n_expert,
        int32_t * __restrict__ state, const int n_slots, const int n_dyn, int32_t * __restrict__ out) {
    int32_t       * key_of = state;
    uint32_t      * stamp  = (uint32_t *) (state + n_slots);
    const int32_t * dyn    = state + 2*n_slots;
    int32_t       * miss   = state + 2*n_slots + n_dyn;
    uint32_t      * cnt    = (uint32_t *) (miss + 1 + 2*MC_MAX_MISS);
    int32_t       * lmap   = map + (int64_t) il*n_expert;

    __shared__ uint32_t need[32];
    __shared__ int32_t  mexp[MC_MAX_MISS];
    __shared__ int      n_m;
    __shared__ int      n_h;
    __shared__ uint32_t red_v[MC_BLOCK/32];
    __shared__ int      red_s[MC_BLOCK/32];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    if (tid < 32) {
        need[tid] = 0;
    }
    if (tid == 0) {
        n_m = 0;
        n_h = 0;
    }
    __syncthreads();
    for (int i = tid; i < n_ids; i += MC_BLOCK) {
        const int e = ids[i];
        if (e >= 0 && e < n_expert) {
            atomicOr(&need[e >> 5], 1u << (e & 31));
        }
    }
    __syncthreads();

    const uint32_t now = cnt[2] + 1;
    if (warp == 0) {
        // misses in expert order, so a lookup is deterministic
        for (int w = 0; w < (n_expert + 31)/32; ++w) {
            const int  e      = w*32 + lane;
            const bool needed = e < n_expert && (need[w] >> lane & 1);
            const int  s      = needed ? lmap[e] : -1;
            const bool is_hit = needed && s >= 0;
            if (is_hit) {
                stamp[s] = now;
            }
            const unsigned mb = __ballot_sync(0xffffffff, needed && !is_hit);
            const unsigned hb = __ballot_sync(0xffffffff, is_hit);
            if (needed && !is_hit) {
                const int r = n_m + __popc(mb & ((1u << lane) - 1u));
                if (r < MC_MAX_MISS) {
                    mexp[r] = e;
                }
            }
            __syncwarp();
            if (lane == 0) {
                n_m += __popc(mb);
                n_h += __popc(hb);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    const int nm = min(n_m, MC_MAX_MISS);
    for (int m = 0; m < nm; ++m) {
        uint32_t bv = 0xffffffffu;
        int      bs = -1;
        for (int j = tid; j < n_dyn; j += MC_BLOCK) {
            const int      s = dyn[j];
            const uint32_t v = stamp[s];
            if (v != now && mc_better(v, s, bv, bs)) {
                bv = v;
                bs = s;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const uint32_t v2 = __shfl_down_sync(0xffffffff, bv, o);
            const int      s2 = __shfl_down_sync(0xffffffff, bs, o);
            if (mc_better(v2, s2, bv, bs)) {
                bv = v2;
                bs = s2;
            }
        }
        if (lane == 0) {
            red_v[warp] = bv;
            red_s[warp] = bs;
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < MC_BLOCK/32; ++w) {
                if (mc_better(red_v[w], red_s[w], bv, bs)) {
                    bv = red_v[w];
                    bs = red_s[w];
                }
            }
            const int e = mexp[m];
            if (bs < 0) {
                cnt[3]++; // no free dynamic slot: the host keeps n_ids <= n_dyn, so this is a bug
                miss[1 + 2*m] = -1;
                miss[2 + 2*m] = -1;
            } else {
                const int old = key_of[bs];
                if (old >= 0) {
                    map[old] = -1;
                }
                key_of[bs] = il*n_expert + e;
                lmap[e]    = bs;
                stamp[bs]  = now;
                miss[1 + 2*m] = e;
                miss[2 + 2*m] = bs;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
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

// all blocks of a bank copy one miss after the other from mapped host memory; loads before stores keep many PCIe reads in flight
static __global__ void k_moe_cache_copy(const int32_t * __restrict__ miss, const int4 * h0, const int4 * h1, const int4 * h2,
        int4 * b0, int4 * b1, int4 * b2, const int64_t n0, const int64_t n1, const int64_t n2) {
    const int nm = miss[0];
    const int t  = blockIdx.y;
    const int4    * h = t == 0 ? h0 : t == 1 ? h1 : h2;
    int4          * b = t == 0 ? b0 : t == 1 ? b1 : b2;
    const int64_t   n = t == 0 ? n0 : t == 1 ? n1 : n2;
    const int64_t step = (int64_t) gridDim.x*blockDim.x*MC_COPY_U;
    for (int m = 0; m < nm; ++m) {
        const int e = miss[1 + 2*m];
        const int s = miss[2 + 2*m];
        if (e < 0 || s < 0) {
            continue;
        }
        const int4 * x = h + (int64_t) e*n;
        int4       * y = b + (int64_t) s*n;
        for (int64_t i0 = (int64_t) blockIdx.x*blockDim.x*MC_COPY_U + threadIdx.x; i0 < n; i0 += step) {
            int4 v[MC_COPY_U];
#pragma unroll
            for (int u = 0; u < MC_COPY_U; ++u) {
                const int64_t i = i0 + (int64_t) u*blockDim.x;
                v[u] = i < n ? x[i] : make_int4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < MC_COPY_U; ++u) {
                const int64_t i = i0 + (int64_t) u*blockDim.x;
                if (i < n) {
                    y[i] = v[u];
                }
            }
        }
    }
}

void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids   = dst->src[0];
    ggml_tensor       * map   = dst->src[1];
    ggml_tensor       * state = dst->src[2];

    const int32_t * p = (const int32_t *) dst->op_params;
    const int il      = p[0];
    const int n_slots = p[1];
    const int n_dyn   = p[2];
    const int n_banks = p[3];
    const void * host[3];
    memcpy(&host[0], p + 4, sizeof(void *));
    memcpy(&host[1], p + 6, sizeof(void *));
    memcpy(&host[2], p + 8, sizeof(void *));

    GGML_ASSERT(n_banks >= 1 && n_banks <= 3 && ggml_nelements(ids) <= MC_MAX_MISS);
    GGML_ASSERT(ggml_nelements(state) >= 2*n_slots + n_dyn + 1 + 2*MC_MAX_MISS + 4);

    cudaStream_t stream = ctx.stream();
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
    const int32_t * miss = (const int32_t *) state->data + 2*n_slots + n_dyn;
    k_moe_cache_copy<<<dim3(128, n_banks), 256, 0, stream>>>(miss, h[0], h[1], h[2], b[0], b[1], b[2], n[0], n[1], n[2]);
    CUDA_CHECK(cudaGetLastError());
}
'''


# ggml: the op
replace('ggml/include/ggml.h', '        GGML_OP_GLU,\n\n        GGML_OP_COUNT,\n',
        '        GGML_OP_GLU,\n\n        GGML_OP_MOE_CACHE_LOOKUP, // DEVCACHE_PATCH\n\n        GGML_OP_COUNT,\n')
patch('ggml/include/ggml.h', '    // top k elements per row\n    // note: the resulting top k indices are in no particular order\n    GGML_API struct ggml_tensor * ggml_top_k(\n', '''    // DEVCACHE_PATCH: MoE cache slots of the experts in ids from a device map, misses get dynamic slots and are copied from mapped host memory
    GGML_API struct ggml_tensor * ggml_moe_cache_lookup(
            struct ggml_context * ctx,
            struct ggml_tensor  * ids,
            struct ggml_tensor  * map,
            struct ggml_tensor  * state,
            struct ggml_tensor  * bank0,
            struct ggml_tensor  * bank1,
            struct ggml_tensor  * bank2,
            int                   il,
            int                   n_slots,
            int                   n_dyn,
            const void          * host0,
            const void          * host1,
            const void          * host2);

''')
replace('ggml/src/ggml.c', '    "GLU",\n};\n\nstatic_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");',
        '    "GLU",\n    "MOE_CACHE_LOOKUP",\n};\n\nstatic_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");')
replace('ggml/src/ggml.c', '    "glu(x)",\n};\n\nstatic_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");',
        '    "glu(x)",\n    "moe_cache_lookup(x)",\n};\n\nstatic_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");')
patch('ggml/src/ggml.c', 'struct ggml_tensor * ggml_top_k(\n', '''// DEVCACHE_PATCH
struct ggml_tensor * ggml_moe_cache_lookup(
        struct ggml_context * ctx,
        struct ggml_tensor  * ids,
        struct ggml_tensor  * map,
        struct ggml_tensor  * state,
        struct ggml_tensor  * bank0,
        struct ggml_tensor  * bank1,
        struct ggml_tensor  * bank2,
        int                   il,
        int                   n_slots,
        int                   n_dyn,
        const void          * host0,
        const void          * host1,
        const void          * host2) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(map->type == GGML_TYPE_I32 && state->type == GGML_TYPE_I32 && bank0 != NULL);

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ggml_nelements(ids));

    int32_t params[16] = { il, n_slots, n_dyn, bank2 ? 3 : bank1 ? 2 : 1 };
    memcpy(params + 4, &host0, sizeof(host0));
    memcpy(params + 6, &host1, sizeof(host1));
    memcpy(params + 8, &host2, sizeof(host2));
    ggml_set_op_params(result, params, sizeof(params));

    result->op     = GGML_OP_MOE_CACHE_LOOKUP;
    result->src[0] = ids;
    result->src[1] = map;
    result->src[2] = state;
    result->src[3] = bank0;
    result->src[4] = bank1;
    result->src[5] = bank2;

    return result;
}

''')

# CPU: never runs it
patch('ggml/src/ggml-cpu/ggml-cpu.c', '        case GGML_OP_TRANSPOSE:\n            {\n                // nop\n            } break;\n        case GGML_OP_COUNT:\n',
      '        case GGML_OP_MOE_CACHE_LOOKUP:\n            {\n                GGML_ABORT("MOE_CACHE_LOOKUP runs on the GPU only"); // DEVCACHE_PATCH\n            }\n')
patch('ggml/src/ggml-cpu/ggml-cpu.cpp', '    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {\n',
      '    if (op->op == GGML_OP_MOE_CACHE_LOOKUP) {\n        return false; // DEVCACHE_PATCH\n    }\n\n')

# CUDA
write('ggml/src/ggml-cuda/moe-cache.cuh', '#pragma once\n\n#include "common.cuh"\n\n// DEVCACHE_PATCH\nvoid ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);\n')
write('ggml/src/ggml-cuda/moe-cache.cu', CU)
patch('ggml/src/ggml-cuda/ggml-cuda.cu', '#include "ggml-cuda/copy-batch.cuh" // COPYBATCH_PATCH\n', '#include "ggml-cuda/moe-cache.cuh" // DEVCACHE_PATCH\n', before=False)
patch('ggml/src/ggml-cuda/ggml-cuda.cu', '        case GGML_OP_TOP_K_MASK:\n            ggml_cuda_op_top_k_mask(ctx, dst);\n            break;\n',
      '        case GGML_OP_MOE_CACHE_LOOKUP:\n            ggml_cuda_op_moe_cache_lookup(ctx, dst); // DEVCACHE_PATCH\n            break;\n')
patch('ggml/src/ggml-cuda/ggml-cuda.cu', '        case GGML_OP_TOP_K_MASK:\n            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;\n',
      '        case GGML_OP_MOE_CACHE_LOOKUP:\n            return op->type == GGML_TYPE_I32 && op->src[0]->type == GGML_TYPE_I32; // DEVCACHE_PATCH\n')


# llama: device state, activation, graph path
patch('src/llama-moe-cache.h', '    float cpu_share() const;\n', '''    // DEVCACHE_PATCH: device-side lookup for small ubatches once the host experts are mapped; dev_poll returns true when it switches on
    bool dev_poll();
    bool dev_active() const;
    ggml_tensor * build_dev_lookup(ggml_context * ctx, ggml_tensor * ids, int32_t il, int64_t n_tokens, int64_t n_used) const;

''')
C = 'src/llama-moe-cache.cpp'
patch(C, '#include "ggml-cpp.h"\n', '#include "ggml-alloc.h" // DEVCACHE_PATCH\n', before=False)
patch(C, '    // PREDICT_PATCH: state of the next-layer prefetch\n', '''    // DEVCACHE_PATCH: device map [n_layer][n_expert] (slot or -1) and per group state, layout in ggml-cuda/moe-cache.cu
    static constexpr int32_t dev_max_miss = 256;
    bool    dev_enabled = false;
    bool    dev_on      = false;
    int64_t dev_next_try = 0;
    ggml_context_ptr dev_ctx;
    ggml_backend_buffer_ptr dev_buf;
    ggml_tensor * dev_map = nullptr;
    std::vector<ggml_tensor *> dev_state;
    std::vector<int32_t> dev_n_dyn;

''')
patch(C, '    void log_stats() const {\n', r'''    // DEVCACHE_PATCH
    void dev_init(bool no_alloc) {
        const char * v = getenv("LLAMA_MOE_DEVCACHE");
        if (v == nullptr || atoi(v) == 0 || no_alloc || groups.empty() || cpu_share > 0.0f || devices.size() != 1 || !devices[0].buf) {
            return;
        }
        int32_t n_expert = 0;
        for (const group & g : groups) {
            n_expert = std::max(n_expert, g.slru.n_expert);
        }
        ggml_init_params params = {
            /*.mem_size   =*/ (1 + groups.size())*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        dev_ctx.reset(ggml_init(params));
        dev_map = ggml_new_tensor_2d(dev_ctx.get(), GGML_TYPE_I32, n_expert, layers.size());
        ggml_set_name(dev_map, "moe_cache.dev_map");
        for (const group & g : groups) {
            dev_state.push_back(ggml_new_tensor_1d(dev_ctx.get(), GGML_TYPE_I32, 3*std::max(g.n_slots, 1) + 1 + 2*dev_max_miss + 4));
        }
        dev_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(dev_ctx.get(), devices[0].buft));
        if (!dev_buf) {
            LLAMA_LOG_WARN("%s: MoE devcache: allocation failed, it is off\n", __func__);
            return;
        }
        ggml_backend_buffer_set_usage(dev_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        dev_n_dyn.assign(groups.size(), 0);
        dev_enabled = true;
        LLAMA_LOG_INFO("%s: MoE devcache: device lookup switches on once the host experts are mapped\n", __func__);
    }

    bool dev_poll() {
        if (dev_on) {
            dev_gen++; // a new ubatch: the device map may have changed
            return false;
        }
        if (!dev_enabled) {
            return false;
        }
        const int64_t t = ggml_time_us();
        if (t < dev_next_try) {
            return false;
        }
        dev_next_try = t + 2000000;
        ggml_backend_t backend = devices[0].backend;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto resolve = (bool (*)(ggml_backend_t, const void *, size_t, uint64_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_device_ptr");
        if (resolve == nullptr) {
            LLAMA_LOG_WARN("%s: MoE devcache: %s cannot resolve host pointers, it is off\n", __func__, ggml_backend_name(backend));
            dev_enabled = false;
            return false;
        }
        // the copy kernel reads host experts through their host addresses
        auto mapped = [&](int32_t il, int32_t e) {
            for (const ggml_tensor * w : layers[il].experts) {
                const size_t es = w->nb[2];
                const uint8_t * p = (const uint8_t *) w->data + e*es;
                uint64_t d = 0;
                if (es % 16 != 0 || ((uintptr_t) p & 15) != 0 || !resolve(backend, p, es, &d) || d != (uint64_t) (uintptr_t) p) {
                    return false;
                }
            }
            return true;
        };
        auto cold = [&](const group & g, int32_t il, int32_t e) {
            const int32_t s = g.slru.slot_map[il][e];
            return !(s < g.n_slots && g.slru.pinned[s]);
        };
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                if (layers[il].ig < 0) {
                    continue;
                }
                for (int32_t e = 0; e < g.slru.n_expert; ++e) {
                    if (cold(g, il, e)) {
                        if (!mapped(il, e)) {
                            return false;
                        }
                        break;
                    }
                }
            }
        }
        size_t n_checked = 0;
        for (const group & g : groups) {
            if (g.n_slots > 0 && g.slru.n_expert != dev_map->ne[0]) {
                LLAMA_LOG_WARN("%s: MoE devcache: groups differ in expert count, it is off\n", __func__);
                dev_enabled = false;
                return false;
            }
            for (int32_t il : g.layers) {
                if (layers[il].ig < 0) {
                    continue;
                }
                for (int32_t e = 0; e < g.slru.n_expert; ++e) {
                    if (cold(g, il, e)) {
                        if (!mapped(il, e)) {
                            LLAMA_LOG_WARN("%s: MoE devcache: layer %d expert %d is not mapped pinned memory, it is off\n", __func__, il, e);
                            dev_enabled = false;
                            return false;
                        }
                        n_checked++;
                    }
                }
            }
        }
        dev_activate();
        LLAMA_LOG_INFO("%s: MoE devcache: on, %zu host experts mapped, dynamic slots per group:", __func__, n_checked);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            LLAMA_LOG_CONT(" %d", dev_n_dyn[ig]);
        }
        LLAMA_LOG_CONT("\n");
        return true;
    }

    void dev_activate() {
        ggml_backend_synchronize(devices[0].backend);
        const int32_t n_expert = dev_map->ne[0];
        std::vector<int32_t> map_h(ggml_nelements(dev_map), -1);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                const int32_t * slots = g.slru.slot_map[il];
                for (int32_t e = 0; slots != nullptr && e < g.slru.n_expert; ++e) {
                    if (slots[e] < g.n_slots) {
                        map_h[il*n_expert + e] = slots[e];
                    }
                }
            }
            std::vector<int32_t> st(ggml_nelements(dev_state[ig]), 0);
            int32_t n_dyn = 0;
            for (int32_t s = 0; s < g.n_slots; ++s) {
                st[s] = g.slru.key_of[s];
                if (!g.slru.pinned[s]) {
                    st[2*g.n_slots + n_dyn++] = s;
                }
            }
            dev_n_dyn[ig] = n_dyn;
            ggml_backend_tensor_set(dev_state[ig], st.data(), 0, ggml_nbytes(dev_state[ig]));
            // from now on the device owns the dynamic slots; large batches read only the pinned experts
            for (int32_t s = 0; s < g.n_slots; ++s) {
                const int32_t key = g.slru.key_of[s];
                if (!g.slru.pinned[s] && key >= 0) {
                    g.slru.slot_map[key / g.slru.n_expert][key % g.slru.n_expert] = g.slru.empty;
                }
            }
        }
        ggml_backend_tensor_set(dev_map, map_h.data(), 0, ggml_nbytes(dev_map));
        dev_mirror.assign(map_h.size(), -1);
        dev_mirror_gen.assign(layers.size(), 0);
        dev_on = true;
    }

    ggml_tensor * build_dev_lookup(ggml_context * ctx, ggml_tensor * ids, int32_t il, int64_t n_tokens, int64_t n_used) const {
        if (!dev_on || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        const int64_t n_ids = n_tokens*n_used;
        if (n_tokens == 0 || n_tokens > max_batch || n_ids > dev_n_dyn[l.ig] || n_ids > dev_max_miss || l.experts.empty() || l.experts.size() > 3) {
            return nullptr;
        }
        const size_t nb = l.experts.size();
        return ggml_moe_cache_lookup(ctx, ids, dev_map, dev_state[l.ig], g.banks[0], nb > 1 ? g.banks[1] : nullptr, nb > 2 ? g.banks[2] : nullptr,
            il, g.n_slots, dev_n_dyn[l.ig], l.experts[0]->data, nb > 1 ? l.experts[1]->data : nullptr, nb > 2 ? l.experts[2]->data : nullptr);
    }

''')
patch(C, '        log("ubatch <= 8", stats_small);\n', '''        if (dev_on) { // DEVCACHE_PATCH
            uint64_t hits = 0, misses = 0, errors = 0;
            double bytes = 0.0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const group & g = groups[ig];
                if (g.n_slots == 0) {
                    continue;
                }
                uint32_t cnt[4] = { 0, 0, 0, 0 };
                ggml_backend_tensor_get(dev_state[ig], cnt, sizeof(int32_t)*(2*g.n_slots + dev_n_dyn[ig] + 1 + 2*dev_max_miss), sizeof(cnt));
                hits += cnt[0];
                misses += cnt[1];
                errors += cnt[3];
                size_t es = 0;
                for (const ggml_tensor * t : g.ref) {
                    es += t->nb[2];
                }
                bytes += (double) cnt[1]*es;
            }
            LLAMA_LOG_INFO("llama_moe_cache: devcache: hits = %llu, misses = %llu, hit rate = %.2f%%, copied = %.2f MiB, errors = %llu\\n",
                (unsigned long long) hits, (unsigned long long) misses, 100.0*hits/std::max<uint64_t>(hits + misses, 1), bytes/1024.0/1024.0, (unsigned long long) errors);
        }
''')
patch(C, '    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {\n', '    pimpl->dev_init(no_alloc); // DEVCACHE_PATCH\n', before=False)
patch(C, 'float llama_moe_cache::cpu_share() const {\n', '''bool llama_moe_cache::dev_poll() {
    return pimpl->dev_poll(); // DEVCACHE_PATCH
}

bool llama_moe_cache::dev_active() const {
    return pimpl->dev_on; // DEVCACHE_PATCH
}

ggml_tensor * llama_moe_cache::build_dev_lookup(ggml_context * ctx, ggml_tensor * ids, int32_t il, int64_t n_tokens, int64_t n_used) const {
    return pimpl->build_dev_lookup(ctx, ids, il, n_tokens, n_used); // DEVCACHE_PATCH
}

''')
patch('src/llama-graph.cpp', '    ggml_tensor * slot_map = moe_cache->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0]);\n', '''    if (moe_cache->dev_active()) { // DEVCACHE_PATCH: the lookup and the miss copies run on the device, no split
        for (ggml_tensor * w : { up_exps, gate_exps, down_exps, gate_up_exps }) {
            if (w != nullptr && moe_cache->get_experts(w) == nullptr) {
                return nullptr;
            }
        }
        ggml_tensor * ids = selected_experts;
        if (!ggml_is_contiguous(ids)) {
            ids = ggml_cont(ctx0, ids);
        }
        ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));
        ggml_tensor * slots = moe_cache->build_dev_lookup(ctx0, ids, il, selected_experts->ne[1], selected_experts->ne[0]);
        if (slots == nullptr || !ggml_backend_supports_op(moe_cache->backend(il), slots)) {
            return nullptr;
        }
        ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend(il));
        cb(slots, "ffn_moe_slots", il);
        ggml_build_forward_expand(gf, slots);
        return ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]);
    }

''')
# ubatches with more expert ids than dynamic slots: the GPU stages their experts, since the CPU would read punched host copies
patch('src/llama-moe-cache.h', '    float cpu_share() const;\n', '''    ggml_backend_t dev_backend() const;

''')
patch(C, '    std::vector<int32_t> dev_n_dyn;\n', '''    mutable std::vector<int32_t>  dev_mirror;
    mutable std::vector<uint64_t> dev_mirror_gen;
    uint64_t dev_gen = 1;

    // slot of a cached expert, n_slots if not cached; with the device lookup on, dynamic slots come from the device map
    int32_t slot_of(const group & g, int32_t il, int64_t e) const {
        const int32_t * slots = g.slru.slot_map[il];
        if (slots == nullptr) {
            return g.n_slots;
        }
        if (!dev_on || slots[e] < g.n_slots) {
            return slots[e];
        }
        const int32_t n_expert = dev_map->ne[0];
        if (dev_mirror_gen[il] != dev_gen) {
            ggml_backend_tensor_get(dev_map, dev_mirror.data() + (size_t) il*n_expert, (size_t) il*n_expert*sizeof(int32_t), n_expert*sizeof(int32_t));
            dev_mirror_gen[il] = dev_gen;
        }
        const int32_t s = dev_mirror[(size_t) il*n_expert + e];
        return s >= 0 && s < g.n_slots ? s : g.n_slots;
    }
''', before=False)
replace(C, '''        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }
''', '''        const int32_t s0 = slot_of(g, b.il, e); // DEVCACHE_PATCH
        if (s0 >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && s0 + n < g.n_slots && slot_of(g, b.il, e + n) == s0 + n) {
            n++;
        }
''')
replace(C, 'ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);',
        'ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], s0*bank->nb[2]);')
replace(C, '''        const int32_t * slots = g.slru.slot_map[b.il];
        if (backend != devices[g.id].backend || slots == nullptr || slots[e] >= g.n_slots) {
            return nullptr;
        }
        const ggml_tensor * bank = g.banks[b.ip];
        return (const char *) bank->data + (size_t) slots[e]*bank->nb[2];''', '''        const int32_t s = slot_of(g, b.il, e); // DEVCACHE_PATCH
        if (backend != devices[g.id].backend || s >= g.n_slots) {
            return nullptr;
        }
        const ggml_tensor * bank = g.banks[b.ip];
        return (const char *) bank->data + (size_t) s*bank->nb[2];''')
patch(C, 'float llama_moe_cache::cpu_share() const {\n', '''ggml_backend_t llama_moe_cache::dev_backend() const {
    return pimpl->dev_on ? pimpl->devices[0].backend : nullptr; // DEVCACHE_PATCH
}

''')
patch('src/llama-graph.cpp', '        ggml_mul_mat_id(ctx0, cached ? moe_cache->get_experts(w) : w, cur_mm, slots);\n', '''    if (slots == nullptr && moe_cache && moe_cache->dev_active() && moe_cache->get_experts(w) != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, res, moe_cache->dev_backend()); // DEVCACHE_PATCH: the host copies of the pinned experts are punched
    }
''', before=False)

patch('src/llama-context.cpp', '    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {\n', '''    if (moe_cache && moe_cache->dev_poll()) {
        gf_res_prev_active = nullptr; // DEVCACHE_PATCH: graphs built for the host slot maps are not reused
    }

''')
