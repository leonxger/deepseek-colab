"""Overlaps expert staging uploads of large batches with compute (LLAMA_MOE_PREFETCH_MIB, off by default; later patches build on it). usage: python3 patch_prefetch.py LLAMA_SRC_DIR"""
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


# ---- MoE cache: the checks of copy_experts without the copy ----
H = 'src/llama-moe-cache.h'
A = '''    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);
'''
patch(H, A, A + '''
    // PREFETCH_PATCH: true if copy_experts would copy expert e of w from the cache for a copy on backend
    bool has_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const;
''')

MC = 'src/llama-moe-cache.cpp'
A = '''    static const ggml_tensor * base_of(const ggml_tensor * t) {
'''
patch(MC, A, '''    // PREFETCH_PATCH: the checks of copy_experts
    bool has_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        const int32_t * slots = g.slru.slot_map[b.il];
        return backend == devices[g.id].backend && slots != nullptr && slots[e] < g.n_slots;
    }

''' + A)
A = '''int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}
'''
patch(MC, A, A + '''
bool llama_moe_cache::has_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e) const {
    return pimpl->has_expert(backend, w, e); // PREFETCH_PATCH
}
''')

# ---- context state ----
CH = 'src/llama-context.h'
A = '''    copy_experts_info copy_experts;
'''
patch(CH, A, A + '''
    // PREFETCH_PATCH: LLAMA_MOE_PREFETCH_MIB, the non-cached experts of the next host expert tensors are uploaded on a second stream
    struct moe_prefetch_slot {
        const ggml_tensor *    w    = nullptr; // host experts held by the slot
        ggml_tensor *          data = nullptr; // I8 [slot_size] in the prefetch buffer
        std::vector<int32_t>   pos;            // position of each expert of w in the slot, -1 if it is not held
        ggml_backend_event_ptr ready;          // recorded on the prefetch stream after the uploads into the slot
        ggml_backend_event_ptr done;           // recorded on the compute stream after the copies out of the slot
    };

    struct moe_prefetch_stats {
        size_t n_copy = 0;
        size_t n_hit  = 0; // non-cached experts copied from the prefetch buffer
        size_t n_miss = 0; // non-cached experts uploaded on the compute stream
        size_t n_fill = 0;
        size_t bytes  = 0; // uploaded on the prefetch stream
    };

    struct moe_prefetch_info {
        ggml_backend_t          backend = nullptr; // compute backend of the expert copies, nullptr if prefetch is off
        ggml_backend_ptr        stream;            // second backend on the same device, with its own stream
        ggml_context_ptr        ctx;
        ggml_context_ptr        ctx_views;
        ggml_backend_buffer_ptr buf;
        std::vector<moe_prefetch_slot> slots;
        size_t slot_size = 0;

        std::map<const ggml_tensor *, const ggml_tensor *> next; // tensor copied after each one, model order until observed
        std::vector<const ggml_tensor *> want;
        const ggml_tensor * prev = nullptr;

        bool                active = false;
        moe_prefetch_slot * cur    = nullptr; // slot holding experts of the current copy
        bool                waited = false;   // the compute stream already waits for the uploads into cur

        moe_prefetch_stats graph;
        moe_prefetch_stats total;

        ~moe_prefetch_info() {
            if (stream) {
                ggml_backend_synchronize(stream.get());
            }
            if (backend != nullptr) {
                LLAMA_LOG_INFO("%s: MoE prefetch total: %zu copies, %zu experts from the prefetch buffer, %zu non-cached uploaded, %zu fills, %.2f GiB prefetched\\n",
                    __func__, total.n_copy, total.n_hit, total.n_miss, total.n_fill, total.bytes/1024.0/1024.0/1024.0);
            }
        }
    };

    moe_prefetch_info moe_prefetch;

    void    moe_prefetch_init(bool enable);
    void    moe_prefetch_begin(ggml_backend_t backend, const ggml_tensor * w);
    int64_t moe_prefetch_copy(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t first, int64_t end);
    void    moe_prefetch_end(ggml_backend_t backend, const ggml_tensor * w);
    void    moe_prefetch_fill(moe_prefetch_slot & slot, const ggml_tensor * w);
    void    moe_prefetch_log();
''')

# ---- context code ----
C = 'src/llama-context.cpp'
patch(C, '#include <cinttypes>\n', '#include <algorithm> // PREFETCH_PATCH\n#include <cinttypes>\n')

A = '''                moe_cache->prefill(prof, getenv("LLAMA_MOE_CACHE_RESIDENT")); // FROZEN_PATCH
            }
        }

        sched_reserve();
'''
patch(C, A, A + '''        moe_prefetch_init(params.ctx_other == nullptr); // PREFETCH_PATCH
''')

A = '''    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\\n", __func__, status);
    }
'''
patch(C, A, A + '''    moe_prefetch_log(); // PREFETCH_PATCH
''')

A = '''    // group consecutive experts and copy them together
'''
patch(C, A, '''    lctx->moe_prefetch_begin(backend, src); // PREFETCH_PATCH

''' + A)

A = '''            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
'''
patch(C, A, '''            next = lctx->moe_prefetch_copy(backend, src, dst, next, e); // PREFETCH_PATCH
''' + A)

A = '''        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
'''
patch(C, A, '''        next = lctx->moe_prefetch_copy(backend, src, dst, next, last + 1); // PREFETCH_PATCH

''' + A)

A = '''        first = last + 1;
    }

    return true;
}
'''
patch(C, A, '''        first = last + 1;
    }

    lctx->moe_prefetch_end(backend, src); // PREFETCH_PATCH

    return true;
}
''')

A = '''llm_graph_cb llama_context::graph_get_cb() const {
'''
patch(C, A, r'''// PREFETCH_PATCH: docs/prefetch.md
void llama_context::moe_prefetch_init(bool enable) {
    const char * env = getenv("LLAMA_MOE_PREFETCH_MIB");
    const long long mib = env != nullptr ? atoll(env) : 0;
    if (mib <= 0 || model.no_alloc()) {
        return;
    }
    if (!enable) {
        LLAMA_LOG_INFO("%s: MoE prefetch is off in a draft context\n", __func__);
        return;
    }

    // the scheduler offloads the matmuls of host experts to the first GPU
    ggml_backend_t backend = nullptr;
    for (ggml_backend_t b : backend_ptrs) {
        if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = b;
            break;
        }
    }
    if (backend == nullptr) {
        LLAMA_LOG_WARN("%s: MoE prefetch requires a GPU, it is off\n", __func__);
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.async || !props.caps.events) {
        LLAMA_LOG_WARN("%s: MoE prefetch requires async copies and events on %s, it is off\n", __func__, ggml_backend_dev_name(dev));
        return;
    }

    // host expert tensors in graph order and the largest sets of non-cached experts
    std::vector<const ggml_tensor *> order;
    int64_t n_max       = 0;
    size_t  max_tensor  = 0;
    size_t  max_layer   = 0;
    size_t  max_expert  = 0;
    size_t  n_per_layer = 0;
    for (const auto & layer : model.layers) {
        size_t layer_size = 0;
        size_t n_tensors  = 0;
        for (const ggml_tensor * w : { layer.ffn_gate_up_exps, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps }) {
            if (w == nullptr || w->buffer == nullptr || !ggml_backend_buffer_is_host(w->buffer) ||
                ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS || !ggml_is_contiguous(w)) {
                continue;
            }
            int64_t n = 0;
            for (int64_t e = 0; e < w->ne[2]; e++) {
                n += moe_cache && moe_cache->has_expert(backend, w, e) ? 0 : 1;
            }
            order.push_back(w);
            n_max       = std::max(n_max, n);
            max_tensor  = std::max(max_tensor, (size_t) n*w->nb[2]);
            max_expert  = std::max(max_expert, w->nb[2]);
            layer_size += (size_t) n*w->nb[2];
            n_tensors++;
        }
        max_layer   = std::max(max_layer, layer_size);
        n_per_layer = std::max(n_per_layer, n_tensors);
    }
    if (order.empty() || max_tensor == 0) {
        LLAMA_LOG_WARN("%s: MoE prefetch: no non-cached host experts, it is off\n", __func__);
        return;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t budget    = (size_t) mib*1024*1024;
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t slot_size = GGML_PAD(max_tensor, alignment);
    size_t n_slots   = std::min(budget/slot_size, order.size());
    if (n_slots == 0) {
        // one slot that holds the first experts of each set, the others are uploaded on the compute stream
        n_slots   = 1;
        slot_size = budget/alignment*alignment;
    }
    if (slot_size < max_expert) {
        LLAMA_LOG_WARN("%s: MoE prefetch: %lld MiB is less than one expert, it is off\n", __func__, mib);
        return;
    }

    auto & pf = moe_prefetch;
    ggml_init_params ip = {
        /*.mem_size   =*/ n_slots*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    pf.ctx.reset(ggml_init(ip));
    ip.mem_size = 2*ggml_tensor_overhead();
    pf.ctx_views.reset(ggml_init(ip));
    GGML_ASSERT(pf.ctx && pf.ctx_views);
    bool ok = true;
    pf.slots.resize(n_slots);
    for (size_t i = 0; i < n_slots; ++i) {
        moe_prefetch_slot & slot = pf.slots[i];
        slot.data = ggml_new_tensor_1d(pf.ctx.get(), GGML_TYPE_I8, (int64_t) slot_size);
        ggml_format_name(slot.data, "moe_prefetch.%zu", i);
        slot.ready.reset(ggml_backend_event_new(dev));
        slot.done.reset(ggml_backend_event_new(dev));
        ok = ok && slot.ready && slot.done;
    }
    pf.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pf.ctx.get(), buft));
    pf.stream.reset(ggml_backend_dev_init(dev, nullptr));
    if (!ok || !pf.buf || !pf.stream) {
        LLAMA_LOG_WARN("%s: MoE prefetch: failed to allocate %.2f MiB, the events or the second stream on %s, it is off\n", __func__,
            n_slots*slot_size/1024.0/1024.0, ggml_backend_dev_name(dev));
        pf.slots.clear();
        pf.buf.reset();
        pf.stream.reset();
        return;
    }

    pf.backend   = backend;
    pf.slot_size = slot_size;
    for (size_t i = 0; i < order.size(); ++i) {
        pf.next[order[i]] = order[(i + 1) % order.size()];
    }
    LLAMA_LOG_INFO("%s: MoE prefetch: %zu slots x %.2f MiB = %.2f MiB in %s, lookahead %zu expert tensors\n", __func__,
        n_slots, slot_size/1024.0/1024.0, n_slots*slot_size/1024.0/1024.0, ggml_backend_buft_name(buft), n_slots);
    LLAMA_LOG_INFO("%s: MoE prefetch: largest non-cached set %" PRId64 " experts, %.2f MiB per tensor, %.2f MiB per layer, one layer ahead needs LLAMA_MOE_PREFETCH_MIB=%zu\n", __func__,
        n_max, max_tensor/1024.0/1024.0, max_layer/1024.0/1024.0, (n_per_layer*GGML_PAD(max_tensor, alignment) + 1024*1024 - 1)/(1024*1024));
}

void llama_context::moe_prefetch_begin(ggml_backend_t backend, const ggml_tensor * w) {
    auto & pf = moe_prefetch;
    pf.active = pf.backend != nullptr && backend == pf.backend && ggml_is_contiguous(w);
    pf.cur    = nullptr;
    pf.waited = false;
    if (!pf.active) {
        return;
    }
    pf.graph.n_copy++;
    if (pf.prev != nullptr && pf.prev != w) {
        pf.next[pf.prev] = w;
    }
    pf.prev = w;
    for (auto & slot : pf.slots) {
        if (slot.w == w) {
            pf.cur = &slot;
            break;
        }
    }
}

// copies the experts of [first, end) held by the current slot, uploads the gaps between them, returns where the uncopied tail starts
int64_t llama_context::moe_prefetch_copy(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t first, int64_t end) {
    auto & pf = moe_prefetch;
    if (!pf.active || first >= end) {
        return first;
    }
    const moe_prefetch_slot * slot = pf.cur;
    const size_t expert_size = w->nb[2];
    int64_t next  = first;
    int64_t n_hit = 0;
    for (int64_t e = first; slot != nullptr && e < end; ) {
        if (slot->pos[e] < 0) {
            e++;
            continue;
        }
        int64_t n = 1;
        while (e + n < end && slot->pos[e + n] == slot->pos[e] + n) {
            n++;
        }
        if (!pf.waited) {
            ggml_backend_event_wait(backend, slot->ready.get());
            pf.waited = true;
        }
        if (next < e) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) w->data + next*expert_size, next*expert_size, (e - next)*expert_size);
        }

        // the slot is I8, the view gets the type of dst so that the layouts match
        ggml_reset(pf.ctx_views.get());
        ggml_tensor * src_view = ggml_new_tensor_3d(pf.ctx_views.get(), dst->type, dst->ne[0], dst->ne[1], n);
        src_view->view_src  = slot->data;
        src_view->view_offs = slot->pos[e]*expert_size;
        GGML_ASSERT(src_view->view_offs + ggml_nbytes(src_view) <= ggml_nbytes(slot->data));
        ggml_tensor * dst_view = ggml_view_3d(pf.ctx_views.get(), dst, dst->ne[0], dst->ne[1], n, dst->nb[1], dst->nb[2], e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        n_hit += n;
        e     += n;
        next   = e;
    }
    pf.graph.n_hit  += n_hit;
    pf.graph.n_miss += end - first - n_hit;
    return next;
}

void llama_context::moe_prefetch_end(ggml_backend_t backend, const ggml_tensor * w) {
    auto & pf = moe_prefetch;
    if (!pf.active) {
        return;
    }
    if (pf.waited) {
        ggml_backend_event_record(pf.cur->done.get(), backend);
    }
    pf.active = false;
    pf.cur    = nullptr;

    // the next tensors in graph order, at most one per slot
    pf.want.clear();
    for (auto it = pf.next.find(w); it != pf.next.end() && pf.want.size() < pf.slots.size(); it = pf.next.find(it->second)) {
        if (it->second == w || std::find(pf.want.begin(), pf.want.end(), it->second) != pf.want.end()) {
            break;
        }
        pf.want.push_back(it->second);
    }
    for (const ggml_tensor * c : pf.want) {
        moe_prefetch_slot * victim = nullptr;
        bool held = false;
        for (auto & slot : pf.slots) {
            held = held || slot.w == c;
            if (victim == nullptr && std::find(pf.want.begin(), pf.want.end(), slot.w) == pf.want.end()) {
                victim = &slot;
            }
        }
        if (!held && victim != nullptr) {
            moe_prefetch_fill(*victim, c);
        }
    }
}

void llama_context::moe_prefetch_fill(moe_prefetch_slot & slot, const ggml_tensor * w) {
    auto & pf = moe_prefetch;
    ggml_backend_t stream = pf.stream.get();
    const std::vector<bool> & used = copy_experts.used;
    const int64_t n_expert    = w->ne[2];
    const size_t  expert_size = w->nb[2];
    const int64_t n_cap       = (int64_t) (pf.slot_size/expert_size);

    // the uploads overwrite the slot, so they wait for the copies out of its previous content
    ggml_backend_event_wait(stream, slot.done.get());

    // the experts used by the current copy stand in for the unknown ids of w, large batches use nearly all of them
    slot.w = w;
    slot.pos.assign(n_expert, -1);
    int32_t n = 0;
    for (int64_t e = 0; e < n_expert && n < n_cap; e++) {
        const bool take = e < (int64_t) used.size() ? used[e] : true;
        if (take && !(moe_cache && moe_cache->has_expert(pf.backend, w, e))) {
            slot.pos[e] = n++;
        }
    }
    for (int64_t e = 0; e < n_expert; ) {
        if (slot.pos[e] < 0) {
            e++;
            continue;
        }
        int64_t k = 1;
        while (e + k < n_expert && slot.pos[e + k] == slot.pos[e] + k) {
            k++;
        }
        ggml_backend_tensor_set_async(stream, slot.data, (const uint8_t *) w->data + e*expert_size, slot.pos[e]*expert_size, k*expert_size);
        e += k;
    }
    ggml_backend_event_record(slot.ready.get(), stream);

    pf.graph.n_fill++;
    pf.graph.bytes += n*expert_size;
}

void llama_context::moe_prefetch_log() {
    auto & pf = moe_prefetch;
    moe_prefetch_stats & g = pf.graph;
    if (g.n_copy == 0) {
        return;
    }
    LLAMA_LOG_INFO("%s: MoE prefetch: %zu copies, %zu experts from the prefetch buffer, %zu non-cached uploaded, %zu fills, %.1f MiB prefetched\n", __func__,
        g.n_copy, g.n_hit, g.n_miss, g.n_fill, g.bytes/1024.0/1024.0);
    pf.total.n_copy += g.n_copy;
    pf.total.n_hit  += g.n_hit;
    pf.total.n_miss += g.n_miss;
    pf.total.n_fill += g.n_fill;
    pf.total.bytes  += g.bytes;
    g = {};
}

''' + A)
