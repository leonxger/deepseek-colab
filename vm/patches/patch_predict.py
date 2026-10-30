"""Predictive MoE prefetch for decode (LLAMA_MOE_PREDICT, off by default; later patches build on its code). usage: python3 patch_predict.py LLAMA_SRC_DIR"""
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


patch('src/models/deepseek41.cpp', '#include "llama-impl.h"\n', '#include "llama-moe-cache.h" // PREDICT_PATCH\n', before=False)
patch('src/models/deepseek41.cpp', '        cb(logits, "ffn_moe_logits", il);\n', '''
        if (moe_cache != nullptr && il + 1 < n_layer && moe_cache->predict_k() > 0 && // PREDICT_PATCH: next layer's routing from this FFN input
                moe_cache->get_slot_map(il, cur->ne[1], hparams.n_expert_used()) != nullptr &&
                moe_cache->get_slot_map(il + 1, cur->ne[1], hparams.n_expert_used()) != nullptr) {
            const auto & nl = model.layers[il + 1];
            // the normalized input takes the next layer's norm weight; hc_pre keeps its single consumer, so its fusion is unchanged
            ggml_tensor * p = ggml_mul(ctx0, cur, ggml_div(ctx0, nl.ffn_norm, layer.ffn_norm));
            p = build_lora_mm(nl.ffn_gate_inp, p);
            switch ((llama_expert_gating_func_type) hparams.expert_gating_func) {
                case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:       p = ggml_soft_max(ctx0, p); break;
                case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:       p = ggml_sigmoid(ctx0, p); break;
                case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: p = ggml_sqrt(ctx0, ggml_softplus(ctx0, p)); break;
                default: break;
            }
            if (nl.ffn_exp_probs_b != nullptr) {
                p = ggml_add(ctx0, p, nl.ffn_exp_probs_b);
            }
            p = ggml_top_k(ctx0, p, moe_cache->predict_k());
            ggml_set_output(p);
            cb(p, "moe_pred_next", il);
            ggml_build_forward_expand(gf, p);
            moe_cache->set_predict(il, p);
        }
''', before=False)

patch('src/llama-graph.cpp', '    ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend(il));\n    cb(slots, "ffn_moe_slots", il);\n',
      '    moe_cache->bind_predict(il, slots); // PREDICT_PATCH\n', before=False)

patch('src/llama-moe-cache.h', '    float cpu_share() const;\n', '''    // PREDICT_PATCH: experts of the next layer predicted in this layer's split, bound to this layer's slot lookup
    int32_t predict_k() const;
    void set_predict(int32_t il, ggml_tensor * pred);
    void bind_predict(int32_t il, const ggml_tensor * lookup);

''')

C = 'src/llama-moe-cache.cpp'
patch(C, '    std::vector<bool> pinned; // PIN_PATCH: slots that are never evicted\n',
      '    std::vector<int32_t> pf_key; // PREDICT_PATCH: key prefetched into a slot, -1 once counted\n', before=False)
patch(C, '        pinned.assign(n_slots, false); // PIN_PATCH\n', '        pf_key.assign(n_slots, -1); // PREDICT_PATCH\n', before=False)
patch(C, '    void touch(int32_t s) {\n', '''    // PREDICT_PATCH: move slot s to the least recently used end of probation, the first victim
    void move_head(int32_t s) {
        const bool from = is_protected[s];
        if (!from && s == head[0]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }
        prev[s] = -1;
        next[s] = head[0];
        if (head[0] >= 0) {
            prev[head[0]] = s;
        } else {
            tail[0] = s;
        }
        head[0] = s;
        is_protected[s] = false;
        n_protected -= int(from);
    }

    // PREDICT_PATCH: like victim, -1 instead of an abort
    int32_t victim_or_none(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                if (!pinned[s] && (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen)) {
                    return s;
                }
            }
        }
        return -1;
    }

''')
patch(C, '    uint32_t cpu_share_q = 0; // cpu_share in 1/65536', '''    // PREDICT_PATCH: state of the next-layer prefetch
    int32_t predict = 0;
    bool    predict_fetch = false;
    int32_t predict_max = 32;
    std::vector<ggml_tensor *> pending_pred;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> pred_of;
    std::vector<int32_t> pred_ids;
    std::vector<uint32_t> pred_mark;
    uint32_t pred_gen = 0;
    int32_t pred_layer = -1;
    std::vector<moe_cache_slru::fill> pf_fills;
    ggml_backend_t pf_backend = nullptr;
    ggml_backend_event_t pf_event = nullptr;
    bool pf_pending = false;
    struct { size_t lookups = 0, pred_uncached = 0, misses = 0, pred_hit = 0, fetched = 0, used = 0, bytes = 0, skipped = 0; } pst;

''')
patch(C, '        frozen = getenv("LLAMA_MOE_CACHE_FROZEN") != nullptr; // FROZEN_PATCH\n', '''        if (const char * v = getenv("LLAMA_MOE_PREDICT"); v != nullptr && cpu_share == 0.0f) { // PREDICT_PATCH
            predict = std::clamp(atoi(v), 0, 32);
            predict_fetch = getenv("LLAMA_MOE_PREDICT_FETCH") != nullptr && atoi(getenv("LLAMA_MOE_PREDICT_FETCH")) != 0;
            predict_max = getenv("LLAMA_MOE_PREDICT_MAX") ? std::max(atoi(getenv("LLAMA_MOE_PREDICT_MAX")), 1) : 32;
            pending_pred.assign(layers.size(), nullptr);
            LLAMA_LOG_INFO("%s: MoE predict: top %d of the next layer, prefetch %d, at most %d per layer\\n", __func__, predict, (int) predict_fetch, predict_max);
        }
''', before=False)
patch(C, '    ~impl() {\n        log_stats();\n', '''        if (pf_backend != nullptr) { // PREDICT_PATCH
            ggml_backend_synchronize(pf_backend);
            ggml_backend_event_free(pf_event);
            ggml_backend_free(pf_backend);
        }
''', before=False)
patch(C, '    void begin() {\n        stash.clear();\n', '''        if (pf_pending) {
            ggml_backend_synchronize(pf_backend); // PREDICT_PATCH: no prefetch crosses a graph
            pf_pending = false;
        }
''', before=False)
patch(C, '        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));\n        ggml_backend_synchronize(backend);\n', '''        const auto it_pred = pred_of.find(lookup); // PREDICT_PATCH
        ggml_tensor * pred = it_pred != pred_of.end() ? it_pred->second : nullptr;
        if (pred != nullptr) {
            pred_ids.resize(ggml_nelements(pred));
            ggml_backend_tensor_get_async(backend, pred, pred_ids.data(), 0, ggml_nbytes(pred));
        }
''')
patch(C, '        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));\n        ggml_backend_synchronize(backend);\n', '''        if (pf_pending) {
            ggml_backend_event_wait(backend, pf_event); // PREDICT_PATCH: the prefetch lands before this layer's uploads and experts
            pf_pending = false;
        }
''', before=False)
patch(C, '        // the next copy synchronizes the backend before it changes the slot map again\n',
      '        predict_step(backend, il, g, pred); // PREDICT_PATCH\n\n')
patch(C, '    void log_stats() const {\n', r'''    // PREDICT_PATCH: count this layer against the last prediction, then predict or prefetch the next layer
    void predict_step(ggml_backend_t backend, int32_t il, group & g, const ggml_tensor * pred) {
        if (predict == 0) {
            return;
        }
        const int32_t n_expert = g.slru.n_expert;
        const int32_t * slots = g.slru.slot_map[il];
        if (pred_layer == il) {
            pst.misses += fills.size();
            for (const auto & f : fills) {
                pst.pred_hit += pred_mark[f.expert] == pred_gen;
            }
            for (int32_t e : g.slru.uniq) {
                const int32_t s = slots[e];
                if (s < g.n_slots && g.slru.pf_key[s] == il*n_expert + e &&
                        std::none_of(fills.begin(), fills.end(), [&](const moe_cache_slru::fill & f) { return f.slot == s; })) {
                    pst.used++;
                    g.slru.pf_key[s] = -1;
                }
            }
        }
        pred_layer = -1;
        const int32_t il2 = il + 1;
        if (pred == nullptr || il2 >= (int32_t) layers.size() || layers[il2].ig < 0) {
            return;
        }
        layer & l2 = layers[il2];
        group & g2 = groups[l2.ig];
        int32_t * slots2 = g2.slru.slot_map[il2];
        if (slots2 == nullptr) {
            return;
        }
        if (++pred_gen == 0) {
            pred_gen = 1;
        }
        pred_mark.resize(g2.slru.n_expert, 0);
        pred_layer = il2;
        pst.lookups++;
        pf_fills.clear();
        for (size_t i = 0; i < pred_ids.size(); ++i) {
            const int32_t e = pred_ids[i];
            if (e < 0 || e >= g2.slru.n_expert || pred_mark[e] == pred_gen) {
                continue;
            }
            pred_mark[e] = pred_gen;
            if (slots2[e] < g2.n_slots) {
                continue;
            }
            pst.pred_uncached++;
            if (!predict_fetch) {
                continue;
            }
            if ((int32_t) pf_fills.size() >= predict_max) {
                pst.skipped++;
                continue;
            }
            const int32_t s = g2.slru.victim_or_none(il);
            if (s < 0) {
                pst.skipped++;
                continue;
            }
            const int32_t old = g2.slru.key_of[s];
            if (old >= 0) {
                g2.slru.slot_map[old / g2.slru.n_expert][old % g2.slru.n_expert] = g2.slru.empty;
            } else {
                g2.slru.n_filled++;
            }
            g2.slru.key_of[s] = il2*g2.slru.n_expert + e;
            g2.slru.pf_key[s] = g2.slru.key_of[s];
            slots2[e] = s;
            g2.slru.move(s, false);
            pf_fills.push_back({ e, s });
        }
        if (pf_fills.empty()) {
            return;
        }
        // wrong predictions are evicted first, used ones are promoted by the next plan
        for (const auto & f : pf_fills) {
            g2.slru.move_head(f.slot);
        }
        if (pf_backend == nullptr) {
            pf_backend = ggml_backend_dev_init(ggml_backend_get_device(backend), nullptr);
            pf_event   = ggml_backend_event_new(ggml_backend_get_device(backend));
            GGML_ASSERT(pf_backend != nullptr && pf_event != nullptr);
        }
        for (size_t ip = 0; ip < l2.experts.size(); ++ip) {
            const ggml_tensor * w = l2.experts[ip];
            const size_t es = w->nb[2];
            for (const auto & f : pf_fills) {
                ggml_backend_tensor_set_async(pf_backend, g2.banks[ip], (const uint8_t *) w->data + f.expert*es, f.slot*es, es);
                pst.bytes += es;
            }
        }
        ggml_backend_event_record(pf_event, pf_backend);
        pf_pending = true;
        pst.fetched += pf_fills.size();
    }

    void set_predict(int32_t il, ggml_tensor * pred) {
        if (il >= 0 && il < (int32_t) pending_pred.size()) {
            pending_pred[il] = pred;
        }
    }

    void bind_predict(int32_t il, const ggml_tensor * lookup) {
        if (il >= 0 && il < (int32_t) pending_pred.size()) {
            pred_of[lookup] = pending_pred[il];
            pending_pred[il] = nullptr;
        }
    }

''')
patch(C, '        log("ubatch <= 8", stats_small);\n', '''        if (predict > 0 && pst.lookups > 0) { // PREDICT_PATCH
            LLAMA_LOG_INFO("llama_moe_cache: predict: %zu lookups, %zu predicted uncached, misses %zu of which predicted %zu (top %d), prefetched %zu (%.2f MiB), used %zu, skipped %zu\\n",
                pst.lookups, pst.pred_uncached, pst.misses, pst.pred_hit, predict, pst.fetched, pst.bytes/1024.0/1024.0, pst.used, pst.skipped);
        }
''')
patch(C, 'float llama_moe_cache::cpu_share() const {\n', '''int32_t llama_moe_cache::predict_k() const {
    return pimpl->predict; // PREDICT_PATCH
}

void llama_moe_cache::set_predict(int32_t il, ggml_tensor * pred) {
    pimpl->set_predict(il, pred); // PREDICT_PATCH
}

void llama_moe_cache::bind_predict(int32_t il, const ggml_tensor * lookup) {
    pimpl->bind_predict(il, lookup); // PREDICT_PATCH
}

''')
