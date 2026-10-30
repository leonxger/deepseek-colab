"""Profile-prefilled hybrid MoE cache on PR #30966: LLAMA_MOE_CACHE_PROFILE fills and pins the hottest experts, LLAMA_MOE_CACHE_RESIDENT lists them. usage: python3 patch_e.py LLAMA_SRC_DIR"""
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


C = 'src/llama-moe-cache.cpp'
patch(C, '        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;\n',
      '''        if (const char * z = getenv("LLAMA_MOE_CACHE_ZERO"); z != nullptr && cpu_share > 0.0f) {
            n_zero = std::clamp<int32_t>(atoi(z), std::min<int32_t>(n_expert_used, 64), 64); // FROZEN_PATCH
        }
        frozen = getenv("LLAMA_MOE_CACHE_FROZEN") != nullptr; // FROZEN_PATCH
''', before=False)
patch(C, '    uint32_t cpu_share_q = 0;', '    bool     frozen = false; // FROZEN_PATCH: no uploads allowed, host copies of resident experts may be gone\n')
patch(C, '        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together\n',
      '''        if (frozen && !fills.empty()) {
            GGML_ABORT("FROZEN_PATCH: frozen MoE cache would upload %zu experts in layer %d", fills.size(), il);
        }
''')
patch(C, '    void begin() {\n', r'''    // FROZEN_PATCH: fill the cache in profile order, then every remaining slot, and write the resident experts
    void prefill(const char * path, const char * out) {
        size_t n = 0, bytes = 0;
        auto put = [&](int32_t il, int32_t e) {
            if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
                return;
            }
            layer & l = layers[il];
            group & g = groups[l.ig];
            int32_t * slots = g.slru.slot_map[il];
            if (slots == nullptr || e < 0 || e >= g.slru.n_expert || slots[e] < g.n_slots || g.slru.n_filled >= g.n_slots) {
                return;
            }
            int32_t s = -1;
            for (int32_t t = g.slru.head[0]; t >= 0 && s < 0; t = g.slru.next[t]) {
                if (g.slru.key_of[t] < 0) {
                    s = t;
                }
            }
            if (s < 0) {
                return;
            }
            g.slru.key_of[s] = il*g.slru.n_expert + e;
            slots[e] = s;
            g.slru.move(s, false);
            g.slru.n_filled++;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t es = w->nb[2];
                ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + e*es, s*es, es);
                bytes += es;
            }
            n++;
        };
        FILE * f = fopen(path, "r");
        if (f == nullptr) {
            throw std::runtime_error(std::string("failed to open the MoE cache profile ") + path);
        }
        int il = 0, e = 0;
        while (fscanf(f, "%d %d", &il, &e) == 2) {
            put(il, e);
        }
        fclose(f);
        const size_t n_profiled = n;
        for (int32_t l2 = 0; l2 < (int32_t) layers.size(); ++l2) {
            if (layers[l2].ig >= 0) {
                for (int32_t e2 = 0; e2 < groups[layers[l2].ig].slru.n_expert; ++e2) {
                    put(l2, e2);
                }
            }
        }
        size_t n_res = 0;
        FILE * fo = out != nullptr && *out != '\0' ? fopen(out, "w") : nullptr;
        for (int32_t l2 = 0; l2 < (int32_t) layers.size(); ++l2) {
            if (layers[l2].ig < 0) {
                continue;
            }
            const group & g = groups[layers[l2].ig];
            for (int32_t e2 = 0; e2 < g.slru.n_expert; ++e2) {
                if (g.slru.slot_map[l2] != nullptr && g.slru.slot_map[l2][e2] < g.n_slots) {
                    n_res++;
                    if (fo) {
                        fprintf(fo, "%d %d\n", l2, e2);
                    }
                }
            }
        }
        if (fo) {
            fclose(fo);
        }
        for (const group & g : groups) {
            if (g.n_slots > 0 && g.slru.n_filled != g.n_slots) {
                throw std::runtime_error("FROZEN_PATCH: MoE cache prefill left empty slots");
            }
        }
        LLAMA_LOG_INFO("%s: prefilled %zu experts (%zu from the profile), %.2f GiB, %zu resident\n", __func__, n, n_profiled, bytes/1024.0/1024.0/1024.0, n_res);
    }

''')
patch(C, 'float llama_moe_cache::cpu_share() const {\n', '''void llama_moe_cache::prefill(const char * path, const char * out) {
    pimpl->prefill(path, out); // FROZEN_PATCH
}

''')
patch('src/llama-moe-cache.h', '    float cpu_share() const;\n', '''    // FROZEN_PATCH: fill every slot in profile order (file of "layer expert" lines) and write the resident experts to out
    void prefill(const char * path, const char * out);

''')
patch('src/llama-context.cpp',
      '            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size, cparams.moe_cache_cpu_share);\n',
      '''            if (const char * prof = getenv("LLAMA_MOE_CACHE_PROFILE"); prof != nullptr && *prof != '\\0' && !model.no_alloc()) {
                moe_cache->prefill(prof, getenv("LLAMA_MOE_CACHE_RESIDENT")); // FROZEN_PATCH
            }
''', before=False)

# PIN_PATCH: LLAMA_MOE_CACHE_PIN_PREFILL=1 pins the profile experts in the cache and keeps LLAMA_MOE_CACHE_DYN slots per group for misses
patch(C, '    int32_t n_filled    = 0; // slots that hold an expert\n', '    std::vector<bool> pinned; // PIN_PATCH: slots that are never evicted\n')
patch(C, '        is_protected.assign(n_slots, false);\n', '        pinned.assign(n_slots, false); // PIN_PATCH\n', before=False)
patch(C, '            for (int32_t s = head[segment]; s >= 0; s = next[s]) {\n                const int32_t key = key_of[s];\n',
      '                if (pinned[s]) {\n                    continue; // PIN_PATCH\n                }\n', before=False)
patch(C, '        size_t n = 0, bytes = 0;\n        auto put = [&](int32_t il, int32_t e) {\n',
      '        const bool pin = getenv("LLAMA_MOE_CACHE_PIN_PREFILL") != nullptr; // PIN_PATCH\n'
      '        const int32_t dyn = getenv("LLAMA_MOE_CACHE_DYN") ? atoi(getenv("LLAMA_MOE_CACHE_DYN")) : 320;\n')
patch(C, '            if (slots == nullptr || e < 0 || e >= g.slru.n_expert || slots[e] < g.n_slots || g.slru.n_filled >= g.n_slots) {\n',
      '            if (pin && g.slru.n_filled >= g.n_slots - std::min(dyn, g.n_slots/4)) {\n                return; // PIN_PATCH: keep the dynamic slots free\n            }\n')
patch(C, '            g.slru.n_filled++;\n            for (size_t ip = 0; ip < l.experts.size(); ++ip) {\n',
      '            g.slru.pinned[s] = pin; // PIN_PATCH\n')
patch(C, '            if (layers[l2].ig >= 0) {\n                for (int32_t e2 = 0; e2 < groups[layers[l2].ig].slru.n_expert; ++e2) {\n                    put(l2, e2);\n',
      '            if (pin) {\n                break; // PIN_PATCH: no top-up when pinning\n            }\n')
patch(C, '            if (g.n_slots > 0 && g.slru.n_filled != g.n_slots) {\n', '            if (pin) {\n                continue; // PIN_PATCH\n            }\n')

# PADZERO_PATCH: LLAMA_MOE_PAD_ZERO=1 takes the 512-byte guard after an uploaded run from a zero buffer, never from host bytes of the next expert
patch('src/llama-context.cpp',
      '        if (size > 0) {\n            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);\n        }\n',
      '''        static const bool pad_zero = getenv("LLAMA_MOE_PAD_ZERO") != nullptr; // PADZERO_PATCH
        static ggml_backend_buffer_t zero_buf = nullptr;
        if (pad_zero && padding > 0 && zero_buf == nullptr) {
            ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
            zero_buf = ggml_backend_buft_alloc_buffer(hbt ? hbt : ggml_backend_cpu_buffer_type(), 4096);
            ggml_backend_buffer_clear(zero_buf, 0);
        }
        if (pad_zero && padding > 0) {
            if (size > padding) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size - padding);
            }
            ggml_backend_tensor_set_async(backend, dst, ggml_backend_buffer_get_base(zero_buf), offset + size - padding, padding);
        } else
''')
