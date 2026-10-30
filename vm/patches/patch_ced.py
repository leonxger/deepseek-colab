"""Exact CED prefill for DeepSeek-V4.1: prompt tokens before the last 2,414 run only the encoder layers (LLAMA_DSV41_CED=1). usage: python3 patch_ced.py LLAMA_SRC_DIR"""
import os, sys

src = sys.argv[1]


def patch(rel, anchor, text, before=True):
    p = os.path.join(src, rel)
    s = open(p).read()
    if text.strip() in s:
        print('already patched', rel, text.strip()[:40])
        return
    assert s.count(anchor) == 1, (rel, anchor)
    s = s.replace(anchor, text + anchor if before else anchor + text)
    open(p, 'w').write(s)
    print('patched', rel)


# staging API used by the server
patch('src/llama-ext.h', '// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.\n', '''// CED_PATCH: decoder tokens a DeepSeek-V4.1 output needs before it (env LLAMA_DSV41_CED, LLAMA_DSV41_CED_REPLAY), 0 = CED off
LLAMA_API int32_t llama_dsv41_ced_replay(const struct llama_context * ctx);

// CED_PATCH: positions of seq_id below cut skip the decoder except its KV projection, cut < 0 never skips
LLAMA_API void llama_dsv41_ced_set_cut(struct llama_context * ctx, llama_seq_id seq_id, llama_pos cut);

''')

# graph parameter: part of the topology, so a skip graph is never reused for a full ubatch
patch('src/llama-graph.h', '    llm_graph_cb cb;\n\n    llm_graph_result * res;\n', '''
    // CED_PATCH: first decoder layer of a DeepSeek-V4.1 encoder-only ubatch, -1 = full graph
    int32_t ced_dec_first = -1;
''', before=False)
patch('src/llama-graph.h', '            cparams.causal_attn             == other.cparams.causal_attn             &&\n',
      '            ced_dec_first                   == other.ced_dec_first                   && // CED_PATCH\n', before=False)

# context: configuration, per-sequence cut, per-ubatch decision
patch('src/llama-context.h', '    int decode(const llama_batch_ext & batch_inp);\n', '''
    // CED_PATCH: DeepSeek-V4.1 encoder-only prefill, see llama_dsv41_ced_replay
    int32_t ced_replay() const;
    void    ced_set_cut(llama_seq_id seq_id, llama_pos cut);
    bool    ced_skip(const llama_ubatch & ubatch) const;
''', before=False)
patch('src/llama-context.h', '    bool graph_reuse_disable = false;\n', '''
    // CED_PATCH: first decoder layer (-1 = off), replay window, dry run, flag of the ubatch being built, per-sequence cut
    int32_t ced_dec_first = -1;
    int32_t ced_replay_n  = 0;
    bool    ced_dry       = false;
    bool    ced_skip_cur  = false;
    std::vector<llama_pos> ced_cut;
''', before=False)

patch('src/llama-context.cpp', '    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732\n', '''    // CED_PATCH: DeepSeek-V4.1 Causal Encoder-Decoder prefill, opt-in (docs/ced.md)
    if (const char * ced = getenv("LLAMA_DSV41_CED"); ced != nullptr && atoi(ced) != 0 && model.arch == LLM_ARCH_DEEPSEEK41) {
        const auto & hp = model.hparams;
        const int32_t n_layer_ced = (int32_t) hp.n_layer();
        int32_t d0 = -1;
        bool swa_ok = hp.n_swa > 1 && hp.swa_type == LLAMA_SWA_TYPE_STANDARD;
        for (int32_t il = 0; il < n_layer_ced; ++il) {
            // the last global KV source starts the decoder: every later layer only reads it
            d0 = hp.dsv41_kv_src[il] == il ? il : d0;
            swa_ok = swa_ok && hp.is_swa(il);
        }
        if (d0 > 0 && d0 + 1 < n_layer_ced && swa_ok) {
            const char * rep = getenv("LLAMA_DSV41_CED_REPLAY");
            ced_dec_first = d0;
            ced_replay_n  = rep ? std::max(1, atoi(rep)) : (n_layer_ced - 1 - d0)*(int32_t) (hp.n_swa - 1) + 1;
            ced_dry       = atoi(ced) == 2;
            ced_cut.assign(LLAMA_MAX_SEQ, -1);
        }
        LLAMA_LOG_INFO("%s: DSV4.1 CED prefill: %s, first decoder layer %d, n_swa = %u, replay = %d tokens\\n", __func__,
                ced_dec_first < 0 ? "unsupported layout" : ced_dry ? "dry run (aligned batches, no skip)" : "on", d0, hp.n_swa, ced_replay_n);
    }

''')
patch('src/llama-context.cpp', '        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);\n',
      '        ced_skip_cur = ced_skip(ubatch); // CED_PATCH\n')
patch('src/llama-context.cpp', '        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);\n',
      '        ced_skip_cur = false; // CED_PATCH: a later graph_reserve must see the full graph\n', before=False)
patch('src/llama-context.cpp', '        /*.res         =*/ res,\n',
      '        /*.ced_dec_first =*/ ced_skip_cur ? ced_dec_first : -1, // CED_PATCH\n', before=False)
patch('src/llama-context.cpp', 'struct ggml_cgraph * llama_graph_reserve(\n', r'''// CED_PATCH: DeepSeek-V4.1 encoder-only prefill

int32_t llama_context::ced_replay() const {
    if (ced_dec_first < 0) {
        return 0;
    }
    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            return 0; // DSpark reads the decoder inputs of every prompt token
        }
    }
    return ced_replay_n;
}

void llama_context::ced_set_cut(llama_seq_id seq_id, llama_pos cut) {
    if (seq_id >= 0 && (size_t) seq_id < ced_cut.size()) {
        ced_cut[seq_id] = cut;
    }
}

// a ubatch skips the decoder when no token is an output and every token lies before the cut of each of its sequences
bool llama_context::ced_skip(const llama_ubatch & ubatch) const {
    if (ced_dec_first < 0 || ced_dry || cparams.embeddings || cparams.embeddings_nextn || !ubatch.token || ubatch.embd) {
        return false;
    }
    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            return false;
        }
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            return false;
        }
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            if (seq_id < 0 || (size_t) seq_id >= ced_cut.size() || ubatch.pos[i] >= ced_cut[seq_id]) {
                return false;
            }
        }
    }
    LLAMA_LOG_DEBUG("%s: encoder-only ubatch, %u tokens from pos %d\n", __func__, ubatch.n_tokens, ubatch.pos[0]);
    return true;
}

int32_t llama_dsv41_ced_replay(const llama_context * ctx) {
    return ctx->ced_replay();
}

void llama_dsv41_ced_set_cut(llama_context * ctx, llama_seq_id seq_id, llama_pos cut) {
    ctx->ced_set_cut(seq_id, cut);
}

''')

# graph: the encoder-only ubatch stops at the first decoder layer after its KV projection
patch('src/models/models.h', '        bool sparse = true;\n', '''        // CED_PATCH: build_attention41 stores the window KV and the compressed rows only (encoder-only ubatch)
        bool ced_proj = false;
''', before=False)
patch('src/models/deepseek41.cpp', '    ggml_tensor * out = nullptr;\n    if (ratio == 0 || !ci.kq_mask) {\n', '''    if (ced_proj) {
        // CED_PATCH: the window KV exactly as both attention paths below store it, no attention
        if (inp_attn->self_k_rot) {
            kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot);
        }
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));
        return nullptr;
    }

''')
patch('src/models/deepseek41.cpp',
      '        ggml_tensor * residual = inpL;\n        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);\n',
      '''        if (params.ced_dec_first >= 0 && il >= params.ced_dec_first) {
            // CED_PATCH: no token of this ubatch is an output or in a replay window: build only what later tokens read (docs/ced.md)
            GGML_ASSERT(il == params.ced_dec_first && pre_prev);
            cur = build_hc_pre(inpL, pre_prev, il);
            cb(cur, "hc_attn_pre", il);
            cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
            ced_proj = true;
            build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il);
            ced_proj = false;
            for (int jl = il + 1; jl < n_layer; ++jl) {
                const int32_t src   = hparams.dsv41_kv_src[jl];
                const int64_t ratio = hparams.dsv4_compress_ratios[jl];
                if (ratio != 0 && src != jl && inp_dsv4->mctx->get_comp_read_layer(jl) == jl && comp_row_of[src]) {
                    // per-device mirror of the kv source, fed as build_attention41 feeds it
                    const auto & ci       = ratio == 2 ? inp_dsv4->get_csa()       : inp_dsv4->get_hca();
                    const auto * comp_ctx = ratio == 2 ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
                    ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, comp_row_of[src], ci.state_write_idxs, jl));
                }
            }
            break;
        }

''')

# server: register the cut per prompt, end a batch at it, keep only exact checkpoints, drop interrupted CED prompts
SC = 'tools/server/server-context.cpp'
patch(SC, '#include "llama.h"\n', '#include "../../src/llama-ext.h" // CED_PATCH: llama_dsv41_ced_*\n', before=False)
patch(SC, '    server_prompt prompt;\n', '''    // CED_PATCH: first prompt token whose decoder runs in full, and the first cache length a continuation may start from
    int32_t ced_cut   = -1;
    int32_t ced_exact = -1;

    bool ced_dirty() const {
        return ced_exact > prompt.n_tokens();
    }

''')
patch(SC, '        spec_is_replay = false;\n', '''        ced_cut   = -1; // CED_PATCH
        ced_exact = -1;
''', before=False)
patch(SC, '            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\\n", prompt.n_tokens(), truncated);\n', '''
            // CED_PATCH: an interrupted CED prompt holds decoder window states that no continuation may read
            if (ced_dirty()) {
                SLT_WRN(*this, "CED prompt stopped at %d tokens, before %d: clearing the cache\\n", prompt.n_tokens(), ced_exact);
                prompt_clear();
            }
''', before=False)
patch(SC, '                        slot.prompt.tokens.keep_first(n_past);\n', '''
                        // CED_PATCH: full decoder only for the R - 1 tokens before the last checkpoint (prompt end - 4) and after it
                        {
                            const int32_t ced_r = llama_dsv41_ced_replay(ctx_tgt);
                            const int32_t ced_c = slot.task->n_tokens() - std::min<int32_t>(n_batch, 4) - (ced_r - 1);
                            const bool ced_on = ced_r > 0 && ced_c > n_past && slot.task->type == SERVER_TASK_TYPE_COMPLETION &&
                                    slot.task->n_tokens_shared == 0 && !input_tokens.has_mtmd && !slot.need_embd();
                            slot.ced_cut   = ced_on ? ced_c : -1;
                            slot.ced_exact = ced_on ? ced_c + ced_r - 1 : -1;
                            llama_dsv41_ced_set_cut(ctx_tgt, slot.id, slot.ced_cut);
                            if (ced_on) {
                                SLT_INF(slot, "CED: decoder in full from token %d of %d (replay %d)\\n", ced_c, slot.task->n_tokens(), ced_r);
                            }
                        }
''', before=False)
patch(SC, '                        // break at the last user message, or at user messages at least min step past the last checkpoint\n', '''                        // CED_PATCH: end the batch at the cut, so no ubatch mixes encoder-only and full tokens
                        if (slot.prompt.n_tokens() == slot.ced_cut) {
                            break;
                        }

''')
patch(SC, '                    do_checkpoint = do_checkpoint && !has_mtmd;\n', '''
                    // CED_PATCH: a checkpoint before ced_exact would restore decoder window states that were never computed
                    do_checkpoint = do_checkpoint && (slot.ced_cut < 0 || (int32_t) n_tokens_start >= slot.ced_exact);
''', before=False)

for rel, mark in [('src/llama-ext.h', 'llama_dsv41_ced_set_cut'), ('src/llama-graph.h', 'other.ced_dec_first'),
                  ('src/llama-context.h', 'ced_skip_cur'), ('src/llama-context.cpp', 'ced_skip_cur = false;'),
                  ('src/models/models.h', 'ced_proj'), ('src/models/deepseek41.cpp', 'params.ced_dec_first'),
                  (SC, 'slot.ced_exact')]:
    assert mark in open(os.path.join(src, rel)).read(), (rel, mark)
print('CED patch complete')
