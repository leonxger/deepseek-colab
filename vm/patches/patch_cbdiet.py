"""Compute-buffer diet for DeepSeek-V4.1 prefill: the indexer block mask is broadcast, not materialized (bit-identical); LLAMA_DSV41_CBDIET=0 restores the old graph. usage: python3 patch_cbdiet.py LLAMA_SRC_DIR"""
import os, sys

src = sys.argv[1]
rel = 'src/models/deepseek41.cpp'
p = os.path.join(src, rel)
s = open(p).read()
if 'CBDIET_PATCH' in s:
    print('already patched', rel)
    sys.exit(0)


def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


sub('''static bool dsv41_env_on(const char * name) {
    const char * v = getenv(name);
    return !v || atoi(v) != 0;
}
''', '''static bool dsv41_env_on(const char * name) {
    const char * v = getenv(name);
    return !v || atoi(v) != 0;
}

static int64_t dsv41_env_int(const char * name, int64_t def) { // CBDIET_PATCH
    const char * v = getenv(name);
    return v ? atoll(v) : def;
}
''')

sub('''            const int64_t chunk = fused_idx ? n_tok :
                std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) });
''', '''            // CBDIET_PATCH: the fused indexer scores each query row in its own blocks, so query chunks are bit-identical and cap the score size
            static const bool    cbdiet        = dsv41_env_on("LLAMA_DSV41_CBDIET");
            static const int64_t idx_chunk_mib = dsv41_env_int("LLAMA_DSV41_IDX_CHUNK_MIB", 0);
            const int64_t chunk = fused_idx ? (cbdiet && idx_chunk_mib > 0 ? std::clamp<int64_t>((idx_chunk_mib << 18)/(n_comp*n_stream), 1, n_tok) : n_tok) :
                std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) });
''')

sub('''                if (fused_idx) {
                    score = ggml_lightning_indexer(ctx0, iq, ik, iw, ci.kq_mask);
''', '''                if (fused_idx) {
                    score = nc == n_tok ? ggml_lightning_indexer(ctx0, iq, ik, iw, ci.kq_mask) :
                        ggml_lightning_indexer(ctx0, ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]),
                                ik, rows(iw), rows(ci.kq_mask));
''')

sub('''                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
''', '''                } else if (use_blk && cbdiet && n_blk*bs == n_comp) {
                    // CBDIET_PATCH: the same 0/-inf block mask added by broadcast over the block size; FILL never reads its source, a view of the score
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
                    const size_t  es  = ggml_element_size(score);
                    ggml_tensor * bm    = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, n_blk, nc*n_stream, 1, es, es*n_blk, es*n_blk*nc*n_stream, 0), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, k_b, nc*n_stream, 1, es, es*k_b, es*k_b*nc*n_stream, 0), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                    cb(bm, "cand_mask", il);
                    score = ggml_reshape_4d(ctx0, ggml_add_inplace(ctx0, ggml_reshape_4d(ctx0, score, bs, n_blk, nc, n_stream),
                            ggml_reshape_4d(ctx0, bm, 1, n_blk, nc, n_stream)), n_comp, nc, 1, n_stream);
                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
''')

open(p, 'w').write(s)
t = open(p).read()
for mark in ['dsv41_env_int("LLAMA_DSV41_IDX_CHUNK_MIB"', 'use_blk && cbdiet && n_blk*bs == n_comp', 'ik, rows(iw), rows(ci.kq_mask)']:
    assert mark in t, mark
print('CBDIET patch complete')
