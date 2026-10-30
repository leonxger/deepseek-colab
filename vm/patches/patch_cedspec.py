"""Keeps CED prefill on with the DSpark draft (LLAMA_DSV41_CED_SPEC=1): the draft reads layers 37-39 over 128 tokens, inside the exact replay window. usage: python3 patch_cedspec.py LLAMA_SRC_DIR"""
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


SPEC = 'getenv("LLAMA_DSV41_CED_SPEC") == nullptr'
C = 'src/llama-context.cpp'
patch(C, '''    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            return 0; // DSpark reads the decoder inputs of every prompt token
        }
    }
''', '''    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled && %s) { // CEDSPEC_PATCH
            return 0; // DSpark reads the decoder inputs of every prompt token
        }
    }
''' % SPEC)
patch(C, '''    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            return false;
        }
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
''', '''    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled && %s) { // CEDSPEC_PATCH
            return false;
        }
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
''' % SPEC)
patch(C, '''        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }
''', '''        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t && !(%s)) {
            // CEDSPEC_PATCH: an encoder-only ubatch has no decoder layer inputs, the draft never attends to these rows
            std::memset(embd_layer_inp[il].data + token_offset*model.hparams.n_embd, 0, n_tokens*model.hparams.n_embd*sizeof(float));
            extracted = true;
            continue;
        }
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }
''' % SPEC)
patch('src/llama-graph.cpp', '''            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
''', '''            if (embeddings_layer_inp[il]) {
                if (t_layer_inp[il] == nullptr && params.ced_dec_first >= 0 && (int32_t) il >= params.ced_dec_first) {
                    continue; // CEDSPEC_PATCH: encoder-only graph
                }
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
''')
