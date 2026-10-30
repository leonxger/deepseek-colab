"""Keeps CED prefill on for text prompts while an mmproj is loaded (llama-server marks every prompt multimodal). usage: python3 patch_cedmm.py LLAMA_SRC_DIR"""
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


patch('tools/server/server-common.h', '    bool empty() const { return tokens.empty(); }\n',
      '    bool empty() const { return tokens.empty(); }\n    bool has_media() const { return !map_idx_to_media.empty(); } // CEDMM_PATCH\n')
patch('tools/server/server-context.cpp', '!input_tokens.has_mtmd && !slot.need_embd();', '!input_tokens.has_media() && !slot.need_embd(); // CEDMM_PATCH')
for rel, mark in [('tools/server/server-common.h', 'has_media() const'), ('tools/server/server-context.cpp', 'input_tokens.has_media()')]:
    assert mark in open(os.path.join(src, rel)).read(), (rel, mark)
print('CEDMM patch complete')
