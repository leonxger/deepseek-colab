"""KV cache seq_rm from the position index (SEQRM_PATCH): usage python3 patch_seqrm.py LLAMA_SRC_DIR, runtime env LLAMA_KV_SEQRM_INDEX=1."""
import os, sys

src = sys.argv[1]


def replace(rel, old, new):
    p = os.path.join(src, rel)
    s = open(p).read()
    if new in s:
        print('already replaced', rel, new[:50])
        return
    assert s.count(old) == 1, (rel, old)
    open(p, 'w').write(s.replace(old, new))
    print('replaced', rel, new[:50])


replace('src/llama-kv-cache.cpp', '''        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
''', '''        uint32_t new_head = cells.size();

        // SEQRM_PATCH: the cells of seq_id in [p0, p1) from its (pos, cell) index instead of a scan of every cell
        static const bool use_index = [] {
            const char * env = getenv("LLAMA_KV_SEQRM_INDEX");
            return env != nullptr && atoi(env) != 0;
        }();
        if (use_index) {
            const auto & sp = cells.seq_pos_get(seq_id);
            std::vector<uint32_t> idx;
            for (auto it = sp.lower_bound({ p0, 0 }); it != sp.end() && it->first < p1; ++it) {
                idx.push_back(it->second);
            }
            for (const uint32_t i : idx) {
                if (cells.seq_rm(i, seq_id) && i < new_head) {
                    new_head = i;
                }
            }
        }

        for (uint32_t i = 0; !use_index && i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
''')
