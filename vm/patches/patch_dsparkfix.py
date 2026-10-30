"""Stops a DSpark draft early on an empty candidate list instead of emitting a garbage token id (outputs unchanged). usage: python3 patch_dsparkfix.py LLAMA_SRC_DIR"""
import os, re, sys

p = os.path.join(sys.argv[1], 'common/speculative.cpp')
s = open(p).read()
if 'DSPARKFIX_PATCH' in s:
    print('already patched common/speculative.cpp')
    sys.exit(0)
pat = re.compile(r'^( +)const auto \* cur_p = common_sampler_get_candidates\(smpl, true\);\n', re.M)
s, n = pat.subn(lambda m: m.group(0) + m.group(1) + 'if (cur_p == nullptr || cur_p->size == 0) {\n' + m.group(1) +
                '    break; // DSPARKFIX_PATCH: empty candidate list\n' + m.group(1) + '}\n', s)
assert n == 5, n
open(p, 'w').write(s)
print('patched common/speculative.cpp DSPARKFIX_PATCH x%d' % n)
