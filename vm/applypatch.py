"""Overwrites the 82 edited tensors in the ISTA shards with the uncensored bytes, sha256 checked. usage: python3 -I applypatch.py SHARD_DIR PATCH_DIR"""
import hashlib, json, os, sys

RUNG = os.environ.get('RUNG', 'IQ2_XXS')
NAME = 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-' + RUNG + '-%05d-of-00006.gguf'
shards, pdir = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(pdir, 'patch-%s.json' % RUNG)))
with open(os.path.join(pdir, 'patch-%s.bin' % RUNG), 'rb') as p:
    for e in man['tensors']:
        p.seek(e['patch_off']); b = p.read(e['nbytes'])
        assert hashlib.sha256(b).hexdigest() == e['sha256'], e['name']
        with open(os.path.join(shards, NAME % e['shard']), 'r+b') as f:
            f.seek(e['off']); f.write(b); f.flush()
            f.seek(e['off']); assert hashlib.sha256(f.read(e['nbytes'])).hexdigest() == e['sha256'], e['name']
print('patched', len(man['tensors']), 'tensors,', man['total_bytes'], 'bytes')
