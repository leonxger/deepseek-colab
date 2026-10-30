"""Builds the uncensor patch: AtomicChat's 82 edited tensors by HTTP range reads, addressed by ISTA shard and offset. usage: python3 -I mkpatch.py OUT_DIR"""
import concurrent.futures, hashlib, json, os, struct, sys, urllib.request

ATOMIC_REV = 'b8bca19979431f96126ceac8cbe0ee7160a28f8a'
ISTA_REV = 'a419d74cd28d5b9f8c5e2b9077d11a609c17bc32'
RUNG = os.environ.get('RUNG', 'IQ2_XXS')
HF_HDR = os.environ.get('HF_HDR')
A = 'https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF/resolve/%s/DeepSeek-V4.1-Flash-1017-GSQ-RCO-%s-%%05d-of-00004.gguf' % (ATOMIC_REV, RUNG)
I = 'https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/%s/DeepSeek-V4.1-Flash-1017-GSQ-RCO-%s-%%05d-of-00006.gguf' % (ISTA_REV, RUNG)
KINDS = ('.attn_output_b.', '.ffn_down_shexp.', '.engram_wkv.')
SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
         21: (256, 110), 22: (256, 82), 23: (256, 136), 26: (1, 4), 27: (1, 8), 29: (256, 56), 30: (1, 2), 39: (32, 17)}


def fetch(url, start, n, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (start, start + n - 1)})
            if HF_HDR:
                req.add_header('Authorization', open(HF_HDR).read().split(':', 1)[1].strip())
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            if len(b) == n:
                return b
        except OSError:
            pass
    raise RuntimeError('range fetch failed: %s %d+%d' % (url, start, n))


class R:
    def __init__(self, b): self.b, self.p = b, 0
    def take(self, n):
        if self.p + n > len(self.b): raise EOFError
        x = self.b[self.p:self.p + n]; self.p += n; return x
    def u(self, f): return struct.unpack(f, self.take(struct.calcsize(f)))[0]
    def skip(self, t):
        if t == 8: self.take(self.u('<Q'))
        elif t == 9:
            at, n = self.u('<I'), self.u('<Q')
            if at in (8, 9):
                for _ in range(n): self.skip(at)
            else: self.take(n * struct.calcsize(SCALAR[at]))
        else: self.take(struct.calcsize(SCALAR[t]))


def tensors(url):
    size = 1 << 20
    buf = b''
    while True:
        buf += fetch(url, len(buf), size - len(buf))
        try:
            r = R(buf)
            assert r.u('<I') == 0x46554747
            r.u('<I'); nt, nkv = r.u('<Q'), r.u('<Q')
            for _ in range(nkv):
                r.take(r.u('<Q')); r.skip(r.u('<I'))
            out = []
            for _ in range(nt):
                name = r.take(r.u('<Q')).decode(); dims = [r.u('<Q') for _ in range(r.u('<I'))]
                out.append((name, dims, r.u('<I'), r.u('<Q')))
            ds = (r.p + 31) // 32 * 32
            res = {}
            for name, dims, gt, off in out:
                pb, bb = BLOCK[gt]; n = 1
                for d in dims: n *= d
                res[name] = {'off': ds + off, 'nbytes': n // pb * bb, 'type': gt, 'dims': dims}
            return res
        except EOFError:
            size *= 4


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    atomic, ista = {}, {}
    for i in range(1, 5):
        for n, t in tensors(A % i).items(): atomic[n] = dict(t, url=A % i)
    for i in range(2, 6):
        for n, t in tensors(I % i).items(): ista[n] = dict(t, shard=i)
    todo = sorted(n for n in atomic if any(k in n for k in KINDS))
    assert len(todo) == 82, len(todo)
    entries, pos = [], 0
    for n in todo:
        a, t = atomic[n], ista[n.replace('.engram_wkv.', '.engram_kv.')]
        assert (a['type'], a['nbytes'], a['dims']) == (t['type'], t['nbytes'], t['dims']), n
        entries.append({'name': n.replace('.engram_wkv.', '.engram_kv.'), 'shard': t['shard'], 'off': t['off'],
                        'nbytes': t['nbytes'], 'patch_off': pos, 'src': a['url'], 'src_off': a['off']})
        pos += t['nbytes']
    blob = os.path.join(out, 'patch-%s.bin' % RUNG)
    with open(blob, 'wb') as f:
        f.truncate(pos)

    def one(e):
        b = fetch(e['src'], e['src_off'], e['nbytes'])
        with open(blob, 'r+b') as f:
            f.seek(e['patch_off']); f.write(b)
        e['sha256'] = hashlib.sha256(b).hexdigest()
        return e['name']

    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for name in ex.map(one, entries):
            print('fetched', name, flush=True)
    for e in entries:
        del e['src'], e['src_off']
    json.dump({'atomic_rev': ATOMIC_REV, 'ista_rev': ISTA_REV, 'total_bytes': pos, 'tensors': entries},
              open(os.path.join(out, 'patch-%s.json' % RUNG), 'w'), indent=1)
    print('patch done', pos, 'bytes', len(entries), 'tensors')


if __name__ == '__main__':
    main()
