"""Frees RAM-disk pages of tensors the server already holds in VRAM (punch holes, file size kept), or lists ranges for pin.c. usage: punch.py CPU_REGEX FILE... | --experts RESIDENT FILE... | --pin-cold OUT RESIDENT FILE..."""
import ctypes, os, re, struct, sys

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
         21: (256, 110), 22: (256, 82), 23: (256, 136), 26: (1, 4), 27: (1, 8), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
KEEP = re.compile(r'^(token_embd|per_layer_token_embd)\.weight$')
libc = ctypes.CDLL('libc.so.6', use_errno=True)
libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]


def tensors(path):
    f = open(path, 'rb')
    u = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    s = lambda: f.read(u('<Q'))

    def skip(t):
        if t == 8: s()
        elif t == 9:
            at, n = u('<I'), u('<Q')
            if at in (8, 9):
                for _ in range(n): skip(at)
            else: f.seek(n * SCALAR[at], 1)
        else: f.seek(SCALAR[t], 1)
    assert u('<I') == 0x46554747
    u('<I'); nt, nkv = u('<Q'), u('<Q')
    for _ in range(nkv):
        s(); skip(u('<I'))
    out = []
    for _ in range(nt):
        name = s().decode(); dims = [u('<Q') for _ in range(u('<I'))]; gt = u('<I'); off = u('<Q')
        n = 1
        for d in dims: n *= d
        out.append((name, off, n // BLOCK[gt][0] * BLOCK[gt][1]))
    base = (f.tell() + 31) // 32 * 32
    return [(name, base + off, nb) for name, off, nb in out]


EXP = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')


def resident(path):
    return set(tuple(map(int, l.split())) for l in open(path) if l.strip())


def expert_ranges(path, keep):
    """(name, start, size) per routed expert of the shard, keep(il, e) selects them."""
    for name, off, nb in tensors(path):
        m = EXP.match(name)
        if not m:
            continue
        il = int(m.group(1))
        es = nb // 384
        for e in range(384):
            if keep(il, e):
                yield name, off + e * es, es


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == '--experts':
        res = resident(args[1])
        freed = 0
        for path in args[2:]:
            fd = os.open(path, os.O_RDWR)
            for name, off, nb in expert_ranges(path, lambda il, e: (il, e) in res):
                a, b = (off + 4095) // 4096 * 4096, (off + nb) // 4096 * 4096
                if b > a:
                    if libc.fallocate(fd, 0x03, a, b - a) != 0:
                        raise OSError(ctypes.get_errno(), 'fallocate punch failed on %s' % path)
                    freed += b - a
            os.close(fd)
        print('punched resident experts %.2f GB' % (freed / 1e9))
        sys.exit(0)
    if args[0] == '--pin-cold':
        out, res = open(args[1], 'w'), resident(args[2])
        n = 0
        for path in args[3:]:
            run = None
            for name, off, nb in expert_ranges(path, lambda il, e: (il, e) not in res):
                if run and run[0] + run[1] == off:
                    run[1] += nb
                    continue
                if run:
                    out.write('%s %d %d\n' % (os.path.realpath(path), run[0], run[1])); n += run[1]
                run = [off, nb]
            if run:
                out.write('%s %d %d\n' % (os.path.realpath(path), run[0], run[1])); n += run[1]
        print('pin request (cold experts) %.2f GB' % (n / 1e9))
        sys.exit(0)
    args = sys.argv[1:]
    if args[0] == '--pin-request':
        out, cpu = open(args[1], 'w'), re.compile(args[2].rsplit('=', 1)[0])
        n = 0
        for path in args[3:]:
            for name, off, nb in tensors(path):
                if cpu.search(name):
                    out.write('%s %d %d\n' % (os.path.realpath(path), off, nb)); n += nb
        print('pin request %.2f GB' % (n / 1e9))
        sys.exit(0)
    cpu = re.compile(args[0].rsplit('=', 1)[0])
    freed = 0
    for path in args[1:]:
        fd = os.open(path, os.O_RDWR)
        for name, off, nb in tensors(path):
            if cpu.search(name) or KEEP.match(name):
                continue
            a, b = (off + 4095) // 4096 * 4096, (off + nb) // 4096 * 4096
            if b > a:
                if libc.fallocate(fd, 0x03, a, b - a) != 0:
                    raise OSError(ctypes.get_errno(), 'fallocate punch failed on %s' % path)
                freed += b - a
        os.close(fd)
    print('punched %.2f GB' % (freed / 1e9))
