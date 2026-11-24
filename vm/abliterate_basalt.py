#!/usr/bin/env python3
"""Apply mrfakename abliteration-edit.npz in-place to a .basalt DeepSeek-V4.1-Flash pack.

Uses the standalone compiled `abliterate` C binary (vm/abliterate.c -DABLITERATE_STANDALONE,
zero llama.cpp / libggml-base dependency) to edit all 62 non-zero Q8_0 / Q5_K / Q6_K target
tensors (`blk.<L>.(attn_output_b|ffn_down_shexp).weight`) inside the pack's `gguf` section.

Usage:
  python3 abliterate_basalt.py MODEL.basalt EDIT.npz ABLITERATE_BIN [--dry-run] [--update-sha]
"""
import argparse, ast, hashlib, json, os, re, struct, subprocess, sys, tempfile, zipfile

TARGET_RE = re.compile(r'^blk\.(\d+)\.(attn_output_b|ffn_down_shexp)\.weight$')
HEADER_BYTES = 65536


def load_npy(raw):
    assert raw[:6] == b'\x93NUMPY', 'not a .npy stream'
    major = raw[6]
    hlen = struct.unpack('<H' if major == 1 else '<I', raw[8:10 if major == 1 else 12])[0]
    off = (10 if major == 1 else 12) + hlen
    hdr = ast.literal_eval(raw[10 if major == 1 else 12:off].decode('latin1').strip())
    descr, fortran, shape = hdr['descr'], bool(hdr['fortran_order']), tuple(hdr['shape'])
    fmt_map = {'<f4': ('<', 'f', 4), '<f8': ('<', 'd', 8), '>f4': ('>', 'f', 4), '>f8': ('>', 'd', 8)}
    assert descr in fmt_map, f'unsupported npy dtype: {descr!r}'
    endian, code, esz = fmt_map[descr]
    n = 1
    for d in shape:
        n *= d
    payload = raw[off:off + n * esz]
    assert len(payload) == n * esz, 'truncated npy payload'
    return shape, fortran, struct.unpack(f'{endian}{n}{code}', payload)


def load_edit_npz(path):
    with zipfile.ZipFile(path, 'r') as zf:
        q_shape, q_f, q_vals = load_npy(zf.read('Q.npy'))
        s_shape, s_f, s_vals = load_npy(zf.read('scales.npy'))
    assert len(q_shape) == 2 and q_shape[1] == 2, f'unexpected Q shape: {q_shape!r}'
    assert len(s_shape) == 2 and s_shape[1] == 2, f'unexpected scales shape: {s_shape!r}'
    dim = q_shape[0]
    if q_f:
        q0, q1 = list(q_vals[:dim]), list(q_vals[dim:2 * dim])
    else:
        q0 = [q_vals[2 * i] for i in range(dim)]
        q1 = [q_vals[2 * i + 1] for i in range(dim)]
    n0 = sum(x * x for x in q0)
    n1 = sum(x * x for x in q1)
    dot = sum(a * b for a, b in zip(q0, q1))
    assert abs(n0 - 1.0) < 1e-4 and abs(n1 - 1.0) < 1e-4 and abs(dot) < 1e-4, (n0, n1, dot)
    nlayers = s_shape[0]
    if s_f:
        scales = [(float(s_vals[i]), float(s_vals[nlayers + i])) for i in range(nlayers)]
    else:
        scales = [(float(s_vals[2 * i]), float(s_vals[2 * i + 1])) for i in range(nlayers)]
    return dim, q0, q1, scales


def parse_basalt_header(hdr_bytes):
    assert len(hdr_bytes) == HEADER_BYTES
    magic, ver, nsec, file_bytes = struct.unpack('<8sIIQ', hdr_bytes[:24])
    assert magic == b'BASALTMF' and ver == 2, f'not a BASALTMF v2 file: magic={magic!r} ver={ver}'
    secs = {}
    for i in range(nsec):
        entry_off = 64 + i * 128
        name_b, soff, sbytes = struct.unpack('<64sQQ', hdr_bytes[entry_off:entry_off + 80])
        name = name_b.split(b'\x00', 1)[0].decode('utf-8')
        secs[name] = (soff, sbytes, entry_off)
    return file_bytes, secs


def build_plan_from_meta(pack_path, secs, meta, npz_path):
    dim, q0, q1, scales = load_edit_npz(npz_path)
    expected = sum(1 for a, b in scales for x in (a, b) if abs(x) > 0.0)
    real = os.path.realpath(pack_path)
    plan = []
    for name, info in sorted(meta['tensors'].items()):
        m = TARGET_RE.match(name)
        if not m:
            continue
        layer, kind = int(m.group(1)), m.group(2)
        if layer >= len(scales):
            continue
        scale = scales[layer][0 if kind == 'attn_output_b' else 1]
        if scale == 0.0:
            continue
        sec_name = info.get('section', 'gguf')
        assert sec_name in secs, f'tensor {name} references missing section {sec_name}'
        sec_off, sec_bytes, _ = secs[sec_name]
        rel_off = int(info['offset'])
        nb = int(info['bytes'])
        assert rel_off + nb <= sec_bytes, f'tensor {name} exceeds section {sec_name}'
        dims = [int(d) for d in info['shape']]
        assert len(dims) == 2 and dims[1] == dim, f'unexpected shape for {name}: {dims} vs dim={dim}'
        plan.append((real, name, int(info['type_id']), dims[0], dims[1], sec_off + rel_off, nb, scale))
    assert len(plan) == expected, f'expected {expected} non-zero abliteration targets, found {len(plan)}'
    return dim, q0, q1, plan


def refresh_section_sha256(pack_path, secs, sec_name='gguf'):
    sec_off, sec_bytes, entry_off = secs[sec_name]
    h = hashlib.sha256()
    with open(pack_path, 'r+b') as f:
        f.seek(sec_off)
        rem = sec_bytes
        while rem > 0:
            buf = f.read(min(rem, 16 << 20))
            assert buf, 'unexpected EOF while hashing section'
            h.update(buf)
            rem -= len(buf)
        digest = h.digest()
        f.seek(entry_off + 80)
        f.write(digest)
    print(f'updated {sec_name} section sha256 -> {digest.hex()[:16]}...')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack')
    ap.add_argument('npz')
    ap.add_argument('abliterate_bin', nargs='?', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--update-sha', action='store_true')
    args = ap.parse_args()

    with open(args.pack, 'rb') as f:
        hdr = f.read(HEADER_BYTES)
        _, secs = parse_basalt_header(hdr)
        moff, mbytes, _ = secs['meta']
        f.seek(moff)
        meta = json.loads(f.read(mbytes))

    dim, q0, q1, plan = build_plan_from_meta(args.pack, secs, meta, args.npz)
    if args.dry_run:
        total_mb = sum(x[6] for x in plan) / (1024 * 1024)
        print(f'DRY-RUN OK: dim={dim} targets={len(plan)} total={total_mb:.1f} MiB')
        return

    assert args.abliterate_bin and os.path.exists(args.abliterate_bin), f'missing abliterate binary: {args.abliterate_bin}'
    with tempfile.TemporaryDirectory(prefix='abliterate-basalt-') as td:
        q_bin = os.path.join(td, 'Q.bin')
        plan_txt = os.path.join(td, 'plan.txt')
        with open(q_bin, 'wb') as f:
            f.write(struct.pack('<q', dim))
            f.write(struct.pack(f'<{dim}d', *q0))
            f.write(struct.pack(f'<{dim}d', *q1))
        with open(plan_txt, 'w') as f:
            for real, name, gt, ne0, ne1, off, nb, scale in plan:
                f.write(f'{real}\t{name}\t{gt}\t{ne0}\t{ne1}\t{off}\t{nb}\t{scale:.12g}\n')
        r = subprocess.run([args.abliterate_bin, q_bin, plan_txt])
        if r.returncode != 0:
            sys.exit(r.returncode)

    if args.update_sha:
        refresh_section_sha256(args.pack, secs, 'gguf')
    print(f'abliteration applied to {args.pack} ({len(plan)} tensors)')


if __name__ == '__main__':
    main()
