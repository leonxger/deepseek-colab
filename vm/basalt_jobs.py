#!/usr/bin/env python3
"""Generate pfetch.py HTTP Range job JSONs for split-storage Basalt DeepSeek-V4.1-Flash packs.

Reads only the 64 KiB BASALTMF v2 header from Hugging Face and splits a 260-285 GB .basalt pack
into two parallel pfetch.py job lists:
  1. Backbone sparse file (logical size = full pack; physical size = 149.4 / 174.9 GB):
     Header + every section EXCEPT `engram` (leaving the 110.6 GB `engram` span as a 0-block hole).
  2. Engram tmpfs file (/mnt/ram/engram.bin, 110.6 GB):
     ONLY the `engram` section at dest_off = 0, read via BASALT_ENGRAM_FILE=/mnt/ram/engram.bin.

Usage:
  python3 basalt_jobs.py --quant iq1_m \
    --backbone-dest /mnt/ram/model.basalt --backbone-jobs /content/res/j_basalt_bb.json \
    --engram-dest /mnt/ram/engram.bin     --engram-jobs /content/res/j_basalt_eng.json
"""
import argparse, json, os, struct, urllib.request

PACKS = {
    'iq1_m': 'https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt/resolve/main/DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt',
}

HEADER_BYTES = 65536


def parse_header(data):
    assert len(data) == HEADER_BYTES, f'expected {HEADER_BYTES} header bytes, got {len(data)}'
    magic, ver, nsec, file_bytes = struct.unpack('<8sIIQ', data[:24])
    assert magic == b'BASALTMF' and ver == 2, f'bad basalt header: magic={magic!r} ver={ver}'
    sections = []
    for i in range(nsec):
        off = 64 + i * 128
        name_b, soff, sbytes, sha = struct.unpack('<64sQQ32s', data[off:off + 112])
        name = name_b.split(b'\x00', 1)[0].decode('utf-8')
        sections.append({'name': name, 'offset': soff, 'bytes': sbytes, 'sha256': sha.hex()})
    return file_bytes, sections


def read_header(url, hdr_file=None):
    headers = {'Range': f'bytes=0-{HEADER_BYTES - 1}'}
    if hdr_file and os.path.exists(hdr_file):
        for line in open(hdr_file):
            k, _, v = line.strip().partition(':')
            if k:
                headers[k.strip()] = v.strip()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return parse_header(data)


def build_jobs(url, file_bytes, sections, backbone_dest, engram_dest):
    backbone_jobs = [{
        'url': url,
        'dest': backbone_dest,
        'dest_size': file_bytes,
        'src_off': 0,
        'dest_off': 0,
        'size': HEADER_BYTES,
    }]
    engram_jobs = []
    for s in sections:
        if s['bytes'] == 0:
            continue
        if s['name'] == 'engram':
            engram_jobs.append({
                'url': url,
                'dest': engram_dest,
                'dest_size': s['bytes'],
                'src_off': s['offset'],
                'dest_off': 0,
                'size': s['bytes'],
            })
        else:
            backbone_jobs.append({
                'url': url,
                'dest': backbone_dest,
                'dest_size': file_bytes,
                'src_off': s['offset'],
                'dest_off': s['offset'],
                'size': s['bytes'],
            })
    assert len(engram_jobs) == 1, f'expected 1 engram section, found {len(engram_jobs)}'
    return backbone_jobs, engram_jobs


def punch_sections(pack_path, names):
    """Inward-4K-aligned FALLOC_FL_PUNCH_HOLE on named BASALTMF sections (e.g. dspark/gguf, dense)."""
    import ctypes
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    libc.fallocate.restype = ctypes.c_int
    FALLOC_FL_KEEP_SIZE = 0x01
    FALLOC_FL_PUNCH_HOLE = 0x02
    with open(pack_path, 'r+b') as f:
        _, sections = parse_header(f.read(HEADER_BYTES))
        fd = f.fileno()
        punched = 0
        for s in sections:
            if s['name'] not in names or s['bytes'] == 0:
                continue
            pa = (s['offset'] + 4095) & ~4095
            pb = (s['offset'] + s['bytes']) & ~4095
            if pb > pa:
                rc = libc.fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, pa, pb - pa)
                if rc == 0:
                    punched += pb - pa
    return punched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quant', choices=tuple(PACKS), default='iq1_m')
    ap.add_argument('--url', default=None)
    ap.add_argument('--backbone-dest', default='/mnt/ram/model.basalt')
    ap.add_argument('--engram-dest', default='/mnt/ram/engram.bin')
    ap.add_argument('--backbone-jobs', default=None)
    ap.add_argument('--engram-jobs', default=None)
    ap.add_argument('--engram-size-out', default=None, help='Optional path to write exact Engram byte size')
    ap.add_argument('--punch-sections', nargs='+', default=None,
                    help='Punch named sections (e.g. dspark/gguf dense) in --backbone-dest after /load')
    args = ap.parse_args()

    if args.punch_sections:
        freed = punch_sections(args.backbone_dest, set(args.punch_sections))
        print(f'punched {args.punch_sections} in {args.backbone_dest}: freed {freed / (1024**3):.2f} GiB')
        return

    assert args.backbone_jobs and args.engram_jobs, '--backbone-jobs and --engram-jobs are required'
    url = args.url or PACKS[args.quant]
    file_bytes, sections = read_header(url, os.environ.get('HF_HDR'))
    bb_jobs, eng_jobs = build_jobs(url, file_bytes, sections, args.backbone_dest, args.engram_dest)

    for path, data in ((args.backbone_jobs, bb_jobs), (args.engram_jobs, eng_jobs)):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    if args.engram_size_out:
        with open(args.engram_size_out, 'w') as f:
            f.write(str(eng_jobs[0]['size']) + '\n')

    bb_gb = sum(j['size'] for j in bb_jobs) / 1e9
    eng_gb = sum(j['size'] for j in eng_jobs) / 1e9
    print(f'basalt_jobs ({args.quant}): logical={file_bytes/1e9:.2f} GB | '
          f'backbone={bb_gb:.2f} GB ({len(bb_jobs)} ranges -> {args.backbone_dest}) | '
          f'engram={eng_gb:.2f} GB (1 range -> {args.engram_dest})')


if __name__ == '__main__':
    main()
