"""Parallel HTTP range fetcher: all jobs share one worker pool, chunks land in place via dd. usage: python3 pfetch.py JOBS.json [--workers 64] [--chunk-mb 64] [--log FILE] [--tag NAME]"""
import argparse, json, os, queue, subprocess, sys, threading, time, urllib.request


def resolve(url, hdr):
    headers = {'Range': 'bytes=0-0'}
    if hdr:
        for line in open(hdr):
            k, _, v = line.strip().partition(':')
            if k:
                headers[k.strip()] = v.strip()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl() if 'huggingface.co' in url else url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--chunk-mb', type=int, default=64)
    ap.add_argument('--log')
    ap.add_argument('--tag', default='pfetch')
    a = ap.parse_args()
    log = open(a.log, 'a') if a.log else sys.stdout
    lk = threading.Lock()

    def say(s):
        with lk:
            log.write(time.strftime('%H:%M:%S ') + a.tag + ' ' + s + '\n'); log.flush()

    jobs = json.load(open(a.jobs))
    for j in jobs:
        if not j.get('hdr') and os.environ.get('HF_HDR') and 'huggingface.co' in j['url']:
            j['hdr'] = os.environ['HF_HDR']
    t0 = time.time()
    final = {}
    for j in jobs:
        fd = os.open(j['dest'], os.O_WRONLY | os.O_CREAT, 0o644)
        if j.get('dest_size') is not None and os.fstat(fd).st_size != j['dest_size']:
            os.ftruncate(fd, j['dest_size'])
        os.close(fd)
    q = queue.Queue()
    chunk = a.chunk_mb << 20
    total = 0
    for i, j in enumerate(jobs):
        s0 = j.get('src_off', 0)
        for s in range(s0, s0 + j['size'], chunk):
            q.put((i, s, min(s + chunk, s0 + j['size'])))
        total += j['size']
    done, errors = [0], []

    def url_of(i, refresh=False):
        u = jobs[i]['url']
        with lk:
            if refresh or u not in final:
                final[u] = None
        if final.get(u) is None:
            for attempt in range(5):
                try:
                    r = resolve(u, jobs[i].get('hdr'))
                    with lk:
                        final[u] = r
                    break
                except Exception as e:
                    say('resolve retry %s: %r' % (os.path.basename(u.split('?')[0]), e))
                    time.sleep(2 ** attempt)
        return final.get(u) or u

    def worker():
        while True:
            try:
                i, s, e = q.get_nowait()
            except queue.Empty:
                return
            j = jobs[i]
            off = j.get('dest_off', 0) + s - j.get('src_off', 0)
            for attempt in range(8):
                u = url_of(i)
                h = " -H @'%s'" % j['hdr'] if j.get('hdr') else ''
                cmd = ("set -o pipefail; curl -sf --connect-timeout 20 --speed-limit 1000000 --speed-time 30%s -r %d-%d '%s' | "
                       "dd of='%s' bs=4M seek=%d oflag=seek_bytes conv=notrunc status=none iflag=fullblock") % (h, s, e - 1, u, j['dest'], off)
                rc = subprocess.run(['bash', '-c', cmd]).returncode
                if rc == 0:
                    with lk:
                        done[0] += e - s
                    break
                if rc == 22:
                    url_of(i, refresh=True)
                time.sleep(min(2 ** attempt, 30))
            else:
                errors.append((i, s, e))

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    [t.start() for t in ths]
    last, last_done = time.time(), 0
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
        now = time.time()
        if now - last >= 5:
            say('%.1f/%.1f GB, last 5 s %.0f MB/s, avg %.0f MB/s' % (done[0] / 1e9, total / 1e9, (done[0] - last_done) / 1e6 / (now - last),
                                                                   done[0] / 1e6 / (now - t0)))
            last, last_done = now, done[0]
    dt = time.time() - t0
    if errors:
        say('FAILED %d chunks' % len(errors))
        sys.exit(1)
    say('DONE %.2f GB in %.1f s = %.0f MB/s' % (done[0] / 1e9, dt, done[0] / 1e6 / dt))


if __name__ == '__main__':
    main()
