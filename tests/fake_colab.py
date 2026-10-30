"""Fake colab CLI for offline tests: sessions live under FAKE_COLAB_DIR, exec runs the shipped code against a stand-in /content."""
import json, os, re, shutil, signal, subprocess, sys, time

D = os.environ['FAKE_COLAB_DIR']
TESTS = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.join(D, 'sessions.json')
REFUSE = ('bash', 'gcc', 'curl', 'wget', 'mount', 'nvidia-smi')


def note(path, obj):
    with open(os.path.join(D, path), 'a') as f:
        f.write(json.dumps(dict(obj, t=time.time(), pid=os.getpid())) + '\n')


def sessions():
    try:
        with open(SESS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(s):
    tmp = '%s.%d' % (SESS, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(s, f)
    os.replace(tmp, SESS)


def vm(name):
    return os.path.join(D, 'vms', name)


def rewrite(cmd, name):
    """Maps VM paths into the fake VM dir and swaps gcc and boot.sh for harmless stand-ins; refuses anything else heavy."""
    cmd = cmd.replace('gcc -shared -fPIC -O2 -o vm/libpin.so vm/pin.c -ldl -lpthread', 'cp vm/pin.c vm/libpin.so')
    cmd = cmd.replace('bash vm/boot.sh', '%s %s' % (sys.executable, os.path.join(TESTS, 'fake_boot.py')))
    cmd = cmd.replace('/content', os.path.join(vm(name), 'content')).replace('/root/', os.path.join(vm(name), 'root') + '/')
    if any(re.search(r'(^|[\s;&(])%s\b' % w, cmd) for w in REFUSE):
        raise SystemExit('fake colab refuses to run: ' + cmd)
    return cmd


def kill_vm(name):
    pg = os.path.join(vm(name), 'content', 'res', 'boot.pgid')
    try:
        os.killpg(int(open(pg).read()), signal.SIGKILL)
    except (OSError, ValueError):
        pass


def main():
    a = sys.argv[1:]
    if a[:1] == ['--auth']:
        a = a[2:]
    note('calls.log', {'argv': a})
    cmd, rest = a[0], a[1:]
    name = rest[rest.index('-s') + 1] if '-s' in rest else None
    pos = [x for i, x in enumerate(rest) if not x.startswith('-') and (i == 0 or rest[i - 1] not in ('-s', '-i', '--timeout', '--gpu'))]
    s = sessions()
    if cmd == 'usage':
        if os.environ.get('FAKE_USAGE_FAIL'):
            print('[colab] Warning: failed to fetch compute-unit info: 401', file=sys.stderr)
            return 1
        print('Current balance: %s compute units\nUsage rate: %.2f/hr\nActive assignments: %d'
              % (os.environ.get('FAKE_BALANCE', '300.00'), 6.5 * len(s), len(s)))
    elif cmd == 'sessions':
        if not s:
            print('[colab] No active sessions found on server.')
        for n in s:
            print('[%s] fake-endpoint-%s | Hardware: NVIDIA G4 | Shape: Standard | Variant: GPU' % (n, n))
    elif cmd == 'new':
        if name in s:
            print('[colab] Session %s already exists.' % name)
            return 1
        os.makedirs(os.path.join(vm(name), 'content'), exist_ok=True)
        os.makedirs(os.path.join(vm(name), 'root'), exist_ok=True)
        s[name] = {'created': time.time(), 'gpu': rest[rest.index('--gpu') + 1] if '--gpu' in rest else None}
        save(s)
        print("[colab] Creating session '%s'...\n[colab] Session READY." % name)
    elif cmd == 'upload':
        if name not in s:
            print("[colab] Session '%s' not found." % name)
            return 1
        dest = rewrite(pos[1], name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(pos[0], dest)
        print("[colab] Uploaded '%s' to '%s'" % (pos[0], pos[1]))
    elif cmd == 'exec':
        if name not in s:
            print("[colab] Session '%s' not found." % name)
            return 1
        code = sys.stdin.read()
        real = subprocess.run
        subprocess.run = lambda c, *x, **kw: real(rewrite(c, name), *x, **kw)
        exec(compile(code, '<colab exec>', 'exec'), {'__name__': '__main__'})
    elif cmd == 'stop':
        if name not in s:
            print("[colab] Session '%s' not found." % name)
            return 0
        kill_vm(name)
        del s[name]
        save(s)
        print("[colab] Stopping session '%s'..." % name)
    elif cmd == 'ssh' and '--proxy-mode' in rest:
        if name not in s:
            note('violations.log', {'what': 'proxy-mode for a missing session would auto-create it', 'name': name})
            print("[colab] Creating runtime '%s'..." % name, file=sys.stderr, flush=True)
            return 1
        while name in sessions():
            time.sleep(0.2)
    else:
        print('fake colab: unsupported %s' % a, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
