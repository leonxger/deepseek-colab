"""Fake ssh for offline tests: validates the options, runs the ProxyCommand like ssh does and forwards -L to FAKE_VM_PORT."""
import json, os, signal, socket, subprocess, sys, threading, time

D = os.environ['FAKE_COLAB_DIR']
VM_PORT = int(os.environ['FAKE_VM_PORT'])


def pump(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    for s in (a, b):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def note(ok):
    with open(os.path.join(D, 'fwd.log'), 'a') as f:
        f.write(json.dumps({'t': time.time(), 'ok': ok}) + '\n')


def main():
    args, opts, i = sys.argv[1:], {}, 0
    while i < len(args):
        if args[i] == '-o':
            k, v = args[i + 1].split('=', 1)
            opts[k] = v
            i += 2
        elif args[i] in ('-L', '-F', '-i'):
            opts[args[i]] = args[i + 1]
            i += 2
        elif args[i] == '-N':
            opts['-N'] = True
            i += 1
        else:
            opts['host'] = args[i]
            i += 1
    with open(os.path.join(D, 'ssh_calls.log'), 'a') as f:
        f.write(json.dumps({'t': time.time(), 'pid': os.getpid(), 'opts': opts}) + '\n')
    need = {'-N': True, '-F': '/dev/null', 'User': 'root', 'StrictHostKeyChecking': 'no', 'UserKnownHostsFile': '/dev/null',
            'ExitOnForwardFailure': 'yes', 'IdentitiesOnly': 'yes', 'ServerAliveInterval': '20', 'LogLevel': 'ERROR'}
    bad = [k for k, v in need.items() if opts.get(k) != v] + [k for k in ('ProxyCommand', '-L', '-i', 'host') if k not in opts]
    lhost, lport, rhost, rport = opts.get('-L', ':::').split(':')
    if bad or rhost != '127.0.0.1' or rport != '18080':
        print('fake ssh: bad options %s' % bad, file=sys.stderr)
        return 255
    proxy = subprocess.Popen(['/bin/sh', '-c', 'exec ' + opts['ProxyCommand'].replace('%%', '%')], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
    time.sleep(0.3)
    if proxy.poll() is not None:
        print('kex_exchange_identification: Connection closed by remote host', file=sys.stderr)
        return 255
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((lhost, int(lport)))
    except OSError as e:
        print('bind [%s]:%s: %s' % (lhost, lport, e), file=sys.stderr)
        proxy.kill()
        return 255
    srv.listen(16)

    def accept():
        while True:
            c, _ = srv.accept()
            try:
                u = socket.create_connection(('127.0.0.1', VM_PORT), timeout=5)
                u.settimeout(None)
            except OSError:
                note(False)
                print('channel 3: open failed: connect failed: Connection refused', file=sys.stderr, flush=True)
                c.close()
                continue
            note(True)
            threading.Thread(target=pump, args=(c, u), daemon=True).start()
            threading.Thread(target=pump, args=(u, c), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *a: (proxy.kill(), os._exit(0)))
    proxy.wait()
    os._exit(255)


if __name__ == '__main__':
    sys.exit(main())
