#!/usr/bin/env python3
"""Runs uncensored DeepSeek-V4.1-Flash on a Google Colab G4 behind a local OpenAI-compatible endpoint; see README.md."""
import sys

if sys.version_info < (3, 9):
    sys.exit('llm.py needs Python 3.9 or newer (this is %d.%d)' % sys.version_info[:2])
import hmac, http.client, io, json, os, re, secrets, shlex, shutil, signal, socket, socketserver, subprocess, tarfile
import threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

USAGE = '''usage: python3 llm.py COMMAND [--fast]
  up            set up on first use, start the GPU and print the endpoint (base URL, API key, models)
  down          stop the GPU and the local endpoint
  status        endpoint state, Colab balance and session
  watch         live view: GPU on or off, idle countdown to the auto-stop, cost so far (up shows it too)
  settings      change the saved defaults: image input, context, reasoning effort, idle stop, hard stop, uncensored
  ask TEXT      send one question and stream the answer (--max for reasoning effort max)
  opencode ...  start opencode with this endpoint configured, in any shell; arguments go to opencode
  logs          local log plus the GPU's boot timeline and server log
  setup         install and log in only, start nothing
  serve         run the endpoint in the foreground (Ctrl+C stops the GPU too)
  --fast        this run only: text only, 131K context, the faster layout (118.7 instead of 105.6 tok/s)
env: LLM_PORT (18080) LLM_IDLE_MIN (10) LLM_MAX_HOURS (6) LLM_MIN_BALANCE (5) LLM_IMAGES (0|1) LLM_CONTEXT (4096-262144)
     LLM_PROFILE (default|fast presets) LLM_BIND (127.0.0.1)
     HF_TOKEN (optional Hugging Face token for the downloads) LLM_YES=1 (answer yes to install prompts) LLM_BOOT_ENV ("K=V ...")
     LLM_SESSION LLM_COLAB LLM_STATE LLM_VM_DIR; tests: LLM_SSH_CMD LLM_TUNNEL_PORT LLM_POLL_S'''

IS_WIN = os.name == 'nt'
ROOT = os.path.dirname(os.path.realpath(__file__))
STATE = os.path.abspath(os.environ.get('LLM_STATE') or os.path.join(ROOT, 'state'))
KEY_FILE, SSH_KEY = os.path.join(STATE, 'api_key'), os.path.join(STATE, 'id_ed25519')
SESSION, LOCK, LOG = (os.path.join(STATE, f) for f in ('session.json', 'serve.lock', 'llm.log'))
SETTINGS_FILE = os.path.join(STATE, 'settings.json')
VM_DIR = os.path.abspath(os.environ.get('LLM_VM_DIR') or os.path.join(ROOT, 'vm'))
DEFAULTS = {'images': True, 'context': 262144, 'effort': '50', 'idle_min': 10, 'max_hours': 6, 'uncensor': True}


def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    return dict(DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}) if isinstance(saved, dict) else dict(DEFAULTS)


SETTINGS = load_settings()
NAME = os.environ.get('LLM_SESSION', 'llm-ds41')
PORT = int(os.environ.get('LLM_PORT', '18080'))
BIND = os.environ.get('LLM_BIND', '127.0.0.1')
HOST = BIND if BIND not in ('', '0.0.0.0', '::') else '127.0.0.1'
LOOPBACK = BIND in ('127.0.0.1', 'localhost', '::1')
IDLE_MIN = float(os.environ.get('LLM_IDLE_MIN') or SETTINGS['idle_min'])
MAX_HOURS = float(os.environ.get('LLM_MAX_HOURS') or SETTINGS['max_hours'])
MIN_BALANCE = float(os.environ.get('LLM_MIN_BALANCE', '5'))
TUNNEL_PORT = int(os.environ.get('LLM_TUNNEL_PORT', '0'))
POLL_S = float(os.environ.get('LLM_POLL_S', '15'))
HEALTH_S = min(1.0, POLL_S)
BOOT_TIMEOUT, FAIL_COOLDOWN, KEEPALIVE_S, MAX_BODY, UPSTREAM_TIMEOUT = 1500, 60, 5, 64 << 20, 3600
COLAB_CLI = 'google-colab-cli==0.7.5'
G4_CU_PER_HOUR = 8.9
# the two GPU memory layouts measured on the G4: 'default' fits images and 262K per request, 'fast' text only up to 131K
LAYOUTS = {'default': {'UBATCH': '2048', 'MOE_CACHE_MIB': '70430', 'DYN': '110', 'DMA': '0', 'SPEED': '105.6'},
           'fast': {'UBATCH': '3072', 'MOE_CACHE_MIB': '74231', 'DYN': '166', 'DMA': '1', 'SPEED': '118.7'}}
CONTEXTS = {'32k': 32768, '64k': 65536, '128k': 131072, '131k': 131072, '256k': 262144, '262k': 262144}
BOOT_KEYS = ('CTX', 'UBATCH', 'MMPROJ', 'MOE_CACHE_MIB', 'DYN', 'DMA', 'DRAFT', 'UNCENSOR', 'WORKERS')
VM_PORT = 18080
NO_WINDOW, DETACHED, NEW_GROUP, BREAKAWAY = 0x08000000, 0x00000008, 0x00000200, 0x01000000
CHILD_FLAGS = NO_WINDOW if IS_WIN else 0
CHILD_ENV = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
HOP = {'connection', 'keep-alive', 'proxy-connection', 'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade',
       'content-length', 'host', 'accept-encoding', 'expect', 'server', 'date'}
STEPS = (('encoder', r'image encoder fetched', 'image encoder downloaded (1 GB)'),
         ('patch', r'uncensor patch fetched', 'uncensored tensors downloaded (1.7 GB)'),
         ('draft', r'draft fetched', 'draft model downloaded (8 GB)'),
         ('model', r'backbone fetched', 'model downloaded (138 GB)'),
         ('build', r'llama\.cpp built', 'llama.cpp built'),
         ('loaded', r'healthy', 'model loaded onto the GPU'),
         ('pinned', r'pin: ', 'RAM freed, cold experts pinned'),
         ('engram', r'engram done', 'Engram table loaded (110 GB)'),
         ('warm', r'first token', 'warmup answer done'))
LATER = (('loaded', 'loading the model onto the GPU'), ('pinned', 'freeing RAM and pinning experts'),
         ('engram', 'loading the Engram table (110 GB)'), ('warm', 'warming up'))
ECHO, TTY, COLOR = [True], [False], [False]
_log_lock = threading.Lock()
_cache = {}


def init_console():
    """Colors and in-place status lines only on a real terminal; a pipe or NO_COLOR gets plain text."""
    TTY[0] = bool(sys.stdout and sys.stdout.isatty())
    if not TTY[0] or os.environ.get('NO_COLOR') or os.environ.get('TERM') == 'dumb':
        return
    if IS_WIN:
        try:
            import ctypes
            k32, mode = ctypes.windll.kernel32, ctypes.c_uint32()
            h = k32.GetStdHandle(-11)
            if not (k32.GetConsoleMode(h, ctypes.byref(mode)) and k32.SetConsoleMode(h, mode.value | 4)):
                return
        except (AttributeError, OSError):
            return
    COLOR[0] = True


def paint(text, *styles):
    codes = {'bold': '1', 'dim': '2', 'red': '31', 'green': '32', 'cyan': '36'}
    return '\x1b[%sm%s\x1b[0m' % (';'.join(codes[s] for s in styles), text) if COLOR[0] else text


def glyph(fancy, plain):
    # the classic Windows console font lacks these symbols; Windows Terminal (WT_SESSION) has them
    utf = 'utf' in (getattr(sys.stdout, 'encoding', None) or '').lower()
    return fancy if utf and (not IS_WIN or os.environ.get('WT_SESSION')) else plain


def mmss(seconds):
    return '%d:%02d' % divmod(max(0, int(float(seconds))), 60)


def units(cu):
    return 'about %s compute units' % ('%.2f' % cu if cu < 1 else '%.1f' % cu)


def minutes(n):
    return '%g minute%s' % (n, '' if n == 1 else 's')


def cli():
    """This script's command as the user would type it here, for the hints we print."""
    py = 'python3'
    if IS_WIN:
        names = [n for n in ('python3', 'python') if shutil.which(n) and 'windowsapps' not in shutil.which(n).lower()]
        py = names[0] if names else 'py -3' if shutil.which('py') else join_cmd([sys.executable])
    elif not shutil.which('python3'):
        py = join_cmd([sys.executable])
    here = os.path.normcase(os.path.abspath(os.getcwd())) == os.path.normcase(ROOT)
    return py + ' ' + ('llm.py' if here else join_cmd([os.path.join(ROOT, 'llm.py')]))


class Live:
    """The one status line under the step list: redrawn in place on a terminal, silent on a pipe."""

    def __init__(self):
        self.width, self.n = 0, 0

    def clear(self):
        if self.width:
            sys.stdout.write('\r' + ' ' * self.width + '\r')
            self.width = 0

    def put(self, parts):
        """Draws (text, styles) parts; a line wider than the terminal is cut and drawn without colors."""
        if not TTY[0]:
            return
        cols, plain = shutil.get_terminal_size((80, 24)).columns - 1, ''.join(t for t, styles in parts)
        self.clear()
        sys.stdout.write(plain[:cols] if len(plain) > cols else ''.join(paint(t, *styles) for t, styles in parts))
        sys.stdout.flush()
        self.width = min(len(plain), cols)

    def show(self, when, text):
        spin = glyph('⣾⣽⣻⢿⡿⣟⣯⣷', '|/-\\')
        self.n += 1
        self.put([('  ', ()), ('%5s' % when, ('dim',)), ('  ', ()), (spin[self.n % len(spin)], ('cyan',)), ('  ' + text, ())])

    def line(self, text):
        self.clear()
        print(text, flush=True)


class StartError(Exception):
    pass


def split_cmd(s, win=IS_WIN):
    if os.path.exists(s):
        return [s]
    if win:
        return [t[1:-1] if t.startswith('"') else t for t in re.findall(r'"[^"]*"|\S+', s)]
    return shlex.split(s)


def join_cmd(args, win=IS_WIN):
    return subprocess.list2cmdline(args) if win else shlex.join(args)


def find_tool(name):
    if IS_WIN:
        p = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'OpenSSH', name + '.exe')
        if os.path.exists(p):
            return p
    return shutil.which(name) or name


def colab_cmd():
    if os.environ.get('LLM_COLAB'):
        return split_cmd(os.environ['LLM_COLAB']) + ['--auth', 'oauth2']
    exe = shutil.which('colab') or os.path.join(os.path.expanduser('~'), '.local', 'bin', 'colab.exe' if IS_WIN else 'colab')
    return [exe, '--auth', 'oauth2']


def context_value(text):
    """Tokens per request: 32K, 64K, 131K, 262K or a number from 4096 to 262144 (rounded up to 256); None otherwise."""
    t = str(text).strip().lower().replace(',', '')
    if t in CONTEXTS:
        return CONTEXTS[t]
    return -(-int(t) // 256) * 256 if t.isdigit() and 4096 <= int(t) <= 262144 else None


def short_context(n):
    return next((k.upper() for k, v in CONTEXTS.items() if v == n and k not in ('128k', '256k')), '{:,}'.format(n))


def choice():
    """Image input and context per request in effect: LLM_PROFILE or --fast presets, then LLM_IMAGES and LLM_CONTEXT, then the settings."""
    images, context = bool(SETTINGS['images']), context_value(SETTINGS['context']) or 262144
    preset = os.environ.get('LLM_PROFILE')
    if preset:
        if preset not in LAYOUTS:
            sys.exit('unknown LLM_PROFILE %r (use default or fast)' % preset)
        images, context = (True, 262144) if preset == 'default' else (False, 131072)
    if os.environ.get('LLM_IMAGES'):
        images = os.environ['LLM_IMAGES'] == '1'
    if os.environ.get('LLM_CONTEXT'):
        context = context_value(os.environ['LLM_CONTEXT']) or sys.exit('LLM_CONTEXT must be 4096-262144 tokens, or 32K, 64K, 131K, 262K')
    return images, context


def layout(images, context):
    return 'fast' if not images and context <= 131072 else 'default'


def describe(images, context):
    return '%s context · image input %s' % (short_context(context), 'on' if images else 'off')


def model_efforts():
    return (('ds41', str(SETTINGS['effort'])), ('ds41-max', 'max'))


def profile():
    """Boot values for the chosen images and context; CTX is the total of the 2 request slots."""
    images, context = choice()
    lay = LAYOUTS[layout(images, context)]
    return {'CTX': str(2 * context), 'UBATCH': lay['UBATCH'], 'MMPROJ': '1' if images else '0', 'MOE_CACHE_MIB': lay['MOE_CACHE_MIB'],
            'DYN': lay['DYN'], 'DMA': lay['DMA']}


COLAB = colab_cmd()


def api_key():
    if 'key' not in _cache:
        with open(KEY_FILE) as f:
            _cache['key'] = f.read().strip()
    return _cache['key']


def redact(text):
    text = re.sub(r'(Bearer\s+)\S+', r'\1***', text)
    try:
        return text.replace(api_key(), '***') if api_key() else text
    except OSError:
        return text


def log(msg, echo=None):
    msg = redact(msg)
    with _log_lock:
        try:
            if os.path.getsize(LOG) > 1000000:
                os.replace(LOG, LOG + '.1')
        except OSError:
            pass
        try:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write(time.strftime('%Y-%m-%dT%H:%M:%S ') + msg + '\n')
        except OSError:
            pass
    if ECHO[0] if echo is None else echo:
        print(msg, flush=True)


def kill_tree(pid, hard=False):
    try:
        if IS_WIN:
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=CHILD_FLAGS, timeout=30)
        else:
            os.killpg(pid, signal.SIGKILL if hard else signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def run(cmd, timeout, input=None):
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL if input is None else subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=CHILD_ENV, creationflags=CHILD_FLAGS, start_new_session=not IS_WIN)
    except OSError as e:
        return 127, '%s: %s' % (cmd[0], e)
    try:
        out = p.communicate(None if input is None else input.encode(), timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        kill_tree(p.pid, hard=True)
        try:
            p.communicate(timeout=10)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        return 124, 'timeout'
    return p.returncode, out.decode('utf-8', 'replace')


def colab(args, timeout=120, input=None):
    return run(COLAB + args, timeout, input)


def vmx(cmd, timeout=100):
    code = ('import subprocess\nr = subprocess.run(%s, shell=True, capture_output=True, text=True, timeout=%d)\n'
            'print(r.stdout[-20000:] + r.stderr[-4000:])\n') % (json.dumps(cmd), timeout)
    return colab(['exec', '-s', NAME, '--timeout', str(timeout + 20)], timeout + 40, code)


def usage():
    rc, out = colab(['usage'], 90)
    bal = re.search(r'balance: ([0-9.]+)', out)
    rate = re.search(r'rate: ([0-9.]+)', out)
    return (float(bal.group(1)) if bal else None), (float(rate.group(1)) if rate else None)


def session_alive(name=NAME):
    rc, out = colab(['sessions'], 90)
    return ('[%s]' % name) in out


def vm_timeline():
    return vmx('cat /content/res/timeline.txt 2>/dev/null; ls /content/res/READY 2>/dev/null', 40)


def _win_start(pid):
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return '?' if ctypes.get_last_error() == 5 else None
    try:
        code, times = wintypes.DWORD(), [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return '?'
        if code.value != 259:
            return None
        if not k32.GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
            return '?'
        return str(times[0].dwHighDateTime << 32 | times[0].dwLowDateTime)
    finally:
        k32.CloseHandle(h)


def proc_start(pid):
    """Start-time token of a live process ('?' if alive but unknown), None if it is gone."""
    if not pid:
        return None
    if IS_WIN:
        try:
            return _win_start(pid)
        except Exception:
            try:
                out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/NH', '/FO', 'CSV'], capture_output=True, text=True,
                                     creationflags=CHILD_FLAGS, timeout=30).stdout
            except (OSError, subprocess.SubprocessError):
                return '?'
            return '?' if '"%d"' % pid in out else None
    try:
        with open('/proc/%d/stat' % pid) as f:
            rest = f.read().rsplit(')', 1)[1].split()
        return None if rest[0] in ('Z', 'X') else rest[19]
    except FileNotFoundError:
        if os.path.isdir('/proc'):
            return None
    except (OSError, IndexError):
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return '?'
    except OSError:
        return None
    return '?'


def me():
    return {'pid': os.getpid(), 'start': proc_start(os.getpid())}


def alive(rec):
    if not rec or not rec.get('pid'):
        return False
    st = proc_start(int(rec['pid']))
    return st is not None and (st == '?' or rec.get('start') in (None, '?', st))


def kill_rec(rec):
    if alive(rec):
        kill_tree(int(rec['pid']))


def remove(path):
    # Windows refuses to delete a file another process has open for a moment (the watchdog reads session.json)
    for k in range(30):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.1)


def _read(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except OSError:
        return None


def read_lock():
    raw = _read(LOCK)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        try:
            young = time.time() - os.path.getmtime(LOCK) < 5
        except OSError:
            return None
        return {'pid': None, 'busy': True} if young else {'pid': None}


def take_lock(info):
    """Creates the serve lock with O_EXCL; returns None when taken, else the record of the live holder."""
    os.makedirs(STATE, exist_ok=True)
    for attempt in range(5):
        try:
            fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raw, held = _read(LOCK), read_lock()
            if held and (held.get('busy') or alive(held)):
                return held
            if _read(LOCK) == raw:
                try:
                    os.remove(LOCK)
                except OSError:
                    time.sleep(0.2)
            continue
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(info, **me()), f)
        return None
    return read_lock() or {'pid': None, 'busy': True}


def release_lock():
    held = read_lock()
    if held and held.get('pid') == os.getpid():
        remove(LOCK)


def serve_running():
    held = read_lock()
    return held if held and held.get('port') and alive(held) else None


def load_session():
    for k in range(5):
        try:
            with open(SESSION) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.1)
    return None


def save_session(s):
    tmp = '%s.%d.%d.tmp' % (SESSION, os.getpid(), threading.get_ident())
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=1)
    for k in range(30):
        try:
            os.replace(tmp, SESSION)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, SESSION)


def cleanup_local(s):
    kill_rec(s.get('tunnel'))
    kill_rec(s.get('watchdog'))
    remove(SESSION)


def stop_cloud(s, reason):
    """Kills the tunnel, wipes secrets on the VM, stops it and logs the cost; returns True if it is still running."""
    s = s or {'name': NAME}
    kill_rec(s.get('tunnel'))
    if session_alive():
        vmx('rm -rf /content/secrets; echo CLEANED', 30)
        for attempt in range(3):
            colab(['stop', '-s', NAME], 120)
            if not session_alive():
                break
            time.sleep(5)
    still = session_alive()
    bal, rate = usage()
    # Colab books usage minutes late, so the cost comes from the running time, not from the balance
    ran = time.time() - s['t0'] if s.get('t0') else None
    cu = ran / 3600 * G4_CU_PER_HOUR if ran is not None else None
    log('down (%s): session %s, balance %s CU%s' % (reason, 'STILL ACTIVE, run: llm.py down' if still else 'stopped', bal,
                                                  ', ran %s, about %.2f CU' % (mmss(ran), cu) if ran is not None else ''))
    _cache['stopped'] = (cu, ran)
    if still:
        s['stopping'] = True
        save_session(s)
    else:
        cleanup_local(s)
    return still


def vm_files():
    out = []
    for d, dirs, files in os.walk(VM_DIR):
        dirs[:] = sorted(x for x in dirs if x != '__pycache__')
        out += [os.path.relpath(os.path.join(d, f), VM_DIR).replace(os.sep, '/') for f in sorted(files) if not f.endswith('.pyc')]
    return out


def missing_vm_files():
    """Files that boot.sh, serve.sh, build.sh or patches/series name but vm/ lacks; a start is refused before allocation."""
    have, refs = set(vm_files()), set()
    for f in ('boot.sh', 'serve.sh', 'build.sh'):
        try:
            with open(os.path.join(VM_DIR, f), encoding='utf-8', errors='replace') as fh:
                refs |= set(re.findall(r'(?:\$V|/content/vm)/([\w.-]+\.(?:py|sh|txt|c))\b', fh.read()))
        except OSError:
            refs.add(f)
    refs |= {f for f in ('boot.sh', 'serve.sh', 'build.sh', 'patches/series')}
    if 'patches/series' in have:
        with open(os.path.join(VM_DIR, 'patches', 'series'), encoding='utf-8') as fh:
            refs |= {'patches/' + l.strip() for l in fh if l.strip() and not l.startswith('#')}
    return sorted(r for r in refs if r not in have and '$' not in r)


def build_bundle(path):
    """Writes the VM tarball with LF line endings (Windows checkouts may have CRLF) and fixed file modes."""
    items = [(os.path.join(VM_DIR, f), 'vm/' + f, 0o644, None) for f in vm_files()] + [(KEY_FILE, 'secrets/key', 0o600, None)]
    if os.environ.get('HF_TOKEN'):
        items.append((None, 'secrets/hf_hdr', 0o600, ('Authorization: Bearer %s\n' % os.environ['HF_TOKEN'].strip()).encode()))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_BINARY', 0), 0o600)
    with os.fdopen(fd, 'wb') as f, tarfile.open(fileobj=f, mode='w:gz') as tar:
        for src, arc, mode, data in items:
            if src:
                with open(src, 'rb') as fh:
                    data = fh.read().replace(b'\r\n', b'\n')
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mode, ti.mtime = len(data), mode, int(time.time())
            tar.addfile(ti, io.BytesIO(data))


def boot_vars():
    env = dict(profile(), UNCENSOR='1' if SETTINGS['uncensor'] else '0')
    env.update({k: os.environ[k] for k in BOOT_KEYS if k in os.environ})
    for kv in os.environ.get('LLM_BOOT_ENV', '').split():
        k, sep, v = kv.partition('=')
        if not sep:
            raise StartError('LLM_BOOT_ENV entry without "=": ' + kv)
        env[k] = v
    for k, v in env.items():
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k) or "'" in v:
            raise StartError('bad boot env %s: names must be shell words, values must not contain single quotes' % k)
    return env


def boot_env():
    return ' '.join("%s='%s'" % (k, v) for k, v in boot_vars().items())


def boot_cmd(envs):
    return ('cd /content && umask 077 && tar xzf bundle.tgz && rm -f bundle.tgz && chmod 700 secrets && chmod 600 secrets/* && '
            'gcc -shared -fPIC -O2 -o vm/libpin.so vm/pin.c -ldl -lpthread && '
            'mkdir -p res && (%s setsid nohup bash vm/boot.sh > res/boot.out 2>&1 < /dev/null &) && echo BOOT_STARTED' % envs)


def ssh_cmd(port, win=IS_WIN):
    proxy = join_cmd(COLAB + ['ssh', '--proxy-mode', '-s', NAME, '-i', SSH_KEY], win).replace('%', '%%')
    base = split_cmd(os.environ['LLM_SSH_CMD'], win) if os.environ.get('LLM_SSH_CMD') else [find_tool('ssh')]
    null = 'NUL' if win else '/dev/null'
    return base + ['-N', '-F', null, '-o', 'ProxyCommand=' + proxy, '-o', 'User=root', '-i', SSH_KEY, '-o', 'IdentitiesOnly=yes',
                   '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=' + null, '-o', 'LogLevel=ERROR', '-o', 'BatchMode=yes',
                   '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes',
                   '-L', '127.0.0.1:%d:127.0.0.1:%d' % (port, VM_PORT), NAME]


def pick_port():
    for p in (range(TUNNEL_PORT, TUNNEL_PORT + 50) if TUNNEL_PORT else [0]):
        s = socket.socket()
        try:
            s.bind(('127.0.0.1', p))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    raise StartError('no free local port for the tunnel')


def http_call(port, method, path, body=None, headers=None, timeout=10, host='127.0.0.1'):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        return conn, conn.getresponse()
    except BaseException:
        conn.close()
        raise


def tunnel_health(port, timeout=10):
    try:
        conn, r = http_call(port, 'GET', '/health', headers={'Authorization': 'Bearer ' + api_key()}, timeout=timeout)
        ok = r.status == 200 and b'ok' in r.read()
        conn.close()
        return ok
    except (OSError, http.client.HTTPException):
        return False


def spawn_detached(args, out=None):
    kw = dict(stdin=subprocess.DEVNULL, stdout=out or subprocess.DEVNULL, stderr=subprocess.STDOUT if out else subprocess.DEVNULL,
              env=CHILD_ENV)
    if not IS_WIN:
        return subprocess.Popen(args, start_new_session=True, **kw)
    try:
        return subprocess.Popen(args, creationflags=DETACHED | NEW_GROUP | NO_WINDOW | BREAKAWAY, **kw)
    except OSError:
        return subprocess.Popen(args, creationflags=DETACHED | NEW_GROUP | NO_WINDOW, **kw)


def spawn_watchdog(t0):
    # the grace lets the proxy's own hard stop (wipes secrets, logs the cost) go first; the watchdog is the backstop
    deadline = t0 + MAX_HOURS * 3600 + min(120.0, MAX_HOURS * 180)
    p = spawn_detached([sys.executable, os.path.realpath(__file__), '_watchdog', NAME, repr(t0), repr(deadline)])
    return {'pid': p.pid, 'start': proc_start(p.pid)}


def watchdog(name, t0, deadline):
    """Detached guard: stops the VM at the hard deadline, or after LLM_IDLE_MIN once the proxy process is gone."""
    poll, orphan = max(1.0, min(30.0, IDLE_MIN * 6)), None
    while os.path.exists(SESSION):
        s = load_session()
        if s is not None and (s.get('name') != name or abs(s.get('t0', 0) - t0) > 1e-3):
            return
        now, why = time.time(), None
        if now >= deadline:
            why = 'hard stop after %.2f h' % MAX_HOURS
        elif s is not None:
            orphan = None if alive(s.get('serve')) else (orphan or now)
            if orphan and now - orphan >= IDLE_MIN * 60:
                why = 'proxy gone for %.1f min' % IDLE_MIN
        if why:
            log('watchdog %s: %s, stopping the VM' % (name, why))
            kill_rec((s or {}).get('tunnel'))
            for k in range(3):
                colab(['stop', '-s', name], 120)
                if not session_alive(name):
                    break
                time.sleep(5)
            log('watchdog %s: session %s' % (name, 'STILL ACTIVE' if session_alive(name) else 'stopped'))
            return
        time.sleep(max(0.2, min(poll, deadline - now)))


def drain(p, on_create=None):
    for raw in p.stderr:
        line = raw.decode('utf-8', 'replace').strip()
        if line and not re.search(r'open failed: connect failed', line):
            log('tunnel: ' + line[-300:], echo=False)
            if on_create and re.search(r'\[colab\] Creating (runtime|session)', line):
                on_create()


class Manager:
    """VM lifecycle inside serve: start on demand, hold waiters, keep the tunnel up, stop on idle, hard cap or request."""

    def __init__(self):
        self.cond = threading.Condition()
        self.state, self.steps, self.attempt, self.errors, self.begun = 'off', [], 0, {}, 0.0
        self.fail_until, self.fail_msg, self.stop_reason = 0.0, '', None
        self.active, self.last_activity, self.served = 0, time.time(), 0
        self.tunnel, self.port, self.tunnel_ok, self.repairing = None, None, False, False
        self.sess, self.closing, self.last_retry = None, False, 0.0
        self.tlock, self.no_tunnel, self.last_stop = threading.Lock(), False, None

    def ready(self):
        return self.state == 'ready' and self.tunnel_ok

    def enter(self):
        with self.cond:
            if self.ready():
                self.active += 1
                return True
        return False

    def leave(self, served=1):
        with self.cond:
            self.active -= 1
            self.served += served
            self.last_activity = time.time()

    def acquire(self, on_tick=None, every=KEEPALIVE_S):
        """Waits until the VM serves requests and counts the caller as active; returns None or an error message."""
        t0 = time.time()
        tick, mine = t0 + every, None
        while True:
            with self.cond:
                if self.ready():
                    self.active += 1
                    return None
                if mine is not None and mine in self.errors:
                    return self.errors[mine]
                if mine is None and self.state in ('off', 'stopping') and time.time() < self.fail_until:
                    return self.fail_msg
                if self.state == 'off':
                    if self.closing:
                        return 'proxy is shutting down'
                    self._begin(True)
                if self.state == 'starting':
                    mine = self.attempt
                self.cond.wait(max(0.05, tick - time.time()))
            if time.time() >= tick:
                tick += every
                if on_tick:
                    on_tick(int(time.time() - t0))

    def wait_off(self, on_tick=None, limit=900, every=KEEPALIVE_S):
        t0 = time.time()
        tick = t0 + every
        while time.time() - t0 < limit:
            with self.cond:
                if self.state == 'off':
                    return os.path.exists(SESSION)
                self.cond.wait(max(0.05, tick - time.time()))
            if time.time() >= tick:
                tick += every
                if on_tick:
                    on_tick(int(time.time() - t0))
        return True

    def adopt(self):
        with self.cond:
            if self.state == 'off':
                self._begin(False)

    def _begin(self, new):
        self.attempt += 1
        self.state, self.steps, self.stop_reason, self.no_tunnel, self.begun = 'starting', [], None, False, time.time()
        threading.Thread(target=self._run_start, args=(self.attempt, new), daemon=True).start()

    def _check_stop(self):
        if self.stop_reason:
            raise StartError('start cancelled (%s)' % self.stop_reason)

    def _run_start(self, attempt, new):
        try:
            ok, err = self._start(new), None
        except StartError as e:
            ok, err = False, str(e)
        except Exception as e:
            ok, err = False, 'internal error: %r' % e
        with self.cond:
            if ok and not self.stop_reason:
                self.state, self.tunnel_ok, self.last_activity = 'ready', True, time.time()
                self.cond.notify_all()
                return
            if ok:
                err = 'start cancelled (%s)' % self.stop_reason
            if err is None:
                self.state = 'off'
                self.cond.notify_all()
                return
            self.errors[attempt], self.fail_msg, self.fail_until = err, err, time.time() + FAIL_COOLDOWN
            self.state = 'stopping'
            self.cond.notify_all()
        log('start failed: ' + err)
        self._stop('start failed')

    def request_stop(self, reason, only_idle=False):
        with self.cond:
            if only_idle and not (self.state == 'ready' and self.active == 0 and
                                  time.time() - self.last_activity > IDLE_MIN * 60):
                return
            if self.state == 'starting':
                self.stop_reason = reason
                return
            if self.state == 'stopping' or (self.state == 'off' and not os.path.exists(SESSION)):
                return
            self.state = 'stopping'
            self.cond.notify_all()
        log('stopping the VM (%s)' % reason)
        threading.Thread(target=self._stop, args=(reason,), daemon=True).start()

    def _stop(self, reason):
        with self.tlock:
            self.no_tunnel = True
            self._kill_tunnel()
        try:
            stop_cloud(load_session(), reason)
            cu, ran = _cache.get('stopped', (None, None))
            self.last_stop = {'reason': reason, 'cu': cu, 'ran': ran, 'at': time.time()}
        except Exception as e:
            log('stop failed: %r' % e)
        with self.cond:
            self.state, self.tunnel_ok, self.sess = 'off', False, None
            self.cond.notify_all()

    def _stage(self, s, key, t0):
        s['stages'][key] = round(time.time() - t0, 1)
        save_session(s)

    def activity(self):
        """What the start is doing now, derived from the steps done so far."""
        done = {key for key, at, label in self.steps}
        if 'allocated' not in done:
            return 'allocating a G4 GPU'
        if 'uploaded' not in done:
            return 'uploading the boot files'
        todo = [w for k, w in (('model', 'downloading the model'), ('build', 'building llama.cpp')) if k not in done]
        rest = [w for k, w in LATER if k not in done]
        todo = [] if len(rest) < len(LATER) else todo
        return ' and '.join(todo) if todo else rest[0] if rest else 'opening the endpoint'

    def _start(self, new):
        s = load_session()
        if session_alive():
            log('session %s is running, adopting it' % NAME)
            s = s or {'name': NAME, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 't0': time.time(), 'stages': {}}
            s.pop('stopping', None)
            s['serve'] = me()
            if not alive(s.get('watchdog')):
                s['watchdog'] = spawn_watchdog(s['t0'])
            save_session(s)
            self.sess = s
            self.steps += [('allocated', time.time(), None), ('uploaded', time.time(), None)]
            self._await_ready(s, adopted=True)
        elif not new:
            if s:
                cleanup_local(s)
                log('cleaned up state of the stopped session %s' % NAME)
            return False
        else:
            if s:
                cleanup_local(s)
            s = self._launch()
            self._await_ready(s, adopted=False)
        return True

    def _launch(self):
        envs = boot_env()
        try:
            missing = missing_vm_files()
        except OSError as e:
            raise StartError('cannot read vm/ (LLM_VM_DIR %s): %s' % (VM_DIR, e))
        if missing:
            raise StartError('vm/ is incomplete, missing: ' + ', '.join(missing))
        for p in (KEY_FILE, SSH_KEY):
            if not os.path.exists(p):
                raise StartError('missing %s; run: python3 llm.py setup' % p)
        bal, rate = usage()
        if bal is None:
            raise StartError('the Colab CLI cannot read your balance (login expired?); run: python3 llm.py setup')
        if bal < MIN_BALANCE:
            raise StartError('Colab balance %.2f compute units is below LLM_MIN_BALANCE %.0f' % (bal, MIN_BALANCE))
        self._check_stop()
        t0 = time.time()
        s = {'name': NAME, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'balance_start': bal, 't0': t0, 'stages': {}, 'serve': me(),
             'choice': list(choice())}
        save_session(s)
        s['watchdog'] = spawn_watchdog(t0)
        save_session(s)
        self.sess = s
        log('up: %s, balance %.2f CU, hard stop after %.1f h, idle stop after %.1f min' % (describe(*choice()), bal, MAX_HOURS, IDLE_MIN))
        for attempt in range(2):
            rc, out = colab(['new', '-s', NAME, '--gpu', 'G4'], 300)
            if rc == 0 and 'READY' in out:
                break
            log('allocation attempt %d failed: %s' % (attempt + 1, out.strip()[-300:]))
            self._check_stop()
        else:
            raise StartError('no G4 GPU from Colab (none free, or the plan has no compute units for it): ' + out.strip()[-200:])
        self._stage(s, 'allocated', t0)
        log('G4 allocated after %.0f s' % s['stages']['allocated'])
        self.steps.append(('allocated', time.time(), 'G4 GPU allocated'))
        bundle = os.path.join(STATE, 'bundle.tgz')
        try:
            build_bundle(bundle)
            rc, out = colab(['upload', '-s', NAME, bundle, '/content/bundle.tgz'], 120)
        finally:
            try:
                os.remove(bundle)
            except OSError:
                pass
        if rc != 0 or 'Upload failed' in out:
            raise StartError('bundle upload failed: ' + out.strip()[-200:])
        self._check_stop()
        rc, out = vmx(boot_cmd(envs), 60)
        if 'BOOT_STARTED' not in out:
            raise StartError('boot start failed: ' + out.strip()[-300:])
        self._stage(s, 'boot_started', t0)
        self.steps.append(('uploaded', time.time(), None))
        log('boot started on the GPU VM after %.0f s; ready in about 4 minutes' % s['stages']['boot_started'])
        return s

    def _await_ready(self, s, adopted):
        """Ready when /health answers through the tunnel: the VM opens port 18080 only after the warmup, so the tunnel can open at once."""
        t0, seen, fails, spawns, next_tl, box = time.time(), 0, 0, 0, time.time(), None
        base = s.get('t0', t0) + s.get('stages', {}).get('boot_started', 0)
        kill_rec(s.get('tunnel'))
        while time.time() - t0 < BOOT_TIMEOUT:
            # a timeline read can take a minute while the VM builds, so it runs beside the health check
            if box is None and time.time() >= next_tl:
                next_tl, box = time.time() + POLL_S, []
                threading.Thread(target=lambda b=box: b.append(vm_timeline()), daemon=True).start()
            if box:
                (rc, out), box = box[0], None
                if rc != 0:
                    log('vm timeline read failed (rc %d)' % rc, echo=False)
                lines = [l for l in out.splitlines() if re.match(r'^\d+s ', l)]
                if adopted and rc == 0 and not lines:
                    raise StartError('running session %s has no boot timeline' % NAME)
                adopted, fails = False, (0 if rc == 0 else fails + 1)
                if fails >= 5 and not session_alive():
                    raise StartError('the VM disappeared during boot')
                seen = self._take(lines, seen, base)
                bad = [l for l in lines if re.search(r'FAILED|died|GUARD', l)]
                if bad:
                    raise StartError('boot failed on VM: ' + bad[0])
            if self.tunnel is None or self.tunnel.poll() is not None:
                # never respawn the bridge for a session colab no longer lists: proxy-mode would create a new one
                if spawns and (spawns >= 5 or not session_alive()):
                    raise StartError('tunnel kept failing or the VM disappeared during boot')
                self._kill_tunnel()
                self._spawn_tunnel(s)
                spawns += 1
            elif tunnel_health(self.port, timeout=5):
                rc, out = vm_timeline()
                self._take([l for l in out.splitlines() if re.match(r'^\d+s ', l)], seen, base)
                self._stage(s, 'endpoint_ready', s.get('t0', t0))
                log('READY after %.0f s (stages %s)' % (time.time() - s.get('t0', t0), s['stages']))
                return
            self._check_stop()
            time.sleep(HEALTH_S)
        raise StartError('boot did not finish in %d min' % (BOOT_TIMEOUT // 60))

    def _take(self, lines, seen, base):
        """Logs the new timeline lines and records the known ones as steps, timed by the VM clock."""
        for l in lines[seen:]:
            log('  vm ' + l)
            m = re.match(r'^(\d+)s \S+ (.*)$', l)
            for key, rx, label in STEPS if m else ():
                if re.match(rx, m.group(2)):
                    self.steps.append((key, min(base + int(m.group(1)), time.time()), label))
        return max(seen, len(lines))

    def _spawn_tunnel(self, s):
        port = pick_port()
        with self.tlock:
            # a ProxyCommand started after colab stop would auto-create a fresh session, so never spawn during a stop
            if self.no_tunnel:
                raise StartError('stop in progress')
            try:
                p = subprocess.Popen(ssh_cmd(port), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     env=CHILD_ENV, creationflags=CHILD_FLAGS, start_new_session=not IS_WIN)
            except OSError as e:
                raise StartError('cannot run ssh: %s' % e)
            self.tunnel, self.port = p, port
        threading.Thread(target=drain, args=(p, self._auto_created), daemon=True).start()
        s['tunnel'] = {'pid': p.pid, 'start': proc_start(p.pid), 'port': port}
        save_session(s)

    def _auto_created(self):
        log('ALERT: colab ssh --proxy-mode created a new session %s because it did not exist; stopping it' % NAME)
        threading.Thread(target=colab, args=(['stop', '-s', NAME], 120), daemon=True).start()
        self.request_stop('tunnel created a session')

    def _kill_tunnel(self):
        p, self.tunnel = self.tunnel, None
        if p and p.poll() is None:
            kill_tree(p.pid)
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                kill_tree(p.pid, hard=True)

    def _tunnel_up(self, s):
        self._spawn_tunnel(s)
        t1 = time.time()
        while time.time() - t1 < 60:
            if tunnel_health(self.port):
                return True
            if self.tunnel is None or self.tunnel.poll() is not None:
                break
            self._check_stop()
            time.sleep(1)
        self._kill_tunnel()
        return False

    def _repair(self):
        try:
            log('tunnel exited, checking the session')
            s = self.sess or load_session() or {'name': NAME}
            reason = 'session gone'
            if session_alive():
                reason = 'tunnel failed 3 times'
                for attempt in range(3):
                    if self.state != 'ready':
                        return
                    if self._tunnel_up(s):
                        with self.cond:
                            self.tunnel_ok = True
                            self.cond.notify_all()
                        log('tunnel restored')
                        return
            log('gpu vm unreachable (%s)' % reason)
            self.request_stop(reason)
        except Exception as e:
            log('tunnel repair failed: %r' % e)
            self.request_stop('tunnel repair failed')
        finally:
            self.repairing = False

    def tunnel_failed(self):
        """Called after an upstream connect error; returns True once a dead tunnel has been replaced."""
        p = self.tunnel
        if p is None or p.poll() is None:
            return False
        self._start_repair()
        t0 = time.time()
        with self.cond:
            while self.state == 'ready' and not self.tunnel_ok and time.time() - t0 < 90:
                self.cond.wait(1)
            return self.ready()

    def _start_repair(self):
        with self.cond:
            if self.state != 'ready' or self.repairing:
                return
            self.repairing, self.tunnel_ok = True, False
        threading.Thread(target=self._repair, daemon=True).start()

    def monitor(self):
        while True:
            time.sleep(0.5)
            now = time.time()
            with self.cond:
                st, s, p = self.state, self.sess, self.tunnel
            if st in ('starting', 'ready') and s and now > s.get('t0', now) + MAX_HOURS * 3600:
                self.request_stop('hard stop after %.2f h' % MAX_HOURS)
            elif st == 'ready':
                self.request_stop('idle %.1f min' % IDLE_MIN, only_idle=True)
                if p is not None and p.poll() is not None:
                    self._start_repair()
            elif st == 'off' and os.path.exists(SESSION) and now - self.last_retry > 60:
                self.last_retry = now
                self.request_stop('retrying the stop')

    def shutdown(self):
        with self.cond:
            self.closing = True
            busy = self.state != 'off' or os.path.exists(SESSION)
        if busy:
            log('proxy exiting: stopping the VM first (Ctrl+C again leaves it to the watchdog)')
            self.request_stop('proxy exit')
            self.wait_off()

    def status(self):
        with self.cond:
            s = self.sess or {}
            state = 'ready (tunnel reconnecting)' if self.state == 'ready' and not self.tunnel_ok else self.state
            ran = time.time() - s['t0'] if s.get('t0') and self.state != 'off' else None
            return {'state': state, 'stage': self.activity() if self.state == 'starting' else '', 'session': NAME, 'active': self.active,
                    'served': self.served, 'ran': ran, 'cu': ran / 3600 * G4_CU_PER_HOUR if ran is not None else None,
                    'last_stop': self.last_stop, 'idle_s': round(time.time() - self.last_activity), 'idle_stop_min': IDLE_MIN, 'max_hours': MAX_HOURS,
                    'started': s.get('started'), 'stages': s.get('stages'), 'last_error': self.fail_msg or None, 'pid': os.getpid()}


def with_effort(body):
    """The model name ds41-max means reasoning effort max for clients that do not send reasoning_effort themselves."""
    try:
        req = json.loads(body)
    except ValueError:
        return body
    if not isinstance(req, dict) or req.get('reasoning_effort') is not None:
        return body
    effort = dict(model_efforts()).get(req.get('model'))
    if effort is None or effort == '50':
        return body
    req['reasoning_effort'] = effort
    return json.dumps(req).encode()


def error_obj(code, msg, kind='unavailable_error'):
    return {'error': {'code': code, 'message': msg, 'type': kind}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    timeout = 900
    disable_nagle_algorithm = True
    server_version, sys_version = 'llm-proxy', ''

    def setup(self):
        self.request.settimeout(30)
        BaseHTTPRequestHandler.setup(self)

    def log_message(self, *args):
        pass

    def chunk(self, data):
        self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def start_chunked(self, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

    def fail(self, code, msg, sse=False, obj=None):
        obj = obj or error_obj(code, msg)
        if not sse:
            return self.send_json(code, obj)
        try:
            self.chunk(b'data: ' + json.dumps(obj).encode() + b'\n\n')
            self.wfile.write(b'0\r\n\r\n')
        except OSError:
            pass
        self.close_connection = True

    def authed(self):
        got = self.headers.get('Authorization', '').encode()
        if hmac.compare_digest(got, ('Bearer ' + api_key()).encode()):
            return True
        self.send_json(401, error_obj(401, 'Invalid API Key', 'authentication_error'))
        return False

    def read_body(self):
        try:
            if 'chunked' in self.headers.get('Transfer-Encoding', '').lower():
                parts, total = [], 0
                while True:
                    n = int(self.rfile.readline(1024).split(b';')[0].strip(), 16)
                    if n == 0:
                        while self.rfile.readline(1024) not in (b'\r\n', b'\n', b''):
                            pass
                        return b''.join(parts)
                    total += n
                    if total > MAX_BODY:
                        break
                    parts.append(self.rfile.read(n))
                    self.rfile.readline(8)
            else:
                n = int(self.headers.get('Content-Length') or 0)
                if n <= MAX_BODY:
                    return self.rfile.read(n) if n > 0 else b''
            self.close_connection = True
            self.fail(413, 'request body too large', obj=error_obj(413, 'request body too large', 'invalid_request_error'))
        except ValueError:
            self.close_connection = True
            self.fail(400, 'bad request body', obj=error_obj(400, 'bad request body', 'invalid_request_error'))
        return None

    def do_GET(self):
        self.route(None)

    def do_POST(self):
        body = self.read_body()
        if body is not None:
            self.route(body)

    def host_ok(self):
        # on loopback, only local host names: a web page that rebinds its DNS name to 127.0.0.1 cannot reach the endpoint
        host = (self.headers.get('Host') or '').lower()
        name = host.split(']')[0] + ']' if host.startswith('[') else host.rsplit(':', 1)[0]
        return not LOOPBACK or name in ('', '127.0.0.1', 'localhost', '[::1]')

    def models(self):
        info = profile()
        return {'object': 'list', 'data': [{'id': m, 'object': 'model', 'owned_by': 'llm.py', 'created': 0, 'reasoning_effort': e,
                                            'context_length': int(info['CTX']) // 2, 'image_input': info['MMPROJ'] == '1'} for m, e in model_efforts()]}

    def route(self, body):
        mgr, path = self.server.mgr, self.path.split('?')[0]
        if not self.host_ok():
            return self.fail(403, 'unexpected Host header', obj=error_obj(403, 'unexpected Host header', 'permission_error'))
        if path.startswith('/_llm/'):
            return self.control(path, body)
        if path == '/v1/models' and body is None:
            if self.authed():
                self.send_json(200, self.models())
            return
        if body and path == '/v1/chat/completions':
            body = with_effort(body)
        if body is not None and path.startswith('/v1/'):
            if self.authed():
                t0 = time.time()
                self.held(body)
                log('%s %s done in %.1f s' % (self.command, path, time.time() - t0), echo=False)
            return
        if body is None and mgr.ready():
            return self.forward(None)
        if body is not None and mgr.enter():
            try:
                return self.forward(body)
            finally:
                mgr.leave()
        if path in ('/health', '/v1/health'):
            return self.send_json(503, dict(error_obj(503, 'gpu vm %s' % mgr.state), status=mgr.state))
        self.fail(503, 'gpu vm %s; a POST to /v1/... starts it' % mgr.state)

    def held(self, body):
        mgr = self.server.mgr
        if mgr.enter():
            try:
                return self.forward(body)
            finally:
                mgr.leave()
        stream = False
        try:
            stream = bool(json.loads(body).get('stream')) if body else False
        except (ValueError, AttributeError):
            pass
        if not stream:
            err = mgr.acquire()
            if err:
                return self.fail(503, err)
        else:
            self.start_chunked('text/event-stream')
            tick = lambda n: self.chunk(b': starting gpu vm (%d s)\n\n' % n)
            try:
                tick(0)
                err = mgr.acquire(tick)
            except OSError:
                self.close_connection = True
                return
            if err:
                return self.fail(503, err, sse=True)
        try:
            self.forward(body, sse=stream)
        finally:
            mgr.leave()

    def forward(self, body, sse=False):
        """Sends the request through the tunnel and pipes the answer back; sse means 200 and SSE headers were already sent."""
        mgr = self.server.mgr
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        for attempt in range(2):
            try:
                conn, resp = http_call(mgr.port, self.command, self.path, body, headers, UPSTREAM_TIMEOUT)
                break
            except (OSError, http.client.HTTPException) as e:
                if attempt == 0 and mgr.tunnel_failed():
                    continue
                return self.fail(502, 'upstream unreachable (%s)' % e.__class__.__name__, sse)
        try:
            if sse:
                if resp.status != 200:
                    raw = resp.read()
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        obj = None
                    if not isinstance(obj, dict) or 'error' not in obj:
                        obj = error_obj(resp.status, raw.decode('utf-8', 'replace')[:500], 'upstream_error')
                    return self.fail(resp.status, '', True, obj)
                self.pipe(resp, True)
                return
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            n = resp.getheader('Content-Length')
            if n is not None and not resp.chunked:
                self.send_header('Content-Length', n)
                self.end_headers()
                self.pipe(resp, False)
            else:
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                self.pipe(resp, True)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            conn.close()

    def pipe(self, resp, chunked):
        while True:
            data = resp.read1(65536) if chunked else resp.read(65536)
            if not data:
                break
            if chunked:
                self.chunk(data)
            else:
                self.wfile.write(data)
        if chunked:
            self.wfile.write(b'0\r\n\r\n')

    def control(self, path, body):
        if not self.authed():
            return
        mgr = self.server.mgr
        if path == '/_llm/status':
            return self.send_json(200, mgr.status())
        if path not in ('/_llm/up', '/_llm/down') or body is None:
            return self.fail(404, 'unknown control request', obj=error_obj(404, 'unknown control request', 'not_found_error'))
        self.start_chunked('text/plain; charset=utf-8')
        say = lambda text: self.chunk((text + '\n').encode())
        try:
            if path == '/_llm/up':
                # protocol for cmd_up: "step SECONDS LABEL", "wait SECONDS ACTIVITY", then "READY SECONDS" or "FAILED MESSAGE"
                with mgr.cond:
                    joined = mgr.state == 'starting'
                    t0, sent = (mgr.begun if joined else time.time()), [0 if joined else len(mgr.steps)]

                def steps():
                    done = mgr.steps
                    sent[0] = 0 if sent[0] > len(done) else sent[0]
                    for key, at, label in done[sent[0]:]:
                        if label:
                            say('step %d %s' % (max(0, at - t0), label))
                    sent[0] = len(done)

                def tick(n):
                    steps()
                    say('wait %d %s' % (n, mgr.activity()))
                err = mgr.acquire(tick, every=1)
                if not err:
                    mgr.leave(served=0)
                    steps()
                say('FAILED ' + err if err else 'READY %d' % (time.time() - t0))
                self.wfile.write(b'0\r\n\r\n')
                return
            try:
                opts = json.loads(body or b'{}')
            except ValueError:
                opts = {}
            _cache.pop('stopped', None)
            mgr.request_stop('down')
            still = mgr.wait_off(lambda n: say('wait %d stopping the GPU' % n), every=1)
            cu, ran = _cache.pop('stopped', (None, None))
            say('STILL' if still else 'STOPPED %s %s' % (cu, ran))
            self.wfile.write(b'0\r\n\r\n')
            if opts.get('exit'):
                self.close_connection = True
                threading.Thread(target=self.server.shutdown, daemon=True).start()
        except OSError:
            self.close_connection = True


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not IS_WIN

    def server_bind(self):
        if IS_WIN:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        e = sys.exc_info()[1]
        if not isinstance(e, (OSError, http.client.HTTPException)):
            log('handler error: %r' % e, echo=False)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def serve(detached=False):
    ECHO[0] = not detached
    if not os.path.exists(KEY_FILE):
        sys.exit('missing %s; run: python3 llm.py setup' % KEY_FILE)
    profile()
    held = take_lock({'port': PORT, 'detached': detached, 'choice': list(choice())})
    if held:
        sys.exit('llm.py serve is already running (pid %s, port %s)' % (held.get('pid'), held.get('port')))
    try:
        srv = Server((BIND, PORT), Handler)
    except OSError as e:
        release_lock()
        sys.exit('cannot serve on %s:%d: %s (another program on this port? set LLM_PORT)' % (BIND, PORT, e))
    srv.mgr = Manager()
    for sig in ('SIGTERM', 'SIGBREAK', 'SIGHUP'):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _interrupt)
    if os.path.exists(SESSION):
        srv.mgr.adopt()
    threading.Thread(target=srv.mgr.monitor, daemon=True).start()
    log('serve: http://%s:%d/v1, %s, session %s (idle stop %.1f min, hard stop %.1f h)'
        % (HOST, PORT, describe(*choice()), NAME, IDLE_MIN, MAX_HOURS))
    try:
        srv.serve_forever(0.5)
    except KeyboardInterrupt:
        log('serve: interrupted')
    finally:
        srv.server_close()
        try:
            srv.mgr.shutdown()
        except KeyboardInterrupt:
            log('serve: second interrupt, leaving the VM to the watchdog')
        release_lock()
        log('serve: exit')


def control_call(port, path, body, timeout):
    data = None if body is None else json.dumps(body).encode()
    return http_call(port, 'GET' if body is None else 'POST', path, data,
                     {'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'}, timeout, HOST)


def follow(resp, header=None):
    """Draws a start or stop stream: finished steps as lines, the current one as a live line; returns the final word and its rest."""
    live = Live()
    for raw in resp:
        kind, _, rest = raw.decode('utf-8', 'replace').strip().partition(' ')
        if kind in ('step', 'wait'):
            if header:
                print(header + '\n', flush=True)
                header = None
            when, _, text = rest.partition(' ')
            if kind == 'step':
                live.line('  %s  %s  %s' % (paint('%5s' % mmss(when), 'dim'), paint(glyph('✓', '+'), 'green'), text))
            else:
                live.show(mmss(when), text)
        elif kind:
            live.clear()
            return kind, rest
    live.clear()
    return '', ''


def confirm(question):
    if os.environ.get('LLM_YES') == '1':
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    try:
        return input(question + ' [Y/n] ').strip().lower() in ('', 'y', 'yes')
    except EOFError:
        return False


def effort_value(text):
    """A reasoning effort DeepSeek V4.1 accepts: 1-100, low (50), high (75) or max; None otherwise."""
    t = str(text).strip().lower()
    return t if t in ('low', 'high', 'max') or (t.isdigit() and 1 <= int(t) <= 100) else None


def number(text, low, high):
    try:
        v = float(text)
    except ValueError:
        return None
    return (int(v) if v == int(v) else v) if low <= v <= high else None


def pick(label, hint, default, parse, options=()):
    """One row of the settings form, with options listed above it; asks again until the answer is valid."""
    for i, (key, text) in enumerate(options):
        print('  %s%s  %s' % (paint('%-12s' % (label if i == 0 else ''), 'bold'), paint(key, 'cyan'), text))
    row = '  %s%s%s %s ' % (paint('%-12s' % ('' if options else label), 'bold'), paint('%-30s' % hint, 'dim'),
                            paint('%6s' % ('[%s]' % default), 'cyan'), paint(glyph('›', '>'), 'cyan'))
    while True:
        try:
            text = input(row).strip().lower()
        except EOFError:
            text = ''
        value = parse(text or str(default))
        if value is not None:
            return value
        print('  %s%s' % (' ' * 12, paint('not valid, try again', 'red')))


def yes_no(text):
    return {'y': True, 'yes': True, 'n': False, 'no': False}.get(text)


def speed_note(images, context):
    """What the chosen pair runs at, and why a smaller context alone is not faster."""
    note = 'about %s tok/s' % LAYOUTS[layout(images, context)]['SPEED']
    if images:
        return note + ' · smaller contexts work but are not faster'
    return note + ('' if context <= 131072 else ' · text only up to 131K runs at %s tok/s' % LAYOUTS['fast']['SPEED'])


def settings_summary():
    return ' · '.join((describe(*choice()), 'effort %s' % SETTINGS['effort'], 'idle stop %g min' % IDLE_MIN, 'hard stop %g h' % MAX_HOURS,
                       'uncensored' if SETTINGS['uncensor'] else 'original model'))


def choose_settings(force=False):
    """On a terminal: the first start asks for each default, later starts show them and Enter keeps them; agents and pipes skip this."""
    global IDLE_MIN, MAX_HOURS
    if not (sys.stdin and sys.stdin.isatty()) or os.environ.get('LLM_YES') == '1':
        return
    if os.path.exists(SETTINGS_FILE) and not force:
        print('%s  %s' % (paint('Settings', 'bold'), settings_summary()))
        try:
            answer = input('%s%s %s ' % (' ' * 10, paint('Enter starts, c changes them', 'dim'), paint(glyph('›', '>'), 'cyan')))
        except EOFError:
            answer = ''
        print()
        if answer.strip().lower() not in ('c', 'change'):
            return
    print('%s  %s\n' % (paint('Settings', 'bold'), paint('Enter keeps the value in brackets', 'dim')))
    old, new = SETTINGS, dict(SETTINGS)
    new['images'] = pick('Images', 'image input, y or n', 'y' if old['images'] else 'n', yes_no)
    new['context'] = pick('Context', '32K, 64K, 131K or 262K', short_context(context_value(old['context']) or 262144), context_value)
    print('  %s%s' % (' ' * 12, paint(speed_note(new['images'], new['context']), 'dim')))
    new['effort'] = pick('Effort', 'ds41: 1-100, low, high, max', old['effort'], effort_value)
    new['idle_min'] = pick('Idle stop', 'minutes, 1-120', old['idle_min'], lambda t: number(t, 1, 120))
    new['max_hours'] = pick('Hard stop', 'hours, 1-24', old['max_hours'], lambda t: number(t, 1, 24))
    new['uncensor'] = pick('Model', 'y uncensored, n original', 'y' if old['uncensor'] else 'n', yes_no)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(new, f, indent=2)
    SETTINGS.update(new)
    IDLE_MIN = float(os.environ.get('LLM_IDLE_MIN') or new['idle_min'])
    MAX_HOURS = float(os.environ.get('LLM_MAX_HOURS') or new['max_hours'])
    print('\n  %s Saved%s\n' % (paint(glyph('✓', '+'), 'green'), paint(' · change them later: %s settings' % cli(), 'dim')))


def watch(port):
    """Live view until Ctrl+C: one status line with the idle countdown, plus a line for each start and stop."""
    if not TTY[0]:
        return
    print(paint('\n  Ctrl+C closes this view; the endpoint keeps running.\n', 'dim'))
    live, prev, dots = Live(), None, (glyph('●', '*'), glyph('◌', '~'), glyph('○', 'o'))
    try:
        while True:
            try:
                conn, r = control_call(port, '/_llm/status', None, 5)
                st = json.loads(r.read())
                conn.close()
            except (OSError, ValueError, http.client.HTTPException):
                live.line('  %s  the endpoint has stopped' % paint(time.strftime('%H:%M'), 'dim'))
                return
            state = 'ready' if st['state'].startswith('ready') else st['state']
            if prev and state != prev:
                live.line('  %s  %s' % (paint(time.strftime('%H:%M'), 'dim'), watch_event(prev, state, st)))
            prev = state
            spent = 'ran %s · %s' % (mmss(st['ran']), units(st['cu'])) if st.get('ran') is not None else ''
            if state == 'ready':
                left = min(st['idle_stop_min'] * 60 - st['idle_s'], st['max_hours'] * 3600 - (st['ran'] or 0))
                doing = ('answering %d request%s' % (st['active'], '' if st['active'] == 1 else 's') if st['active']
                         else 'idle %s · stops in %s' % (mmss(st['idle_s']), mmss(left)))
                live.put([('  ', ()), (dots[0] + ' GPU running', ('green', 'bold')), ('   ' + doing + '   ', ()), (spent, ('dim',))])
            elif state in ('starting', 'stopping'):
                what = 'starting · ' + st['stage'] if state == 'starting' else 'stopping'
                live.put([('  ', ()), (dots[1] + ' GPU ' + what, ('cyan',)), ('   ' + spent, ('dim',))])
            else:
                live.put([('  ', ()), (dots[2] + ' GPU off', ('bold',)), ('   starts on the next request, about 4 minutes', ('dim',))])
            time.sleep(1)
    except KeyboardInterrupt:
        live.clear()
        print('\n  Closed the live view; the endpoint keeps running.\n  %s %s watch   %s %s down'
              % (paint('Live view', 'dim'), paint(cli(), 'cyan'), paint('Stop', 'dim'), paint(cli(), 'cyan')))


def watch_event(prev, state, st):
    stop = st.get('last_stop') or {}
    if state in ('stopping', 'off') and prev in ('ready', 'starting'):
        if st.get('last_error') and prev == 'starting':
            return paint('start failed: ' + st['last_error'], 'red')
        cost = ': ran %s, %s' % (mmss(stop['ran']), units(stop['cu'])) if stop.get('ran') is not None else ''
        why = 'after %s without requests' % minutes(st['idle_stop_min']) if stop.get('reason', '').startswith('idle') else stop.get('reason', '')
        return 'GPU stopped %s%s' % (why, cost)
    if state == 'starting':
        return 'GPU starting for a new request'
    if state == 'ready':
        return paint('GPU ready', 'green')
    return 'GPU ' + state


def install_colab():
    """Installs the pinned Colab CLI with uv (installing uv first if needed) or pipx; returns True when it is installed."""
    home_bin = os.path.join(os.path.expanduser('~'), '.local', 'bin')
    uv = shutil.which('uv') or next((p for p in (os.path.join(home_bin, 'uv.exe' if IS_WIN else 'uv'),) if os.path.exists(p)), None)
    pipx = shutil.which('pipx')
    tool = 'uv' if uv or not pipx else 'pipx'
    print("Google's Colab CLI is not installed yet. llm.py uses it to start and stop the GPU.", flush=True)
    if not confirm('Install %s with %s%s?' % (COLAB_CLI, tool, ' (installs uv first)' if tool == 'uv' and not uv else '')):
        return False
    print('Installing...', end=' ', flush=True)
    if tool == 'uv' and not uv:
        rc, out = run(['powershell', '-ExecutionPolicy', 'ByPass', '-c', 'irm https://astral.sh/uv/install.ps1 | iex'] if IS_WIN
                      else ['sh', '-c', 'curl -LsSf https://astral.sh/uv/install.sh | sh'], 900)
        if rc != 0:
            print('failed:\n' + out[-2000:], flush=True)
            return False
        uv = shutil.which('uv') or os.path.join(home_bin, 'uv.exe' if IS_WIN else 'uv')
    rc, out = run([uv, 'tool', 'install', COLAB_CLI] if tool == 'uv' else [pipx, 'install', COLAB_CLI], 900)
    print('done.' if rc == 0 else 'failed:\n' + out[-2000:], flush=True)
    return rc == 0


def colab_present():
    return bool(os.environ.get('LLM_COLAB')) or os.path.exists(COLAB[0]) or bool(shutil.which(COLAB[0]))


def colab_login():
    """Runs the CLI's one-time Google login in this terminal and hides the balance lines it prints afterwards."""
    hide = (b'Current balance', b'Usage rate', b'Active assignments')
    p = subprocess.Popen(COLAB + ['usage'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(CHILD_ENV, PYTHONUNBUFFERED='1'))
    buf = b''
    while True:
        chunk = os.read(p.stdout.fileno(), 4096)
        if not chunk:
            break
        *lines, buf = (buf + chunk).split(b'\n')
        keep = [l + b'\n' for l in lines if not l.lstrip().startswith(hide)]
        # a line without newline is the code prompt, unless it may still turn into a hidden line
        if buf and not any(h.startswith(buf.lstrip()) for h in hide):
            keep, buf = keep + [buf], b''
        sys.stdout.flush()
        sys.stdout.buffer.write(b''.join(keep))
        sys.stdout.buffer.flush()
    p.wait()


def setup():
    """First-run setup, idempotent: tools, Colab CLI, Google login, API key and ssh key; returns the Colab balance."""
    global COLAB
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    missing = [t for t in ('ssh', 'ssh-keygen') if not (shutil.which(t) or os.path.exists(find_tool(t)))]
    if missing:
        sys.exit('Missing %s. Install an OpenSSH client: Ubuntu/Debian "sudo apt install openssh-client", Fedora "sudo dnf install '
                 'openssh-clients", Windows "Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"' % ' and '.join(missing))
    if not colab_present():
        if not install_colab():
            sys.exit('The Colab CLI is needed. Install it with "uv tool install %s" (or pipx), then run this again.' % COLAB_CLI)
        COLAB = colab_cmd()
        if not colab_present():
            sys.exit('Installed, but "colab" is not on PATH yet. Open a new terminal and run this again.')
    bal, rate = usage()
    if bal is None:
        if not (sys.stdin and sys.stdin.isatty()):
            sys.exit('The Colab CLI is installed but not logged in yet. Run this in a terminal: %s setup' % cli())
        print('\n%s: open the link below, sign in with the Google account of your Colab plan and paste the code here.'
              % paint('Google login (only once)', 'bold'), flush=True)
        colab_login()
        bal, rate = usage()
        if bal is None:
            sys.exit('The login did not work. Try again: %s setup' % cli())
        print('%s Colab balance: %.1f compute units.\n' % (paint('Logged in.', 'bold', 'green'), bal), flush=True)
    if not os.path.exists(KEY_FILE):
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_BINARY', 0), 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(('sk-ds41-' + secrets.token_hex(24) + '\n').encode())
    if not os.path.exists(SSH_KEY):
        rc, out = run([find_tool('ssh-keygen'), '-q', '-t', 'ed25519', '-N', '', '-C', 'llm-ds41', '-f', SSH_KEY], 30)
        if rc:
            sys.exit('ssh-keygen failed: ' + out)
    write_opencode()
    return bal


def write_opencode(port=PORT):
    info = profile()
    image = {'attachment': True, 'modalities': {'input': ['text', 'image'], 'output': ['text']}} if info['MMPROJ'] == '1' else {}
    models = {m: dict({'name': 'DeepSeek-V4.1-Flash uncensored, reasoning effort %s' % e, 'tool_call': True, 'reasoning': True,
                       'interleaved': {'field': 'reasoning_content'}, 'options': {'reasoningEffort': e},
                       'limit': {'context': int(info['CTX']) // 2, 'output': 32768}}, **image) for m, e in model_efforts()}
    cfg = {'$schema': 'https://opencode.ai/config.json',
           'provider': {'ds41': {'npm': '@ai-sdk/openai-compatible', 'name': 'DeepSeek-V4.1-Flash on Colab',
                                 'options': {'baseURL': 'http://%s:%d/v1' % (HOST, port), 'apiKey': '{file:%s}' % KEY_FILE.replace('\\', '/'),
                                             'headerTimeout': 1800000, 'chunkTimeout': 900000},
                                 'models': models}},
           'model': 'ds41/ds41', 'small_model': 'ds41/ds41'}
    with open(os.path.join(STATE, 'opencode.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def endpoint_info(port, bal=None):
    info, me_cmd = profile(), cli()
    rows = (('Base URL', 'http://%s:%d/v1' % (HOST, port)), ('API key', api_key()),
            ('Models', 'ds41 (effort %s) · ds41-max (effort max)' % SETTINGS['effort']),
            ('Context', '%s tokens per request · image input %s · about %s tok/s' % (
                '{:,}'.format(int(info['CTX']) // 2), 'on' if info['MMPROJ'] == '1' else 'off', LAYOUTS[layout(*choice())]['SPEED'])))
    print()
    for name, value in rows:
        print('  %-10s %s' % (name, paint(value, 'bold')))
    print()
    for name, value in (('Try it', me_cmd + ' ask "Hello!"'), ('opencode', me_cmd + ' opencode [project folder]'), ('Stop', me_cmd + ' down')):
        print('  %-10s %s' % (name, paint(value, 'cyan')))
    print('\n' + paint('  About %.0f compute units per hour%s · stops after %s without requests'
                       % (G4_CU_PER_HOUR, ' · balance %.1f' % bal if bal is not None else '', minutes(IDLE_MIN)), 'dim'))


def stopped_note(kind, rest):
    if kind != 'STOPPED':
        sys.exit('The GPU may still be running. Run this again: %s down' % cli())
    cu, ran = (rest.split() + ['None', 'None'])[:2]
    done = paint(glyph('✓', '+') + ' Stopped', 'bold', 'green')
    print(done + (' · the GPU ran %s · %s' % (mmss(float(ran)), units(float(cu))) if cu != 'None'
                  else ' · nothing is running or billed now'))


def cmd_setup():
    bal = setup()
    print('Setup done. Colab balance: %.1f compute units, about %.0f GPU hours.' % (bal, bal / G4_CU_PER_HOUR))
    print('Next: %s up' % cli())


def cmd_up():
    print(paint('DeepSeek V4.1 Flash on Colab', 'bold') + '\n')
    bal = setup()
    held = serve_running()
    if held and held.get('choice') != list(choice()):
        sys.exit('The endpoint runs with %s. To change that, stop it first: %s down'
                 % (describe(*held['choice']) if held.get('choice') else 'other settings', cli()))
    if bal < MIN_BALANCE and not (held and load_session()):
        sys.exit('Colab balance %.2f compute units is below LLM_MIN_BALANCE %.0f' % (bal, MIN_BALANCE))
    if not held:
        choose_settings()
        with open(os.path.join(STATE, 'serve.out'), 'wb') as out:
            p = spawn_detached([sys.executable, os.path.realpath(__file__), 'serve', '--detached'], out)
        t0 = time.time()
        while not held:
            if p.poll() is not None or time.time() - t0 > 30:
                sys.exit('The local endpoint did not start: %s' % (_read(os.path.join(STATE, 'serve.out')) or b'').decode('utf-8', 'replace'))
            time.sleep(0.3)
            held = serve_running()
    conn, r = control_call(held['port'], '/_llm/up', {}, 120)
    kind, rest = follow(r, paint('Starting a Colab G4 GPU', 'bold') + paint(' · about 4 minutes', 'dim'))
    conn.close()
    if kind != 'READY':
        print(paint('Start failed: ' + (rest or 'the local endpoint closed the connection'), 'red'))
        sys.exit('Details: %s logs' % cli())
    secs = int(rest or 0)
    print(paint('%s The GPU is already running' % glyph('●', '*') if secs < 2 else '\n%s Ready after %s' % (glyph('✓', '+'), mmss(secs)),
                'bold', 'green'))
    endpoint_info(held['port'], bal)
    watch(held['port'])


def cmd_down(everything=False):
    held = serve_running()
    if held:
        leave = everything or bool(held.get('detached'))
        try:
            conn, r = control_call(held['port'], '/_llm/down', {'exit': leave}, 120)
            kind, rest = follow(r)
            conn.close()
            t0 = time.time()
            while leave and alive(held) and time.time() - t0 < 30:
                time.sleep(0.3)
            if leave and alive(held):
                kill_tree(int(held['pid']), hard=True)
            return stopped_note(kind, rest)
        except (OSError, http.client.HTTPException) as e:
            log('endpoint pid %s did not answer (%s); killing it and stopping directly' % (held['pid'], e.__class__.__name__))
            kill_tree(int(held['pid']), hard=True)
            time.sleep(1)
    held = take_lock({'port': 0})
    if held:
        sys.exit('Another llm.py command holds the lock (pid %s).' % held.get('pid'))
    try:
        still = stop_cloud(load_session(), 'down')
    finally:
        release_lock()
    cu, ran = _cache.pop('stopped', (None, None))
    stopped_note('STILL' if still else 'STOPPED', '%s %s' % (cu, ran))


def cmd_status():
    held, rows = serve_running(), []
    if held:
        try:
            conn, r = control_call(held['port'], '/_llm/status', None, 10)
            st = json.loads(r.read())
            conn.close()
            gpu = {'off': 'off, starts on the next request', 'starting': 'starting: ' + st['stage']}.get(st['state'], st['state'])
            if st['state'].startswith('ready'):
                gpu = '%s, idle %s of %s, %d request%s served, ran %s, %s' % (
                    st['state'], mmss(st['idle_s']), minutes(st['idle_stop_min']), st['served'], '' if st['served'] == 1 else 's',
                    mmss(st['ran'] or 0), units(st['cu'] or 0))
            rows += [('Endpoint', 'http://%s:%d/v1 · %s' % (HOST, held['port'], describe(*held['choice']) if held.get('choice') else '')),
                     ('GPU', gpu)]
            if st.get('last_error'):
                rows.append(('Last error', st['last_error']))
        except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
            rows.append(('Endpoint', 'pid %s not answering (%s)' % (held['pid'], e.__class__.__name__)))
    else:
        rows.append(('Endpoint', 'not running (start it: %s up)' % cli()))
    bal, rate = usage()
    rows.append(('Colab', 'session %s %s, balance %s compute units' % (NAME, 'running' if session_alive() else 'not running',
                                                                        '%.1f' % bal if bal is not None else 'unknown')))
    if held and os.path.exists(KEY_FILE):
        rows.append(('API key', api_key()))
    for name, value in rows:
        print('  %-10s %s' % (name, value))


def cmd_logs():
    raw = _read(LOG)
    print(redact(raw.decode('utf-8', 'replace')[-3000:]) if raw is not None else '(no local log)')
    if session_alive():
        rc, out = vmx('cat /content/res/timeline.txt; tail -n 25 /content/res/server.log', 40)
        print(redact(out))


def cmd_settings():
    if serve_running():
        sys.exit('The endpoint is running with: %s\nStop it first to change them: %s down' % (settings_summary(), cli()))
    if not (sys.stdin and sys.stdin.isatty()):
        sys.exit('Settings: %s\nChange them in a terminal, or edit %s' % (settings_summary(), SETTINGS_FILE))
    choose_settings(force=True)


def cmd_watch():
    held = serve_running()
    if not held:
        sys.exit('The endpoint is not running. Start it first: %s up' % cli())
    if not TTY[0]:
        sys.exit('watch needs a terminal; use "%s status" for a snapshot' % cli())
    watch(held['port'])


def cmd_ask(words):
    held = serve_running()
    if not held:
        sys.exit('The endpoint is not running. Start it first: %s up' % cli())
    model = 'ds41-max' if '--max' in words else 'ds41'
    body = {'model': model, 'stream': True, 'messages': [{'role': 'user', 'content': ' '.join(w for w in words if w != '--max') or 'Hello!'}]}
    conn, r = http_call(held['port'], 'POST', '/v1/chat/completions', json.dumps(body).encode(),
                        {'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'}, UPSTREAM_TIMEOUT, HOST)
    if r.status != 200:
        sys.exit('HTTP %d: %s' % (r.status, r.read()[:500].decode('utf-8', 'replace')))
    live, part, timings = Live(), None, None
    for raw in r:
        line = raw.decode('utf-8', 'replace').strip()
        wait = re.match(r': starting gpu vm \((\d+) s\)', line)
        if wait:
            live.show(mmss(wait.group(1)), 'starting the GPU, about 4 minutes')
        if not line.startswith('data: {'):
            continue
        event = json.loads(line[6:])
        if event.get('error'):
            live.clear()
            sys.exit('\nerror: %s' % event['error'].get('message', event['error']))
        timings = event.get('timings') or timings
        for choice in event.get('choices', []):
            delta = choice.get('delta') or {}
            # the reasoning is shown dimmed on a terminal and left out on a pipe
            for kind, text in (('think', delta.get('reasoning_content')), ('say', delta.get('content'))):
                if not text or (kind == 'think' and not TTY[0]):
                    continue
                live.clear()
                if part not in (None, kind):
                    sys.stdout.write('\n\n')
                part = kind
                sys.stdout.write(paint(text, 'dim') if kind == 'think' else text)
                sys.stdout.flush()
    conn.close()
    print()
    if timings and timings.get('predicted_n'):
        print(paint('%s: %d tokens at %.1f tokens/s' % (model, timings['predicted_n'], timings.get('predicted_per_second', 0)), 'dim'))


def find_opencode():
    for p in (shutil.which('opencode'), os.path.join(os.path.expanduser('~'), '.opencode', 'bin', 'opencode.exe' if IS_WIN else 'opencode')):
        if p and os.path.exists(p):
            return p
    return None


def cmd_opencode(args):
    exe = find_opencode()
    if not exe:
        sys.exit('opencode is not installed. Get it from https://opencode.ai, then run this again.')
    held = serve_running()
    if not held:
        sys.exit('The endpoint is not running. Start it first: %s up' % cli())
    if held.get('choice'):
        os.environ.pop('LLM_PROFILE', None)
        os.environ['LLM_IMAGES'], os.environ['LLM_CONTEXT'] = '1' if held['choice'][0] else '0', str(held['choice'][1])
    write_opencode(held['port'])
    sys.exit(subprocess.call([exe] + args, env=dict(os.environ, OPENCODE_CONFIG=os.path.join(STATE, 'opencode.json'))))


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else 'help'
    if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if '--fast' in args and cmd not in ('ask', 'opencode'):
        os.environ['LLM_PROFILE'] = CHILD_ENV['LLM_PROFILE'] = 'fast'
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    init_console()
    commands = {'up': cmd_up, 'down': lambda: cmd_down('--all' in args), 'status': cmd_status, 'logs': cmd_logs, 'setup': cmd_setup,
                'settings': cmd_settings, 'watch': cmd_watch,
                'ask': lambda: cmd_ask(args[1:]), 'opencode': lambda: cmd_opencode(args[1:])}
    try:
        if cmd == 'serve':
            serve('--detached' in args)
        elif cmd == '_watchdog' and len(args) == 4:
            ECHO[0] = False
            watchdog(args[1], float(args[2]), float(args[3]))
        elif cmd in commands:
            commands[cmd]()
        else:
            sys.exit(USAGE)
    except KeyboardInterrupt:
        print()
        if cmd == 'up' and serve_running():
            print('Interrupted. The GPU may keep starting in the background. Stop it with: %s down' % cli())
        sys.exit(130)


if __name__ == '__main__':
    main()
