# Setup and reference

Everything beyond the quick start: each platform, the commands, connecting apps, settings, costs and troubleshooting.

## What you need

- **Google AI Pro or Colab Pro.** Google AI Pro includes 200 Colab compute units every month, Colab Pro 100. The script uses Colab's G4 GPU (96 GB of VRAM, 190 GB of RAM) at about 9 units per hour, so that is about 22 or 11 GPU hours a month. Pay-as-you-go units work too.
- **Python 3.9 or newer.** `llm.py` uses only the standard library.
- **An OpenSSH client** (`ssh`, `ssh-keygen`). It's built into macOS, Windows 10/11 and most Linux desktops.
- **Google's Colab CLI** (`google-colab-cli` 0.7.5). `llm.py` offers to install it on the first run, using [uv](https://docs.astral.sh/uv/) or pipx.

Nothing heavy runs on your machine. The model, the downloads and the build all happen on the Colab VM.

## Install

### Linux
```bash
sudo apt install -y python3 git openssh-client curl    # Debian/Ubuntu; Fedora: sudo dnf install python3 git openssh-clients curl
git clone https://github.com/leonxger/deepseek-colab.git deepseek-colab && cd deepseek-colab
python3 llm.py up
```

### macOS
Python 3 comes from [python.org](https://www.python.org/downloads/) or `brew install python`. git comes with the Xcode command line tools (`xcode-select --install`).
```bash
git clone https://github.com/leonxger/deepseek-colab.git deepseek-colab && cd deepseek-colab
python3 llm.py up
```

### Windows
Python comes from [python.org](https://www.python.org/downloads/) (or Anaconda), git from [git-scm.com](https://git-scm.com/download/win). The OpenSSH client is built into Windows 10 and 11. In PowerShell or cmd:
```powershell
git clone https://github.com/leonxger/deepseek-colab.git deepseek-colab
cd deepseek-colab
python llm.py up
```
Use `python3` or `py -3` instead of `python` if that is how your Python is called. Google's Colab CLI officially supports only Linux and macOS, but setup, login and start were tested on Windows in PowerShell. WSL2 is an alternative (not tested): run the Linux commands inside Ubuntu, and Windows apps reach `http://127.0.0.1:18080/v1` through WSL2's localhost forwarding.

### First run
`python3 llm.py up` does this, once:

1. Checks for `ssh` and `ssh-keygen`.
2. Offers to install the Colab CLI (`uv tool install google-colab-cli==0.7.5`, installing uv first if needed). Answer `y`, or set `LLM_YES=1` to skip the question.
3. Starts the Google login. Open the printed link, sign in with the account that holds your Colab plan, and paste the code back into the terminal. The token is cached in `~/.config/colab-cli/`.
4. Creates `state/api_key` (your endpoint's key), `state/id_ed25519` (the key for the SSH tunnel to the VM) and `state/opencode.json`.
5. Asks for your defaults (see [Settings](#settings)). Enter keeps the suggested value.

From then on, `up` shows your settings in one line: Enter starts, `c` changes them. `python3 llm.py setup` does steps 1-4 without starting anything.

## Commands

| Command | What it does |
| --- | --- |
| `python3 llm.py up` | Set up if needed, start the GPU, wait until it serves, print base URL, API key and models, then show the live status (Ctrl+C closes the view). The endpoint keeps running in the background. |
| `python3 llm.py up --fast` | Same, for this run text only with 131K context: the faster layout, about 12% faster decode. |
| `python3 llm.py down` | Stop the GPU and the background endpoint. |
| `python3 llm.py status` | Endpoint state, Colab balance and session, base URL and key. |
| `python3 llm.py watch` | The live status again: GPU running or off, countdown to the automatic stop, cost so far. |
| `python3 llm.py settings` | Change the saved defaults. |
| `python3 llm.py ask "question"` | Send one question and stream the answer. `--max` uses `ds41-max`. |
| `python3 llm.py opencode [folder]` | Start opencode connected to the endpoint, in any shell. |
| `python3 llm.py logs` | Local log plus the VM's boot timeline and the server log. |
| `python3 llm.py serve` | Run the endpoint in the foreground. The GPU starts on the first request; Ctrl+C stops it. |
| `python3 llm.py setup` | Install and log in only. |

To switch image input or context, run `down`, then `up` (or `up --fast`).

## What happens on `up`

About 4 minutes from the command to the first answer:

1. Colab allocates a G4 VM (a few seconds when one is free).
2. `llm.py` uploads `vm/` and your API key, and starts `vm/boot.sh` there.
3. In parallel, the VM downloads the model from Hugging Face (138 GB of weights, 8 GB draft model, 1 GB image encoder, at about 3.5 GB/s), fetches AtomicChat's 82 uncensored tensors, and builds llama.cpp with the patches in `vm/patches/`.
4. The server loads. Weights already copied to the GPU are freed from RAM, the cold experts are pinned for fast copies, and the 110.6 GB Engram table streams in.
5. Only after a warmup answer does the VM open its port, so no request ever reaches a half-loaded model.

Requests sent while the GPU starts are held, not refused. Streaming requests get a `: starting gpu vm (N s)` comment every 5 seconds so clients don't time out.

## Use the endpoint

Base URL `http://127.0.0.1:18080/v1`, API key from `state/api_key` (also shown by `up` and `status`).

| Model name | Reasoning effort | Use it for |
| --- | --- | --- |
| `ds41` | 50 | everyday work, faster answers |
| `ds41-max` | max | hard problems; it thinks much longer |

A `reasoning_effort` field in the request overrides both. The reasoning arrives separately in `reasoning_content`.

**curl**
```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Authorization: Bearer $(cat state/api_key)" -H "Content-Type: application/json" \
  -d '{"model": "ds41", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Python (`pip install openai`)**
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key=open("state/api_key").read().strip())
reply = client.chat.completions.create(model="ds41", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

**Images** (when image input is on): send them the OpenAI way, as `image_url` parts with a `data:image/png;base64,...` URL.

**opencode:** `python3 llm.py opencode [project folder]` starts opencode with the generated config: both models, tool calling, the right context limit, images when image input is on. It works the same in bash, PowerShell and cmd. To use the endpoint from your own opencode config instead, copy the `provider` block from `state/opencode.json` into it.

**Other apps** (Open WebUI, Continue, LibreChat, Chatbox, ...): add an "OpenAI-compatible" provider with the base URL, the key and the model name `ds41` or `ds41-max`. Listing models (`GET /v1/models`) is answered locally and never starts the GPU; only a real request does.

**Other devices.** By default the endpoint listens on 127.0.0.1 only. The safe way to reach it from another machine is an SSH tunnel to the computer running `llm.py` (`ssh -L 18080:127.0.0.1:18080 you@that-computer`) or a VPN such as Tailscale. `LLM_BIND=0.0.0.0` opens it on your network directly. The traffic is then plain HTTP, so anyone on that network can read the key.

## Settings

The first `up` asks for these and saves them in `state/settings.json`; `python3 llm.py settings` changes them. They apply when the endpoint starts, so run `down` first.

| Setting | Default | Choices |
| --- | --- | --- |
| Images | on | image input on or off |
| Context | 262K | tokens per request: 32K, 64K, 131K, 262K or any number from 4,096 to 262,144. Text only up to 131K runs on the faster measured memory layout (118.7 tok/s); everything else on the 262K-with-images layout (105.6 tok/s). A smaller context works but is not faster, because only these two layouts are measured. |
| Reasoning effort of `ds41` | 50 | 1-100, `low` (50), `high` (75) or `max`. `ds41-max` always uses max. A request's own `reasoning_effort` wins. |
| Idle stop | 10 minutes | 1-120 minutes without requests, then the GPU stops |
| Hard stop | 6 hours | 1-24 hours, then the GPU stops whatever happens |
| Uncensored | yes | no = ISTA-DASLab's original tensors |

Without a terminal (scripts, agents) nothing is asked and the saved values or defaults apply. Environment variables override the saved values for one run, for example `LLM_IDLE_MIN=30 python3 llm.py up`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_PORT` | 18080 | local port of the endpoint |
| `LLM_IDLE_MIN` | saved idle stop | minutes without requests before the GPU stops |
| `LLM_MAX_HOURS` | saved hard stop | hard stop after this many hours |
| `LLM_MIN_BALANCE` | 5 | refuse to start below this many compute units |
| `LLM_IMAGES` | saved | `1` image input on, `0` off |
| `LLM_CONTEXT` | saved | tokens per request, 4096-262144, or 32K, 64K, 131K, 262K |
| `LLM_PROFILE` | unset | presets: `default` (images, 262K) or `fast` (text only, 131K, same as `--fast`) |
| `LLM_BIND` | 127.0.0.1 | listen address (see "Other devices") |
| `HF_TOKEN` | none | your Hugging Face token, if anonymous downloads get rate-limited; sent only to the VM, deleted at stop |
| `UNCENSOR` | saved | `0` serves ISTA-DASLab's original (censored) tensors |
| `DRAFT` | 1 | `0` turns off the draft model (slower) |
| `LLM_YES` | unset | `1` answers yes to the install question and skips the settings questions |
| `LLM_SESSION` | llm-ds41 | Colab session name |

## Costs and safety nets

- The G4 bills about 9 compute units per hour from allocation to stop. The live view and `status` show the running time and cost so far, and every stop prints what the session cost (from the running time, because Colab updates the balance a few minutes late).
- **Idle stop:** after `LLM_IDLE_MIN` minutes without requests the GPU stops. The endpoint stays up and starts the GPU again on the next request (about 4 minutes).
- **Hard stop:** after `LLM_MAX_HOURS` hours.
- **Watchdog:** a small background process stops the VM if the endpoint crashes, and also enforces the hard stop.
- **Before you shut down, log off or put the computer to sleep, run `python3 llm.py down`.** The watchdog lives on your computer: when that sleeps, the VM keeps billing until Colab reclaims it.
- Use one computer per Google account at a time. The Colab CLI keeps its session list per machine, so a VM started elsewhere is invisible here.

## Troubleshooting

| You see | Why | Fix |
| --- | --- | --- |
| `no G4 GPU from Colab` | no G4 free right now, or the plan has no compute units | wait a few minutes and retry; check the balance with `status` |
| `the Colab CLI cannot read your balance` | Google login expired | `python3 llm.py setup` |
| `boot failed on VM: ... FAILED` | a download or the build failed on the VM | `python3 llm.py logs` shows the step; `up` again usually works |
| `GUARD: ... RAM available` | the VM ran short of RAM while loading | `up` again; if it repeats, use `--fast` |
| `cannot serve on 127.0.0.1:18080` | another program uses the port | `LLM_PORT=18081 python3 llm.py up` |
| `401 Invalid API Key` | wrong or missing key | use the key from `state/api_key` |
| `403 unexpected Host header` | the app used a host name other than 127.0.0.1 or localhost | use `http://127.0.0.1:18080/v1` |
| first request waits ~4 minutes | the GPU is starting | normal; later requests are fast |
| `The endpoint runs with ...` | changing image input or context while running | `python3 llm.py down`, then `up` |
| `The endpoint is not running` (from `ask` or `opencode`) | nothing started yet, or it was stopped | `python3 llm.py up` |

The full local log is `state/llm.log`. Keys and tokens are redacted there.

## Files

```text
llm.py               the whole client: setup, endpoint, VM lifecycle
vm/boot.sh           cold start on the VM: downloads, build, server, memory layout
vm/serve.sh          llama-server flags
vm/build.sh          builds llama.cpp b11963 with vm/patches/series
vm/patches/          PR #30966 plus 21 patches, applied in series order
vm/*.py, vm/pin.c    downloader, uncensor splice, RAM punching, page pinning, port gate
docs/                how it works, graphics, the video
tests/               offline tests with a fake Colab CLI and a mock server
state/               your API key, SSH key, logs (created on first run, never committed)
```

Tests (Linux, offline, about 80 s): `python3 -B -m unittest -v tests/test_llm.py`

## Uninstall

`python3 llm.py down`, then delete the folder. To remove the Colab CLI too: `uv tool uninstall google-colab-cli` and delete `~/.config/colab-cli/`.
