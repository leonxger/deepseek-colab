# AGENT_START_HERE

You are an AI agent helping someone use this repo. Read this file fully; it is enough to set the project up, run it, connect apps to it and fix the usual problems. Deeper references: `SETUP.md` (user guide), `docs/how-it-works.md` (internals).

## What this is

`llm.py` turns a Google Colab G4 GPU into a private OpenAI-compatible endpoint on the user's computer, serving uncensored DeepSeek V4.1 Flash (2-bit GGUF, about 249 GB including its Engram table). The user needs Google AI Pro (200 Colab compute units a month) or Colab Pro (100 a month), Python 3.9+ and an OpenSSH client. The GPU costs about 9 compute units per hour. Nothing heavy runs locally.

```text
app --HTTP--> llm.py on 127.0.0.1:18080 --SSH tunnel (colab ssh --proxy-mode)--> G4 VM: gate.py :18080 --> llama-server :18081
```

- `llm.py` (stdlib only) is the whole client. It installs Google's Colab CLI if asked, logs in, creates keys, allocates the VM with `colab new --gpu G4`, uploads `vm/`, runs `vm/boot.sh`, holds the tunnel, proxies requests, and stops the VM when idle.
- `vm/boot.sh` downloads the weights from Hugging Face into a RAM disk, splices in the uncensored tensors (`vm/mkpatch.py`, `vm/applypatch.py`), builds llama.cpp b11963 with `vm/patches/series` (`vm/build.sh`), starts the server (`vm/serve.sh`), frees RAM that the GPU already holds (`vm/punch.py`), pins the cold experts (`vm/pin.c`), fills the Engram table, and finally opens port 18080 (`vm/gate.py`).
- Image input and context per request are separate settings, mapped to two measured GPU memory layouts in `LAYOUTS` in `llm.py`: text only up to 131,072 tokens runs at 118.7 tok/s; image input on, or more context, runs at 105.6 tok/s (up to 262,144 tokens per request). `--fast` is the text-only 131K preset.

## Commands

| Command | Effect | Costs money? |
| --- | --- | --- |
| `python3 llm.py setup` | installs the CLI (asks), Google login, keys, `state/opencode.json` | no |
| `python3 llm.py up [--fast]` | setup if needed, starts the GPU, prints base URL, API key, models; in a human terminal it then stays in a live view until Ctrl+C | **yes, about 9 compute units per hour until stopped** |
| `python3 llm.py status` | endpoint state, Colab balance, session, base URL and key | no |
| `python3 llm.py watch` | live view for a human terminal (countdown to the auto-stop, cost so far) | no |
| `python3 llm.py settings` | asks for the saved defaults (image input, context, effort, idle stop, hard stop, uncensored) | no |
| `python3 llm.py ask "question"` | one streamed answer from the running endpoint (`--max` for effort max) | only while the GPU runs |
| `python3 llm.py opencode [folder]` | starts opencode with the endpoint configured, any shell | only while the GPU runs |
| `python3 llm.py logs` | local log plus the VM boot timeline and server log tail | no |
| `python3 llm.py down` | stops GPU, tunnel, watchdog and background endpoint | no, it saves |
| `python3 llm.py serve` | foreground endpoint; the GPU starts on the first request | when a request arrives |

## Rules for agents

1. **Ask before `up` or `serve`.** A running G4 spends the user's compute units, about 9 per hour. Say so in one line and wait for a yes.
2. **Leave nothing running by accident.** When the user is done, or before they shut down or sleep the computer, run `python3 llm.py down` and confirm with `status` that the session shows "not running".
3. **The Google login is interactive.** `setup` prints a URL and waits for a pasted code. Ask the user to run it in their own terminal; you cannot complete it for them.
4. **Treat `state/api_key` as a password.** Show it to the user when they need it for an app; never paste it into public places, issues or commits. `state/` is git-ignored.
5. **Never start `colab ssh --proxy-mode` yourself for a session that `colab sessions` does not list.** It silently creates a new paid VM. `llm.py` guards against this; don't bypass it.
6. **Don't reorder or edit `vm/patches/`.** Later patches anchor on code that earlier ones insert, so `series` order matters, and the build commit is pinned.
7. **Windows works natively** in PowerShell or cmd (`python llm.py up`); setup, login and start were tested there, although Google's CLI officially supports only Linux and macOS. WSL2 is the fallback.
8. **Settings belong to the user.** Without a terminal, `up` asks nothing and uses `state/settings.json` or the defaults, and returns right after printing the endpoint. To change them, have the user run `python3 llm.py settings`, or edit `state/settings.json` with their OK: `images` (true or false), `context` (4096-262144 tokens), `effort` (1-100, `low`, `high`, `max`), `idle_min` (1-120), `max_hours` (1-24), `uncensor` (true or false).

## Helping a user from zero

1. Check the prerequisites: `python3 --version` (3.9+; on Windows `python --version`), `ssh -V`.
2. Ask the user to run `python3 llm.py setup` in their terminal. It installs `google-colab-cli==0.7.5` via uv or pipx and starts the login.
3. With their OK, run `python3 llm.py up`. Wait about 4 minutes; progress lines start with `vm`.
4. Hand over the printed base URL and key, and configure their app:
   - opencode: `python3 llm.py opencode [project folder]`, or merge the `provider` block of `state/opencode.json` into their config.
   - OpenAI SDK, curl or any "OpenAI-compatible" provider: base URL `http://127.0.0.1:18080/v1`, key from `state/api_key`.
   - Models are `ds41` (effort 50) and `ds41-max` (effort max).
5. Check it with `python3 llm.py ask "Hello!"`. Tell them: the GPU stops after 10 idle minutes and restarts on the next request (about 4 minutes); `down` stops it now.

## When something fails

Start with `python3 llm.py status`, then `python3 llm.py logs`. The VM timeline lines (`<seconds>s <time> <event>`) show how far the boot got. A line containing `FAILED`, `died` or `GUARD` aborts the start, and `llm.py` stops the VM by itself.

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `no G4 GPU from Colab` | no free G4, or no compute units | retry in a few minutes; check the balance |
| `cannot read your balance` | Colab login expired | user runs `python3 llm.py setup` |
| `llama.cpp build FAILED` | build error on the VM | `logs`; retry `up` once; if it repeats, report it with the log |
| `backbone FAILED`, `engram FAILED`, `draft FAILED` | Hugging Face download failed or was rate-limited | retry; if it repeats, set `HF_TOKEN` to the user's token |
| `GUARD: ... MiB RAM available` | VM RAM ran short while loading | retry; then `up --fast` |
| `boot did not finish` | stuck boot | `down`, then `up` |
| `401`, `403` | wrong key, or the app uses a host other than 127.0.0.1 or localhost | fix the app's base URL and key |
| port in use | another program on 18080 | `LLM_PORT=18081` |
| endpoint dead but the VM may still run | crash or reboot of the user's computer | `python3 llm.py down`; it adopts and stops the session |

Settings are environment variables (`LLM_IDLE_MIN`, `LLM_MAX_HOURS`, `LLM_PORT`, `LLM_IMAGES`, `LLM_CONTEXT`, `LLM_PROFILE`, `LLM_BIND`, `HF_TOKEN`, `UNCENSOR`, `DRAFT`, `LLM_MIN_BALANCE`). The full table is in `SETUP.md`. They take effect when the background endpoint starts, so run `down` first.

## Facts you can quote

These were measured on the G4 at reasoning effort max, one answer stream at a time:

- **Decode:** 105.6 tok/s (default) and 118.7 tok/s (`--fast`), averages of six fixed answers.
- **Time to first token:** 9.9 s at 8K and 27.8 s at 32K (default); 8.8 s and 23.4 s (`--fast`). About 1 s for cached follow-ups. 208 s for a 244K-token prompt.
- **Long context:** needle-in-a-haystack 5/5 at 247K tokens. Image tests 6/6.
- **Cost:** about 8.9 compute units per GPU hour. Each start takes about 4 minutes and downloads about 260 GB (free).
- **Weights:** ISTA-DASLab's GSQ-RCO IQ2_XXS (2.0 bits per weight on average), which matched the original on their AIME25, GPQA-Diamond and LiveCodeBench v6 runs at effort 50. AtomicChat's 82 uncensored tensors are spliced in.
- **Lossless speedups:** every speedup was checked against the build before it on the same weights: same tokens, or a greedy KL divergence of at most 0.0025. The 2-bit quantization itself is the one loss against the original model.

## Code map

| File | Role |
| --- | --- |
| `llm.py` | `setup()` first run; `Manager` VM lifecycle (start, tunnel, idle stop, repair); `Handler` HTTP proxy (`/v1/*`, local `/v1/models`, `/_llm/*` control); `watchdog()` detached guard; `LAYOUTS`, `BOOT_KEYS` |
| `vm/boot.sh` | the boot sequence and its timeline (`/content/res/timeline.txt`) |
| `vm/serve.sh` | llama-server flags and the runtime switches of the patches |
| `vm/build.sh`, `vm/patches/` | llama.cpp b11963 at a pinned commit, with the patch series |
| `vm/jobs.py`, `vm/pfetch.py` | download jobs at pinned Hugging Face revisions, parallel range fetcher |
| `vm/mkpatch.py`, `vm/applypatch.py` | builds and applies the uncensor splice (sha256 checked) |
| `vm/punch.py`, `vm/pin.c` | frees RAM the GPU already holds; pins cold experts for DMA |
| `vm/gate.py` | opens the public port only when everything is loaded |
| `vm/profile-IQ2_XXS.txt` | expert hotness order used to fill the GPU cache |
| `tests/` | offline tests: `python3 -B -m unittest -v tests/test_llm.py` (Linux) |
