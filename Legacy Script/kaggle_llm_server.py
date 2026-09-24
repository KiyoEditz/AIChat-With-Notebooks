# -*- coding: utf-8 -*-
"""kaggle_llm_server.py

Kaggle Notebook version of the Colab LLM Server.
Adapted for Kaggle's dual T4 GPU accelerator (2x T4, ~15GB VRAM each = ~30GB total).

Stack: Ollama (multi-GPU) -> LiteLLM Proxy -> Cloudflare Tunnel

# Setup checklist before running this cell
1. Kaggle notebook settings -> Accelerator -> "GPU T4 x2"
2. Kaggle notebook settings -> Internet -> "On"
3. Run this cell and wait for the model pull + load (uses `ollama pull`,
   same as the original Colab script -- no HuggingFace token needed)
4. Copy the BASE_URL / API_KEY printed at the end into your local client env

# Fixes in this version vs the original Colab script
- [Memory leak] Added a cleanup() routine that stops loaded Ollama models,
  kills all background processes, and clears CUDA cache. It runs
  automatically on normal shutdown (Ctrl+C / stop cell), on any exception
  during model download/load, and is registered via atexit + SIGTERM so it
  fires even if Kaggle kills the kernel from the UI.
- [Slow downloads] Initially replaced `ollama pull` with a direct
  `huggingface_hub` + `hf_transfer` download for speed. Reverted: on
  Xet-backed HF repos, huggingface_hub reconstructs the file from chunks in
  a separate temp location before moving it into place, briefly needing
  ~2x the model's size in free disk space. Kaggle's ~57GB disk doesn't have
  that headroom for larger models and crashed with "No space left on
  device". `ollama pull` writes blobs straight into /root/.ollama/models
  with no such duplicate-copy step, so we're back to it -- slower, but
  storage-safe. A free-disk-space check now runs before the pull starts so
  a low-space situation fails fast with a clear message instead of
  crashing mid-download.
- [Log spam] `ollama pull`'s progress bar prints a new line per tick when
  its output isn't a real TTY (e.g. redirected through a subprocess pipe),
  causing the wall of repeated "pulling ... 4%" lines. Fixed by capturing
  the subprocess output ourselves and only printing an updated line at
  most once every ~2 seconds via carriage-return overwrite, instead of
  printing every single tick.
"""

import subprocess, time, os, re, sys, gc, atexit, signal, shutil, requests

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Pick a model that needs BOTH T4s to fit comfortably. 32B-class models
# at Q4_K_M/Q5_K_M land in the 19-23GB range, which exceeds a single T4's
# ~15GB usable VRAM and forces a real multi-GPU split.

# Confirmed working/tested by the user on the previous ollama-pull-based
# script -- same EVA-Qwen2.5-32B-v0.2 family, i1 (imatrix) quant, Q6_K.
MODEL = "hf.co/mradermacher/EVA-Qwen2.5-32B-v0.2-i1-GGUF:Q6_K"   # ~27GB

# Alternatives (uncomment one to switch):
# MODEL = "hf.co/bartowski/EVA-Qwen2.5-32B-v0.2-GGUF:Q4_K_M"        # ~20GB, smaller/faster, less disk headroom needed
# MODEL = "hf.co/bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M"        # general purpose, not abliterated
# MODEL = "hf.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M"  # coding-focused

OLLAMA_PORT = 11434
PROXY_PORT = 4000
API_KEY = "sk-kaggle-local"

# Context window (tokens). With 2x T4 you have more VRAM headroom than the
# single-T4 Colab setup, so this can go higher than the original 8192 --
# but leave margin: every extra 1k tokens of context costs a few hundred MB
# of KV-cache VRAM on top of the model weights themselves.
NUM_CTX = 12288

# Explicitly expose both GPUs to Ollama. Do NOT set this to just "0" or
# the model will try to fit on a single T4 and may OOM or silently skip
# GPU 1 entirely.
GPU_IDS = "0,1"

# Minimum free disk space (GB) required before attempting the pull. Ollama
# writes blobs directly (no duplicate temp copy), so this only needs to be
# a bit larger than the model itself -- but Kaggle's disk is shared with
# preinstalled packages/datasets, so we check up front and fail fast with
# a clear message instead of crashing mid-download.
MIN_FREE_DISK_GB = 32

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def sh(cmd, check=True, quiet=False, capture=False):
    """Run shell command."""
    kw = dict(shell=True, check=check)
    if capture:
        kw.update(capture_output=True, text=True)
    elif quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, **kw)


def bg(cmd, log=None, env=None):
    """Run shell command in background."""
    out = open(log, "w") if log else subprocess.DEVNULL
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.Popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, env=full_env)


def wait_http(url, timeout=60, name=""):
    """Poll URL until it responds (< 500 status)."""
    for i in range(timeout):
        try:
            if requests.get(url, timeout=2).status_code < 500:
                print(f"  \u2705 {name} is ready")
                return True
        except Exception:
            pass
        if i > 0 and i % 15 == 14:
            print(f"  \u231b Still waiting for {name}... ({i+1}s)")
        time.sleep(1)
    print(f"  \u26a0\ufe0f {name} not confirmed after {timeout}s -- continuing")
    return False


def section(n, total, title):
    bar = "\u2500" * 64
    print(f"\n{bar}\n [{n}/{total}] {title}\n{bar}")


def check_dual_gpu_usage():
    """Verify both GPUs actually have memory allocated after model load."""
    r = sh(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total "
        "--format=csv,noheader",
        check=False, capture=True,
    )
    if r.returncode != 0:
        print("  \u26a0\ufe0f Could not query nvidia-smi")
        return
    print("  GPU memory usage:")
    used_gpus = 0
    for line in [l for l in r.stdout.strip().splitlines() if l.strip()]:
        print(f"    {line}")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            used_mb = float(parts[2].replace("MiB", "").strip())
            if used_mb > 500:
                used_gpus += 1
    if used_gpus >= 2:
        print("  \u2705 Both GPUs are being used")
    else:
        print(
            "  \u26a0\ufe0f Only one GPU shows significant usage. The model may be "
            "small enough to fit on a single T4. Pick a larger model/quant "
            "if you want a forced split across both GPUs."
        )


# ------------------------------------------------------------------
# FIX 1: Memory / VRAM cleanup handler
# ------------------------------------------------------------------
# Runs on: normal shutdown (Ctrl+C / stop cell), SIGTERM (Kaggle killing the
# kernel), and any unhandled exception during download/load (registered via
# atexit further below). Safe to call multiple times.

_procs = {"ollama": None, "litellm": None, "cloudflared": None}
_cleaned_up = False


def cleanup(*_args):
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    print("\n  \U0001f9f9 Cleaning up processes and freeing RAM/VRAM...")

    # Ask Ollama to unload any resident models (frees VRAM immediately,
    # rather than waiting for the process kill below).
    r = sh("ollama ps", check=False, capture=True)
    if r.returncode == 0 and r.stdout:
        for line in r.stdout.strip().splitlines()[1:]:  # skip header
            model_name = line.split()[0] if line.split() else None
            if model_name:
                sh(f"ollama stop {model_name}", check=False, quiet=True)

    # Terminate tracked background processes gracefully, then force-kill.
    for name, proc in _procs.items():
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # Belt-and-suspenders: kill by process name too, in case a proc wasn't
    # tracked (e.g. cleanup() called before a stage started).
    sh("pkill -9 -f 'ollama serve'", check=False, quiet=True)
    sh("pkill -9 -f litellm", check=False, quiet=True)
    sh("pkill -9 -f cloudflared", check=False, quiet=True)

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    print("  \u2705 Cleanup complete -- RAM/VRAM released")


atexit.register(cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Pre-cleanup: in case this cell was re-run after a previous crashed/failed
# attempt in the same kernel, clear out any leftover processes first.
sh("pkill -9 -f 'ollama serve'", check=False, quiet=True)
sh("pkill -9 -f litellm", check=False, quiet=True)
sh("pkill -9 -f cloudflared", check=False, quiet=True)
time.sleep(1)

# ------------------------------------------------------------------
# BANNER
# ------------------------------------------------------------------
print("\u2554" + "\u2550" * 64 + "\u2557")
print("\u2551  Kaggle LLM Server \u00b7 Dual T4 Multi-GPU Bridge" + " " * 15 + "\u2551")
print("\u2551  Stack: Ollama (2x T4) -> LiteLLM Proxy -> Cloudflare Tunnel  \u2551")
print("\u255a" + "\u2550" * 64 + "\u255d")

r = sh(
    "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    check=False, capture=True,
)
if r.returncode == 0:
    print(f"\n  GPUs detected:\n{r.stdout.strip()}")
    gpu_count = len([l for l in r.stdout.strip().splitlines() if l.strip()])
    if gpu_count < 2:
        print(
            "\n  \u26a0\ufe0f Only 1 GPU detected. Go to Notebook Settings -> "
            "Accelerator -> 'GPU T4 x2' and restart the session."
        )
else:
    print("\n  \u26a0\ufe0f No GPU detected -- enable 'GPU T4 x2' in Notebook Settings")

print(f"\n  Model    : {MODEL}")
print(f"  GPUs     : {GPU_IDS}\n")

TOTAL_STEPS = 6

try:
    # -- 1. INSTALL OLLAMA BINARY --------------------------------------
    section(1, TOTAL_STEPS, "Install Ollama Binary")
    sh("apt-get update -qq && apt-get install -y -qq zstd", check=False, quiet=True)
    sh("rm -rf ollama.tar.zst /usr/local/bin/ollama /usr/bin/ollama", check=False)
    url = "https://github.com/ollama/ollama/releases/download/v0.30.10/ollama-linux-amd64.tar.zst"
    print(f"  \u231b Downloading Ollama binary...")
    sh(f"wget -q {url} -O ollama.tar.zst")
    sh("tar -C /usr -xaf ollama.tar.zst")
    r = sh("ollama --version", capture=True, check=False)
    if r.returncode == 0:
        print(f"  \u2705 Ollama installed: {r.stdout.strip()}")
    else:
        raise RuntimeError("Ollama binary extraction failed -- check the release version in the URL")

    # -- 2. START OLLAMA SERVER (both GPUs visible) --------------------
    section(2, TOTAL_STEPS, "Start Ollama Server (multi-GPU)")
    ollama_env = {
        "OLLAMA_HOST": "0.0.0.0:11434",
        "OLLAMA_ORIGINS": "*",
        "CUDA_VISIBLE_DEVICES": GPU_IDS,   # <-- exposes both T4s to Ollama
    }
    _procs["ollama"] = bg("ollama serve", "/tmp/ollama.log", env=ollama_env)
    time.sleep(3)
    wait_http(f"http://localhost:{OLLAMA_PORT}", 30, "Ollama")

    # -- 3. PULL MODEL VIA OLLAMA (stable storage, throttled log) -------
    # NOTE: reverted from the direct huggingface_hub/Xet download. On Xet-
    # backed repos, huggingface_hub reconstructs the final file from chunks
    # in a *separate* temp location before moving it into place -- for a
    # brief window that means roughly 2x the model size on disk (e.g. a
    # 20GB file needs ~40GB free). Kaggle's ~57GB disk (shared with other
    # pre-installed packages/datasets) isn't enough headroom for larger
    # models, which is what caused the "No space left on device" crash.
    # `ollama pull` streams and writes blobs directly into
    # /root/.ollama/models with no such duplicate-copy step, so it's the
    # safer option even though it's slower. We keep the download itself
    # unchanged, but throttle how often we print progress so it doesn't
    # spam the notebook output.
    section(3, TOTAL_STEPS, "Pull Model via Ollama")

    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    print(f"  Free disk space: {free_gb:.1f} GB (minimum required: {MIN_FREE_DISK_GB} GB)")
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free, need at least {MIN_FREE_DISK_GB} GB for this model. "
            f"Free up space (e.g. remove unused Kaggle datasets/outputs attached to this "
            f"notebook) or pick a smaller quant."
        )

    print(f"  \u231b Pulling {MODEL} -- slower than the HF-direct method, but avoids duplicate")
    print(f"     temp files on disk. Progress below refreshes every ~2s instead of every tick.\n")

    def pull_model_with_throttled_log(model, max_attempts=3):
        # NOTE: `ollama pull` has a known intermittent bug (ollama/ollama
        # issues #3628, #4898, #14177) where it fails with
        # "Error: remove .../blobs/sha256-...-partial-N: no such file or
        # directory" -- this fires during the cleanup/finalize step for a
        # blob that actually finished downloading fine; the partial temp
        # file was already removed/renamed by the time ollama tries to
        # remove it again (a benign race in its parallel chunk downloader).
        # It is NOT a disk-space or corruption problem. Simply retrying the
        # pull almost always succeeds immediately, since ollama resumes and
        # skips blobs it already has.
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"  \u21bb Retry {attempt}/{max_attempts} after a transient ollama pull error...")
                time.sleep(3)

            proc = subprocess.Popen(
                f"ollama pull {model}",
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            last_print = 0.0
            last_line = ""
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                last_line = line
                now = time.time()
                if now - last_print >= 2:
                    print("\r  " + line[:110].ljust(110), end="", flush=True)
                    last_print = now
            proc.wait()
            print()  # finish the throttled line with a newline

            if proc.returncode == 0:
                return
            if attempt == max_attempts:
                raise RuntimeError(f"ollama pull failed after {max_attempts} attempts (exit {proc.returncode}): {last_line}")

    pull_model_with_throttled_log(MODEL)
    print(f"  \u2705 Pull complete: {MODEL}")

    with open("/tmp/Modelfile", "w") as f:
        f.write(f"FROM {MODEL}\nPARAMETER num_ctx {NUM_CTX}\n")
    result = sh("ollama create character -f /tmp/Modelfile", check=False, capture=True)
    if result.returncode == 0:
        print("  \u2705 Model ready, aliased as 'character'")
        ollama_model_name = "character"
    else:
        print("  \u2139\ufe0f Alias failed, referencing the model name directly")
        ollama_model_name = MODEL

    # -- 4. LOAD MODEL AND VERIFY DUAL-GPU SPLIT -----------------------
    section(4, TOTAL_STEPS, "Verify Dual-GPU Split")
    print("  Sending a warm-up request to force the model into VRAM...")
    warm = requests.post(
        f"http://localhost:{OLLAMA_PORT}/api/generate",
        json={"model": ollama_model_name, "prompt": "hi", "stream": False},
        timeout=300,
    )
    if warm.status_code != 200:
        raise RuntimeError(f"Model failed to load: HTTP {warm.status_code} -- {warm.text[:300]}")
    time.sleep(2)
    check_dual_gpu_usage()

    # -- 5. LITELLM PROXY -----------------------------------------------
    section(5, TOTAL_STEPS, "LiteLLM: API -> Ollama Bridge")
    print("  Installing litellm...")
    sh("pip install -q 'litellm[proxy]'", check=False)

    entries = "\n".join(
        f"  - model_name: {n}\n"
        f"    litellm_params:\n"
        f"      model: ollama/{ollama_model_name}\n"
        f"      api_base: http://localhost:{OLLAMA_PORT}"
        for n in ["character1", MODEL]
    )
    litellm_yaml = f"""
model_list:
{entries}

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: "{API_KEY}"
"""
    with open("/tmp/litellm.yaml", "w") as f:
        f.write(litellm_yaml)

    _procs["litellm"] = bg(
        f"litellm --config /tmp/litellm.yaml --port {PROXY_PORT} --host 0.0.0.0",
        "/tmp/litellm.log",
    )
    time.sleep(6)
    wait_http(f"http://localhost:{PROXY_PORT}/health", 45, "LiteLLM proxy")

    # -- 6. CLOUDFLARE TUNNEL -------------------------------------------
    section(6, TOTAL_STEPS, "Cloudflare Tunnel -> Public HTTPS URL")
    sh(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared "
        "&& chmod +x /usr/local/bin/cloudflared"
    )
    _procs["cloudflared"] = bg(f"cloudflared tunnel --url http://localhost:{PROXY_PORT}", "/tmp/cloudflared.log")
    print("  \u231b Waiting for public URL...")
    tunnel_url = None
    for _ in range(90):
        time.sleep(1)
        try:
            text = open("/tmp/cloudflared.log").read()
            m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", text)
            if m:
                tunnel_url = m.group(0)
                break
        except Exception:
            pass

except Exception as e:
    print(f"\n  \u274c Startup failed: {e}")
    cleanup()
    sys.exit(1)

# ------------------------------------------------------------------
# READY
# ------------------------------------------------------------------
print()
print("\u2554" + "\u2550" * 64 + "\u2557")
if tunnel_url:
    print("\u2551  \u2705 SERVER IS READY" + " " * 47 + "\u2551")
else:
    print("\u2551  \u26a0\ufe0f SERVER STARTED (no tunnel URL found)" + " " * 29 + "\u2551")
print("\u255a" + "\u2550" * 64 + "\u255d")

if tunnel_url:
    print(f"""
  URL     : {tunnel_url}
  Model   : {MODEL}
  API Key : {API_KEY}

  BASE_URL="{tunnel_url}"
  API_KEY="{API_KEY}"

  IMPORTANT:
  - Keep this cell RUNNING while using the remote LLM
  - The Cloudflare URL is temporary -- it changes every restart
  - First request after startup needs 30-60s while the model warms up
  - Stopping the cell (or a crash) triggers automatic cleanup of
    RAM/VRAM -- see the "Cleaning up..." message
""")
else:
    print(f"\n  \u26a0\ufe0f Could not obtain a tunnel URL.")
    print(f"  Check log: !cat /tmp/cloudflared.log")
    print(f"  Local LiteLLM proxy: http://localhost:{PROXY_PORT}")

print("  \u231b Cell running continuously. Stop the cell to shut down (cleanup runs automatically).\n")

# ------------------------------------------------------------------
# KEEP ALIVE (Heartbeat)
# ------------------------------------------------------------------
tick = 0
try:
    while True:
        time.sleep(60)
        tick += 1
        ts = time.strftime("%H:%M:%S")
        try:
            requests.get(f"http://localhost:{PROXY_PORT}/health", timeout=5)
            status = "healthy"
        except Exception:
            status = "unreachable"
        print(f"  [{ts}] heartbeat #{tick:04d} | proxy {status} | {tunnel_url or 'no tunnel'}")
except KeyboardInterrupt:
    cleanup()