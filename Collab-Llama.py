# -*- coding: utf-8 -*-
"""Collab-Llama.py

Colab Notebook LLM Server -- llama.cpp edition, one-click.

Active path: AIChat -> Cloudflare Tunnel -> llama-server -> GGUF (local file)
GPU: any Colab GPU works -- T4, L4, A100, etc. This script auto-detects the
attached GPU's compute capability at runtime and picks the right CUDA arch
itself. No manual editing needed when you switch GPU type in Runtime ->
Change runtime type. See Build-Llama-CUDA-Release.py for how the prebuilt
binary covers multiple GPU generations in one fat tarball.

## Why this exists (vs. colab_llm_server.py)
Full rationale in the project README ("Pilihan Stack Server LLM"). Short
version: drops Ollama + LiteLLM, talks to llama.cpp's own llama-server
directly -- no translation layer for sampling parameters, native DRY
sampling for long-roleplay repetition control, one process instead of two.
The original colab_llm_server.py / .ipynb are untouched -- this is a
separate, opt-in alternative.

## What's new in this version
- Prefers a prebuilt CUDA binary (see Build-Llama-CUDA-Release.py) and only
  falls back to compiling from source if no prebuilt is configured, the
  download fails, or it doesn't match this environment (verified against
  the bundled VERSION.txt + a real `--version` sanity run -- never trusted
  blindly).
- Model download uses huggingface_hub's `snapshot_download(local_dir=...)`
  directly (not llama-server's built-in `-hf` downloader), so the model
  lives ONLY at MODEL_DIR -- never duplicated into the default HF cache.
  Xet chunk caching is disabled by default (HF_XET_CHUNK_CACHE_SIZE_BYTES=0)
  to avoid retaining a second copy of chunk data during download.
- Explicit readiness state machine (PROCESS_STARTED -> MODEL_LOADING ->
  HEALTH_READY -> TUNNEL_READY) using llama-server's actual documented
  /health semantics (503 "Loading model" vs 200 {"status":"ok"}) instead of
  a generic "any non-5xx response" guess. "SERVER IS READY" is only ever
  printed after a real (1-token) completion request succeeds end-to-end.

## Setup checklist
1. Enable GPU: Runtime -> Change runtime type -> T4 GPU
2. (Optional) Set GITHUB_RELEASE_URL below to your own prebuilt tarball from
   Build-Llama-CUDA-Release.py for a much faster first run.
3. Run this cell and wait.
4. Copy the BASE_URL / API_KEY printed at the end into your local .env
5. Keep this cell running while you use the remote LLM
"""

import subprocess, time, os, re, sys, shutil, glob, hashlib, tarfile, requests, socket

# =====================================================================
# CONFIGURATION
# =====================================================================

# Hugging Face repo + quant (llama.cpp's -hf syntax: <user>/<repo>:<tag>).
MODEL_REPO = "mradermacher/EVA-Qwen2.5-14B-v0.2-i1-GGUF:Q6_K"   # ~12.2 GB

# Alternatives (uncomment one to switch):
# MODEL_REPO = "bartowski/EVA-Qwen2.5-14B-v0.2-GGUF:Q6_K"
# MODEL_REPO = "bartowski/Qwen2.5-Coder-14B-Instruct-abliterated-GGUF:Q6_K"
# MODEL_REPO = "richardyoung/Qwen3-14B-abliterated-GGUF:Q6_K"
# MODEL_REPO = "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K"

LLAMA_PORT  = 8080
API_KEY     = "sk-colab-local"          # unchanged -- existing .env files
                                         # keep working if you only swap BASE_URL.

# Pinned llama.cpp release tag -- reproducibility, and the value the prebuilt
# binary's VERSION.txt is checked against.
LLAMA_CPP_TAG = "b10605"

# CUDA_ARCH is auto-detected at runtime from whatever GPU Colab actually
# attached (see detect_gpu_arch() below, called in MAIN before anything
# else runs) -- do NOT hardcode this. It's only ever used as the build
# target if the prebuilt tarball turns out unusable and this notebook has
# to compile llama.cpp from source itself as a fallback, in which case it
# builds ONLY for the GPU that's actually here (fast, single-arch). The
# prebuilt path doesn't need this value at all -- it's a fat multi-arch
# binary that already covers T4/A100/L4/RTX/H100 in one file.
CUDA_ARCH = None   # filled in automatically -- see detect_gpu_arch()
_FALLBACK_ARCH_IF_UNDETECTABLE = "75"   # last resort if nvidia-smi query fails

# OPTIONAL: URL of a prebuilt tarball produced by Build-Llama-CUDA-Release.py
# (a GitHub Release asset URL you control). Leave empty to always build from
# source. This is a FAT multi-arch build -- one URL covers every GPU these
# notebooks might attach to, so you never need to change it when you switch
# GPU type in Colab. Example:
#   GITHUB_RELEASE_URL = "https://github.com/<user>/<repo>/releases/download/v1.0.0/llama-cuda-multiarch-b10605.tar.gz"
GITHUB_RELEASE_URL = "https://github.com/KiyoEditz/AIChat-With-Notebooks/releases/download/v1.0.0/llama-cuda-multiarch-b10605.tar.gz"
# NOTE: rebuild + re-upload with Build-Llama-CUDA-Release.py's new
# multi-arch CUDA_ARCH first -- the old single-arch (sm75) tarball at the
# previous URL won't satisfy the arch-coverage check below on non-T4 GPUs.

NUM_CTX = 8192   # Adjust down if VRAM is tight on a smaller GPU (T4/L4);
                  # a big-VRAM GPU (A100) can comfortably go higher.

# Scratch storage -- ephemeral, local-disk, NOT Google Drive. Wiped when the
# runtime disconnects, which is fine/expected (re-downloaded next run, or
# reused from a prebuilt release for the binary).
SCRATCH_DIR = "/content" if os.path.isdir("/content") else "/tmp"
MODEL_DIR = f"{SCRATCH_DIR}/models"
XET_CACHE_DIR = f"{SCRATCH_DIR}/hf-xet"
LLAMA_SRC_DIR = f"{SCRATCH_DIR}/llama.cpp"
PREBUILT_DIR = f"{SCRATCH_DIR}/llama-prebuilt"

# --- Repetition control (server-side defaults; server.js overrides these
# per-request -- MUST match .env.example's defaults) ---------------------
REPEAT_LAST_N, REPEAT_PENALTY = 256, 1.05
PRESENCE_PENALTY, FREQUENCY_PENALTY = 0.0, 0.0
DRY_MULTIPLIER, DRY_BASE = 0.8, 1.75
DRY_ALLOWED_LENGTH, DRY_PENALTY_LAST_N = 2, 1024

MIN_FREE_DISK_MARGIN_GB = 3   # extra headroom beyond the model's exact size

# ====================================================================
# Utility functions
# ====================================================================

def sh(cmd, check=True, quiet=False, capture=False):
    kw = dict(shell=True, check=check)
    if capture:
        kw.update(capture_output=True, text=True)
    elif quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, **kw)


def bg(cmd, log=None, cwd=None, env=None):
    out = open(log, "w") if log else subprocess.DEVNULL
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.Popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, cwd=cwd, env=full_env)


def section(n, total, title):
    bar = "\u2500" * 64
    print(f"\n{bar}\n  [{n}/{total}]  {title}\n{bar}")


def run_with_retry(fn, max_attempts=3, label="operation"):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait_s = attempt * 5
            print(f"  \u21bb {label} failed ({e}); retry {attempt}/{max_attempts} in {wait_s}s...")
            time.sleep(wait_s)


def acquire_free_port(preferred_port=8080, fallbacks=(8081, 8082, 8088, 8888, 5000, 18080)):
    """Ensure a free, bindable TCP port for llama-server. First kills any lingering
    instances and clears the preferred port. If still blocked, falls back to the next
    available port so startup never fails with 'couldn't bind HTTP server socket'."""
    sh("pkill -9 -f 'llama-server'", check=False, quiet=True)
    sh("pkill -9 -f cloudflared", check=False, quiet=True)
    sh(f"fuser -k -9 {preferred_port}/tcp 2>/dev/null", check=False, quiet=True)
    time.sleep(1)

    candidates = [preferred_port] + [p for p in fallbacks if p != preferred_port]
    for p in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                if p != preferred_port:
                    print(f"  ⚠️ Port {preferred_port} in use; automatically switched to port {p}")
                return p
            except OSError:
                sh(f"fuser -k -9 {p}/tcp 2>/dev/null", check=False, quiet=True)
                time.sleep(0.5)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                        s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s2.bind(("0.0.0.0", p))
                        if p != preferred_port:
                            print(f"  ⚠️ Port {preferred_port} in use; automatically switched to port {p}")
                        return p
                except OSError:
                    continue

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        p = s.getsockname()[1]
        print(f"  ⚠️ Preferred ports unavailable; assigned free port {p}")
        return p


def detect_gpu_arch():
    """Detect every attached GPU's CUDA compute capability as 'XY' strings
    (e.g. '86' for an RTX 3090, '80' for an A100). Returns a list -- usually
    length 1 on Colab (single GPU), but this stays correct if that ever
    changes. Returns [] if nvidia-smi can't be queried (no GPU, driver not
    ready yet, etc.) -- callers must handle that gracefully, not assume."""
    r = sh("nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader",
           check=False, capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    archs = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, name, cap = parts[0], parts[1], parts[2]
            arch = cap.replace(".", "")   # "8.0" -> "80"
            archs.append(arch)
            print(f"  GPU {idx}: {name} -> compute capability {cap} (sm_{arch})")
    return archs


def _parse_embedded_archs(archs_field):
    """Parse a prebuilt tarball's VERSION.txt 'cuda_archs' field, e.g.
    '75-real;80-real;86-real;89-real;90-real;90-virtual', into:
      - real_archs: set of arch strings with native (SASS) code embedded
      - virtual_max: highest arch with PTX embedded (JIT-compiles forward
        to that arch or newer at load time), or None if none present
    Also tolerates a legacy single-value field with no suffix (old
    single-arch builds) or the old 'sm_75' style value for backward compat
    with tarballs built before this multi-arch update."""
    real, virtual_max = set(), None
    for tok in archs_field.replace("sm_", "").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("-real"):
            real.add(tok[:-len("-real")])
        elif tok.endswith("-virtual"):
            num = tok[:-len("-virtual")]
            if virtual_max is None or int(num) > int(virtual_max):
                virtual_max = num
        else:
            real.add(tok)
    return real, virtual_max


# ====================================================================
# STEP: llama-server binary (prebuilt-first, source-build fallback)
# ====================================================================

def verify_prebuilt(pkg_dir):
    """Check the prebuilt tarball's own VERSION.txt against what THIS run
    expects, then actually execute it -- never trust a downloaded binary
    just because it extracted successfully."""
    version_file = f"{pkg_dir}/VERSION.txt"
    if not os.path.isfile(version_file):
        raise RuntimeError("prebuilt package has no VERSION.txt -- can't verify, refusing to use it")

    meta = {}
    for line in open(version_file):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            meta[k] = v

    if meta.get("llama_cpp_tag") != LLAMA_CPP_TAG:
        raise RuntimeError(f"prebuilt is tag {meta.get('llama_cpp_tag')!r}, this run wants {LLAMA_CPP_TAG!r}")

    # Multi-arch coverage check: this GPU's detected arch must either have
    # native SASS code embedded, or be new enough for the embedded PTX
    # ("virtual") entry to JIT-compile forward to it. Falls back to the
    # legacy 'cuda_arch' field for tarballs built before this update.
    archs_field = meta.get("cuda_archs") or meta.get("cuda_arch", "")
    real_archs, virtual_max = _parse_embedded_archs(archs_field)
    if not real_archs and not virtual_max:
        raise RuntimeError("prebuilt VERSION.txt has no cuda_archs/cuda_arch field -- can't verify compatibility")

    covered_desc = f"sm_{{{','.join(sorted(real_archs))}}}" + (f" (+PTX from sm_{virtual_max})" if virtual_max else "")
    if not CUDA_ARCH:
        raise RuntimeError("this runtime's GPU arch could not be detected -- refusing to trust a prebuilt blindly")
    elif CUDA_ARCH in real_archs:
        print(f"  GPU sm_{CUDA_ARCH}: native match in prebuilt ({covered_desc})")
    elif virtual_max and int(CUDA_ARCH) >= int(virtual_max):
        print(f"  GPU sm_{CUDA_ARCH}: no native build, but covered via PTX JIT from sm_{virtual_max} "
              f"(first request may warm up a little slower while the driver compiles it)")
    else:
        raise RuntimeError(f"prebuilt covers {covered_desc}, this GPU is sm_{CUDA_ARCH} -- not covered")

    run_sh = f"{pkg_dir}/run.sh"
    if not os.path.isfile(run_sh):
        raise RuntimeError("prebuilt package has no run.sh launcher")
    os.chmod(run_sh, 0o755)
    os.chmod(f"{pkg_dir}/llama-server", 0o755)

    r = sh(f"{run_sh} --version", check=False, capture=True)
    if r.returncode != 0:
        raise RuntimeError(f"prebuilt binary failed a real --version run (exit {r.returncode}): {r.stderr[:200]}")

    print(f"  \u2705 Prebuilt verified: {meta.get('llama_cpp_tag')} / covers {covered_desc} "
          f"(built {meta.get('built_at', '?')}, CUDA VMM: {meta.get('cuda_vmm', 'unknown')})")
    return run_sh


def try_prebuilt():
    if not GITHUB_RELEASE_URL:
        raise RuntimeError("GITHUB_RELEASE_URL not set")

    os.makedirs(PREBUILT_DIR, exist_ok=True)
    tarball_path = f"{PREBUILT_DIR}/prebuilt.tar.gz"

    print(f"  \u231b Downloading prebuilt binary from GITHUB_RELEASE_URL...")

    def download_step():
        r = requests.get(GITHUB_RELEASE_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(tarball_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    run_with_retry(download_step, label="prebuilt download")

    extract_dir = f"{PREBUILT_DIR}/extracted"
    sh(f"rm -rf {extract_dir}", check=False, quiet=True)
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball_path) as tar:
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(extract_dir, filter='data')
        else:
            tar.extractall(extract_dir)

    # The tarball contains one top-level dir (see Build-Llama-CUDA-Release.py)
    entries = [d for d in glob.glob(f"{extract_dir}/*") if os.path.isdir(d)]
    if not entries:
        raise RuntimeError("prebuilt tarball had no top-level directory after extraction")

    return verify_prebuilt(entries[0])


def find_cuda_driver_lib():
    """Locate a libcuda.so the linker can actually consume.

    CMake's FindCUDAToolkit only creates the CUDA::cuda_driver imported
    target if find_library(NAMES cuda) succeeds. Colab/Kaggle GPU images
    ship the CUDA toolkit without the driver stub package, and the real
    driver is injected as a versioned 'libcuda.so.1' with no unversioned
    dev symlink (commonly at /usr/local/nvidia/lib64) -- so ggml-cuda's
    `target_link_libraries(... CUDA::cuda_driver)` fails the CMake *generate*
    step with "target was not found" if we don't hand it a discoverable path
    ourselves.

    Returns (path, is_versioned).
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    dirs = []
    for base in [cuda_home, "/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda*")):
        dirs += [
            f"{base}/lib64/stubs", f"{base}/lib/stubs",
            f"{base}/targets/x86_64-linux/lib/stubs",
            f"{base}/lib64", f"{base}/targets/x86_64-linux/lib",
        ]
    dirs += [
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib64",
        "/lib",
    ]
    for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if p.strip():
            dirs.append(p.strip())

    seen = set()
    unique_dirs = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    dirs = unique_dirs

    for d in dirs:
        p = f"{d}/libcuda.so"
        if os.path.exists(p):
            return p, False

    versioned = []
    for d in dirs:
        if os.path.isfile(f"{d}/libcuda.so.1"):
            versioned.append(f"{d}/libcuda.so.1")
        versioned += sorted(glob.glob(f"{d}/libcuda.so.*"))
    r = sh("ldconfig -p", check=False, capture=True)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if "libcuda.so" in line and "=>" in line:
                versioned.append(line.split("=>")[-1].strip())

    if not versioned:
        r = sh("find /usr/local /usr/lib /lib -name 'libcuda.so*' 2>/dev/null", check=False, capture=True)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                candidate = line.strip()
                if os.path.isfile(candidate):
                    if candidate.endswith("/libcuda.so"):
                        return candidate, False
                    versioned.append(candidate)

    for p in versioned:
        if os.path.isfile(p):
            return p, True
    return None, False


def cuda_driver_cmake_flags():
    """Extra -D flags that make CUDA::cuda_driver resolvable, or [] if no
    driver library exists on this image at all."""
    path, is_versioned = find_cuda_driver_lib()
    if not path:
        print("  \u26a0\ufe0f No libcuda.so / libcuda.so.1 found -- will build with CUDA VMM disabled if needed.")
        return []
    if is_versioned:
        link_dir = "/tmp/cuda-driver-link"
        os.makedirs(link_dir, exist_ok=True)
        link_path = f"{link_dir}/libcuda.so"
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(path, link_path)
        print(f"  Driver lib: {path} (symlinked -> {link_path})")
    else:
        link_path, link_dir = path, os.path.dirname(path)
        print(f"  Driver lib: {path}")
    return [f"-DCMAKE_LIBRARY_PATH={link_dir}", f"-DCUDA_cuda_driver_LIBRARY={link_path}"]


def build_from_source():
    server_bin = f"{LLAMA_SRC_DIR}/build/bin/llama-server"
    if os.path.isfile(server_bin):
        print(f"  \u2705 Source build already cached at {server_bin} -- skipping rebuild")
        return server_bin

    print("  Installing build tools (cmake, build-essential)...")
    sh("apt-get update -qq && apt-get install -y -qq cmake build-essential git", check=False, quiet=True)

    r = sh("nvcc --version", check=False, capture=True)
    if r.returncode != 0:
        print("  \u26a0\ufe0f nvcc not found -- CUDA toolkit may be missing on this runtime; "
              "the build below may fail.")

    def clone_step():
        sh(f"rm -rf {LLAMA_SRC_DIR}", check=False, quiet=True)
        sh(f"git clone --depth 1 --branch {LLAMA_CPP_TAG} "
           f"https://github.com/ggml-org/llama.cpp.git {LLAMA_SRC_DIR}")

    print(f"  \u231b Cloning llama.cpp @ {LLAMA_CPP_TAG}...")
    run_with_retry(clone_step, label="git clone")

    print(f"  \u231b Configuring build (CUDA, sm_{CUDA_ARCH} for this runtime's GPU)...")
    driver_flags = cuda_driver_cmake_flags()

    def configure(extra_flags):
        sh(f"rm -rf {LLAMA_SRC_DIR}/build", check=False, quiet=True)
        return sh(
            f'cmake -B {LLAMA_SRC_DIR}/build -S {LLAMA_SRC_DIR} '
            f'-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCH}" '
            f'-DCMAKE_BUILD_TYPE=Release '
            f'-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF '
            + " ".join(extra_flags),
            check=False,
        ).returncode == 0

    if not configure(driver_flags):
        # ggml's own escape hatch: skips the libcuda.so link requirement at
        # the cost of the VMM-backed memory pool (slightly less efficient
        # allocator) -- correctness is unaffected.
        print("  \u26a0\ufe0f Configure failed with the CUDA driver library -- retrying with "
              "GGML_CUDA_NO_VMM=ON.")
        if not configure(["-DGGML_CUDA_NO_VMM=ON"]):
            raise RuntimeError("CMake configure failed even with CUDA VMM disabled -- see the build log above.")

    print("  \u231b Compiling llama-server (~5-10 min)...")
    sh(f"cmake --build {LLAMA_SRC_DIR}/build --config Release -j$(nproc) --target llama-server")

    if not os.path.isfile(server_bin):
        raise RuntimeError(f"Build finished but {server_bin} was not produced")

    r = sh(f"{server_bin} --version", check=False, capture=True)
    if r.returncode != 0:
        raise RuntimeError(f"Freshly built binary failed --version (exit {r.returncode})")

    print(f"  \u2705 Source build verified: {server_bin}")
    return server_bin


def ensure_llama_server():
    """Prebuilt-first (point 1.1), source build only as fallback (1.3),
    cached for the rest of this session (1.4), verified before trust (1.5)."""
    if GITHUB_RELEASE_URL:
        try:
            return try_prebuilt()
        except Exception as e:
            print(f"  \u26a0\ufe0f Prebuilt unusable ({e}) -- falling back to source build.")
    else:
        print("  GITHUB_RELEASE_URL not set -- building from source. "
              "See Build-Llama-CUDA-Release.py to make future runs faster.")
    return build_from_source()


# ====================================================================
# STEP: Model download (HF/Xet, disk-efficient, no duplicate storage)
# ====================================================================

def ensure_model():
    """snapshot_download(local_dir=...) writes real files directly into
    MODEL_DIR -- the default HF cache (~/.cache/huggingface) is never used
    to store the model a second time. Xet chunk caching is disabled by
    default (HF_XET_CHUNK_CACHE_SIZE_BYTES=0) so no extra chunk-store copy
    accumulates during a one-time download. Disk space is checked against
    the EXACT bytes this run needs (via dry_run), not a flat guess, and a
    rerun with the model already present downloads nothing."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(XET_CACHE_DIR, exist_ok=True)

    # Must be set before huggingface_hub/hf_xet read them.
    os.environ.setdefault("HF_XET_CACHE", XET_CACHE_DIR)
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1")
    # HF_XET_HIGH_PERFORMANCE intentionally left unset -- Colab's free tier
    # RAM is limited, no reason to opt into its extra parallelism/memory use.

    sh("pip install -q -U huggingface_hub hf_xet", check=False, quiet=True)
    from huggingface_hub import snapshot_download

    repo_id, _, quant = MODEL_REPO.partition(":")
    patterns = [f"*{quant}*.gguf"] if quant else ["*.gguf"]

    print(f"  Repo   : {repo_id}")
    print(f"  Quant  : {quant or '(any)'}")

    def dry_run_step():
        return snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR,
                                  allow_patterns=patterns, dry_run=True)

    plan = run_with_retry(dry_run_step, label="HF metadata fetch")
    to_download = [f for f in plan if f.will_download]

    if not to_download:
        print("  \u2705 Model already fully present in MODEL_DIR -- nothing to download")
    else:
        total_bytes = sum(f.file_size for f in to_download)
        total_gb = total_bytes / (1024 ** 3)
        free_gb = shutil.disk_usage(MODEL_DIR).free / (1024 ** 3)
        print(f"  Need to download: {len(to_download)} file(s), {total_gb:.1f} GB")
        print(f"  Free disk space : {free_gb:.1f} GB")

        if free_gb < total_gb + MIN_FREE_DISK_MARGIN_GB:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free, need ~{total_gb:.1f} GB + "
                f"{MIN_FREE_DISK_MARGIN_GB} GB margin. Free up space or pick a smaller quant."
            )

        def download_step():
            snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR, allow_patterns=patterns)

        print(f"  \u231b Downloading (resumable -- a rerun continues, doesn't restart)...")
        run_with_retry(download_step, max_attempts=3, label="model download")

        # Verify: every file we expected is present with the right size.
        for f in to_download:
            local_path = os.path.join(MODEL_DIR, f.filename)
            if not os.path.isfile(local_path):
                raise RuntimeError(f"Expected file missing after download: {f.filename}")
            actual = os.path.getsize(local_path)
            if actual != f.file_size:
                raise RuntimeError(
                    f"Size mismatch for {f.filename}: expected {f.file_size}, got {actual} "
                    f"-- download likely corrupted, delete {local_path} and rerun."
                )
        print(f"  \u2705 Verified {len(to_download)} file(s) present with correct size")

    ggufs = sorted(glob.glob(f"{MODEL_DIR}/**/*.gguf", recursive=True))
    if not ggufs:
        raise RuntimeError(f"No .gguf files found in {MODEL_DIR} after download")

    # Multi-shard GGUF: point llama-server at shard 1, it auto-loads the rest.
    shard1 = [g for g in ggufs if "-00001-of-" in g]
    entrypoint = shard1[0] if shard1 else ggufs[0]
    print(f"  Model file: {entrypoint}")
    return entrypoint


# ====================================================================
# STEP: Readiness state machine
# ====================================================================
# PROCESS_STARTED -> MODEL_LOADING -> HEALTH_READY -> (completion test) -> TUNNEL_READY
# llama-server's own documented /health semantics (verified against the
# b10605 server README): 503 {"error":{"message":"Loading model"...}} while
# loading, 200 {"status":"ok"} once truly ready. We do NOT treat "any
# non-5xx" as ready -- 503 is explicitly "still loading", not an error to
# shrug off.

def wait_for_ready(proc, log_path, base_url, api_key, timeout=2700):
    start = time.time()
    last_print = 0.0
    state = "PROCESS_STARTED"
    print(f"  State: {state}")

    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = ""
            try:
                with open(log_path) as f:
                    tail = f.read()[-1500:]
            except Exception:
                pass
            raise RuntimeError(f"llama-server process exited early (code {proc.returncode}).\n"
                                f"Last log output:\n{tail}")

        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                if state != "HEALTH_READY":
                    state = "HEALTH_READY"
                    print(f"  State: {state} (/health confirms model loaded)")
                break
            elif r.status_code == 503:
                if state != "MODEL_LOADING":
                    state = "MODEL_LOADING"
                    print(f"  State: {state} (server up, model still loading)")
        except Exception:
            pass

        now = time.time()
        if now - last_print >= 10:
            elapsed = int(now - start)
            try:
                size = os.path.getsize(log_path)
                with open(log_path, "rb") as f:
                    f.seek(max(0, size - 200))
                    tail_line = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    tail_line = tail_line[-1][:100] if tail_line else ""
            except Exception:
                tail_line = ""
            print(f"  \u231b [{elapsed}s] {state}... {tail_line}")
            last_print = now
        time.sleep(2)
    else:
        raise RuntimeError(f"Timed out after {timeout}s waiting for readiness (last state: {state})")

    # /v1/models -- confirm a model is actually attached, not just that the
    # HTTP server answered.
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        models = r.json().get("data", [])
        if models:
            print(f"  \u2705 /v1/models: {models[0].get('id', '?')}")
    except Exception as e:
        print(f"  \u26a0\ufe0f /v1/models check failed non-fatally: {e}")

    # Real end-to-end completion test -- /health=200 only confirms the
    # model loaded into memory, not that inference actually runs (e.g. a
    # CUDA layer-split issue could still fail at generation time).
    print("  \u231b Running a small end-to-end completion test...")
    try:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
            timeout=60,
        )
        r.raise_for_status()
        r.json()["choices"][0]["message"]["content"]
        print("  \u2705 Completion test succeeded -- inference is actually working")
    except Exception as e:
        raise RuntimeError(f"Server is up but a real completion request failed: {e}")

    return True


# ====================================================================
# MAIN
# ====================================================================

print("\u2554" + "\u2550" * 64 + "\u2557")
print("\u2551   \U0001f680  Collab-Llama \u00b7 llama.cpp direct (one-click)          \u2551")
print("\u255a" + "\u2550" * 64 + "\u255d")

r = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False, capture=True)
if r.returncode == 0:
    print(f"\n  \U0001f3ae  GPU    : {r.stdout.strip()}")
else:
    print("\n  \u26a0\ufe0f  No GPU — aktifkan GPU apa saja di: Runtime → Change runtime type")

# Auto-detect this runtime's GPU compute capability -- this is what makes
# switching GPU type (T4 <-> L4 <-> A100) in Colab's Runtime settings work
# without ever touching CUDA_ARCH by hand. Sets the module-level CUDA_ARCH
# used by verify_prebuilt() / build_from_source() below.
_detected = detect_gpu_arch()
if _detected:
    CUDA_ARCH = _detected[0]   # single-GPU on Colab -- first entry is it
    if len(set(_detected)) > 1:
        print(f"  \u26a0\ufe0f  Mixed GPU archs detected ({_detected}) -- unusual for Colab, using sm_{CUDA_ARCH}")
else:
    CUDA_ARCH = _FALLBACK_ARCH_IF_UNDETECTABLE
    print(f"  \u26a0\ufe0f  Could not auto-detect GPU arch via nvidia-smi -- defaulting to sm_{CUDA_ARCH}. "
          f"If this is wrong, the prebuilt/source-build steps below will fail loudly rather than silently misbuild.")

print(f"  \U0001f4e6  Model  : {MODEL_REPO}\n")

TOTAL_STEPS = 5
cf_proc = None
llama_proc = None

# Pre-cleanup in case this cell was re-run after a cancelled/crashed attempt
sh("pkill -9 -f 'llama-server'", check=False, quiet=True)
sh("pkill -9 -f cloudflared", check=False, quiet=True)
sh(f"fuser -k -9 {LLAMA_PORT}/tcp 2>/dev/null", check=False, quiet=True)
time.sleep(1)

try:
    # ── 1. LLAMA-SERVER BINARY ──────────────────────────────────
    section(1, TOTAL_STEPS, "llama-server binary (prebuilt-first, source fallback)")
    server_cmd = ensure_llama_server()

    # ── 2. MODEL DOWNLOAD ────────────────────────────────────────
    section(2, TOTAL_STEPS, "Model (HF/Xet, disk-efficient)")
    model_path = ensure_model()

    # ── 3. START LLAMA-SERVER ────────────────────────────────────
    section(3, TOTAL_STEPS, "Start llama-server")
    LLAMA_PORT = acquire_free_port(LLAMA_PORT)

    llama_log = "/tmp/llama_server.log"
    llama_cmd = (
        f"{server_cmd} "
        f"-m {model_path} "
        f"--host 0.0.0.0 --port {LLAMA_PORT} "
        f"--api-key {API_KEY} "
        f"--ctx-size {NUM_CTX} "
        f"-ngl 99 "
        f"--repeat-last-n {REPEAT_LAST_N} --repeat-penalty {REPEAT_PENALTY} "
        f"--presence-penalty {PRESENCE_PENALTY} --frequency-penalty {FREQUENCY_PENALTY} "
        f"--dry-multiplier {DRY_MULTIPLIER} --dry-base {DRY_BASE} "
        f"--dry-allowed-length {DRY_ALLOWED_LENGTH} --dry-penalty-last-n {DRY_PENALTY_LAST_N} "
        f"--jinja"
    )
    llama_proc = bg(llama_cmd, llama_log)
    wait_for_ready(llama_proc, llama_log, f"http://localhost:{LLAMA_PORT}", API_KEY, timeout=2700)

    # ── 4. CLOUDFLARE TUNNEL ─────────────────────────────────────
    section(4, TOTAL_STEPS, "Cloudflare Tunnel → Public HTTPS URL")
    sh(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared "
        "&& chmod +x /usr/local/bin/cloudflared"
    )
    cf_proc = bg(f"cloudflared tunnel --url http://localhost:{LLAMA_PORT}", "/tmp/cloudflared.log")

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

    if not tunnel_url:
        raise RuntimeError("Could not obtain a Cloudflare tunnel URL — check /tmp/cloudflared.log")
    print(f"  State: TUNNEL_READY ({tunnel_url})")

    # ── 5. READY ─────────────────────────────────────────────────
    section(5, TOTAL_STEPS, "Ready")
    gguf_size_gb = os.path.getsize(model_path) / (1024 ** 3)
    disk_free_gb = shutil.disk_usage(MODEL_DIR).free / (1024 ** 3)
    r = sh(f"{server_cmd} --version", check=False, capture=True)
    llama_ver = (r.stdout + r.stderr).strip().splitlines()[0] if r.returncode == 0 else "unknown"

    print()
    print("\u2554" + "\u2550" * 64 + "\u2557")
    print("\u2551   \u2705  SERVER IS READY                                            \u2551")
    print("\u255a" + "\u2550" * 64 + "\u255d")
    print(f"""
  llama.cpp    : {llama_ver}
  Model        : {MODEL_REPO}
  Model path   : {model_path} ({gguf_size_gb:.1f} GB)
  GPU          : {r.stdout.strip() if False else ''}
  Context      : {NUM_CTX}
  Sampler      : repeat_penalty={REPEAT_PENALTY} (last {REPEAT_LAST_N}) | dry_multiplier={DRY_MULTIPLIER}
  Disk free    : {disk_free_gb:.1f} GB
  Cloudflare   : {tunnel_url}
  API Key      : {API_KEY}

  ╔══════════════════════════════════════════════════════════════╗
  ║   Paste ke .env lokal kamu:                                  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║   BASE_URL="{tunnel_url}"
  ║   API_KEY="{API_KEY}"
  ╚══════════════════════════════════════════════════════════════╝

  ⚠️  Biarkan cell ini RUNNING selama memakai LLM.
  ⚠️  URL Cloudflare berubah setiap restart.
""")


    # ── KEEP ALIVE ──────────────────────────────────────────────
    tick = 0
    while True:
        time.sleep(60)
        tick += 1
        ts = time.strftime("%H:%M:%S")
        try:
            r = requests.get(f"http://localhost:{LLAMA_PORT}/health", timeout=5)
            status = "🟢 healthy" if r.status_code == 200 else f"⚠️  HTTP {r.status_code}"
        except Exception:
            status = "⚠️  unreachable"
        print(f"  [{ts}] heartbeat #{tick:04d} | server {status} | {tunnel_url}")

except KeyboardInterrupt:
    print("\n  🛑 Shutting down...")
    for proc in [cf_proc, llama_proc]:
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass
    print("  ✅ Stopped.")

except Exception as e:
    print(f"\n  ❌ Startup failed: {e}")
    for proc in [cf_proc, llama_proc]:
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass