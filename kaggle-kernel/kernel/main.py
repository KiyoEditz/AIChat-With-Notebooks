# -*- coding: utf-8 -*-
"""kaggle-kernel/kernel/main.py

Kaggle Kernel LLM Server for AIChat (beta-kaggle edition)
Active path: AIChat -> Cloudflare Tunnel -> Asyncio Gateway (8080) -> llama-server (8081) -> GGUF

Features:
1. Dual-T4 GPU auto-detection & prebuilt multi-arch llama-server binary (with source fallback).
2. Disk-efficient Hugging Face Xet GGUF download to /kaggle/tmp/models (supports optional HF_TOKEN).
3. Transparent Asyncio TCP/HTTP Gateway on GATEWAY_PORT (8080):
   - Forwards all OpenAI/llama-server requests & SSE streams with zero buffering overhead.
   - Tracks active inference streams so slow generation never triggers a false-positive idle shutdown.
   - Exposes GET  /__aichat_admin/heartbeat (updates activity timestamp).
   - Exposes POST /__aichat_admin/shutdown  (triggers immediate cleanup & kernel exit when AIChat stops).
4. Auto-commits the live Cloudflare Tunnel URL (tunnel_state.json) to a private Kaggle Dataset
   (<username>/aichat-kaggle-tunnel) without stopping the server.
5. 30-Minute Auto-Idle Watchdog (IDLE_TIMEOUT_SECONDS = 1800) to protect weekly Kaggle GPU quota.
"""

import asyncio
import atexit
import gc
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import requests

# =====================================================================
# INJECTED ORCHESTRATOR METADATA (populated in .staging/ before push)
# =====================================================================
RUN_ID = os.environ.get("AICHAT_RUN_ID", "__AICHAT_RUN_ID__")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "__KAGGLE_USERNAME__")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "__KAGGLE_KEY__")
TUNNEL_DATASET_SLUG = os.environ.get("AICHAT_TUNNEL_DATASET_SLUG", "aichat-kaggle-tunnel")

# =====================================================================
# SERVER & MODEL CONFIGURATION
# =====================================================================
MODEL_REPO = "mradermacher/Gemma-4-Dark-Gemistry-31B-i1-GGUF:Q6_K"  # ~27 GB
# Alternatives:
# MODEL_REPO = "mradermacher/EVA-Qwen2.5-32B-v0.2-i1-GGUF:Q6_K"
# MODEL_REPO = "bartowski/EVA-Qwen2.5-32B-v0.2-GGUF:Q4_K_M"
# MODEL_REPO = "bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M"

GATEWAY_PORT = 8080
LLAMA_INTERNAL_PORT = 8081
API_KEY = "sk-kaggle-local"

# Auto-Idle Watchdog: 30 minutes (1800 seconds)
IDLE_TIMEOUT_SECONDS = 1800

LLAMA_CPP_TAG = "b12076"
CUDA_ARCH = None
_FALLBACK_ARCH_IF_UNDETECTABLE = "75"

GITHUB_RELEASE_URL = (
    "https://github.com/KiyoEditz/AIChat-With-Notebooks/releases/download/"
    "v1.4.0/llama-cuda-multiarch-b12076.tar.gz"
)
GITHUB_RELEASES_API = "https://api.github.com/repos/KiyoEditz/AIChat-With-Notebooks/releases"

PARALLEL_SLOTS = 4
CTX_SLOT = 8192
NUM_CTX = PARALLEL_SLOTS * CTX_SLOT
CACHE_RAM = 16384
GPU_IDS = "0,1"

SCRATCH_DIR = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else "/tmp"
MODEL_DIR = f"{SCRATCH_DIR}/models"
XET_CACHE_DIR = f"{SCRATCH_DIR}/hf-xet"
LLAMA_SRC_DIR = f"{SCRATCH_DIR}/llama.cpp"
PREBUILT_DIR = f"{SCRATCH_DIR}/llama-prebuilt"

REPEAT_LAST_N, REPEAT_PENALTY = 256, 1.05
PRESENCE_PENALTY, FREQUENCY_PENALTY = 0.0, 0.0
DRY_MULTIPLIER, DRY_BASE = 0.8, 1.75
DRY_ALLOWED_LENGTH, DRY_PENALTY_LAST_N = 2, 1024

ENABLE_THINKING = False
MIN_FREE_DISK_MARGIN_GB = 5

START_TIME = time.time()
_gateway_state = {
    "last_activity": time.time(),
    "active_connections": 0,
    "shutdown_requested": False,
    "shutdown_reason": None,
}

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def sh(cmd, check=True, quiet=False, capture=False):
    kw = dict(shell=isinstance(cmd, str), check=check)
    if capture:
        kw.update(capture_output=True, text=True)
    elif quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, **kw)


def bg(cmd, log=None, env=None, cwd=None):
    out = open(log, "w") if log else subprocess.DEVNULL
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.Popen(
            cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, env=full_env, cwd=cwd
        )
    finally:
        if log and out != subprocess.DEVNULL:
            out.close()


def section(n, total, title):
    bar = "\u2500" * 64
    print(f"\n{bar}\n [{n}/{total}] {title}\n{bar}", flush=True)


def run_with_retry(fn, max_attempts=3, label="operation"):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait_s = attempt * 5
            print(f"  \u21bb {label} failed ({e}); retry {attempt}/{max_attempts} in {wait_s}s...", flush=True)
            time.sleep(wait_s)


def acquire_free_port(preferred_port, fallbacks=(8082, 8083, 8088, 18080, 18081)):
    candidates = [preferred_port] + [p for p in fallbacks if p != preferred_port]
    for p in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                if p != preferred_port:
                    print(f"  \u26a0\ufe0f Port {preferred_port} in use; switched to port {p}", flush=True)
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def detect_gpu_arch():
    r = sh(
        "nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader",
        check=False,
        capture=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return []
    archs = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, name, cap = parts[0], parts[1], parts[2]
            arch = cap.replace(".", "")
            archs.append(arch)
            print(f"  GPU {idx}: {name} -> compute capability {cap} (sm_{arch})", flush=True)
    return archs


def _arch_num(val):
    m = re.match(r"^(\d+)", str(val))
    return int(m.group(1)) if m else 0


def _parse_embedded_archs(archs_field):
    real, virtual_max = set(), None
    for tok in archs_field.replace("sm_", "").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("-real"):
            real.add(tok[:-len("-real")])
        elif tok.endswith("-virtual"):
            num = tok[:-len("-virtual")]
            if virtual_max is None or _arch_num(num) > _arch_num(virtual_max):
                virtual_max = num
        else:
            real.add(tok)
    return real, virtual_max


def check_dual_gpu_usage():
    r = sh(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader",
        check=False,
        capture=True,
    )
    if r.returncode != 0:
        print("  \u26a0\ufe0f Could not query nvidia-smi", flush=True)
        return
    print("  GPU memory usage:", flush=True)
    used_gpus = 0
    for line in [l for l in r.stdout.strip().splitlines() if l.strip()]:
        print(f"    {line}", flush=True)
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            used_mb = float(parts[2].replace("MiB", "").strip())
            if used_mb > 500:
                used_gpus += 1
    if used_gpus >= 2:
        print("  \u2705 Both GPUs are being used", flush=True)
    else:
        print("  \u26a0\ufe0f Single GPU active or model fits on 1 GPU.", flush=True)


# ------------------------------------------------------------------
# Cleanup handler
# ------------------------------------------------------------------

_procs = {"llama": None, "cloudflared": None}
_cleaned_up = False


def cleanup(*_args):
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    print("\n  \U0001f9f9 Cleaning up processes and freeing RAM/VRAM...", flush=True)
    for _name, proc in _procs.items():
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
    sh("pkill -9 -f '[l]lama-server'", check=False, quiet=True)
    sh("pkill -9 -f '[c]loudflared'", check=False, quiet=True)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    print("  \u2705 Cleanup complete -- RAM/VRAM released", flush=True)


atexit.register(cleanup)
signal.signal(signal.SIGTERM, cleanup)

sh("pkill -9 -f '[l]lama-server'", check=False, quiet=True)
sh("pkill -9 -f '[c]loudflared'", check=False, quiet=True)
time.sleep(1)


# ====================================================================
# STEP 1: llama-server binary (prebuilt-first, source-build fallback)
# ====================================================================

def ensure_prebuilt_cuda_libs(pkg_dir):
    """Ensure libcudart.so.XX, libcublas.so.XX, and libcublasLt.so.XX required by
    the prebuilt binary are present in pkg_dir/lib (bridges CUDA 12 prebuilts on
    Kaggle's CUDA 13 / Ubuntu 24.04 containers and vice versa)."""
    lib_dir = f"{pkg_dir}/lib"
    os.makedirs(lib_dir, exist_ok=True)
    llama_bin = f"{pkg_dir}/llama-server"

    def find_missing_cuda_sonames():
        ld_env = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        targets = [llama_bin] + glob.glob(f"{lib_dir}/libggml-cuda.so*")
        missing = set()
        for t in targets:
            if not os.path.isfile(t):
                continue
            r = sh(f'LD_LIBRARY_PATH="{ld_env}" ldd "{t}"', check=False, capture=True)
            if r.returncode != 0:
                continue
            for line in r.stdout.splitlines():
                if "=> not found" in line:
                    soname = line.split("=>")[0].strip()
                    if any(soname.startswith(p) for p in ("libcudart.so", "libcublas.so", "libcublasLt.so")):
                        missing.add(soname)
        return sorted(missing)

    missing = find_missing_cuda_sonames()
    if not missing:
        return

    print(f"  \u231b Resolving CUDA runtime libraries for prebuilt: {', '.join(missing)}...", flush=True)

    search_dirs = (
        glob.glob("/tmp/cuda-compat-cu*/nvidia/*/lib")
        + glob.glob("/usr/local/lib/python*/*-packages/nvidia/*/lib")
        + glob.glob(f"{sys.prefix}/lib/python*/*-packages/nvidia/*/lib")
        + glob.glob("/opt/conda/lib/python*/site-packages/nvidia/*/lib")
        + glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib")
        + glob.glob("/usr/local/cuda*/lib64")
        + ["/opt/conda/lib", "/usr/lib/x86_64-linux-gnu"]
    )

    def link_from_dirs(dirs_to_scan):
        for d in dirs_to_scan:
            if not os.path.isdir(d):
                continue
            for f_name in os.listdir(d):
                if any(f_name.startswith(p) for p in ("libcudart.so", "libcublas.so", "libcublasLt.so")):
                    src = os.path.join(d, f_name)
                    dst = os.path.join(lib_dir, f_name)
                    if os.path.isfile(src) and not os.path.lexists(dst):
                        try:
                            os.symlink(os.path.realpath(src), dst)
                        except Exception:
                            shutil.copy2(src, dst)

    link_from_dirs(search_dirs)
    missing = find_missing_cuda_sonames()
    if not missing:
        print("  \u2705 Linked CUDA runtime libraries from local environment.", flush=True)
        return

    cu_major = "12"
    for m_name in missing:
        parts = m_name.split(".so.")
        if len(parts) >= 2 and parts[1].split(".")[0].isdigit():
            cu_major = parts[1].split(".")[0]
            break

    compat_dir = f"/tmp/cuda-compat-cu{cu_major}"
    pkgs = [f"nvidia-cuda-runtime-cu{cu_major}", f"nvidia-cublas-cu{cu_major}"]
    print(f"  \u231b Installing {', '.join(pkgs)} for prebuilt compatibility...", flush=True)
    sh(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            compat_dir,
        ]
        + pkgs,
        check=False,
    )
    link_from_dirs(glob.glob(f"{compat_dir}/nvidia/*/lib"))
    if not find_missing_cuda_sonames():
        print(f"  \u2705 Installed and linked CUDA {cu_major} runtime libraries.", flush=True)


def verify_prebuilt(pkg_dir):
    version_file = f"{pkg_dir}/VERSION.txt"
    if not os.path.isfile(version_file):
        raise RuntimeError("prebuilt package has no VERSION.txt")

    meta = {}
    for line in open(version_file):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            meta[k] = v

    if meta.get("llama_cpp_tag") != LLAMA_CPP_TAG:
        raise RuntimeError(f"prebuilt is tag {meta.get('llama_cpp_tag')!r}, expected {LLAMA_CPP_TAG!r}")

    archs_field = meta.get("cuda_archs") or meta.get("cuda_arch", "")
    real_archs, virtual_max = _parse_embedded_archs(archs_field)
    if not real_archs and not virtual_max:
        raise RuntimeError("prebuilt VERSION.txt has no cuda_archs/cuda_arch field")

    covered_desc = f"sm_{{{','.join(sorted(real_archs))}}}" + (
        f" (+PTX from sm_{virtual_max})" if virtual_max else ""
    )
    if not CUDA_ARCH:
        raise RuntimeError("GPU arch could not be detected")
    elif CUDA_ARCH in real_archs:
        print(f"  GPU sm_{CUDA_ARCH}: native match in prebuilt ({covered_desc})", flush=True)
    elif virtual_max and _arch_num(CUDA_ARCH) >= _arch_num(virtual_max):
        print(f"  GPU sm_{CUDA_ARCH}: covered via PTX JIT from sm_{virtual_max}", flush=True)
    else:
        raise RuntimeError(f"prebuilt covers {covered_desc}, this GPU is sm_{CUDA_ARCH}")

    run_sh = f"{pkg_dir}/run.sh"
    if not os.path.isfile(run_sh):
        raise RuntimeError("prebuilt package has no run.sh launcher")
    os.chmod(run_sh, 0o755)
    os.chmod(f"{pkg_dir}/llama-server", 0o755)

    ensure_prebuilt_cuda_libs(pkg_dir)

    r = sh(f"{run_sh} --version", check=False, capture=True)
    if r.returncode != 0:
        raise RuntimeError(f"prebuilt binary failed --version: {r.stderr[:200]}")

    print(f"  \u2705 Prebuilt verified: {meta.get('llama_cpp_tag')} / {covered_desc}", flush=True)
    return run_sh


def resolve_prebuilt_url():
    """Return a valid download URL for llama-cuda-multiarch-{LLAMA_CPP_TAG}.tar.gz.
    Checks GITHUB_RELEASE_URL first; if it 404s (e.g. uploaded under a newer release
    tag on KiyoEditz/AIChat-With-Notebooks), auto-discovers the asset via GitHub API."""
    wanted_name = f"llama-cuda-multiarch-{LLAMA_CPP_TAG}.tar.gz"
    if GITHUB_RELEASE_URL:
        try:
            r = requests.head(GITHUB_RELEASE_URL, allow_redirects=True, timeout=15)
            if r.status_code == 200:
                return GITHUB_RELEASE_URL
        except Exception:
            pass

    try:
        r = requests.get(GITHUB_RELEASES_API, timeout=15)
        if r.status_code == 200:
            for rel in r.json():
                for asset in rel.get("assets", []):
                    if asset.get("name") == wanted_name and asset.get("browser_download_url"):
                        url = asset["browser_download_url"]
                        print(f"  \u2705 Auto-discovered prebuilt in release {rel.get('tag_name')}: {wanted_name}", flush=True)
                        return url
    except Exception:
        pass

    if GITHUB_RELEASE_URL:
        return GITHUB_RELEASE_URL
    raise RuntimeError("GITHUB_RELEASE_URL not set and no matching release asset found")


def try_prebuilt():
    url = resolve_prebuilt_url()

    os.makedirs(PREBUILT_DIR, exist_ok=True)
    tarball_path = f"{PREBUILT_DIR}/prebuilt.tar.gz"
    print(f"  \u231b Downloading prebuilt binary ({os.path.basename(url)})...", flush=True)

    def download_step():
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(tarball_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    run_with_retry(download_step, label="prebuilt download")

    extract_dir = f"{PREBUILT_DIR}/extracted"
    sh(f"rm -rf {extract_dir}", check=False, quiet=True)
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball_path) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(extract_dir, filter="data")
        else:
            tar.extractall(extract_dir)

    entries = [d for d in glob.glob(f"{extract_dir}/*") if os.path.isdir(d)]
    if not entries:
        raise RuntimeError("prebuilt tarball had no top-level directory")
    return verify_prebuilt(entries[0])


def find_cuda_driver_lib():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    dirs = []
    for base in [cuda_home, "/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda*")):
        dirs += [
            f"{base}/lib64/stubs",
            f"{base}/lib/stubs",
            f"{base}/targets/x86_64-linux/lib/stubs",
            f"{base}/lib64",
            f"{base}/targets/x86_64-linux/lib",
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

    for d in unique_dirs:
        p = f"{d}/libcuda.so"
        if os.path.exists(p):
            return p, False

    versioned = []
    for d in unique_dirs:
        if os.path.isfile(f"{d}/libcuda.so.1"):
            versioned.append(f"{d}/libcuda.so.1")
        versioned += sorted(glob.glob(f"{d}/libcuda.so.*"))

    for p in versioned:
        if os.path.isfile(p):
            return p, True
    return None, False


def cuda_driver_cmake_flags():
    path, _is_versioned = find_cuda_driver_lib()
    if not path:
        return []
    link_dir = "/tmp/cuda-driver-link"
    os.makedirs(link_dir, exist_ok=True)
    for name in ("libcuda.so", "libcuda.so.1"):
        dst = f"{link_dir}/{name}"
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(path), dst)
    curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if link_dir not in curr_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{link_dir}:{curr_ld}" if curr_ld else link_dir
    rpath_flag = f"-Wl,-rpath-link,{link_dir} -Wl,--allow-shlib-undefined"
    return [
        f"-DCMAKE_LIBRARY_PATH={link_dir}",
        f"-DCUDA_cuda_driver_LIBRARY={link_dir}/libcuda.so",
        f"-DCMAKE_EXE_LINKER_FLAGS={rpath_flag}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={rpath_flag}",
    ]


def build_from_source():
    server_bin = f"{LLAMA_SRC_DIR}/build/bin/llama-server"
    if os.path.isfile(server_bin):
        return server_bin

    shutil.rmtree("/tmp/cuda-toolkit", ignore_errors=True)
    for _env_key in ("PATH", "LD_LIBRARY_PATH"):
        if _env_key in os.environ:
            os.environ[_env_key] = ":".join(
                p for p in os.environ[_env_key].split(":") if p and "/tmp/cuda-toolkit" not in p
            )
    if os.path.isdir("/usr/local/cuda"):
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["CUDA_PATH"] = "/usr/local/cuda"
        os.environ["CUDAToolkit_ROOT"] = "/usr/local/cuda"

    sh("apt-get update -qq && apt-get install -y -qq cmake build-essential git", check=False, quiet=True)

    def clone_step():
        sh(f"rm -rf {LLAMA_SRC_DIR}", check=False, quiet=True)
        sh(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                LLAMA_CPP_TAG,
                "https://github.com/ggml-org/llama.cpp.git",
                LLAMA_SRC_DIR,
            ]
        )

    run_with_retry(clone_step, label="git clone")
    driver_flags = cuda_driver_cmake_flags()

    def configure(extra_flags):
        sh(f"rm -rf {LLAMA_SRC_DIR}/build", check=False, quiet=True)
        cmd = [
            "cmake",
            "-B",
            f"{LLAMA_SRC_DIR}/build",
            "-S",
            LLAMA_SRC_DIR,
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ]
        if os.path.isdir("/usr/local/cuda"):
            cmd.append("-DCUDAToolkit_ROOT=/usr/local/cuda")
        if os.path.isfile("/usr/local/cuda/bin/nvcc"):
            cmd.append("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
        cmd += extra_flags
        return sh(cmd, check=False).returncode == 0

    def compile_step():
        nproc = min(os.cpu_count() or 4, 4)
        return (
            sh(
                [
                    "cmake",
                    "--build",
                    f"{LLAMA_SRC_DIR}/build",
                    "--config",
                    "Release",
                    f"-j{nproc}",
                    "--target",
                    "llama-server",
                ],
                check=False,
            ).returncode
            == 0
        )

    vmm_ok = bool(driver_flags) and configure(driver_flags)
    if not vmm_ok or not compile_step():
        if not configure(["-DGGML_CUDA_NO_VMM=ON"]) or not compile_step():
            raise RuntimeError("CMake configure/build failed")

    if not os.path.isfile(server_bin):
        raise RuntimeError(f"Build finished but {server_bin} was not produced")
    return server_bin


def ensure_llama_server():
    if GITHUB_RELEASE_URL:
        try:
            return try_prebuilt()
        except Exception as e:
            print(f"  \u26a0\ufe0f Prebuilt unusable ({e}) -- falling back to source build.", flush=True)
    return build_from_source()


# ====================================================================
# STEP 2: Model download (HF/Xet, disk-efficient)
# ====================================================================

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(XET_CACHE_DIR, exist_ok=True)

    os.environ.setdefault("HF_XET_CACHE", XET_CACHE_DIR)
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1")

    sh("pip install -q -U huggingface_hub hf_xet", check=False, quiet=True)
    from huggingface_hub import snapshot_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except Exception:
            pass

    if hf_token:
        print("  \u2705 Found HF_TOKEN, logging in to Hugging Face...", flush=True)
        login(token=hf_token)

    repo_id, _, quant = MODEL_REPO.partition(":")
    patterns = [f"*{quant}*.gguf"] if quant else ["*.gguf"]

    print(f"  Repo   : {repo_id}", flush=True)
    print(f"  Quant  : {quant or '(any)'}", flush=True)

    def dry_run_step():
        return snapshot_download(
            repo_id=repo_id, local_dir=MODEL_DIR, allow_patterns=patterns, dry_run=True, token=hf_token
        )

    plan = run_with_retry(dry_run_step, label="HF metadata fetch")
    to_download = [f for f in plan if f.will_download]

    if not to_download:
        print("  \u2705 Model already present in MODEL_DIR", flush=True)
    else:
        total_bytes = sum(f.file_size for f in to_download)
        total_gb = total_bytes / (1024 ** 3)
        free_gb = shutil.disk_usage(MODEL_DIR).free / (1024 ** 3)
        print(f"  Need to download: {len(to_download)} file(s), {total_gb:.1f} GB (Free: {free_gb:.1f} GB)", flush=True)

        if free_gb < total_gb + MIN_FREE_DISK_MARGIN_GB:
            raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free, need {total_gb:.1f} GB")

        def download_step():
            snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR, allow_patterns=patterns, token=hf_token)

        print("  \u231b Downloading model...", flush=True)
        run_with_retry(download_step, max_attempts=3, label="model download")

    planned_ggufs = sorted(
        os.path.join(MODEL_DIR, f.filename)
        for f in plan
        if f.filename.lower().endswith(".gguf") and os.path.isfile(os.path.join(MODEL_DIR, f.filename))
    )
    ggufs = planned_ggufs or sorted(glob.glob(f"{MODEL_DIR}/**/*.gguf", recursive=True))
    if not ggufs:
        raise RuntimeError(f"No .gguf files found in {MODEL_DIR}")

    shard1 = [g for g in ggufs if "-00001-of-" in g]
    entrypoint = shard1[0] if shard1 else ggufs[0]
    print(f"  Model file: {entrypoint}", flush=True)
    return entrypoint


# ====================================================================
# STEP 3: Readiness check against internal llama-server
# ====================================================================

def wait_for_ready(proc, log_path, base_url, api_key, timeout=3000):
    start = time.time()
    last_print = 0.0
    state = "PROCESS_STARTED"
    print(f"  State: {state}", flush=True)

    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = ""
            try:
                with open(log_path) as f:
                    tail = f.read()[-1500:]
            except Exception:
                pass
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}).\n{tail}")

        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("  State: HEALTH_READY (/health confirms model loaded)", flush=True)
                break
            elif r.status_code == 503 and state != "MODEL_LOADING":
                state = "MODEL_LOADING"
                print(f"  State: {state} (loading weights across GPUs...)", flush=True)
        except Exception:
            pass

        now = time.time()
        if now - last_print >= 15:
            elapsed = int(now - start)
            print(f"  \u231b [{elapsed}s] {state}...", flush=True)
            last_print = now
        time.sleep(2)
    else:
        raise RuntimeError(f"Timed out after {timeout}s waiting for llama-server readiness")

    print("  \u231b Running end-to-end completion self-test...", flush=True)
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    r.raise_for_status()
    print("  \u2705 Completion self-test succeeded", flush=True)
    return True


# ====================================================================
# STEP 4: Asyncio TCP/HTTP Gateway (Heartbeat, Shutdown, Stream Proxy)
# ====================================================================

def _send_http_json(writer, status_code, status_text, payload):
    body = json.dumps(payload).encode("utf-8")
    headers = (
        f"HTTP/1.1 {status_code} {status_text}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8")
    writer.write(headers + body)


async def _pipe_stream(reader, writer, track_activity=True):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            if track_activity:
                _gateway_state["last_activity"] = time.time()
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle_gateway_client(client_reader, client_writer, llama_port, api_key):
    try:
        initial = await asyncio.wait_for(client_reader.read(8192), timeout=30.0)
    except Exception:
        client_writer.close()
        return

    if not initial:
        client_writer.close()
        return

    _gateway_state["last_activity"] = time.time()

    # Inspect HTTP request line & headers if present
    header_End = initial.find(b"\r\n\r\n")
    header_bytes = initial[:header_End] if header_End != -1 else initial[:1024]
    header_text = header_bytes.decode("latin-1", errors="ignore")
    lines = header_text.splitlines()
    first_line = lines[0] if lines else ""
    parts = first_line.split(" ")
    method = parts[0].upper() if len(parts) >= 1 else ""
    path = parts[1] if len(parts) >= 2 else ""

    # Check admin control routes
    if path.startswith("/__aichat_admin/"):
        auth_ok = False
        for line in lines[1:]:
            if line.lower().startswith("authorization:"):
                val = line.split(":", 1)[1].strip()
                if val == f"Bearer {api_key}" or val == api_key:
                    auth_ok = True
                    break

        if path.startswith("/__aichat_admin/heartbeat"):
            _gateway_state["last_activity"] = time.time()
            _send_http_json(
                client_writer,
                200,
                "OK",
                {
                    "ok": True,
                    "run_id": RUN_ID,
                    "uptime_seconds": int(time.time() - START_TIME),
                    "active_connections": _gateway_state["active_connections"],
                },
            )
            await client_writer.drain()
            client_writer.close()
            return

        if path.startswith("/__aichat_admin/shutdown"):
            if not auth_ok:
                _send_http_json(client_writer, 401, "Unauthorized", {"ok": False, "error": "Invalid API Key"})
                await client_writer.drain()
                client_writer.close()
                return

            _send_http_json(
                client_writer,
                200,
                "OK",
                {"ok": True, "message": "Shutdown signal received. Stopping Kaggle kernel..."},
            )
            await client_writer.drain()
            client_writer.close()
            _gateway_state["shutdown_reason"] = "Received /__aichat_admin/shutdown from AIChat server"
            _gateway_state["shutdown_requested"] = True
            return

    # Standard OpenAI / llama-server proxy path
    _gateway_state["active_connections"] += 1
    llama_writer = None
    try:
        llama_reader, llama_writer = await asyncio.open_connection("127.0.0.1", llama_port)
        llama_writer.write(initial)
        await llama_writer.drain()
        await asyncio.gather(
            _pipe_stream(client_reader, llama_writer, track_activity=True),
            _pipe_stream(llama_reader, client_writer, track_activity=True),
        )
    except Exception:
        try:
            client_writer.close()
        except Exception:
            pass
        if llama_writer is not None:
            try:
                llama_writer.close()
            except Exception:
                pass
    finally:
        _gateway_state["active_connections"] = max(0, _gateway_state["active_connections"] - 1)
        _gateway_state["last_activity"] = time.time()


def start_gateway_background(gateway_port, llama_port, api_key):
    async def _runner():
        server = await asyncio.start_server(
            lambda r, w: _handle_gateway_client(r, w, llama_port, api_key),
            host="0.0.0.0",
            port=gateway_port,
        )
        async with server:
            await server.serve_forever()

    t = threading.Thread(target=lambda: asyncio.run(_runner()), daemon=True)
    t.start()
    return t


# ====================================================================
# STEP 5: Publish Cloudflare Tunnel URL to Kaggle Private Dataset
# ====================================================================

def publish_tunnel_to_dataset(tunnel_url):
    username = (KAGGLE_USERNAME or "").strip()
    key = (KAGGLE_KEY or "").strip()
    if not username or username.startswith("__KAGGLE_"):
        print("  \u26a0\ufe0f No KAGGLE_USERNAME injected -- skipping dataset URL publish.", flush=True)
        return False

    ds_ref = f"{username}/{TUNNEL_DATASET_SLUG}"
    ds_dir = "/tmp/aichat_tunnel_dataset"
    shutil.rmtree(ds_dir, ignore_errors=True)
    os.makedirs(ds_dir, exist_ok=True)

    # Configure Kaggle credentials in environment & ~/.kaggle/kaggle.json
    os.environ["KAGGLE_USERNAME"] = username
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key
    else:
        os.environ["KAGGLE_KEY"] = key

    kaggle_cfg_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_cfg_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_cfg_dir, "kaggle.json")
    with open(kaggle_json_path, "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json_path, 0o600)

    state_payload = {
        "run_id": RUN_ID,
        "tunnel_url": tunnel_url,
        "api_key": API_KEY,
        "model_repo": MODEL_REPO,
        "updated_at": int(time.time()),
        "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(ds_dir, "tunnel_state.json"), "w", encoding="utf-8") as f:
        json.dump(state_payload, f, indent=2)

    meta_payload = {
        "title": "AIChat Kaggle Tunnel State",
        "id": ds_ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(ds_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"  \u231b Publishing tunnel URL to private dataset: {ds_ref}...", flush=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        exists = False
        try:
            api.dataset_status(ds_ref)
            exists = True
        except Exception:
            exists = False

        if exists:
            api.dataset_create_version(
                ds_dir,
                version_notes=f"run {RUN_ID} -> {tunnel_url}",
                quiet=True,
                dir_mode="skip",
            )
        else:
            api.dataset_create_new(
                ds_dir,
                public=False,
                quiet=True,
                dir_mode="skip",
            )
        print(f"  \u2705 Published tunnel_state.json to {ds_ref} (run_id={RUN_ID})", flush=True)
        return True
    except Exception as e:
        print(f"  \u26a0\ufe0f Failed to publish via KaggleApi ({e}), trying CLI fallback...", flush=True)
        r = sh(
            f'kaggle datasets version -p "{ds_dir}" -m "run {RUN_ID}" --dir-mode skip',
            check=False,
            capture=True,
        )
        if r.returncode != 0:
            r = sh(
                f'kaggle datasets create -p "{ds_dir}" --dir-mode skip',
                check=False,
                capture=True,
            )
        if r.returncode == 0:
            print(f"  \u2705 Published tunnel_state.json via CLI to {ds_ref}", flush=True)
            return True
        print(f"  \u274c Could not publish tunnel_state.json: {r.stderr}", flush=True)
        return False


# ====================================================================
# MAIN EXECUTION
# ====================================================================

print("\u2554" + "\u2550" * 64 + "\u2557")
print("\u2551  AIChat Kaggle Kernel (Beta Auto-Tunnel + 30m Watchdog)" + " " * 8 + "\u2551")
print("\u255a" + "\u2550" * 64 + "\u255d", flush=True)

r = sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", check=False, capture=True)
if r.returncode == 0 and r.stdout.strip():
    print(f"\n  GPUs detected:\n{r.stdout.strip()}", flush=True)
    gpu_count = len([l for l in r.stdout.strip().splitlines() if l.strip()])
    if gpu_count < 2:
        GPU_IDS = "0"
        print("  \u2139\ufe0f Single GPU runtime detected -- using GPU_IDS='0'", flush=True)
else:
    print("\n  \u26a0\ufe0f No GPU detected!", flush=True)

_detected = detect_gpu_arch()
CUDA_ARCH = _detected[0] if _detected else _FALLBACK_ARCH_IF_UNDETECTABLE

print(f"\n  Run ID       : {RUN_ID}", flush=True)
print(f"  Model        : {MODEL_REPO}", flush=True)
print(f"  GPUs         : {GPU_IDS}", flush=True)
print(f"  Idle Timeout : {IDLE_TIMEOUT_SECONDS // 60} minutes\n", flush=True)

TOTAL_STEPS = 6

try:
    # -- 1. LLAMA-SERVER BINARY ------------------------------------------
    section(1, TOTAL_STEPS, "llama-server binary")
    server_cmd = ensure_llama_server()

    # -- 2. MODEL DOWNLOAD ------------------------------------------------
    section(2, TOTAL_STEPS, "Model download (HF/Xet)")
    model_path = ensure_model()

    # -- 3. START LLAMA-SERVER & GATEWAY ----------------------------------
    section(3, TOTAL_STEPS, "Start llama-server & Asyncio Gateway")
    GATEWAY_PORT = acquire_free_port(GATEWAY_PORT)
    LLAMA_INTERNAL_PORT = acquire_free_port(LLAMA_INTERNAL_PORT, fallbacks=(8082, 8083, 18081, 18082))

    llama_log = "/tmp/llama_server.log"
    llama_env = {"CUDA_VISIBLE_DEVICES": GPU_IDS}
    reasoning_flags = (
        "--reasoning off --chat-template-kwargs '{\"enable_thinking\": false}'"
        if not ENABLE_THINKING
        else "--reasoning on"
    )
    llama_cmd = (
        f"{server_cmd} "
        f"-m {model_path} "
        f"--host 127.0.0.1 --port {LLAMA_INTERNAL_PORT} "
        f"--api-key {API_KEY} "
        f"--ctx-size {NUM_CTX} "
        f"--parallel {PARALLEL_SLOTS} "
        f"--load-mode none "
        f"--cache-ram {CACHE_RAM} "
        f"--flash-attn on -ctk q8_0 -ctv q8_0 "
        f"-ngl 99 --split-mode layer "
        f"--repeat-last-n {REPEAT_LAST_N} --repeat-penalty {REPEAT_PENALTY} "
        f"--presence-penalty {PRESENCE_PENALTY} --frequency-penalty {FREQUENCY_PENALTY} "
        f"--dry-multiplier {DRY_MULTIPLIER} --dry-base {DRY_BASE} "
        f"--dry-allowed-length {DRY_ALLOWED_LENGTH} --dry-penalty-last-n {DRY_PENALTY_LAST_N} "
        f"--jinja "
        f"{reasoning_flags}"
    )
    _procs["llama"] = bg(llama_cmd, llama_log, env=llama_env)
    wait_for_ready(_procs["llama"], llama_log, f"http://127.0.0.1:{LLAMA_INTERNAL_PORT}", API_KEY, timeout=3000)

    start_gateway_background(GATEWAY_PORT, LLAMA_INTERNAL_PORT, API_KEY)
    print(f"  \u2705 Asyncio Gateway listening on 0.0.0.0:{GATEWAY_PORT} -> 127.0.0.1:{LLAMA_INTERNAL_PORT}", flush=True)

    # -- 4. VERIFY GPU SPLIT ----------------------------------------------
    section(4, TOTAL_STEPS, "Verify GPU Split")
    time.sleep(2)
    check_dual_gpu_usage()

    # -- 5. CLOUDFLARE TUNNEL & DATASET PUBLISH ---------------------------
    section(5, TOTAL_STEPS, "Cloudflare Tunnel & Private Dataset Publish")
    sh(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared "
        "&& chmod +x /usr/local/bin/cloudflared"
    )
    _procs["cloudflared"] = bg(
        f"cloudflared tunnel --url http://127.0.0.1:{GATEWAY_PORT}", "/tmp/cloudflared.log"
    )
    print("  \u231b Waiting for public Cloudflare URL...", flush=True)
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
        if _procs["cloudflared"].poll() is not None:
            break

    if not tunnel_url:
        raise RuntimeError("Failed to establish Cloudflare tunnel within 90s.")

    print(f"  \u2705 Tunnel established: {tunnel_url}", flush=True)
    publish_tunnel_to_dataset(tunnel_url)

    # -- 6. READY & KEEP-ALIVE WATCHDOG -----------------------------------
    section(6, TOTAL_STEPS, "Ready & Active Watchdog")
    _gateway_state["last_activity"] = time.time()

    print(
        f"""
  ================================================================
  SERVER READY
  Run ID       : {RUN_ID}
  BASE_URL     : {tunnel_url}
  API_KEY      : {API_KEY}
  Auto-Idle    : {IDLE_TIMEOUT_SECONDS // 60} minutes
  ================================================================
""",
        flush=True,
    )

    last_log_ts = time.time()
    tick = 0
    while True:
        time.sleep(2)

        # 1. Immediate shutdown if requested by AIChat (Ctrl+C)
        if _gateway_state["shutdown_requested"]:
            print(f"\n  \U0001f6d1 Shutdown triggered: {_gateway_state['shutdown_reason']}", flush=True)
            cleanup()
            sys.exit(0)

        # 2. Do not count idle time while active inference/streaming connections exist
        if _gateway_state["active_connections"] > 0:
            _gateway_state["last_activity"] = time.time()

        # 3. Check 30-minute Auto-Idle Watchdog
        idle_sec = time.time() - _gateway_state["last_activity"]
        if idle_sec >= IDLE_TIMEOUT_SECONDS:
            print(
                f"\n  \u23f0 Auto-Idle Watchdog triggered: No heartbeat or chat activity for "
                f"{int(idle_sec)}s ({IDLE_TIMEOUT_SECONDS // 60}m limit). Stopping kernel to save GPU quota.",
                flush=True,
            )
            cleanup()
            sys.exit(0)

        # 4. Periodic status log every 60 seconds
        if time.time() - last_log_ts >= 60:
            last_log_ts = time.time()
            tick += 1
            ts = time.strftime("%H:%M:%S")
            try:
                r = requests.get(f"http://127.0.0.1:{LLAMA_INTERNAL_PORT}/health", timeout=5)
                status = "healthy" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception:
                status = "unreachable"
            print(
                f"  [{ts}] tick #{tick:04d} | llama={status} | active_conns={_gateway_state['active_connections']} "
                f"| idle={int(idle_sec)}s/{IDLE_TIMEOUT_SECONDS}s | {tunnel_url}",
                flush=True,
            )

except KeyboardInterrupt:
    cleanup()
except Exception as e:
    print(f"\n  \u274c Startup failed: {e}", flush=True)
    cleanup()
    sys.exit(1)
