# -*- coding: utf-8 -*-
"""Build-Llama-CUDA-Release.py

One-time builder: produces a portable, self-contained llama-server (CUDA)
tarball you upload to your own GitHub Release, so future Kaggle/Colab runs
can download a prebuilt binary instead of compiling from source every time.

## Multi-GPU support (fat binary, no more per-arch rebuilds)
This build embeds native (SASS) code for EVERY GPU generation these
notebooks might run on -- T4, A100, RTX 30xx/40xx, L4, H100 -- in a single
tarball, plus a PTX ("virtual") fallback for the newest arch so even GPUs
released after this build (e.g. Blackwell RTX 50xx / B200) can still run
it via the driver's PTX JIT instead of needing a rebuild.

## Build Time Note (Kaggle 4-Core CPU)
- Building this full multi-arch fat binary (5 SASS architectures + PTX) takes
  approximately **1 hour 30 minutes (~90 minutes)** on Kaggle's 4-core CPU
  because nvcc compiles every CUDA kernel template 5 times.
- You only need to run this ONCE per LLAMA_CPP_TAG upgrade; after uploading
  the resulting tarball to GitHub Releases, all future Kaggle/Colab sessions
  download and verify it in ~10 seconds.

## What you do with the output
1. Run this cell on Kaggle (Settings -> Accelerator -> GPU T4 x2 or T4 x1).
2. When it finishes (~1 hour 30 minutes on 4-core Kaggle CPU), a tarball +
   .sha256 file sit in /kaggle/working/. Download both from the Kaggle
   notebook's Output panel.
3. On GitHub: your repo -> Releases -> attach both files as release assets ->
   Publish.
"""

import subprocess
import time
import os
import sys
import shutil
import hashlib
import tarfile
import glob

# =====================================================================
# CONFIGURATION -- keep this in sync with Collab-Llama.py / Kaggle-Llama.py
# =====================================================================

# Latest official llama.cpp release tag (v0.8.0 / b12076)
LLAMA_CPP_TAG = "b12076"

# Fat multi-arch build:
#   75 = Turing     (T4, RTX 20xx)
#   80 = Ampere DC  (A100)
#   86 = Ampere     (RTX 30xx, A10/A40)
#   89 = Ada        (L4, RTX 40xx)
#   90 = Hopper     (H100) + PTX virtual for forward-compat
# NOTE: Compiling all 5 architectures + PTX takes ~1 hour 30 minutes (~90 min)
# on Kaggle's 4-core CPU. This is a one-time build per release tag.
CUDA_ARCH = "75-real;80-real;86-real;89-real;90-real;90-virtual"
CUDA_ARCH_HUMAN = (
    "Turing(T4/RTX20xx)=75 Ampere-DC(A100)=80 "
    "Ampere(RTX30xx)=86 Ada(L4/RTX40xx)=89 Hopper(H100)=90 "
    "PTX-forward-compat-from=90"
)

# Final tarball + sha256 go to OUTPUT_DIR (/kaggle/working on Kaggle so they
# appear in the notebook Output tab).
if os.path.isdir("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working"
elif os.path.isdir("/content"):
    OUTPUT_DIR = "/content"
else:
    OUTPUT_DIR = "/tmp"

# Intermediate build tree goes to scratch space (/kaggle/tmp or /tmp) so
# multi-arch .cu.o files never exhaust /kaggle/working's 19.5 GB quota.
SCRATCH_DIR = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else "/tmp"
BUILD_DIR = f"{SCRATCH_DIR}/llama-build-{LLAMA_CPP_TAG}"

# Clean up any leftover /tmp/cuda-toolkit state in os.environ or on disk from
# earlier cell runs in the same Jupyter kernel session.
shutil.rmtree("/tmp/cuda-toolkit", ignore_errors=True)
for _env_key in ("PATH", "LD_LIBRARY_PATH"):
    if _env_key in os.environ:
        os.environ[_env_key] = ":".join(
            p for p in os.environ[_env_key].split(":") if p and "/tmp/cuda-toolkit" not in p
        )
os.environ.pop("NVCC_APPEND_FLAGS", None)
if os.path.isdir("/usr/local/cuda"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["CUDAToolkit_ROOT"] = "/usr/local/cuda"


# =====================================================================
# Utility functions (Jupyter-friendly output streaming & diagnostics)
# =====================================================================

def sh(cmd, check=True, quiet=False, capture=False):
    is_str = isinstance(cmd, str)
    if capture:
        return subprocess.run(cmd, shell=is_str, check=check, capture_output=True, text=True)
    if quiet:
        return subprocess.run(
            cmd,
            shell=is_str,
            check=check,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Stream output through Python's sys.stdout so Jupyter/Kaggle notebooks
    # always display compiler/CMake errors instead of swallowing them.
    proc = subprocess.Popen(
        cmd,
        shell=is_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail_lines.append(line)
        if len(tail_lines) > 80:
            tail_lines.pop(0)
    proc.wait()
    if check and proc.returncode != 0:
        print("\n" + "!" * 64, flush=True)
        print(f"  ❌ Command failed with exit code {proc.returncode}: {cmd}", flush=True)
        print("  Last output lines:", flush=True)
        for l in tail_lines[-40:]:
            print("    " + l, end="", flush=True)
        print("!" * 64 + "\n", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output="".join(tail_lines))
    return subprocess.CompletedProcess(cmd, proc.returncode)


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


def find_cuda_driver_lib():
    """Locate a libcuda.so or libcuda.so.1 (or CUDA toolkit stub libcuda.so)
    so CMake's CUDA::cuda_driver target and GNU ld's transitive DT_NEEDED
    check on libcuda.so.1 both succeed."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    dirs = []
    for base in [cuda_home, "/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda*"), reverse=True):
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

    # 1. Prefer real versioned driver libcuda.so.1 first if present on GPU host
    for d in unique_dirs:
        p = f"{d}/libcuda.so.1"
        if os.path.isfile(p):
            return os.path.realpath(p)
        for cand in sorted(glob.glob(f"{d}/libcuda.so.*")):
            if os.path.isfile(cand):
                return os.path.realpath(cand)

    # 2. Check ldconfig cache
    r = sh("ldconfig -p", check=False, capture=True)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if "libcuda.so" in line and "=>" in line:
                cand = line.split("=>")[-1].strip()
                if os.path.isfile(cand):
                    return os.path.realpath(cand)

    # 3. Check unversioned libcuda.so (e.g. /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so)
    for d in unique_dirs:
        p = f"{d}/libcuda.so"
        if os.path.isfile(p):
            return os.path.realpath(p)

    # 4. Last-resort filesystem search
    r = sh("find /usr/local /usr/lib /lib -name 'libcuda.so*' 2>/dev/null", check=False, capture=True)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            cand = line.strip()
            if os.path.isfile(cand):
                return os.path.realpath(cand)

    return None


def cuda_driver_cmake_flags():
    """Create /tmp/cuda-driver-link containing BOTH libcuda.so and libcuda.so.1
    symlinks, and return CMake flags that satisfy:
      1. CMake FindCUDAToolkit's find_library(NAMES cuda) -> needs libcuda.so
      2. GNU ld's transitive DT_NEEDED resolution when linking llama-server
         against libggml-cuda.so -> needs libcuda.so.1 via -Wl,-rpath-link
    """
    path = find_cuda_driver_lib()
    if not path:
        print(
            "  \u26a0\ufe0f No libcuda.so / libcuda.so.1 found on this image -- "
            "will build with GGML_CUDA_NO_VMM=ON.",
            flush=True,
        )
        return []

    link_dir = "/tmp/cuda-driver-link"
    os.makedirs(link_dir, exist_ok=True)
    for name in ("libcuda.so", "libcuda.so.1"):
        dst = f"{link_dir}/{name}"
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(path, dst)

    # Add link_dir to LD_LIBRARY_PATH for the build & sanity-check steps
    curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if link_dir not in curr_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{link_dir}:{curr_ld}" if curr_ld else link_dir

    print(f"  Driver lib : {path}", flush=True)
    print(f"  Driver link: {link_dir}/libcuda.so & libcuda.so.1", flush=True)

    rpath_flag = f"-Wl,-rpath-link,{link_dir} -Wl,--allow-shlib-undefined"
    return [
        f"-DCMAKE_LIBRARY_PATH={link_dir}",
        f"-DCUDA_cuda_driver_LIBRARY={link_dir}/libcuda.so",
        f"-DCMAKE_EXE_LINKER_FLAGS={rpath_flag}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={rpath_flag}",
    ]


# =====================================================================
# MAIN PACKAGER
# =====================================================================

print("\u2554" + "\u2550" * 64 + "\u2557")
print("\u2551  Build-Llama-CUDA-Release \u00b7 multi-arch CUDA 12/13 packager    \u2551")
print("\u255a" + "\u2550" * 64 + "\u255d", flush=True)

TOTAL = 6
SERVER_BIN = f"{BUILD_DIR}/build/bin/llama-server"

# ── 1. GPU CHECK ──────────────────────────────────────────────────
section(1, TOTAL, "GPU & CUDA Check")
r = sh(
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
    check=False,
    capture=True,
)
if r.returncode != 0:
    print("  \u26a0\ufe0f No GPU detected. A GPU is required to build the CUDA backend.", flush=True)
    sys.exit(1)
first_gpu = r.stdout.strip().splitlines()[0] if r.stdout.strip() else "unknown"
gpu_parts = [p.strip() for p in first_gpu.split(",")]
gpu_name = f"{gpu_parts[0]}, {gpu_parts[1]}" if len(gpu_parts) >= 2 else first_gpu
driver_version = gpu_parts[2] if len(gpu_parts) >= 3 else "unknown"
print(f"  GPU   : {gpu_name}", flush=True)
print(f"  Driver: {driver_version}", flush=True)

r = sh("nvcc --version", check=False, capture=True)
if r.returncode != 0:
    print("  \u274c nvcc not found. CUDA toolkit is required to build the CUDA backend.", flush=True)
    sys.exit(1)
nvcc_version_line = [l for l in r.stdout.splitlines() if "release" in l.lower()]
nvcc_version = nvcc_version_line[0].strip() if nvcc_version_line else r.stdout.strip().splitlines()[-1]
print(f"  nvcc  : {nvcc_version}", flush=True)
print(f"  Target: {CUDA_ARCH}", flush=True)

# ── 2. INSTALL BUILD TOOLS ────────────────────────────────────────
section(2, TOTAL, "Install Build Tools")
sh(
    "apt-get update -qq && apt-get install -y -qq cmake build-essential git ccache 2>/dev/null || true",
    check=False,
    quiet=True,
)
r = sh("cmake --version", check=False, capture=True)
print(f"  {r.stdout.splitlines()[0] if r.returncode == 0 else 'cmake install failed'}", flush=True)

# ── 3. CLONE + BUILD ──────────────────────────────────────────────
section(3, TOTAL, f"Clone + Build llama.cpp @ {LLAMA_CPP_TAG} (multi-arch: {CUDA_ARCH})")


def clone_step():
    sh(f"rm -rf {BUILD_DIR}", check=False, quiet=True)
    sh(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            LLAMA_CPP_TAG,
            "https://github.com/ggml-org/llama.cpp.git",
            BUILD_DIR,
        ]
    )


print(f"  \u231b Cloning llama.cpp @ {LLAMA_CPP_TAG} into {BUILD_DIR}...", flush=True)
run_with_retry(clone_step, label="git clone")

r = sh(f"git -C {BUILD_DIR} rev-parse --short HEAD", check=False, capture=True)
commit_hash = r.stdout.strip() if r.returncode == 0 else "unknown"
print(f"  Commit: {commit_hash}", flush=True)

print("  \u231b Configuring (CUDA, Release)...", flush=True)
driver_flags = cuda_driver_cmake_flags()


def configure(extra_flags):
    sh(f"rm -rf {BUILD_DIR}/build", check=False, quiet=True)
    # Pass as a list (shell=False) so semicolons in CUDA_ARCH are never split by /bin/sh
    cmd = [
        "cmake",
        "-B",
        f"{BUILD_DIR}/build",
        "-S",
        BUILD_DIR,
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


def build_target():
    nproc = os.cpu_count() or 4
    parallel_jobs = min(nproc, 4)
    cmd = [
        "cmake",
        "--build",
        f"{BUILD_DIR}/build",
        "--config",
        "Release",
        f"-j{parallel_jobs}",
        "--target",
        "llama-server",
    ]
    return sh(cmd, check=False).returncode == 0


cuda_vmm = bool(driver_flags)
if cuda_vmm and configure(driver_flags):
    print("  \u2705 Configured (CUDA VMM: on)", flush=True)
else:
    print(
        "  \u26a0\ufe0f Configure with CUDA driver VMM unavailable/failed -- configuring with GGML_CUDA_NO_VMM=ON...",
        flush=True,
    )
    cuda_vmm = False
    if not configure(["-DGGML_CUDA_NO_VMM=ON"]):
        print("  \u274c CMake configure failed even with VMM disabled -- see the log above.", flush=True)
        sys.exit(1)
    print("  \u2705 Configured (CUDA VMM: off)", flush=True)

print("  \u231b Compiling multi-arch llama-server (NOTE: takes ~1 hour 30 minutes on Kaggle 4-core CPU)...", flush=True)
t0 = time.time()
if not build_target():
    if cuda_vmm:
        print(
            "\n  \u26a0\ufe0f Build failed with CUDA VMM enabled (likely stub driver symbol mismatch) -- "
            "retrying cleanly with GGML_CUDA_NO_VMM=ON...",
            flush=True,
        )
        cuda_vmm = False
        if not configure(["-DGGML_CUDA_NO_VMM=ON"]) or not build_target():
            print("  \u274c Build failed even with GGML_CUDA_NO_VMM=ON -- see compiler output above.", flush=True)
            sys.exit(1)
    else:
        print("  \u274c Build failed -- see compiler output above.", flush=True)
        sys.exit(1)

build_seconds = int(time.time() - t0)
print(f"  \u2705 Build finished in {build_seconds}s", flush=True)

if not os.path.isfile(SERVER_BIN):
    print(f"  \u274c Build finished but {SERVER_BIN} was not produced.", flush=True)
    sys.exit(1)

# ── 4. SANITY CHECK ───────────────────────────────────────────────
section(4, TOTAL, "Sanity Check")
r = sh([SERVER_BIN, "--version"], check=False, capture=True)
if r.returncode != 0:
    print(f"  \u274c llama-server --version exited with code {r.returncode}:", flush=True)
    print(f"  stdout: {r.stdout}\n  stderr: {r.stderr}", flush=True)
    sys.exit(1)
version_output = (r.stdout + r.stderr).strip()
print(f"  \u2705 llama-server runs: {version_output.splitlines()[0] if version_output else '(active)'}", flush=True)

# ── 5. PACKAGE (binary + project .so deps + SONAME symlinks) ──────
section(5, TOTAL, "Package Binary + Dependencies")

pkg_name = f"llama-cuda-multiarch-{LLAMA_CPP_TAG}"
pkg_dir = f"{SCRATCH_DIR}/{pkg_name}"
sh(f"rm -rf {pkg_dir}", check=False, quiet=True)
os.makedirs(f"{pkg_dir}/lib", exist_ok=True)

shutil.copy2(SERVER_BIN, f"{pkg_dir}/llama-server")
os.chmod(f"{pkg_dir}/llama-server", 0o755)

# Copy all project shared libraries and preserve versioned SONAME symlinks
build_bin_dir = f"{BUILD_DIR}/build/bin"
bundled_files = set()
if os.path.isdir(build_bin_dir):
    for item in sorted(os.listdir(build_bin_dir)):
        if item.endswith(".so") or ".so." in item:
            src_path = os.path.join(build_bin_dir, item)
            dst_path = os.path.join(f"{pkg_dir}/lib", item)
            if os.path.islink(src_path):
                link_target = os.readlink(src_path)
                if not os.path.isabs(link_target):
                    if os.path.lexists(dst_path):
                        os.remove(dst_path)
                    os.symlink(link_target, dst_path)
                    bundled_files.add(item)
                    continue
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                bundled_files.add(item)

# Also inspect ldd output to catch any remaining build-tree libraries + libcudart.so
# (libcudart is small ~700 KB; bundling it prevents immediate --version failures across
# CUDA 12 <-> CUDA 13 containers).
ldd_targets = [SERVER_BIN]
if os.path.isfile(f"{pkg_dir}/lib/libggml-cuda.so"):
    ldd_targets.append(f"{pkg_dir}/lib/libggml-cuda.so")
for ldd_target in ldd_targets:
    r = sh(["ldd", ldd_target], check=False, capture=True)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if "=>" not in line:
                continue
            parts = line.split("=>")
            soname = parts[0].strip()
            resolved = parts[1].strip().split(" ")[0]
            if resolved and os.path.isabs(resolved) and os.path.isfile(resolved):
                if BUILD_DIR in resolved or soname.startswith("libcudart.so"):
                    dst_soname = os.path.join(f"{pkg_dir}/lib", soname)
                    if not os.path.lexists(dst_soname):
                        shutil.copy2(os.path.realpath(resolved), dst_soname)
                        bundled_files.add(soname)

print(f"  Bundled {len(bundled_files)} project shared library entries into lib/:", flush=True)
for f_name in sorted(bundled_files):
    print(f"    - {f_name}", flush=True)

launcher = (
    "#!/bin/bash\n"
    'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'export LD_LIBRARY_PATH="$DIR/lib:$DIR:$LD_LIBRARY_PATH"\n'
    'exec "$DIR/llama-server" "$@"\n'
)
run_sh = f"{pkg_dir}/run.sh"
with open(run_sh, "w") as f:
    f.write(launcher)
os.chmod(run_sh, 0o755)

sh([run_sh, "--version"])

built_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
version_txt = (
    f"llama_cpp_tag={LLAMA_CPP_TAG}\n"
    f"commit={commit_hash}\n"
    f"cuda_archs={CUDA_ARCH}\n"
    f"cuda_archs_human={CUDA_ARCH_HUMAN}\n"
    f"cuda_vmm={'on' if cuda_vmm else 'off'}\n"
    f"built_on_gpu={gpu_name}\n"
    f"driver_version={driver_version}\n"
    f"nvcc={nvcc_version}\n"
    f"built_at={built_at}\n"
    f"build_seconds={build_seconds}\n"
    f"llama_server_version_output={version_output.splitlines()[0] if version_output else ''}\n"
)
with open(f"{pkg_dir}/VERSION.txt", "w") as f:
    f.write(version_txt)

tarball_path = f"{OUTPUT_DIR}/{pkg_name}.tar.gz"
with tarfile.open(tarball_path, "w:gz") as tar:
    tar.add(pkg_dir, arcname=pkg_name)

sha256 = hashlib.sha256()
with open(tarball_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        sha256.update(chunk)
digest = sha256.hexdigest()
sha_path = f"{tarball_path}.sha256"
with open(sha_path, "w") as f:
    f.write(f"{digest}  {os.path.basename(tarball_path)}\n")

# Clean up intermediate build tree so only the final release artifacts remain
sh(f"rm -rf {BUILD_DIR} {pkg_dir}", check=False, quiet=True)

tar_size_mb = os.path.getsize(tarball_path) / (1024 * 1024)
print(f"\n  \u2705 Packaged: {tarball_path} ({tar_size_mb:.1f} MB)", flush=True)
print(f"  \u2705 Checksum: {sha_path}", flush=True)

# ── 6. UPLOAD INSTRUCTIONS ─────────────────────────────────────────
section(6, TOTAL, "Next Steps: Upload to GitHub Releases")
print(
    f"""
  1. Unduh 2 file berikut dari output notebook (tab 'Output' di Kaggle):
       - {os.path.basename(tarball_path)}
       - {os.path.basename(sha_path)}

  2. Di GitHub repo AIChat-With-Notebooks:
       Releases -> Draft a new release (atau edit release v1.4.0)
       -> Lampirkan kedua file tersebut sebagai release assets
       -> Publish release

  VERSION.txt yang ikut terbundel:
{version_txt}
  Selesai -- script ini HANYA memaketkan binary prebuilt CUDA.
""",
    flush=True,
)
