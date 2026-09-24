# -*- coding: utf-8 -*-
"""Build-Llama-CUDA-Release.py

One-time builder: produces a portable, self-contained llama-server (CUDA)
tarball you upload to your own GitHub Release, so future Kaggle/Colab runs
can download a prebuilt binary instead of compiling from source every time.

## Multi-GPU support (fat binary, no more per-arch rebuilds)
This build embeds native (SASS) code for EVERY GPU generation these
notebooks might run on -- T4, A100, RTX 30xx/40xx, L4, H100 -- in a single
tarball, plus a PTX ("virtual") fallback for the newest arch so even GPUs
released after this build (e.g. a future Blackwell RTX 50xx) can still run
it via the driver's PTX JIT instead of needing a rebuild. The consumer
notebooks (Collab-Llama.py / Kaggle-Llama.py) auto-detect whichever GPU
they're actually running on and verify it's covered -- you never edit
CUDA_ARCH by hand again when switching between T4 / L4 / A100 / RTX / etc.

Run this ONCE, on Kaggle (any GPU is fine -- building a fat binary doesn't
require the target hardware to be present), whenever you want a new build
(e.g. after bumping LLAMA_CPP_TAG). It does NOT start a chat server and
does NOT need a model download -- it only builds + packages the binary.

## Why this exists
ggml-org/llama.cpp does not publish an official prebuilt Linux+CUDA binary
(only Linux CPU-only, and CUDA for Windows) -- confirmed by inspecting their
release assets directly. So "prebuilt binary as the default install path"
(the goal) has to mean *your own* prebuilt, not a random third-party one.
This script builds it once; the consumer notebooks then prefer downloading
that release asset, falling back to a source build only if it's missing,
incompatible, or fails a sanity check.

## What you do with the output
1. Run this cell on Kaggle (Settings -> Accelerator -> GPU T4 x2, or T4 x1
   is fine too -- this only needs ONE GPU to build for sm_75).
2. When it finishes, a tarball + .sha256 file sit in /kaggle/working/.
   Download both from the Kaggle notebook's Output panel.
3. On GitHub: your repo -> Releases -> Draft a new release -> pick/create a
   tag (e.g. "llama-cuda-b10605") -> attach both files as release assets ->
   Publish.
4. Copy the release asset's download URL (see the printed instructions at
   the end) into GITHUB_RELEASE_URL in Collab-Llama.py / Kaggle-Llama.py.

Re-run this script whenever you bump LLAMA_CPP_TAG and want an updated
prebuilt; each run's tarball is self-describing (VERSION.txt) so the
consumer notebooks can tell whether a given release asset is still a good
match before trusting it.
"""

import subprocess, time, os, sys, shutil, hashlib, tarfile, glob

# =====================================================================
# CONFIGURATION -- keep this in sync with Collab-Llama.py / Kaggle-Llama.py
# =====================================================================

LLAMA_CPP_TAG = "b10605"

# Fat multi-arch build: real (SASS) code for every GPU generation these
# notebooks are known to run on, plus a PTX ("virtual") entry on the newest
# one for forward compatibility with GPUs released after this build.
#   75 = Turing     (T4, RTX 20xx)
#   80 = Ampere DC  (A100)
#   86 = Ampere     (RTX 30xx, A10/A40)
#   89 = Ada        (L4, RTX 40xx)
#   90 = Hopper     (H100) -- also embedded as PTX ("-virtual") so a future
#                             arch newer than Hopper can still JIT-compile
#                             against it instead of hard-failing.
# This takes longer to compile (~15-25 min vs ~5-10 for a single arch) and
# produces a bigger binary, but you only pay that cost ONCE per
# LLAMA_CPP_TAG bump -- every consumer notebook after that auto-detects its
# GPU and just works, no manual CUDA_ARCH editing ever again.
CUDA_ARCH = "75-real;80-real;86-real;89-real;90-real;90-virtual"

if os.path.isdir("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working"
elif os.path.isdir("/content"):
    OUTPUT_DIR = "/content"
else:
    OUTPUT_DIR = "/tmp"

BUILD_DIR = f"{OUTPUT_DIR}/llama.cpp"

# ── Utility functions (same shapes as the other notebook scripts) ───

def sh(cmd, check=True, quiet=False, capture=False):
    kw = dict(shell=True, check=check)
    if capture:
        kw.update(capture_output=True, text=True)
    elif quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, **kw)


def section(n, total, title):
    bar = "\u2500" * 64
    print(f"\n{bar}\n [{n}/{total}] {title}\n{bar}")


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


def find_cuda_driver_lib():
    """Locate a libcuda.so the linker can actually consume.

    CMake's FindCUDAToolkit only creates the CUDA::cuda_driver imported target
    if find_library(NAMES cuda) succeeds -- i.e. if a file literally named
    'libcuda.so' exists in one of its search paths. Kaggle/Colab GPU images
    ship the CUDA *toolkit* without the driver stub package
    (cuda-driver-dev-*), and the real driver is injected by the NVIDIA
    container runtime as a versioned 'libcuda.so.1' (commonly at
    /usr/local/nvidia/lib64). So the target never gets created, and llama.cpp's
    `target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)` blows up the
    CMake *generate* step with "but the target was not found".

    Returns (path, is_versioned); is_versioned=True means the caller must
    create the missing 'libcuda.so' symlink pointing at it.
    """
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
    dirs = unique_dirs

    # Prefer an unversioned stub / dev symlink: linking against the toolkit
    # stub is the standard way to build a portable CUDA binary, since it only
    # records a DT_NEEDED on libcuda.so.1 and lets the host's real driver
    # satisfy it at run time.
    for d in dirs:
        p = f"{d}/libcuda.so"
        if os.path.exists(p):
            return p, False

    # Otherwise fall back to the runtime driver and symlink it ourselves. The
    # SONAME baked into the binary is still libcuda.so.1, so the tarball stays
    # exactly as portable as a stub-linked one -- we are not freezing this
    # machine's driver version into the package.
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
        print("  \u26a0\ufe0f No libcuda.so / libcuda.so.1 found on this image -- "
              "will need to build with CUDA VMM disabled.")
        return []
    if is_versioned:
        link_dir = "/tmp/cuda-driver-link"
        os.makedirs(link_dir, exist_ok=True)
        link_path = f"{link_dir}/libcuda.so"
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(path, link_path)
        print(f"  Driver lib: {path}")
        print(f"  No unversioned 'libcuda.so' here -- symlinked one at {link_path}")
    else:
        link_path, link_dir = path, os.path.dirname(path)
        print(f"  Driver lib: {path}")
    # CMAKE_LIBRARY_PATH is what actually fixes find_library(NAMES cuda) -- it
    # is searched ahead of FindCUDAToolkit's own HINTS. Seeding the cache
    # variable as well short-circuits the search outright, so this holds up
    # across CMake versions that name it differently.
    return [f"-DCMAKE_LIBRARY_PATH={link_dir}", f"-DCUDA_cuda_driver_LIBRARY={link_path}"]


print("\u2554" + "\u2550" * 64 + "\u2557")
print("\u2551  Build-Llama-CUDA-Release \u00b7 one-time prebuilt packager" + " " * 7 + "\u2551")
print("\u255a" + "\u2550" * 64 + "\u255d")

TOTAL = 6
SERVER_BIN = f"{BUILD_DIR}/build/bin/llama-server"

# ── 1. GPU CHECK ──────────────────────────────────────────────────
section(1, TOTAL, "GPU Check")
r = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False, capture=True)
if r.returncode != 0:
    print("  \u26a0\ufe0f No GPU detected. A GPU is required to build the CUDA backend.")
    sys.exit(1)
gpu_name = r.stdout.strip().splitlines()[0] if r.stdout.strip() else "unknown"
print(f"  GPU: {gpu_name}")

r = sh("nvcc --version", check=False, capture=True)
if r.returncode != 0:
    print("  \u274c nvcc not found. CUDA toolkit is required to build the CUDA backend "
          "(this should already be present on Kaggle's GPU image).")
    sys.exit(1)
nvcc_version_line = [l for l in r.stdout.splitlines() if "release" in l.lower()]
nvcc_version = nvcc_version_line[0].strip() if nvcc_version_line else r.stdout.strip().splitlines()[-1]
print(f"  {nvcc_version}")

# ── 2. INSTALL BUILD TOOLS ────────────────────────────────────────
section(2, TOTAL, "Install Build Tools")
sh("apt-get update -qq && apt-get install -y -qq cmake build-essential git", check=False, quiet=True)
r = sh("cmake --version", check=False, capture=True)
print(f"  {r.stdout.splitlines()[0] if r.returncode == 0 else 'cmake install failed'}")

# ── 3. CLONE + BUILD ──────────────────────────────────────────────
section(3, TOTAL, f"Clone + Build llama.cpp @ {LLAMA_CPP_TAG} (multi-arch: {CUDA_ARCH})")

def clone_step():
    sh(f"rm -rf {BUILD_DIR}", check=False, quiet=True)
    sh(f"git clone --depth 1 --branch {LLAMA_CPP_TAG} "
       f"https://github.com/ggml-org/llama.cpp.git {BUILD_DIR}")

print(f"  \u231b Cloning...")
run_with_retry(clone_step, label="git clone")

r = sh(f"git -C {BUILD_DIR} rev-parse --short HEAD", check=False, capture=True)
commit_hash = r.stdout.strip() if r.returncode == 0 else "unknown"
print(f"  Commit: {commit_hash}")

print("  \u231b Configuring (CUDA, Release)...")
driver_flags = cuda_driver_cmake_flags()


def configure(extra_flags):
    # Wipe the build dir between attempts: a failed generate leaves a
    # CMakeCache.txt behind with the NOTFOUND driver lookup cached in it.
    sh(f"rm -rf {BUILD_DIR}/build", check=False, quiet=True)
    return sh(
        f'cmake -B {BUILD_DIR}/build -S {BUILD_DIR} '
        f'-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCH}" '
        f'-DCMAKE_BUILD_TYPE=Release '
        f'-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF '
        + " ".join(extra_flags),
        check=False,
    ).returncode == 0


cuda_vmm = True
if not configure(driver_flags):
    # Last resort: ggml's own escape hatch. GGML_CUDA_NO_VMM=ON is precisely
    # the switch that makes ggml-cuda skip its
    # `target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)` line, so the
    # build stops needing libcuda.so at link time. It costs the VMM-backed
    # memory pool (a slightly less efficient allocator), which is a far better
    # outcome than no binary at all -- and VERSION.txt records that it happened
    # so the consumer notebooks aren't misled about what they downloaded.
    print("  \u26a0\ufe0f Configure failed with the CUDA driver library -- retrying with "
          "GGML_CUDA_NO_VMM=ON (skips the libcuda.so link requirement).")
    cuda_vmm = False
    if not configure(["-DGGML_CUDA_NO_VMM=ON"]):
        print("  \u274c CMake configure failed even with VMM disabled -- see the log above.")
        sys.exit(1)
print(f"  \u2705 Configured (CUDA VMM: {'on' if cuda_vmm else 'off'})")

print("  \u231b Compiling llama-server (this is the slow part -- grab a coffee)...")
t0 = time.time()
sh(f"cmake --build {BUILD_DIR}/build --config Release -j$(nproc) --target llama-server")
build_seconds = int(time.time() - t0)
print(f"  \u2705 Build finished in {build_seconds}s")

if not os.path.isfile(SERVER_BIN):
    print(f"  \u274c Build finished but {SERVER_BIN} was not produced. Check the build log above.")
    sys.exit(1)

# ── 4. SANITY CHECK ───────────────────────────────────────────────
section(4, TOTAL, "Sanity Check")
r = sh(f"{SERVER_BIN} --version", check=False, capture=True)
if r.returncode != 0:
    print(f"  \u274c llama-server --version exited with code {r.returncode} -- build is broken, not packaging it.")
    print(f"  stdout: {r.stdout}\n  stderr: {r.stderr}")
    sys.exit(1)
version_output = (r.stdout + r.stderr).strip()
print(f"  \u2705 llama-server runs: {version_output.splitlines()[0] if version_output else '(no output)'}")

# ── 5. PACKAGE (binary + its own .so deps, NOT system CUDA libs) ──
section(5, TOTAL, "Package Binary + Dependencies")

r = sh(f"ldd {SERVER_BIN}", check=False, capture=True)
if r.returncode != 0:
    print("  \u26a0\ufe0f ldd failed to run -- packaging binary alone (may be statically linked, or ldd unavailable)")
    project_libs = []
else:
    # Keep only .so files that live INSIDE the build tree (this project's own
    # libggml*/libllama/libmtmd), skip system libs (libc, libcuda, libcudart,
    # libcublas, etc.) -- those should come from whatever CUDA/driver is
    # already on the host machine at run time, not be frozen into this
    # tarball, since bundling a mismatched CUDA runtime is exactly the kind
    # of "silent incompatibility" this whole verify-before-trust design is
    # meant to avoid.
    project_libs = []
    for line in r.stdout.splitlines():
        if "=>" in line:
            path = line.split("=>")[-1].strip().split(" ")[0]
        else:
            path = line.strip().split(" ")[0]
        if path and os.path.isabs(path) and BUILD_DIR in path and os.path.isfile(path):
            project_libs.append(path)
    print(f"  Found {len(project_libs)} project-owned shared libraries to bundle:")
    for p in project_libs:
        print(f"    - {os.path.basename(p)}")

pkg_name = f"llama-cuda-multiarch-{LLAMA_CPP_TAG}"
pkg_dir = f"/tmp/{pkg_name}"
sh(f"rm -rf {pkg_dir}", check=False, quiet=True)
os.makedirs(f"{pkg_dir}/lib", exist_ok=True)

shutil.copy2(SERVER_BIN, f"{pkg_dir}/llama-server")
for p in project_libs:
    shutil.copy2(p, f"{pkg_dir}/lib/")

# Portable launcher -- points LD_LIBRARY_PATH at the bundled lib/ dir next
# to wherever the tarball ends up extracted, so no patchelf/RPATH surgery
# is needed and it works regardless of extraction path.
launcher = (
    "#!/bin/bash\n"
    'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'export LD_LIBRARY_PATH="$DIR/lib:$LD_LIBRARY_PATH"\n'
    'exec "$DIR/llama-server" "$@"\n'
)
with open(f"{pkg_dir}/run.sh", "w") as f:
    f.write(launcher)
os.chmod(f"{pkg_dir}/run.sh", 0o755)
os.chmod(f"{pkg_dir}/llama-server", 0o755)

# Self-describing metadata -- the consumer notebook's verification step
# reads this before trusting the release asset (spec: "verify binary
# version + CUDA backend" before use, don't blindly trust a downloaded bin).
built_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
version_txt = (
    f"llama_cpp_tag={LLAMA_CPP_TAG}\n"
    f"commit={commit_hash}\n"
    # Raw archs string, parsed by the consumer notebooks' verify_prebuilt()
    # to check "is THIS GPU covered?" -- semicolon-separated, each entry
    # tagged -real (native SASS) or -virtual (PTX, JIT-compiled at load
    # time for that arch or newer).
    f"cuda_archs={CUDA_ARCH}\n"
    f"cuda_archs_human=Turing(T4/RTX20xx)=75 Ampere-DC(A100)=80 "
    f"Ampere(RTX30xx)=86 Ada(L4/RTX40xx)=89 Hopper(H100)=90 "
    f"PTX-forward-compat-from=90\n"
    f"cuda_vmm={'on' if cuda_vmm else 'off'}\n"
    f"built_on_gpu={gpu_name}\n"
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

tar_size_mb = os.path.getsize(tarball_path) / (1024 * 1024)
print(f"\n  \u2705 Packaged: {tarball_path} ({tar_size_mb:.1f} MB)")
print(f"  \u2705 Checksum: {sha_path}")

# ── 6. UPLOAD INSTRUCTIONS ─────────────────────────────────────────
section(6, TOTAL, "Next Steps: Upload to GitHub Releases")
print(f"""
  1. Unduh 2 file berikut dari output notebook (tab 'Output' di Kaggle, atau panel Files di Colab):
       - {os.path.basename(tarball_path)}
       - {os.path.basename(sha_path)}

  2. Di GitHub repo kamu:
       Releases -> Draft a new release
       -> Tag: llama-cuda-multiarch-{LLAMA_CPP_TAG}  (atau tag lain sesukamu)
       -> Attach both files sebagai release assets
       -> Publish release

  3. Salin URL release asset (klik kanan file -> Copy link) -- bentuknya:
       https://github.com/<user>/<repo>/releases/download/llama-cuda-multiarch-{LLAMA_CPP_TAG}/{os.path.basename(tarball_path)}

  4. Tempel URL itu ke GITHUB_RELEASE_URL di Collab-Llama.py / Kaggle-Llama.py.
     SATU URL ini berlaku buat T4, A100, L4, RTX 30xx/40xx, H100 sekaligus --
     gak perlu ganti URL atau CUDA_ARCH lagi tiap pindah GPU. Notebook
     konsumen akan:
       - download tarball itu (sekali per session runtime, di-cache)
       - auto-detect GPU yang lagi aktif via nvidia-smi (compute capability)
       - cocokkan hasil deteksi itu dengan daftar cuda_archs di VERSION.txt
         (native SASS match, ATAU >= arch PTX-forward-compat -> masih jalan
         lewat JIT)
       - kalau cocok & lolos sanity check ('llama-server --version') -> pakai
       - kalau GPU-nya lebih tua dari semua arch yang di-embed (mismatch),
         atau tag/download gagal -> otomatis fallback build dari source,
         khusus utk arch GPU yang terdeteksi saat itu saja (cepat, satu arch)

  VERSION.txt yang ikut terbundel:
{version_txt}
  Selesai -- script ini TIDAK menjalankan server / download model apa pun.
""")