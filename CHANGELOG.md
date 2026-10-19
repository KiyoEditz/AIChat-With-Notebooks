# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [v1.5.5]

### Added
- **Custom Persona Avatar System**:
  - Added custom avatar support for user personas across settings, active chat messages, and memory selectors.
  - **Persona Avatar Management (`settings.html` / `settings.js`)**:
    - Added dedicated avatar upload container with preview circle and remove button in `#persona-editor-card`.
    - Integrated automatic client-side canvas image resizing to 256x256 WebP data URLs.
    - Integrated Pinterest image search and picker directly within the persona editor (`/api/pinterest/search` & `/api/pinterest/image`).
    - Added avatar preview thumbnails on persona cards in the Settings persona list.
  - **Dynamic Chat Message Avatars (`public/app.js` / `public/styles.css`)**:
    - Upgraded user message bubbles (`.message.user .message-avatar`) to dynamically render the active persona's avatar image instead of the static `👤` emoji.
    - Preserved seamless fallback to `👤` whenever a persona has no avatar assigned.
  - **Memory Selector Integration (`public/app.js` / `public/styles.css`)**:
    - Added avatar preview icons inside persona selector chips (`.persona-chip-avatar`) and in the active persona summary card (`.persona-active-avatar`) within the chat memory modal.
    - Selecting a new persona chip immediately updates user avatars in all rendered chat messages reactively.
  - **Backend API & Schema (`server.js`)**:
    - Added `avatar` field handling to `getPersonas()`, `POST /api/personas`, and `PUT /api/personas/:id`.
- **Chat Message Deletion Endpoint & UI**:
  - Added REST endpoint `DELETE /api/chat/:chatId/message/:index` in `server.js` with index boundary validation and atomic JSON disk persistence.
  - Added message deletion button (`🗑️`) beside the edit button (`✏️`) in message metadata (`.message-meta`) with confirmation prompts and localized toast alerts.
- **Full 12-Language i18n Synchronization (`public/Lang/`)**:
  - Fully translated and synchronized all 12 supported languages (`AR`, `DE`, `EN`, `ES`, `FR`, `HI`, `ID`, `JP`, `KO`, `PT`, `RU`, `ZH`).
  - Added localized translation keys across all languages for welcome actions, message editing, message deletion, and async waiting:
    - `welcome.btn_donate`
    - `chat.action.regenerate`
    - `chat.action.delete`
    - `chat.action.edit`
    - `chat.confirm_delete_msg`
    - `chat.msg_deleted`
    - `btn.edit`
    - `toast.please_wait`
  - Normalized all 12 language dictionaries to 105 identical keys with zero missing entries.
- **`Molab-Llama.py` (Beta / Staging)**:
  - One-click LLM server script tailored for **Molab** (`molab.marimo.io`), featuring the **NVIDIA RTX PRO 6000 Blackwell Server Edition** (96 GB VRAM) GPU with CUDA 13.3.
  - Marimo-native pure Python syntax avoiding syntax errors caused by IPython magics like `!cmd`.
  - Automatic GPU compute capability detection (`sm_120` / `sm_100`).
  - Robust CUDA toolkit environment resolution (`setup_cuda_toolkit_env()`) resolving `CUDAToolkit_ROOT`, `CUDA_CUDART_LIBRARY`, and driver symlinks so CMake natively discovers CUDA 13.3 runtime without missing `libcudart.so`.
  - Massive context window default (`NUM_CTX = 16384`) leveraging 96 GB VRAM, with options for 32k/64k.
  - Multi-tier model configurations optimized for 96GB VRAM (32B Q6_K default, 70B/72B flagship options).
  - Built-in Hugging Face token resolution (`HF_TOKEN`) from environment variables, Molab Secrets tab, and `/marimo/.env`.
- **`Build-Molab-CUDA-Release.py` (Dedicated Molab Builder)**:
  - Standalone CUDA release packager specifically for building and packaging native Blackwell `sm_120` llama-server binary tarballs inside Molab.
  - Automatically exports release packages to `/marimo/` ready for upload to GitHub Releases.

### Changed & Improved
- **Persona UI Streamlining & Bug Fixes**:
  - Removed redundant `Persona #` badge from the persona editor card in Settings and from the active persona summary card in the chat memory modal for a cleaner, minimal appearance.
  - Resolved cramped layout between the `+ Tambah` extra attribute button and `💾 Simpan Persona` by adding dedicated vertical spacing and an elegant divider border.
- **Mobile Welcome Viewport & Responsive Layout (`#view-welcome`)**:
  - Scaled mobile diamond icon (`✦`) up from `2.6rem` to `3rem` for well-balanced visual hierarchy on mobile screens.
  - Resolved content cut-off and cramped layout on mobile screens by enabling native vertical scrolling (`overflow-y: auto`, `-webkit-overflow-scrolling: touch`) and safe centering (`min-height: 100%; height: auto; margin: auto 0; box-sizing: border-box;`).
  - Optimized mobile icon and title spacing and aligned "Buka Settings" and "Donasi Proyek" buttons side-by-side on mobile devices (`flex-direction: row; gap: 8px; height: 40px;`).
- **Clean Regenerate Button UI**:
  - Preserved standalone regenerate button (`🔄`) on assistant message bubbles and removed adjacent delete triggers to prevent accidental deletions during chat.

---

## [v1.5.0] — Centralized Persona System & Structured User Attributes

Refactors the persona architecture from duplicating `userPreferences` inside every character's memory file into a centralized, reusable persona pool in `data/persona/`, with a dedicated Settings management panel and compact chat memory selector.

### Added
- **Centralized Persona Storage (`data/persona/`)**:
  - Persona profiles are now stored independently (e.g. `1.json`) and shared globally across all characters.
  - Characters only store a concise `personaId` reference in their memory file, eliminating duplicated persona data across memory files.
- **Dedicated Persona Management Panel (`settings.html`)**:
  - Added new **👤 Persona** tab positioned directly below **Configuration** and above **Appearance**.
  - Provides intuitive persona cards with index badges (e.g. `Persona #1`, `Persona #2`).
  - Supports adding, editing, and deleting multiple personas.
  - Strict input validation:
    - **Name**: String (supports letters and numbers).
    - **Age (Umur)**: Strictly number-only.
    - **Height (Tinggi Badan)**: Strictly number-only in cm.
    - **Weight (Berat Badan)**: Strictly number-only in kg.
    - **Gender (Jenis Kelamin)**: Dropdown selection (`Male`, `Female`, `Non-binary`).
    - **Extra Attributes**: Attribute selector list (Pekerjaan, Hobi, Status, Kepribadian, or custom key-value).
- **Compact Persona Selector in Chat (`index.html`)**:
  - Replaced the bulky key-value memory editor in the chat memory modal with a compact 1-click persona selector.
  - Displays selected persona chips with index and name, plus an active summary badge and direct link to Settings.
- **Dedicated Donation Page (`donate.html` & `donate.css`)**:
  - Standalone donation page supporting Saweria (QRIS, e-wallet, bank transfer) and Ko-fi (PayPal, Card), with GitHub Sponsors noted as under review.
  - Interactive quick-copy link buttons with clipboard feedback toast notifications.
  - Added direct donation button beside the Settings button on the welcome screen (`index.html`) and in the sidebar header.
- **REST Endpoints for Persona in `server.js`**:
  - `GET /api/personas`, `GET /api/personas/:id`, `POST /api/personas`, `PUT /api/personas/:id`, `DELETE /api/personas/:id`.

### Changed
- **Memory Cleaner & Heuristic Regex Removal (`server.js`)**:
  - Migrated 135 memory files to reference central `personaId` instead of bulky redundant `userPreferences` blocks.
  - Removed imprecise regex heuristic auto-filling of name and age from `autoExtractMemory()`, ensuring persona attributes remain exact and user-controlled.
  - Chat completions (`/api/chat/send` and `/api/chat/regenerate`) now resolve and format active persona details cleanly into `## What You Know About the User`.

---

## [v1.4.1] — Thinking Mode Off by Default & Response Length Tuning

Disables chain-of-thought / reasoning tokens (`<think>...</think>`) by default
across all notebook servers and client requests, and refines the character roleplay
system prompt to enforce concise 4–6 sentence responses.

### Added
- **Native thinking / reasoning toggle (`ENABLE_THINKING = False`)**:
  - Configurable in `Collab-Llama.py`, `Collab-Llama-Private.py`, `Kaggle-Llama.py`,
    and `Kaggle-Llama-Private.py`.
  - Added `--reasoning off --chat-template-kwargs '{"enable_thinking": false}'`
    to `llama-server` CLI startup flags when disabled, turning off reasoning at
    both the llama.cpp engine and Jinja chat template levels.
  - Added `chat_template_kwargs: {"enable_thinking": False}` to the startup
    1-token completion test in all notebook scripts.
  - Added `Thinking : OFF` indicator in the server ready banner.
- **Environment configuration (`ENABLE_THINKING`)** in `.env` and `.env.example`:
  - Set to `false` by default to disable thinking tokens on reasoning-capable
    models (e.g. Qwen 2.5/3, DeepSeek-R1).
- **Client-side request kwargs and fallback filter** in `server.js`:
  - Automatically sends `chat_template_kwargs: { enable_thinking: false }` via
    `getModelParams()` on every `/api/chat` and `/api/chat/regenerate` request.
  - Added `cleanAssistantResponse()` helper to strip any completed or unclosed
    `<think>...</think>` tags if a reasoning model leaks thought tokens into the output text.

### Changed
- **Response length accuracy in roleplay rules** (`server.js`):
  - Refined system prompt in `/api/chat` and `/api/chat/regenerate` from
    `1-2 short paragraphs` to `1-2 short paragraphs (4-6 sentences total)` to provide
    a concrete, measurable sentence-count target for more consistent response lengths.

---

## [v1.3.1] — Multi-GPU auto-detect build (T4 / A100 / L4 / RTX / H100)

No more manually editing `CUDA_ARCH` every time you switch GPU type in
Colab or Kaggle. One prebuilt tarball now covers every GPU generation
these notebooks are likely to run on, and the consumer notebooks detect
your hardware automatically.

### Added
- **Fat multi-arch CUDA build.** `Build-Llama-CUDA-Release.py` now compiles
  native code for Turing (T4, `sm_75`), Ampere-DC (A100, `sm_80`), Ampere
  (RTX 30-series, `sm_86`), Ada (L4 / RTX 40-series, `sm_89`), and Hopper
  (H100, `sm_90`) into a single tarball, plus an `sm_90` PTX ("virtual")
  entry so GPUs newer than Hopper can still run it via the driver's JIT
  instead of requiring a rebuild.
- **Runtime GPU auto-detection** in `Collab-Llama.py` and `Kaggle-Llama.py`
  (`detect_gpu_arch()`), reading compute capability straight from
  `nvidia-smi`. `CUDA_ARCH` is no longer a hardcoded config value — it's
  resolved automatically before anything else runs.
- **Coverage-aware prebuilt verification.** `verify_prebuilt()` now checks
  whether the detected GPU is covered by the tarball's embedded arch list
  (native match, or new enough for the PTX forward-compat entry to JIT)
  instead of requiring an exact single-arch string match. A prebuilt from
  before this update, or one that genuinely doesn't cover the attached GPU,
  is rejected with a clear reason and the notebook falls back to compiling
  from source for that GPU's specific arch.
- `VERSION.txt` now records a `cuda_archs` field (semicolon-separated,
  `-real`/`-virtual` tagged) instead of a single `cuda_arch` value, plus a
  `cuda_archs_human` line for quick eyeballing.

### Fixed
- **Critical: `CMAKE_CUDA_ARCHITECTURES` value was silently truncated to
  the first architecture.** The cmake invocation is run through a shell,
  and the semicolon-separated arch list (`75-real;80-real;86-real;...`)
  was passed unquoted — so the shell parsed everything after the first
  semicolon as separate commands (`80-real: not found`, `86-real: not
  found`, etc.) instead of one argument. Only `sm_75` ever actually got
  built, silently. Fixed by quoting the value
  (`-DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCH}"`), confirmed to affect every
  multi-arch build produced before this patch.
- **`find_cuda_driver_lib()` missed the driver library location on some
  Kaggle images.** `nvidia-smi` succeeds (kernel driver is loaded), but no
  `libcuda.so`/`libcuda.so.1` existed in any of the previously-checked
  toolkit/system paths, so the CUDA configure step failed at
  `target_link_libraries(ggml-cuda ... CUDA::cuda_driver)` with "target not
  found" — and the `GGML_CUDA_NO_VMM=ON` retry hit the identical error,
  since that flag doesn't actually drop the driver-link requirement in this
  llama.cpp revision. Added the `nvidia-container-toolkit` mount paths
  (`/usr/local/nvidia/lib64`, `/usr/lib/nvidia`, etc.) to the search list,
  plus a bounded filesystem sweep as a last resort before giving up.

### Changed
- Prebuilt tarball naming: `llama-cuda-sm{N}-{tag}.tar.gz` →
  `llama-cuda-multiarch-{tag}.tar.gz` (one asset now serves every GPU type
  instead of one asset per arch).
- Build time for `Build-Llama-CUDA-Release.py` increases from ~5-10 min
  (single arch) to ~15-25 min (five archs + PTX) — paid once per
  `LLAMA_CPP_TAG` bump, not per notebook run.

### Migration notes
- Re-run `Build-Llama-CUDA-Release.py` and re-upload the resulting tarball
  — old single-arch releases won't satisfy the new coverage check on any
  GPU other than the one they were originally built for.
- Update `GITHUB_RELEASE_URL` in `Collab-Llama.py` / `Kaggle-Llama.py` to
  point at the new `llama-cuda-multiarch-*.tar.gz` asset.
- No other config changes needed — `CUDA_ARCH` is auto-managed from here on.

## [v1.0.0] — llama.cpp migration + stability pass

### Added
- **`Collab-Llama.py`** — new one-click Colab notebook script. Runs `llama-server`
  (llama.cpp) directly, no Ollama/LiteLLM in the path.
- **`Kaggle-Llama.py`** — same, for Kaggle's dual-T4 environment. Preserves the
  original `kaggle_llm_server.py`'s dual-GPU verification and RAM/VRAM cleanup
  handler (`atexit`/`SIGTERM`).
- **`Build-Llama-CUDA-Release.py`** — one-time builder that packages a portable
  `llama-server` (CUDA) tarball for upload to your own GitHub Release, so
  future runs can skip compiling from source.
- **CUDA driver linking fix** (`find_cuda_driver_lib()` / `cuda_driver_cmake_flags()`)
  in all three build scripts above: works around `CUDA::cuda_driver` target
  not found during CMake configure on Kaggle/Colab images that don't expose
  an unversioned `libcuda.so`. Two-tier fallback — symlink a found driver lib
  first, or disable the CUDA VMM allocator (`GGML_CUDA_NO_VMM=ON`) as a last
  resort if no driver library is discoverable at all. Verified against a
  real Kaggle T4 build.
- `safePath()` helper in `server.js` — strict UUID-allowlist path
  construction, applied to every character/chat/memory file endpoint.
- Native llama.cpp sampling parameters in `server.js` / `.env.example`:
  `TOP_K`, `TOP_P`, `MIN_P`, `TYPICAL_P`, `TFS_Z`, `TOP_N_SIGMA`,
  `REPEAT_LAST_N`, `REPEAT_PENALTY`, `DRY_*` (full DRY sampling),
  `XTC_*`, `DYNATEMP_*`, `MIROSTAT*`, `SEED`, `SAMPLERS` — with type/range/
  NaN/Infinity validation on every value (logs a warning and falls back to
  default instead of silently misbehaving or crashing).
- "Unread" indicator (sidebar dot) in `app.js` for chats whose reply arrived
  while you were viewing a different chat.
- Character/chat ownership check in `send`/`regenerate`/`summarize` — a chat
  can no longer be processed under the wrong character's persona/memory.

### Changed
- **Default LLM stack is now `llama-server` (llama.cpp) via Cloudflare
  Tunnel** — no LiteLLM proxy, no Ollama. README and `.env.example` rewritten
  to describe this as the primary path.
- `server.js`'s `getModelParams()` fully rewritten: one canonical native
  param set instead of an OpenAI/Ollama-flavored mix.
- Model download (in the new notebooks) uses `huggingface_hub`'s
  `snapshot_download(local_dir=...)` directly instead of relying on
  `llama-server`'s built-in `-hf` downloader — keeps the model out of the
  default HF cache entirely (no duplicate on-disk copy), with Xet chunk
  caching disabled by default (`HF_XET_CHUNK_CACHE_SIZE_BYTES=0`).
- Notebook readiness check replaced with an explicit state machine
  (`PROCESS_STARTED → MODEL_LOADING → HEALTH_READY → TUNNEL_READY`) using
  `llama-server`'s real `/health` semantics (503 while loading, 200 once
  ready) instead of a generic "any non-5xx" guess. "SERVER IS READY" is only
  printed after a real 1-token completion request succeeds end-to-end.
- `.env` writer (`POST /api/config`) no longer uses a raw string in
  `String.replace()` — a pasted URL containing `$&`/`$1` used to corrupt the
  `.env` file; now uses a replacer function.
- `writeJSON()` is atomic (write to `.tmp`, then rename) — an interrupted
  write can no longer leave a truncated/corrupted chat or character file.
- `/api/chat/send` and `/api/chat/regenerate` now re-read the chat from disk
  immediately before writing, instead of reusing the copy loaded before the
  5–120s LLM call — an edit or regenerate made on the same chat while
  waiting is no longer silently reverted.
- Regenerate's alternate-resolution now happens *before* building the
  message history sent to the model, not after — a rejected/stale reply is
  no longer resent to the model on the next turn.
- `appendMessage()` in `app.js` now sets `data-index` on creation — the
  edit (✏️) button works immediately instead of only after a chat
  switch/reload.
- `escapeHtml()` now also escapes quotes (`"`/`'`), not just `&`/`<`/`>` —
  fixes several call sites that interpolate its output inside quoted HTML
  attributes.
- Avatar `src` values are now escaped before interpolation — a malformed or
  malicious avatar string (e.g. from an imported character card) can no
  longer break out of the `<img>` tag.
- `api()` in `app.js` now checks `res.ok` and handles non-JSON error bodies
  instead of throwing an unhandled parse error that silently broke the
  init chain (empty sidebar, dead buttons, no visible error).
- `sendMessage()`/`regenerateLastMessage()` capture `state.currentChatId`/
  `state.currentCharacterId` into local variables before any `await`, and
  re-check before rendering — switching chats mid-request no longer renders
  (or saves) a reply into the wrong chat.
- Failed sends now remove the optimistic "phantom" user bubble and restore
  the typed text, instead of leaving a bubble on screen for a message that
  was never actually saved.

### Fixed
- **Path traversal (C1)** — every file-path construction from user input now
  goes through `safePath()`. Verified with a live server against `../`,
  `../../`, `..%2f`, `%2e%2e`, `..%5c`, absolute paths, `package.json`,
  `karakter`, and more — all correctly rejected with 400, no file touched,
  server stays up.
- **Crash on malformed `chat.messages` (C2)** — `PUT /api/chat/:chatId` now
  rejects a non-array `messages` field at the source; defensive guards added
  in `send`/`regenerate`/`summarize` for any already-corrupted data.
- **Crash on `null` entries in `importantEvents` (C3)** — filtered at save
  time and defensively at read time.
- **Prototype pollution via `messageIndex` (H1)** — `messageIndex` is now
  validated as a real non-negative integer before use; `"__proto__"` and
  similar can no longer reach `Array.prototype`.
- **Double-click race in alternate navigation** — concurrent clicks used to
  each read a stale `selectedAlternate` and could overwrite each other,
  leaving the counter stuck. Now guarded with a per-message in-flight lock.
- **Regenerate had no busy-state feedback** — pressing Enter during a
  regenerate silently no-op'd; now shows a toast and disables the send
  button.
- **Regenerating a chat's first message sent zero conversation turns** to
  the model (only the system prompt) — now injects a minimal opening
  instruction so the model always has something to respond to.
- **Heuristic memory-extraction false positive** — `"hai suka gak sama
  aku?"` was misread as a "likes" fact because the pronoun pattern lacked a
  word boundary (matched the trailing "i" in "hai"). Fixed with `\b`.
- **`abort` timer leak** on every failed/errored LLM fetch attempt (was only
  cleared on the success path).
- **API key mismatch confusion (H11)** — Colab and Kaggle notebooks issue
  different default keys (`sk-colab-local` / `sk-kaggle-local`); a 401 now
  gets an explicit, actionable error message instead of a generic failure.
- **Kaggle warm-up check accepted a 404 as "ready" (H12)** — now requires an
  exact `200`.
- **Colab's `ollama pull` died on a known transient blob-race error (H10)**
  — retry logic ported from the Kaggle script; stale server processes from a
  re-run are now killed before starting a fresh one.
- Hardcoded `lang="id"` in `index.html` → now defaults to `en` and is set
  dynamically once the actual UI language loads; RTL (`dir="rtl"`) applied
  for Arabic.
- `[data-i18n-title]` attributes (4 elements) were never translated —
  handler added.
- `state.config` lost `uiLang`/`showLangWarning` after saving settings
  (object replacement instead of merge) — fixed.

### Deprecated
- `colab_llm_server.py` / `colab_llm_server.ipynb` / `kaggle_llm_server.py`
  (the Ollama + LiteLLM stack) are kept in the repo as an explicitly-labeled
  optional alternative — no longer the default path, not actively developed
  further.

### Known remaining issues (see project bug tracker / next session)
- Memory write-back (`autoExtractMemory`, `/summarize`) still writes using
  the copy of `memory` loaded at the start of the request rather than
  re-reading fresh immediately before `writeJSON` — a manual memory edit
  made while the LLM is generating can still be overwritten. (Chat writes
  already got this fix; memory did not yet.)
- Frontend async race-condition audit (capture-ID-before-`await` pattern) not
  yet confirmed complete for every async function — only `sendMessage`,
  `regenerateLastMessage`, and `changeAlternate` are confirmed fixed.
- Full `app.js` XSS audit (memory fields, event dates, facts, preferences,
  locale values) not yet done end-to-end.
- LLM request retry/timeout audit (distinguishing timeout from HTTP failure,
  unsafe POST retries) not yet done.
