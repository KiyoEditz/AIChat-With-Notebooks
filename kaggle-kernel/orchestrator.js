/**
 * ═══════════════════════════════════════════════════════════
 *  AIChat — Beta Kaggle Auto-Tunnel Orchestrator
 *  Folder: kaggle-kernel/orchestrator.js
 * ═══════════════════════════════════════════════════════════
 *
 * Lifecycle when `npm run beta-kaggle` (`node server.js --beta-kaggle`) runs:
 *  1. Load credentials from `kaggle-kernel/config/kaggle.json` (fallback: `~/.kaggle/kaggle.json`).
 *  2. Stage `kaggle-kernel/kernel/main.py` into `kaggle-kernel/.staging/kernel/` with a unique
 *     `RUN_ID` and the user's Kaggle credentials so the kernel can publish `tunnel_state.json`
 *     to the private dataset `<username>/aichat-kaggle-tunnel`.
 *  3. Push & start the Kaggle GPU kernel (`NvidiaTeslaT4`, Internet ON, Private).
 *  4. Poll kernel status & `<username>/aichat-kaggle-tunnel/tunnel_state.json` until the new
 *     `RUN_ID` appears with the live `https://xxxx.trycloudflare.com` URL.
 *  5. Automatically update `process.env.BASE_URL`, `process.env.API_KEY`, and `.env`.
 *  6. Send a heartbeat every 60s (`GET /__aichat_admin/heartbeat`) to keep the 30-minute
 *     Auto-Idle Watchdog alive.
 *  7. On `Ctrl + C` (`SIGINT` / `SIGTERM`), send `POST /__aichat_admin/shutdown` to the Kaggle
 *     kernel so it terminates (`COMPLETE`) immediately and preserves GPU quota.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');

const KAGGLE_ROOT = __dirname;
const CONFIG_DIR = path.join(KAGGLE_ROOT, 'config');
const KERNEL_TEMPLATE_DIR = path.join(KAGGLE_ROOT, 'kernel');
const STAGING_DIR = path.join(KAGGLE_ROOT, '.staging');
const STAGING_KERNEL_DIR = path.join(STAGING_DIR, 'kernel');
const STAGING_DOWNLOAD_DIR = path.join(STAGING_DIR, 'download');

const KERNEL_SLUG = 'aichat-kaggle-server';
const KERNEL_TITLE = 'aichat kaggle server';
const TUNNEL_DATASET_SLUG = 'aichat-kaggle-tunnel';
const HEARTBEAT_INTERVAL_MS = 60 * 1000; // 60 seconds
const POLL_INTERVAL_MS = 12 * 1000;      // 12 seconds

let orchestratorState = {
    active: false,
    phase: 'idle', // 'idle' | 'authenticating' | 'pushing' | 'waiting_tunnel' | 'connected' | 'stopping' | 'error'
    runId: null,
    username: null,
    kernelRef: null,
    datasetRef: null,
    tunnelUrl: null,
    apiKey: 'sk-kaggle-local',
    kernelStatus: null,
    error: null,
    startedAt: null,
    heartbeatTimer: null,
    abortPolling: false
};

function logBox(title, lines = []) {
    const bar = '═'.repeat(62);
    console.log(`\n╔${bar}╗`);
    console.log(`║  ${title.padEnd(60)}║`);
    if (lines.length > 0) {
        console.log(`╠${bar}╣`);
        for (const l of lines) {
            console.log(`║  ${String(l).slice(0, 60).padEnd(60)}║`);
        }
    }
    console.log(`╚${bar}╝`);
}

function logStep(step, total, msg) {
    console.log(`\n  [Beta-Kaggle ${step}/${total}] ${msg}`);
}

function logInfo(msg) {
    console.log(`    ↳ ${msg}`);
}

/**
 * Build isolated environment variables for Python `kaggle` CLI / `KaggleApi`
 * so the selected `kaggle.json` (Priority #1: `kaggle-kernel/config/kaggle.json`,
 * Priority #2: `~/.kaggle/kaggle.json`) never conflicts with the other account.
 */
function buildKagglePythonEnv(creds) {
    const env = {
        ...process.env,
        KAGGLE_CONFIG_DIR: path.dirname(creds.sourcePath),
        KAGGLE_USERNAME: creds.username,
        KAGGLE_KEY: creds.key,
        PYTHONIOENCODING: 'utf-8'
    };
    if (creds.key.startsWith('KGAT_')) {
        env.KAGGLE_API_TOKEN = creds.key;
    } else {
        delete env.KAGGLE_API_TOKEN;
    }
    return env;
}

/**
 * Locate Python executable that has the `kaggle` package installed (optional helper,
 * falls back to pure Node.js REST API calls if Python/kaggle is not installed).
 */
function findPythonWithKaggle(creds) {
    const userHome = os.homedir();
    const candidates = [
        path.join(userHome, 'AppData', 'Local', 'Programs', 'Python', 'Python314', 'python.exe'),
        path.join(userHome, 'AppData', 'Local', 'Programs', 'Python', 'Python313', 'python.exe'),
        path.join(userHome, 'AppData', 'Local', 'Programs', 'Python', 'Python312', 'python.exe'),
        path.join(userHome, 'AppData', 'Local', 'Programs', 'Python', 'Python311', 'python.exe'),
        'python3',
        'python'
    ];

    const pyEnv = creds ? buildKagglePythonEnv(creds) : { ...process.env, PYTHONIOENCODING: 'utf-8' };

    for (const cmd of candidates) {
        try {
            if (cmd.includes(path.sep) && !fs.existsSync(cmd)) continue;
            const res = spawnSync(cmd, ['-c', 'import kaggle; print("OK")'], {
                encoding: 'utf8',
                timeout: 8000,
                env: pyEnv
            });
            if (res.status === 0 && (res.stdout || '').includes('OK')) {
                return cmd;
            }
        } catch {
            // Try next candidate
        }
    }
    return null;
}

/**
 * Load Kaggle credentials with strict priority:
 *  - Priority #1: `kaggle-kernel/config/kaggle.json` (project config — e.g. 2nd account)
 *  - Priority #2: `~/.kaggle/kaggle.json` (system userdata fallback — e.g. main account)
 */
function loadCredentials() {
    const primaryPath = path.join(CONFIG_DIR, 'kaggle.json');
    const fallbackPath = path.join(os.homedir(), '.kaggle', 'kaggle.json');

    const candidateSources = [
        { filePath: primaryPath, priorityLabel: 'Prioritas #1 (kaggle-kernel/config/kaggle.json)' },
        { filePath: fallbackPath, priorityLabel: 'Prioritas #2 Fallback (~/.kaggle/kaggle.json)' }
    ];

    for (const src of candidateSources) {
        const p = src.filePath;
        if (!fs.existsSync(p)) continue;
        try {
            const raw = fs.readFileSync(p, 'utf8');
            const data = JSON.parse(raw);
            const username = (data.username || '').trim();
            const key = (data.key || data.token || '').trim();
            if (
                username &&
                key &&
                username !== 'ISI_USERNAME_KAGGLE_KAMU' &&
                key !== 'ISI_API_KEY_KAGGLE_KAMU'
            ) {
                return { username, key, sourcePath: p, priorityLabel: src.priorityLabel };
            }
        } catch (err) {
            throw new Error(`File kredensial Kaggle tidak valid (${p}): ${err.message}`);
        }
    }

    throw new Error(
        `Kredensial Kaggle belum ditemukan atau masih berisi template placeholder!\n` +
        `    1. Prioritas #1: Isi kredensial di ${primaryPath}\n` +
        `    2. Prioritas #2 (Fallback): Atau simpan di ${fallbackPath}\n` +
        `    3. Lihat panduan lengkap di: ${path.join(CONFIG_DIR, 'README.md')}`
    );
}

/**
 * Build Authorization header for direct Kaggle REST API v1 requests.
 */
function getKaggleAuthHeaders(creds) {
    const basic = Buffer.from(`${creds.username}:${creds.key}`).toString('base64');
    return {
        'Authorization': `Basic ${basic}`,
        'Accept': 'application/json',
        'User-Agent': 'AIChat-Beta-Kaggle/1.0'
    };
}

/**
 * Prepare staging folder (`kaggle-kernel/.staging/kernel/`) with injected `RUN_ID` and credentials.
 */
function stageKernelFiles(creds, runId) {
    fs.mkdirSync(STAGING_KERNEL_DIR, { recursive: true });
    fs.mkdirSync(STAGING_DOWNLOAD_DIR, { recursive: true });

    const templateMainPath = path.join(KERNEL_TEMPLATE_DIR, 'main.py');
    if (!fs.existsSync(templateMainPath)) {
        throw new Error(`Template kernel tidak ditemukan di: ${templateMainPath}`);
    }

    let scriptBody = fs.readFileSync(templateMainPath, 'utf8');
    scriptBody = scriptBody
        .replace('__AICHAT_RUN_ID__', runId)
        .replace('__KAGGLE_USERNAME__', creds.username)
        .replace('__KAGGLE_KEY__', creds.key);

    const stagedMainPath = path.join(STAGING_KERNEL_DIR, 'main.py');
    fs.writeFileSync(stagedMainPath, scriptBody, 'utf8');

    const kernelMeta = {
        id: `${creds.username}/${KERNEL_SLUG}`,
        title: KERNEL_TITLE,
        code_file: 'main.py',
        language: 'python',
        kernel_type: 'script',
        is_private: true,
        enable_gpu: true,
        enable_tpu: false,
        enable_internet: true,
        machine_shape: 'NvidiaTeslaT4',
        dataset_sources: [],
        competition_sources: [],
        kernel_sources: [],
        model_sources: [],
        keywords: ['llm', 'llama-cpp', 'aichat']
    };

    const stagedMetaPath = path.join(STAGING_KERNEL_DIR, 'kernel-metadata.json');
    fs.writeFileSync(stagedMetaPath, JSON.stringify(kernelMeta, null, 2), 'utf8');

    return { scriptBody, kernelMeta, stagedMainPath, stagedMetaPath };
}

/**
 * Push kernel to Kaggle using Python `KaggleApi` (if available) or direct Kaggle REST API v1.
 */
async function pushKernelToKaggle(creds, staged, pyExe) {
    if (pyExe) {
        const pyCode = [
            'import os, json, sys',
            'from kaggle.api.kaggle_api_extended import KaggleApi',
            'api = KaggleApi()',
            'api.authenticate()',
            `resp = api.kernels_push(${JSON.stringify(STAGING_KERNEL_DIR)}, timeout="28800", acc="NvidiaTeslaT4")`,
            'ver = getattr(resp, "version_number", None) or getattr(resp, "versionNumber", None) or 1',
            'url = getattr(resp, "url", None) or ""',
            'print(json.dumps({"ok": True, "version": int(ver), "url": str(url)}))'
        ].join('; ');

        const res = spawnSync(pyExe, ['-c', pyCode], {
            encoding: 'utf8',
            timeout: 60000,
            env: buildKagglePythonEnv(creds)
        });

        if (res.status === 0) {
            const lines = (res.stdout || '').trim().split(/\r?\n/);
            const lastLine = lines[lines.length - 1] || '';
            try {
                const parsed = JSON.parse(lastLine);
                if (parsed && parsed.ok) return parsed;
            } catch {
                // Fall through to REST API if output wasn't JSON
            }
        } else {
            const errText = (res.stderr || res.stdout || '').trim();
            logInfo(`Info KaggleApi Python: ${errText.slice(0, 200)} — mencoba jalur REST API langsung...`);
        }
    }

    // Pure Node.js REST API push (works on Mobile/Termux without Python!)
    const pushUrl = 'https://www.kaggle.com/api/v1/kernels/push';
    const payload = {
        slug: `${creds.username}/${KERNEL_SLUG}`,
        newTitle: KERNEL_TITLE,
        text: staged.scriptBody,
        language: 'python',
        kernelType: 'script',
        isPrivate: true,
        enableGpu: true,
        enableTpu: false,
        enableInternet: true,
        machineShape: 'NvidiaTeslaT4',
        sessionTimeoutSeconds: 28800,
        datasetDataSources: [],
        competitionDataSources: [],
        kernelDataSources: [],
        categoryIds: []
    };

    const response = await fetch(pushUrl, {
        method: 'POST',
        headers: {
            ...getKaggleAuthHeaders(creds),
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000)
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) {
        throw new Error(data.error || `HTTP ${response.status} saat push kernel ke Kaggle`);
    }

    return {
        ok: true,
        version: data.versionNumber || 1,
        url: data.url || `https://www.kaggle.com/code/${creds.username}/${KERNEL_SLUG}`
    };
}

/**
 * Query kernel status (`RUNNING`, `QUEUED`, `COMPLETE`, `ERROR`).
 */
async function getKernelStatus(creds, pyExe) {
    const kernelRef = `${creds.username}/${KERNEL_SLUG}`;
    if (pyExe) {
        const pyCode = [
            'import json',
            'from kaggle.api.kaggle_api_extended import KaggleApi',
            'api = KaggleApi()',
            'api.authenticate()',
            `resp = api.kernels_status(${JSON.stringify(kernelRef)})`,
            'st = getattr(resp, "status", "UNKNOWN")',
            'name = str(getattr(st, "name", None) or st).split(".")[-1].upper()',
            'msg = getattr(resp, "failure_message", None) or getattr(resp, "failureMessage", None)',
            'print(json.dumps({"status": name, "failureMessage": msg}))'
        ].join('; ');

        const res = spawnSync(pyExe, ['-c', pyCode], {
            encoding: 'utf8',
            timeout: 20000,
            env: buildKagglePythonEnv(creds)
        });
        if (res.status === 0) {
            const lines = (res.stdout || '').trim().split(/\r?\n/);
            try {
                return JSON.parse(lines[lines.length - 1]);
            } catch { }
        }
    }

    const url = `https://www.kaggle.com/api/v1/kernels/status?userName=${encodeURIComponent(creds.username)}&kernelSlug=${encodeURIComponent(KERNEL_SLUG)}`;
    const response = await fetch(url, {
        headers: getKaggleAuthHeaders(creds),
        signal: AbortSignal.timeout(15000)
    });
    if (!response.ok) return { status: 'UNKNOWN', failureMessage: null };
    const data = await response.json().catch(() => ({}));
    const rawStatus = String(data.status || 'UNKNOWN').split('.').pop().toUpperCase();
    return {
        status: rawStatus,
        failureMessage: data.failureMessage || null
    };
}

/**
 * Fetch `tunnel_state.json` from the user's private dataset `<username>/aichat-kaggle-tunnel`.
 */
async function fetchTunnelStateFromDataset(creds, pyExe) {
    const dsRef = `${creds.username}/${TUNNEL_DATASET_SLUG}`;

    if (pyExe) {
        const localFile = path.join(STAGING_DOWNLOAD_DIR, 'tunnel_state.json');
        try {
            if (fs.existsSync(localFile)) fs.unlinkSync(localFile);
        } catch { }

        const pyCode = [
            'import json',
            'from kaggle.api.kaggle_api_extended import KaggleApi',
            'api = KaggleApi()',
            'api.authenticate()',
            `api.dataset_download_file(${JSON.stringify(dsRef)}, "tunnel_state.json", path=${JSON.stringify(STAGING_DOWNLOAD_DIR)}, force=True, quiet=True)`,
            'print("DOWNLOAD_OK")'
        ].join('; ');

        const res = spawnSync(pyExe, ['-c', pyCode], {
            encoding: 'utf8',
            timeout: 25000,
            env: buildKagglePythonEnv(creds)
        });

        if (res.status === 0 && fs.existsSync(localFile)) {
            try {
                return JSON.parse(fs.readFileSync(localFile, 'utf8'));
            } catch { }
        }
    }

    // Pure Node.js REST fallback
    const downloadUrl = `https://www.kaggle.com/api/v1/datasets/download/${encodeURIComponent(creds.username)}/${encodeURIComponent(TUNNEL_DATASET_SLUG)}/tunnel_state.json`;
    try {
        const response = await fetch(downloadUrl, {
            headers: getKaggleAuthHeaders(creds),
            redirect: 'follow',
            signal: AbortSignal.timeout(15000)
        });
        if (!response.ok) return null;
        const text = await response.text();
        return JSON.parse(text);
    } catch {
        return null;
    }
}

/**
 * Persist the discovered Cloudflare Tunnel URL & API Key into `.env` and `process.env`.
 */
function applyTunnelConfigToEnv(projectRoot, tunnelUrl, apiKey) {
    process.env.BASE_URL = tunnelUrl;
    if (apiKey) process.env.API_KEY = apiKey;

    const envPath = path.join(projectRoot, '.env');
    let envContent = '';
    try {
        envContent = fs.readFileSync(envPath, 'utf8');
    } catch { }

    const updates = {
        BASE_URL: tunnelUrl,
        API_KEY: apiKey || process.env.API_KEY || 'sk-kaggle-local'
    };

    for (const [key, val] of Object.entries(updates)) {
        const regex = new RegExp(`^${key}=.*$`, 'm');
        if (regex.test(envContent)) {
            envContent = envContent.replace(regex, () => `${key}=${val}`);
        } else {
            envContent += (envContent.endsWith('\n') || !envContent ? '' : '\n') + `${key}=${val}\n`;
        }
    }

    fs.writeFileSync(envPath, envContent, 'utf8');
}

/**
 * Start periodic heartbeat (`GET /__aichat_admin/heartbeat`) every 60s
 * so the 30-minute Auto-Idle Watchdog on Kaggle knows AIChat is still running.
 */
function startHeartbeatLoop() {
    if (orchestratorState.heartbeatTimer) {
        clearInterval(orchestratorState.heartbeatTimer);
    }

    orchestratorState.heartbeatTimer = setInterval(async () => {
        if (!orchestratorState.tunnelUrl) return;
        try {
            const url = `${orchestratorState.tunnelUrl.replace(/\/+$/, '')}/__aichat_admin/heartbeat`;
            await fetch(url, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${orchestratorState.apiKey}`
                },
                signal: AbortSignal.timeout(10000)
            });
        } catch {
            // Transient network hiccup; 30-minute window allows plenty of retries
        }
    }, HEARTBEAT_INTERVAL_MS);

    if (orchestratorState.heartbeatTimer.unref) {
        orchestratorState.heartbeatTimer.unref();
    }
}

/**
 * Send immediate shutdown signal (`POST /__aichat_admin/shutdown`) to the running Kaggle kernel.
 */
async function shutdownKaggleServer(reason = 'AIChat server stopped') {
    orchestratorState.abortPolling = true;
    if (orchestratorState.heartbeatTimer) {
        clearInterval(orchestratorState.heartbeatTimer);
        orchestratorState.heartbeatTimer = null;
    }

    if (!orchestratorState.tunnelUrl) {
        return false;
    }

    const targetUrl = `${orchestratorState.tunnelUrl.replace(/\/+$/, '')}/__aichat_admin/shutdown`;
    orchestratorState.phase = 'stopping';
    console.log(`\n  🛑 [Beta-Kaggle] Mengirim sinyal shutdown ke server Kaggle (${orchestratorState.tunnelUrl})...`);

    try {
        const res = await fetch(targetUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${orchestratorState.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ reason }),
            signal: AbortSignal.timeout(8000)
        });
        if (res.ok) {
            console.log('  ✅ [Beta-Kaggle] Server Kaggle berhasil dimatikan (kuota GPU aman).');
            orchestratorState.tunnelUrl = null;
            return true;
        }
        console.warn(`  ⚠️ [Beta-Kaggle] Server Kaggle merespons HTTP ${res.status}. Auto-Idle Watchdog (30m) tetap menjaga kuota.`);
    } catch (err) {
        console.warn(`  ⚠️ [Beta-Kaggle] Gagal menghubungi endpoint shutdown (${err.message}). Auto-Idle Watchdog (30m) akan mematikan kernel otomatis.`);
    }
    return false;
}

/**
 * Register process signal handlers (`SIGINT` / `SIGTERM`) so `Ctrl + C` cleanly stops Kaggle first.
 */
let _signalsRegistered = false;
function registerShutdownHooks() {
    if (_signalsRegistered) return;
    _signalsRegistered = true;

    let shuttingDown = false;
    const handleExitSignal = async (sig) => {
        if (shuttingDown) return;
        shuttingDown = true;
        console.log(`\n  [Beta-Kaggle] Menerima sinyal ${sig}, menutup koneksi...`);
        await shutdownKaggleServer(`Process terminated via ${sig}`);
        process.exit(0);
    };

    process.on('SIGINT', () => handleExitSignal('SIGINT'));
    process.on('SIGTERM', () => handleExitSignal('SIGTERM'));
}

/**
 * Main entry point invoked by `server.js` when `--beta-kaggle` is passed.
 */
async function startBetaKaggleOrchestrator(projectRoot) {
    orchestratorState.active = true;
    orchestratorState.phase = 'authenticating';
    orchestratorState.startedAt = Date.now();
    registerShutdownHooks();

    logBox('🚀 BETA-KAGGLE AUTO-SERVER ORCHESTRATOR', [
        'Mode     : Automated Kaggle GPU Kernel + Auto-Tunnel',
        'Watchdog : 30 Menit Auto-Idle + Shutdown saat Ctrl+C'
    ]);

    try {
        // 1. Validate credentials
        logStep(1, 4, 'Memeriksa kredensial Kaggle (Prioritas #1: config/kaggle.json -> Prioritas #2: ~/.kaggle/kaggle.json)...');
        const creds = loadCredentials();
        const pyExe = findPythonWithKaggle(creds);
        orchestratorState.username = creds.username;
        orchestratorState.kernelRef = `${creds.username}/${KERNEL_SLUG}`;
        orchestratorState.datasetRef = `${creds.username}/${TUNNEL_DATASET_SLUG}`;
        logInfo(`Username Kaggle : ${creds.username}`);
        logInfo(`Prioritas Aktif : ${creds.priorityLabel}`);
        logInfo(`File Kredensial : ${creds.sourcePath}`);
        logInfo(`Engine API      : ${pyExe ? `Python KaggleApi (${pyExe})` : 'Node.js Direct REST API'}`);

        // 2. Stage kernel with unique RUN_ID
        const runId = `run_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        orchestratorState.runId = runId;
        orchestratorState.phase = 'pushing';
        logStep(2, 4, `Menyiapkan & mengirim kernel ke Kaggle (${orchestratorState.kernelRef})...`);
        const staged = stageKernelFiles(creds, runId);

        const pushRes = await pushKernelToKaggle(creds, staged, pyExe);
        logInfo(`Kernel berhasil di-push! (Version #${pushRes.version}, Run ID: ${runId})`);
        if (pushRes.url) logInfo(`URL Kernel: ${pushRes.url}`);

        // 3. Poll kernel status & dataset tunnel_state.json
        orchestratorState.phase = 'waiting_tunnel';
        logStep(3, 4, 'Menunggu server Kaggle menyala & menerbitkan URL Cloudflare Tunnel...');
        logInfo('Proses unduh model (~27 GB) ke VRAM T4 x2 biasanya memakan waktu 3–7 menit.');
        logInfo('Server lokal AIChat sudah aktif — URL akan otomatis disambungkan begitu siap.');

        const timeoutMs = 25 * 60 * 1000; // 25 minutes max wait for startup
        const pollStart = Date.now();
        let lastPrintedStatus = '';

        while (!orchestratorState.abortPolling && (Date.now() - pollStart) < timeoutMs) {
            await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
            if (orchestratorState.abortPolling) return;

            const elapsedSec = Math.round((Date.now() - pollStart) / 1000);
            const mins = Math.floor(elapsedSec / 60);
            const secs = elapsedSec % 60;
            const timeLabel = `${mins}m ${String(secs).padStart(2, '0')}s`;

            // Check kernel status
            const kStatus = await getKernelStatus(creds, pyExe);
            orchestratorState.kernelStatus = kStatus.status;

            if (kStatus.status !== lastPrintedStatus || elapsedSec % 36 < 12) {
                console.log(`    ⏳ [${timeLabel}] Status Kernel Kaggle: ${kStatus.status}`);
                lastPrintedStatus = kStatus.status;
            }

            if (['ERROR', 'FAILED', 'CANCELLED', 'CANCELED'].includes(kStatus.status)) {
                throw new Error(
                    `Kernel Kaggle berhenti dengan status ${kStatus.status}` +
                    (kStatus.failureMessage ? `: ${kStatus.failureMessage}` : '. Pastikan akun Kaggle sudah Phone Verified & GPU T4 tersedia.')
                );
            }

            // Check private dataset for matching run_id
            const tunnelState = await fetchTunnelStateFromDataset(creds, pyExe);
            if (tunnelState && tunnelState.run_id === runId && tunnelState.tunnel_url) {
                orchestratorState.tunnelUrl = tunnelState.tunnel_url;
                orchestratorState.apiKey = tunnelState.api_key || 'sk-kaggle-local';
                orchestratorState.phase = 'connected';

                // 4. Apply to AIChat environment & start heartbeat
                logStep(4, 4, 'URL Cloudflare Tunnel diterima dari Kaggle!');
                applyTunnelConfigToEnv(projectRoot, orchestratorState.tunnelUrl, orchestratorState.apiKey);
                startHeartbeatLoop();

                logBox('✅ KAGGLE SERVER TERHUBUNG OTOMATIS!', [
                    `Run ID   : ${runId}`,
                    `BASE_URL : ${orchestratorState.tunnelUrl}`,
                    `Model    : ${tunnelState.model_repo || 'Active GGUF'}`,
                    ` Shutdown : Tekan Ctrl+C untuk mematikan server Kaggle`
                ]);
                return;
            }
        }

        if (!orchestratorState.abortPolling) {
            throw new Error('Waktu tunggu (25 menit) habis saat menunggu URL Cloudflare dari Kaggle.');
        }
    } catch (err) {
        orchestratorState.phase = 'error';
        orchestratorState.error = err.message;
        console.error(`\n  ❌ [Beta-Kaggle Error] ${err.message}\n`);
    }
}

function getBetaKaggleStatus() {
    return {
        active: orchestratorState.active,
        phase: orchestratorState.phase,
        runId: orchestratorState.runId,
        username: orchestratorState.username,
        kernelRef: orchestratorState.kernelRef,
        datasetRef: orchestratorState.datasetRef,
        tunnelUrl: orchestratorState.tunnelUrl,
        kernelStatus: orchestratorState.kernelStatus,
        error: orchestratorState.error,
        startedAt: orchestratorState.startedAt
    };
}

module.exports = {
    startBetaKaggleOrchestrator,
    shutdownKaggleServer,
    getBetaKaggleStatus
};
