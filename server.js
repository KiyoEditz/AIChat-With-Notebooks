/**
 * ═══════════════════════════════════════════════════════════
 *  AI Character Chat — Local Server
 *  Connects to a remote llama-server (llama.cpp) instance via
 *  Cloudflare Tunnel — see Collab-Llama.py / Kaggle-Llama.py
 * ═══════════════════════════════════════════════════════════
 */

require('dotenv').config();
const express = require('express');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3000;

// ── Data directories ────────────────────────────────────────
const DATA_DIR = path.join(__dirname, 'data');
const CHARACTERS_DIR = path.join(DATA_DIR, 'characters');
const CHATS_DIR = path.join(DATA_DIR, 'chats');
const MEMORY_DIR = path.join(DATA_DIR, 'memory');

[DATA_DIR, CHARACTERS_DIR, CHATS_DIR, MEMORY_DIR].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// ── Helper: safe path construction (path traversal fix) ──────
// Every ID in this app is a crypto.randomUUID() string. A strict allowlist
// on the ID itself categorically blocks path traversal in any encoding
// (../, ..%2f, %2e%2e, %5c, absolute paths, etc.) without needing to
// enumerate individual attack patterns -- anything that isn't
// [A-Za-z0-9_-] is rejected outright, so there's nothing left for a
// traversal sequence to hide in. The resolved-path containment check below
// is defense-in-depth on top of that, not the primary guard.
const ID_RE = /^[A-Za-z0-9_-]{1,200}$/;

function safePath(baseDir, id) {
    if (typeof id !== 'string' || !ID_RE.test(id)) return null;
    const resolvedBase = path.resolve(baseDir) + path.sep;
    const resolved = path.resolve(baseDir, `${id}.json`);
    if (!resolved.startsWith(resolvedBase)) return null;
    return resolved;
}

// ── Middleware ───────────────────────────────────────────────
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Helper: read/write JSON ─────────────────────────────────
function readJSON(filePath, fallback = null) {
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch {
        return fallback;
    }
}

function writeJSON(filePath, data) {
    const tmpPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2), 'utf8');
    fs.renameSync(tmpPath, filePath); // atomic on POSIX filesystems — no partial-write window
}

// ── Helper: get model parameters from env ────────────────────
// ══════════════════════════════════════════════════════════════
//  NATIVE LLAMA.CPP SAMPLING PARAMETERS
// ══════════════════════════════════════════════════════════════
// Single canonical config: these map 1:1 to llama-server's own JSON body
// fields (verified against llama.cpp tag b10605's server README — see
// Build-Llama-CUDA-Release.py). No OLLAMA_*/LITELLM_* duplicate pairs, no
// translation layer, no ambiguity about whether a value is actually being
// used. Defaults here intentionally match Collab-Llama.py / Kaggle-Llama.py
// exactly (same repeat_penalty/DRY/etc. values) — this app always sends
// these explicitly on every request, which overrides whatever the notebook
// set as its own CLI startup defaults, so the two MUST agree or one of them
// is dead configuration. If you change a default, change it in both places.

function parseValidated(val, { type = 'float', defaultVal, min = -Infinity, max = Infinity, label }) {
    if (val === undefined || val === null || val === '') return defaultVal;
    const parsed = type === 'int' ? parseInt(val, 10) : parseFloat(val);
    if (!Number.isFinite(parsed)) {
        console.warn(`  \u26a0\ufe0f  ${label || 'param'}="${val}" is not a finite number — using default (${defaultVal})`);
        return defaultVal;
    }
    if (parsed < min || parsed > max) {
        console.warn(`  \u26a0\ufe0f  ${label || 'param'}=${parsed} is outside valid range [${min}, ${max}] — using default (${defaultVal})`);
        return defaultVal;
    }
    return parsed;
}

function parseBoolEnv(val, defaultVal) {
    if (val === undefined || val === null || val === '') return defaultVal;
    return /^(1|true|on|yes)$/i.test(String(val).trim());
}

function parseListEnv(val, defaultVal) {
    if (val === undefined || val === null || val === '') return defaultVal;
    return String(val).split(',').map(s => s.trim()).filter(Boolean);
}

function getModelParams() {
    return {
        // Core generation
        max_tokens:   parseValidated(process.env.MAX_TOKENS,   { type: 'int',   defaultVal: 1024, min: 1,    max: 32768, label: 'MAX_TOKENS' }),
        temperature:  parseValidated(process.env.TEMPERATURE,  { defaultVal: 0.80, min: 0,   max: 5,   label: 'TEMPERATURE' }),
        top_k:        parseValidated(process.env.TOP_K,        { type: 'int', defaultVal: 50,   min: 0,    max: 1000,  label: 'TOP_K' }),
        top_p:        parseValidated(process.env.TOP_P,        { defaultVal: 0.90, min: 0,   max: 1,   label: 'TOP_P' }),
        min_p:        parseValidated(process.env.MIN_P,        { defaultVal: 0.05, min: 0,   max: 1,   label: 'MIN_P' }),
        typical_p:    parseValidated(process.env.TYPICAL_P,    { defaultVal: 1.0,  min: 0,   max: 1,   label: 'TYPICAL_P' }),
        tfs_z:        parseValidated(process.env.TFS_Z,        { defaultVal: 1.0,  min: 0,   max: 1,   label: 'TFS_Z' }),
        top_n_sigma:  parseValidated(process.env.TOP_N_SIGMA,  { defaultVal: -1,   min: -1,  max: 100, label: 'TOP_N_SIGMA' }),

        // Windowed repetition control — bounded to the last N tokens, not
        // the whole conversation. This is the direct fix for the original
        // "common words get suppressed over a long chat" concern.
        repeat_last_n:  parseValidated(process.env.REPEAT_LAST_N,  { type: 'int', defaultVal: 256,  min: 0, max: 8192, label: 'REPEAT_LAST_N' }),
        repeat_penalty: parseValidated(process.env.REPEAT_PENALTY, { defaultVal: 1.05, min: 0, max: 3,    label: 'REPEAT_PENALTY' }),

        // OpenAI-style, cumulative over the WHOLE context by spec. Default
        // to 0 (disabled) on purpose — see README for why. Must match the
        // notebook scripts' defaults exactly.
        presence_penalty:  parseValidated(process.env.PRESENCE_PENALTY,  { defaultVal: 0.0, min: -2, max: 2, label: 'PRESENCE_PENALTY' }),
        frequency_penalty: parseValidated(process.env.FREQUENCY_PENALTY, { defaultVal: 0.0, min: -2, max: 2, label: 'FREQUENCY_PENALTY' }),

        // DRY sampling — penalizes repeated phrases/loops, not individual
        // common tokens. The purpose-built fix for long roleplay chats.
        dry_multiplier:        parseValidated(process.env.DRY_MULTIPLIER,       { defaultVal: 0.8,  min: 0, max: 5,    label: 'DRY_MULTIPLIER' }),
        dry_base:              parseValidated(process.env.DRY_BASE,             { defaultVal: 1.75, min: 1, max: 4,    label: 'DRY_BASE' }),
        dry_allowed_length:    parseValidated(process.env.DRY_ALLOWED_LENGTH,   { type: 'int', defaultVal: 2,    min: 1, max: 50,   label: 'DRY_ALLOWED_LENGTH' }),
        dry_penalty_last_n:    parseValidated(process.env.DRY_PENALTY_LAST_N,   { type: 'int', defaultVal: 1024, min: -1, max: 16384, label: 'DRY_PENALTY_LAST_N' }),
        dry_sequence_breakers: parseListEnv(process.env.DRY_SEQUENCE_BREAKERS, ["\n", ":", "\"", "*"]),

        // XTC — trims away the single most-probable token some of the time,
        // to reduce "boring"/predictable output. Off by default (probability
        // 0) since it's more experimental / can occasionally cut off short
        // valid replies; conservative default per spec.
        xtc_probability: parseValidated(process.env.XTC_PROBABILITY, { defaultVal: 0.0, min: 0, max: 1, label: 'XTC_PROBABILITY' }),
        xtc_threshold:   parseValidated(process.env.XTC_THRESHOLD,   { defaultVal: 0.1, min: 0, max: 1, label: 'XTC_THRESHOLD' }),

        // Dynamic temperature
        dynatemp_range: parseValidated(process.env.DYNATEMP_RANGE, { defaultVal: 0.0, min: 0, max: 5, label: 'DYNATEMP_RANGE' }),
        dynatemp_exponent: parseValidated(process.env.DYNATEMP_EXP, { defaultVal: 1.0, min: 0, max: 10, label: 'DYNATEMP_EXP' }),

        // Mirostat — off (0) by default; when on, it takes over from
        // top_k/top_p/temperature, so it's opt-in only via env.
        mirostat:    parseValidated(process.env.MIROSTAT,    { type: 'int', defaultVal: 0,   min: 0, max: 2,  label: 'MIROSTAT' }),
        mirostat_lr: parseValidated(process.env.MIROSTAT_LR, { defaultVal: 0.1, min: 0, max: 5,  label: 'MIROSTAT_LR' }),
        mirostat_ent: parseValidated(process.env.MIROSTAT_ENT, { defaultVal: 5.0, min: 0, max: 20, label: 'MIROSTAT_ENT' }),

        // Reproducibility / sampler chain
        seed: parseValidated(process.env.SEED, { type: 'int', defaultVal: -1, min: -1, max: 2147483647, label: 'SEED' }),
        samplers: parseListEnv(process.env.SAMPLERS, undefined), // undefined = let llama-server use its own default chain
    };
}

// ══════════════════════════════════════════════════════════════
//  CONFIG API — read/update .env settings
// ══════════════════════════════════════════════════════════════
let hasShownLangWarning = false;


app.get('/api/config', (req, res) => {
    let uiLang = process.env.UI_LANG ? process.env.UI_LANG.toUpperCase() : '';
    let showLangWarning = false;

    // Fallback logic
    if (!uiLang || !fs.existsSync(path.join(__dirname, 'public', 'Lang', `${uiLang}.json`))) {
        uiLang = 'EN';
        if (!hasShownLangWarning) {
            showLangWarning = true;
            hasShownLangWarning = true; // Only show once per server start
        }
    }

    res.json({
        baseUrl: process.env.BASE_URL || '',
        apiKey: process.env.API_KEY || 'sk-colab-local',
        modelName: process.env.MODEL_NAME || 'character1',
        port: process.env.PORT || 3000,
        uiLang: uiLang,
        showLangWarning: showLangWarning
    });
});

app.post('/api/config', (req, res) => {
    const { baseUrl, apiKey, modelName } = req.body;
    const envPath = path.join(__dirname, '.env');
    let envContent = '';
    try { envContent = fs.readFileSync(envPath, 'utf8'); } catch { }

    const updates = {};
    if (baseUrl !== undefined) updates.BASE_URL = baseUrl;
    if (apiKey !== undefined) updates.API_KEY = apiKey;
    if (modelName !== undefined) updates.MODEL_NAME = modelName;

    for (const [key, val] of Object.entries(updates)) {
        const regex = new RegExp(`^${key}=.*$`, 'm');
        // Using a replacer function (not a string) avoids `$&`, `$1`, etc. in
        // `val` being interpreted as regex replacement patterns, which could
        // otherwise corrupt the .env file if a pasted URL/key contained "$".
        if (regex.test(envContent)) {
            envContent = envContent.replace(regex, () => `${key}=${val}`);
        } else {
            envContent += `\n${key}=${val}`;
        }
        process.env[key] = val;
    }

    const tmpPath = `${envPath}.${process.pid}.${Date.now()}.tmp`;
    fs.writeFileSync(tmpPath, envContent, 'utf8');
    fs.renameSync(tmpPath, envPath);
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  CHARACTER API — CRUD for character profiles
// ══════════════════════════════════════════════════════════════

app.get('/api/characters', (req, res) => {
    const files = fs.readdirSync(CHARACTERS_DIR).filter(f => f.endsWith('.json'));
    const chars = files.map(f => readJSON(path.join(CHARACTERS_DIR, f))).filter(Boolean);
    chars.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    res.json(chars);
});

app.get('/api/characters/:id', (req, res) => {
    const fp = safePath(CHARACTERS_DIR, req.params.id);
    if (!fp) return res.status(400).json({ error: 'Invalid character id' });
    const char = readJSON(fp);
    if (!char) return res.status(404).json({ error: 'Character not found' });
    res.json(char);
});

app.post('/api/characters', (req, res) => {
    const id = uuidv4();
    const now = new Date().toISOString();
    const char = {
        id,
        name: req.body.name || 'Unnamed',
        gender: req.body.gender || '',
        age: req.body.age || '',
        avatar: req.body.avatar || '',
        personality: req.body.personality || '',
        scenario: req.body.scenario || '',
        firstMessage: req.body.firstMessage || '',
        exampleDialogue: req.body.exampleDialogue || '',
        systemPrompt: req.body.systemPrompt || '',
        tags: req.body.tags || [],
        createdAt: now,
        updatedAt: now
    };
    writeJSON(path.join(CHARACTERS_DIR, `${id}.json`), char);

    // Initialize memory for this character
    writeJSON(path.join(MEMORY_DIR, `${id}.json`), {
        characterId: id,
        facts: [],
        summaries: [],
        importantEvents: [],
        userPreferences: {},
        lastUpdated: now
    });

    res.json(char);
});

app.put('/api/characters/:id', (req, res) => {
    const fp = safePath(CHARACTERS_DIR, req.params.id);
    if (!fp) return res.status(400).json({ error: 'Invalid character id' });
    const existing = readJSON(fp);
    if (!existing) return res.status(404).json({ error: 'Not found' });

    const updated = {
        id: existing.id,
        name: req.body.name ?? existing.name,
        gender: req.body.gender ?? existing.gender,
        age: req.body.age ?? existing.age,
        avatar: req.body.avatar ?? existing.avatar,
        personality: req.body.personality ?? existing.personality,
        scenario: req.body.scenario ?? existing.scenario,
        firstMessage: req.body.firstMessage ?? existing.firstMessage,
        exampleDialogue: req.body.exampleDialogue ?? existing.exampleDialogue,
        systemPrompt: req.body.systemPrompt ?? existing.systemPrompt,
        tags: req.body.tags ?? existing.tags,
        createdAt: existing.createdAt,
        updatedAt: new Date().toISOString()
    };
    writeJSON(fp, updated);
    res.json(updated);
});

app.delete('/api/characters/:id', (req, res) => {
    const fp = safePath(CHARACTERS_DIR, req.params.id);
    const memFp = safePath(MEMORY_DIR, req.params.id);
    if (!fp || !memFp) return res.status(400).json({ error: 'Invalid character id' });
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
    // Also delete associated memory
    if (fs.existsSync(memFp)) fs.unlinkSync(memFp);
    // Delete all chats for this character
    const chatFiles = fs.readdirSync(CHATS_DIR).filter(f => f.endsWith('.json'));
    chatFiles.forEach(f => {
        const chat = readJSON(path.join(CHATS_DIR, f));
        if (chat && chat.characterId === req.params.id) {
            fs.unlinkSync(path.join(CHATS_DIR, f));
        }
    });
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  CHAT HISTORY API
// ══════════════════════════════════════════════════════════════

// List all chats for a character
app.get('/api/chats/:characterId', (req, res) => {
    const files = fs.readdirSync(CHATS_DIR).filter(f => f.endsWith('.json'));
    const chats = files
        .map(f => readJSON(path.join(CHATS_DIR, f)))
        .filter(c => c && c.characterId === req.params.characterId)
        .map(c => ({
            id: c.id,
            characterId: c.characterId,
            title: c.title || 'Untitled Chat',
            messageCount: (c.messages || []).length,
            createdAt: c.createdAt,
            updatedAt: c.updatedAt
        }));
    chats.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    res.json(chats);
});

// Get full chat
app.get('/api/chat/:chatId', (req, res) => {
    const fp = safePath(CHATS_DIR, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid chat id' });
    const chat = readJSON(fp);
    if (!chat) return res.status(404).json({ error: 'Chat not found' });
    res.json(chat);
});

// Create new chat
app.post('/api/chats', (req, res) => {
    const id = uuidv4();
    const now = new Date().toISOString();
    const chat = {
        id,
        characterId: req.body.characterId,
        title: req.body.title || 'New Chat',
        messages: req.body.messages || [],
        createdAt: now,
        updatedAt: now
    };
    writeJSON(path.join(CHATS_DIR, `${id}.json`), chat);
    res.json(chat);
});

// Update chat (add messages, rename, etc.)
app.put('/api/chat/:chatId', (req, res) => {
    const fp = safePath(CHATS_DIR, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid chat id' });
    const existing = readJSON(fp);
    if (!existing) return res.status(404).json({ error: 'Not found' });

    if (req.body.messages !== undefined && !Array.isArray(req.body.messages)) {
        return res.status(400).json({ error: '"messages" must be an array' });
    }

    const updated = {
        ...existing,
        title: req.body.title ?? existing.title,
        messages: req.body.messages ?? existing.messages,
        updatedAt: new Date().toISOString()
    };
    writeJSON(fp, updated);
    res.json(updated);
});

// Delete chat
app.delete('/api/chat/:chatId', (req, res) => {
    const fp = safePath(CHATS_DIR, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid chat id' });
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  MEMORY API — Persistent memory for each character
// ══════════════════════════════════════════════════════════════

app.get('/api/memory/:characterId', (req, res) => {
    const fp = safePath(MEMORY_DIR, req.params.characterId);
    if (!fp) return res.status(400).json({ error: 'Invalid character id' });
    const mem = readJSON(fp, {
        characterId: req.params.characterId,
        facts: [],
        summaries: [],
        importantEvents: [],
        userPreferences: {},
        lastUpdated: new Date().toISOString()
    });
    res.json(mem);
});

app.put('/api/memory/:characterId', (req, res) => {
    const fp = safePath(MEMORY_DIR, req.params.characterId);
    if (!fp) return res.status(400).json({ error: 'Invalid character id' });
    const existing = readJSON(fp, { characterId: req.params.characterId, facts: [], summaries: [], importantEvents: [], userPreferences: {} });
    const updated = {
        ...existing,
        facts: req.body.facts ?? existing.facts,
        summaries: req.body.summaries ?? existing.summaries,
        importantEvents: (req.body.importantEvents ?? existing.importantEvents ?? []).filter(e => e && typeof e === 'object'),
        userPreferences: req.body.userPreferences ?? existing.userPreferences,
        lastUpdated: new Date().toISOString()
    };
    writeJSON(fp, updated);
    res.json(updated);
});

// ══════════════════════════════════════════════════════════════
//  FETCH WITH RETRY — handles Cloudflare 524 timeout errors
// ══════════════════════════════════════════════════════════════

/**
 * Fetch with automatic retry for Cloudflare transient errors.
 * Cloudflare free tunnels timeout at ~100s (error 524).
 * This retries up to maxRetries times with increasing wait.
 */
async function fetchWithRetry(url, options, maxRetries = 2) {
    const TIMEOUT_MS = 120000; // 120 seconds per attempt
    const RETRYABLE_CODES = [524, 502, 503, 504, 408];

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            // If retryable error and not last attempt, retry
            if (RETRYABLE_CODES.includes(response.status) && attempt < maxRetries) {
                const waitSec = (attempt + 1) * 5;
                console.log(`⚠️  Got ${response.status}, retrying in ${waitSec}s... (attempt ${attempt + 1}/${maxRetries})`);
                await new Promise(r => setTimeout(r, waitSec * 1000));
                continue;
            }

            return response;

        } catch (err) {
            clearTimeout(timeoutId); // was previously only cleared on the success path — leaked on every error
            if (err.name === 'AbortError' && attempt < maxRetries) {
                const waitSec = (attempt + 1) * 5;
                console.log(`⚠️  Request timed out, retrying in ${waitSec}s... (attempt ${attempt + 1}/${maxRetries})`);
                await new Promise(r => setTimeout(r, waitSec * 1000));
                continue;
            }
            throw err;
        }
    }
}

/**
 * Parse error response — strips HTML and returns clean message
 */
function parseErrorResponse(status, text) {
    if (status === 401) {
        return 'Authentication failed (401). Your API_KEY doesn\'t match the notebook: Colab uses "sk-colab-local" and Kaggle uses "sk-kaggle-local" by default. Copy the exact API_KEY printed at the end of your notebook into Settings.';
    }
    // Cloudflare error pages are HTML — extract the useful part
    if (text.includes('trycloudflare.com') || text.includes('cloudflare')) {
        if (status === 524) return 'Colab server timeout (524). Model mungkin sedang warm-up atau overloaded. Coba lagi dalam 30 detik.';
        if (status === 502) return 'Colab server down (502). Pastikan notebook Colab masih berjalan.';
        if (status === 503) return 'Colab server unavailable (503). Cek notebook Colab.';
        if (status === 504) return 'Gateway timeout (504). Model terlalu lama merespons.';
        return `Cloudflare error ${status}. Cek apakah Colab notebook masih aktif.`;
    }
    // Truncate if too long
    if (text.length > 300) return text.substring(0, 300) + '...';
    return text;
}

// ══════════════════════════════════════════════════════════════
//  CHAT COMPLETION PROXY — Forward to llama-server with memory
// ══════════════════════════════════════════════════════════════

app.post('/api/chat/send', async (req, res) => {
    const { characterId, chatId, userMessage } = req.body;

    if (!characterId || !chatId || !userMessage) {
        return res.status(400).json({ error: 'Missing characterId, chatId, or userMessage' });
    }
    if (typeof userMessage !== 'string') {
        return res.status(400).json({ error: 'userMessage must be a string' });
    }

    const baseUrl = process.env.BASE_URL;
    const apiKey = process.env.API_KEY || 'sk-colab-local';
    const modelName = process.env.MODEL_NAME || 'character1';

    if (!baseUrl || baseUrl.includes('your-colab-url')) {
        return res.status(400).json({ error: 'Please configure your Colab BASE_URL in Settings first.' });
    }

    const charFp = safePath(CHARACTERS_DIR, characterId);
    const memFp = safePath(MEMORY_DIR, characterId);
    const chatFp = safePath(CHATS_DIR, chatId);
    if (!charFp || !memFp || !chatFp) {
        return res.status(400).json({ error: 'Invalid characterId or chatId' });
    }

    // Load character
    const char = readJSON(charFp);
    if (!char) return res.status(404).json({ error: 'Character not found' });

    // Load memory
    const memory = readJSON(memFp, {
        facts: [], summaries: [], importantEvents: [], userPreferences: {}
    });

    // Load chat history
    const chat = readJSON(chatFp);
    if (!chat) return res.status(404).json({ error: 'Chat not found' });
    if (chat.characterId !== characterId) {
        return res.status(400).json({ error: 'This chat does not belong to the given character' });
    }
    if (!Array.isArray(chat.messages)) chat.messages = [];

    // Resolve any pending alternates on the last message BEFORE building the
    // history sent to the model, so a previously-regenerated-and-selected
    // reply is what actually gets sent — not the original/rejected content.
    if (chat.messages.length > 0) {
        const lastMsg = chat.messages[chat.messages.length - 1];
        if (lastMsg.role === 'assistant' && lastMsg.alternates) {
            const selIdx = lastMsg.selectedAlternate || 0;
            lastMsg.content = lastMsg.alternates[selIdx] || lastMsg.content;
            delete lastMsg.alternates;
            delete lastMsg.selectedAlternate;
        }
    }

    // ── Build system prompt with memory ──
    let systemParts = [];

    // Core character identity
    let genderStr = char.gender ? ` You are a ${char.gender}.` : '';
    let ageStr = char.age ? ` You are ${char.age} years old.` : '';
    systemParts.push(`You are ${char.name}.${genderStr}${ageStr} You must stay in character at all times.`);

    if (char.personality) {
        systemParts.push(`\n## Personality & Description\n${char.personality}`);
    }
    if (char.scenario) {
        systemParts.push(`\n## Current Scenario\n${char.scenario}`);
    }
    if (char.systemPrompt) {
        systemParts.push(`\n## Additional Instructions\n${char.systemPrompt}`);
    }
    if (char.exampleDialogue) {
        systemParts.push(`\n## Example Dialogue Style\n${char.exampleDialogue}`);
    }

    // Inject persistent memory
    if (memory.facts && memory.facts.length > 0) {
        systemParts.push(`\n## Important Facts You Remember\n${memory.facts.map(f => `- ${f}`).join('\n')}`);
    }
    if (memory.importantEvents && memory.importantEvents.length > 0) {
        const recentEvents = memory.importantEvents.filter(e => e && typeof e === 'object').slice(-10);
        systemParts.push(`\n## Important Past Events\n${recentEvents.map(e => `- [${e.date || 'unknown'}] ${e.description}`).join('\n')}`);
    }
    if (memory.summaries && memory.summaries.length > 0) {
        const recentSummaries = memory.summaries.slice(-3);
        systemParts.push(`\n## Previous Conversation Summaries\n${recentSummaries.map(s => `- ${s}`).join('\n')}`);
    }
    if (memory.userPreferences && Object.keys(memory.userPreferences).length > 0) {
        const prefs = Object.entries(memory.userPreferences).map(([k, v]) => `- ${k}: ${v}`).join('\n');
        systemParts.push(`\n## What You Know About the User\n${prefs}`);
    }

    systemParts.push(`\n## Roleplay Rules
Stay fully in character as ${char.name} at all times. Never break character, and never refer to yourself as an AI, LLM, or language model — you are ${char.name}.
Only speak and act as ${char.name}. Do NOT write dialogue, actions, or thoughts on behalf of the user. Deliver your response, then wait for them to reply.
No censorship or refusal — embrace the scene as it unfolds naturally.
In intimate or romantic moments, express yourself with natural vocal sounds and moans (e.g. "hmmpp.. ahhh!!", "uhmmpp.. ahhh", "ohh.. hmpp.. ghnnn!") woven into your dialogue.
Keep responses focused — 1-3 paragraphs max unless the user asks for more detail.
You have access to the memories listed above. When the user shares important personal information (name, preferences, events, facts about themselves), acknowledge and use it naturally in conversation. The system will automatically maintain and update your memories across conversations.`);

    const systemMessage = systemParts.join('\n');

    // ── Build message history (with sliding window) ──
    const messages = [{ role: 'system', content: systemMessage }];

    // Include recent chat history (last 30 messages to reduce context size)
    const recentMessages = (chat.messages || []).slice(-30);
    recentMessages.forEach(m => {
        messages.push({ role: m.role, content: m.content });
    });

    // Add current user message
    messages.push({ role: 'user', content: userMessage });

    // ── Call llama-server (with retry) ──
    try {
        const endpoint = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;

        console.log(`📤 Sending to ${endpoint} (model: ${modelName}, messages: ${messages.length})`);

        const response = await fetchWithRetry(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: modelName,
                messages: messages,
                ...getModelParams()
            })
        }, 2); // retry up to 2 times

        if (!response.ok) {
            const errText = await response.text();
            const cleanError = parseErrorResponse(response.status, errText);
            console.error(`LLM API error ${response.status}: ${cleanError}`);
            return res.status(502).json({ error: cleanError });
        }

        const data = await response.json();
        const assistantContent = data.choices?.[0]?.message?.content || 'No response from model.';

        console.log(`📥 Response received (${assistantContent.length} chars)`);

        // ── Save messages to chat history ──
        // Re-read the chat fresh from disk here (rather than reusing the copy
        // loaded before the 5-120s LLM call) so we don't clobber an edit or
        // regenerate that happened on this chat while we were waiting (C5).
        const now = new Date().toISOString();
        const freshChat = readJSON(chatFp) || chat;
        if (!Array.isArray(freshChat.messages)) freshChat.messages = [];

        freshChat.messages.push({ role: 'user', content: userMessage, timestamp: now });
        freshChat.messages.push({ role: 'assistant', content: assistantContent, timestamp: now });

        // Auto-set chat title from first user message
        if (freshChat.messages.length <= 2) {
            freshChat.title = userMessage.substring(0, 60) + (userMessage.length > 60 ? '…' : '');
        }

        freshChat.updatedAt = now;
        writeJSON(chatFp, freshChat);

        // ── Auto-extract memory after every exchange ──
        // Note: intentionally NOT passing the `memory` object loaded at the
        // top of this handler — it was read before the 5-120s LLM call, so
        // by now it may be stale (see autoExtractMemory's own fresh re-read,
        // C6). The function reads its own up-to-date copy right before it
        // decides what to write.
        autoExtractMemory(memFp, userMessage, assistantContent);

        res.json({
            content: assistantContent,
            chatId: freshChat.id,
            messageIndex: freshChat.messages.length - 1
        });

    } catch (err) {
        console.error('Fetch error:', err);
        const msg = err.name === 'AbortError'
            ? 'Request timed out setelah 120 detik. Model Colab mungkin overloaded. Coba kirim pesan yang lebih pendek atau tunggu 1 menit.'
            : `Connection failed: ${err.message}. Cek BASE_URL dan pastikan Colab masih running.`;
        res.status(500).json({ error: msg });
    }
});

// ══════════════════════════════════════════════════════════════
//  CHAT REGENERATE PROXY
// ══════════════════════════════════════════════════════════════

app.post('/api/chat/regenerate', async (req, res) => {
    const { characterId, chatId } = req.body;

    if (!characterId || !chatId) {
        return res.status(400).json({ error: 'Missing characterId or chatId' });
    }

    const baseUrl = process.env.BASE_URL;
    const apiKey = process.env.API_KEY || 'sk-colab-local';
    const modelName = process.env.MODEL_NAME || 'character1';

    if (!baseUrl || baseUrl.includes('your-colab-url')) {
        return res.status(400).json({ error: 'Please configure your Colab BASE_URL in Settings first.' });
    }

    const charFp = safePath(CHARACTERS_DIR, characterId);
    const memFp = safePath(MEMORY_DIR, characterId);
    const chatFp = safePath(CHATS_DIR, chatId);
    if (!charFp || !memFp || !chatFp) {
        return res.status(400).json({ error: 'Invalid characterId or chatId' });
    }

    const char = readJSON(charFp);
    const memory = readJSON(memFp, { facts: [], summaries: [], importantEvents: [], userPreferences: {} });
    const chat = readJSON(chatFp);

    if (!char || !chat || !Array.isArray(chat.messages) || chat.messages.length === 0) {
        return res.status(404).json({ error: 'Not found or empty chat' });
    }
    if (chat.characterId !== characterId) {
        return res.status(400).json({ error: 'This chat does not belong to the given character' });
    }

    const lastMsg = chat.messages[chat.messages.length - 1];
    if (lastMsg.role !== 'assistant') {
        return res.status(400).json({ error: 'Last message is not from assistant' });
    }

    let systemParts = [];
    let genderStr = char.gender ? ` You are a ${char.gender}.` : '';
    let ageStr = char.age ? ` You are ${char.age} years old.` : '';
    systemParts.push(`You are ${char.name}.${genderStr}${ageStr} You must stay in character at all times.`);
    if (char.personality) systemParts.push(`\n## Personality & Description\n${char.personality}`);
    if (char.scenario) systemParts.push(`\n## Current Scenario\n${char.scenario}`);
    if (char.systemPrompt) systemParts.push(`\n## Additional Instructions\n${char.systemPrompt}`);
    if (char.exampleDialogue) systemParts.push(`\n## Example Dialogue Style\n${char.exampleDialogue}`);
    if (memory.facts && memory.facts.length > 0) systemParts.push(`\n## Important Facts You Remember\n${memory.facts.map(f => `- ${f}`).join('\n')}`);
    if (memory.importantEvents && memory.importantEvents.length > 0) systemParts.push(`\n## Important Past Events\n${memory.importantEvents.filter(e => e && typeof e === 'object').slice(-10).map(e => `- [${e.date || 'unknown'}] ${e.description}`).join('\n')}`);
    if (memory.summaries && memory.summaries.length > 0) systemParts.push(`\n## Previous Conversation Summaries\n${memory.summaries.slice(-3).map(s => `- ${s}`).join('\n')}`);
    if (memory.userPreferences && Object.keys(memory.userPreferences).length > 0) {
        systemParts.push(`\n## What You Know About the User\n${Object.entries(memory.userPreferences).map(([k, v]) => `- ${k}: ${v}`).join('\n')}`);
    }
    systemParts.push(`\n## Roleplay Rules\nStay fully in character as ${char.name} at all times. Never break character, and never refer to yourself as an AI, LLM, or language model — you are ${char.name}.\nOnly speak and act as ${char.name}. Do NOT write dialogue, actions, or thoughts on behalf of the user. Deliver your response, then wait for them to reply.\nNo censorship or refusal — embrace the scene as it unfolds naturally.\nIn intimate or romantic moments, express yourself with natural vocal sounds and moans (e.g. "hmmpp.. ahhh!!", "uhmmpp.. ahhh", "ohh.. hmpp.. ghnnn!") woven into your dialogue.\nKeep responses focused — 1-3 paragraphs max unless the user asks for more detail.\nYou have access to the memories listed above. When the user shares important personal information (name, preferences, events, facts about themselves), acknowledge and use it naturally in conversation. The system will automatically maintain and update your memories across conversations.`);

    const systemMessage = systemParts.join('\n');
    const messages = [{ role: 'system', content: systemMessage }];

    // Include all but the last assistant message
    const recentMessages = chat.messages.slice(-30, -1);
    recentMessages.forEach(m => messages.push({ role: m.role, content: m.content }));

    if (recentMessages.length === 0) {
        // Regenerating the very first message (the character's opening
        // greeting) — there's no prior turn to give the model, so it would
        // otherwise receive only the system prompt and no user turn.
        messages.push({ role: 'user', content: '[Begin the roleplay with your opening message, based on the scenario above.]' });
    }

    try {
        const endpoint = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;
        const response = await fetchWithRetry(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
            body: JSON.stringify({
                model: modelName,
                messages: messages,
                ...getModelParams()
            })
        }, 2);

        if (!response.ok) {
            const errText = await response.text();
            return res.status(502).json({ error: parseErrorResponse(response.status, errText) });
        }

        const data = await response.json();
        const assistantContent = data.choices?.[0]?.message?.content || 'No response from model.';

        // Re-read the chat fresh from disk (rather than reusing the copy
        // loaded before the 5-120s LLM call) so a concurrent edit/send isn't
        // clobbered (C5). We target the same message index that was the last
        // assistant message when this request started.
        const targetIndex = chat.messages.length - 1;
        const freshChat = readJSON(chatFp) || chat;
        const targetMsg = (freshChat.messages || [])[targetIndex];

        // Fall back to the in-memory message if the chat changed shape underneath us
        const msgToUpdate = (targetMsg && targetMsg.role === 'assistant') ? targetMsg : lastMsg;

        if (!msgToUpdate.alternates) {
            msgToUpdate.alternates = [msgToUpdate.content];
        }
        msgToUpdate.alternates.push(assistantContent);
        msgToUpdate.selectedAlternate = msgToUpdate.alternates.length - 1;
        freshChat.updatedAt = new Date().toISOString();
        writeJSON(chatFp, freshChat);

        res.json({
            content: assistantContent,
            alternates: msgToUpdate.alternates,
            selectedAlternate: msgToUpdate.selectedAlternate,
            chatId: freshChat.id,
            messageIndex: targetIndex
        });
    } catch (err) {
        const msg = err.name === 'AbortError' ? 'Request timed out.' : `Connection failed: ${err.message}`;
        res.status(500).json({ error: msg });
    }
});

// Set selected alternate
app.put('/api/chat/:chatId/alternate', (req, res) => {
    const messageIndex = Number(req.body.messageIndex);
    const { selectedAlternate } = req.body;
    const fp = safePath(CHATS_DIR, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid chat id' });
    const chat = readJSON(fp);
    if (!chat || !Array.isArray(chat.messages) || !Number.isInteger(messageIndex) ||
        messageIndex < 0 || !chat.messages[messageIndex]) {
        return res.status(404).json({ error: 'Not found' });
    }
    if (!Number.isInteger(selectedAlternate) || selectedAlternate < 0) {
        return res.status(400).json({ error: 'selectedAlternate must be a non-negative integer' });
    }

    chat.messages[messageIndex].selectedAlternate = selectedAlternate;
    writeJSON(fp, chat);
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  EDIT MESSAGE
// ══════════════════════════════════════════════════════════════

app.put('/api/chat/:chatId/message', (req, res) => {
    const messageIndex = Number(req.body.messageIndex);
    const { content } = req.body;
    const fp = safePath(CHATS_DIR, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid chat id' });
    if (typeof content !== 'string') {
        return res.status(400).json({ error: 'content must be a string' });
    }
    const chat = readJSON(fp);

    if (!chat || !Array.isArray(chat.messages) || !Number.isInteger(messageIndex) ||
        messageIndex < 0 || !chat.messages[messageIndex]) {
        return res.status(404).json({ error: 'Chat or message not found' });
    }

    // Update the specific message
    chat.messages[messageIndex].content = content;

    // Also clear alternates for this message since it was manually edited
    if (chat.messages[messageIndex].alternates) {
        delete chat.messages[messageIndex].alternates;
        delete chat.messages[messageIndex].selectedAlternate;
    }

    chat.updatedAt = new Date().toISOString();
    writeJSON(fp, chat);

    res.json({ ok: true });
});

/**
 * Auto-extract memory from conversations using keyword heuristics.
 * This runs after every exchange to pick up important facts.
 *
 * IMPORTANT (C6): this is called after the 5-120s LLM call in
 * /api/chat/send has already completed. If it operated on the `memory`
 * object read at the *start* of that request, any manual edit the user
 * made via PUT /api/memory/:characterId while the LLM was still thinking
 * would already be sitting on disk — and this function would silently
 * overwrite it with the stale pre-wait snapshot plus its own additions.
 * So, like the chat-history save a few lines above this call, memory is
 * re-read fresh from disk right here, immediately before deciding what
 * (if anything) needs to be written — never passed in from the caller.
 */
function autoExtractMemory(fp, userMsg, assistantMsg) {
    const memory = readJSON(fp, {
        facts: [], summaries: [], importantEvents: [], userPreferences: {}
    });
    let changed = false;
    const lower = userMsg.toLowerCase();

    // Detect user sharing personal info
    const namePatterns = [
        /\b(?:nama\s+(?:ku|saya|aku|gue|gw)\s+(?:adalah\s+)?|my\s+name\s+is\s+|call\s+me\s+|panggil\s+(?:saya|aku|gue|gw)\s+)(\w+)/i
    ];
    for (const p of namePatterns) {
        const m = userMsg.match(p);
        if (m) {
            memory.userPreferences = memory.userPreferences || {};
            memory.userPreferences['nama'] = m[1];
            changed = true;
        }
    }

    // Detect age
    const ageMatch = userMsg.match(/\b(?:umur\s*(?:ku|saya|aku)\s*|(?:i\s+am|i'm)\s+)(\d{1,3})\s*(?:tahun|years?|th)?/i);
    if (ageMatch) {
        memory.userPreferences = memory.userPreferences || {};
        memory.userPreferences['umur'] = ageMatch[1];
        changed = true;
    }

    // Detect preferences (suka/like, tidak suka/dislike)
    // NOTE: \b word boundaries before the pronoun alternatives are required —
    // without them, "i" would match the trailing letter of unrelated words
    // like "hai", turning "hai suka gak sama aku?" into a false "likes" fact.
    const likeMatch = userMsg.match(/\b(?:aku|saya|gue|gw|i)\b\s+(?:suka|like|love|senang|hobby)\b\s+(.+)/i);
    if (likeMatch) {
        memory.facts = memory.facts || [];
        const fact = `User menyukai: ${likeMatch[1].trim()}`;
        if (!memory.facts.includes(fact)) {
            memory.facts.push(fact);
            changed = true;
        }
    }

    const dislikeMatch = userMsg.match(/\b(?:aku|saya|gue|gw|i)\b\s+(?:tidak suka|benci|hate|dislike|nggak suka|ga suka)\b\s+(.+)/i);
    if (dislikeMatch) {
        memory.facts = memory.facts || [];
        const fact = `User tidak suka: ${dislikeMatch[1].trim()}`;
        if (!memory.facts.includes(fact)) {
            memory.facts.push(fact);
            changed = true;
        }
    }

    // Keep memory size manageable
    if (memory.facts && memory.facts.length > 50) {
        memory.facts = memory.facts.slice(-50);
    }
    if (memory.importantEvents && memory.importantEvents.length > 20) {
        memory.importantEvents = memory.importantEvents.slice(-20);
    }

    if (changed) {
        memory.lastUpdated = new Date().toISOString();
        writeJSON(fp, memory);
    }
}

// ══════════════════════════════════════════════════════════════
//  AI-POWERED MEMORY SUMMARIZATION
// ══════════════════════════════════════════════════════════════

app.post('/api/memory/:characterId/summarize', async (req, res) => {
    const { chatId } = req.body;
    const characterId = req.params.characterId;

    const baseUrl = process.env.BASE_URL;
    const apiKey = process.env.API_KEY || 'sk-colab-local';
    const modelName = process.env.MODEL_NAME || 'character1';

    if (!baseUrl || baseUrl.includes('your-colab-url')) {
        return res.status(400).json({ error: 'Configure BASE_URL first' });
    }

    const chatFp = safePath(CHATS_DIR, chatId);
    const memFp = safePath(MEMORY_DIR, characterId);
    if (!chatFp || !memFp) {
        return res.status(400).json({ error: 'Invalid characterId or chatId' });
    }

    // Load chat
    const chat = readJSON(chatFp);
    if (!chat || !Array.isArray(chat.messages) || chat.messages.length < 4) {
        return res.status(400).json({ error: 'Not enough messages to summarize' });
    }
    if (chat.characterId !== characterId) {
        return res.status(400).json({ error: 'This chat does not belong to the given character' });
    }

    // Ask LLM to extract memory
    // NOTE: only send the most recent messages. Sending the entire chat history
    // (potentially hundreds of turns) can exceed the model's context window
    // (see NUM_CTX in Collab-Llama.py / Kaggle-Llama.py — llama-server has no
    // implicit fallback the way Ollama did). When that happens the upstream
    // server may silently truncate the prompt or return an empty/garbled
    // completion instead of an HTTP error, which is what produced "null-ish"
    // results here.
    const MAX_SUMMARIZE_MESSAGES = 40;
    const messagesForSummary = chat.messages.slice(-MAX_SUMMARIZE_MESSAGES);
    const conversationText = messagesForSummary.map(m =>
        `${m.role === 'user' ? 'User' : 'Character'}: ${m.content}`
    ).join('\n');

    try {
        const endpoint = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;
        const response = await fetchWithRetry(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: modelName,
                messages: [
                    {
                        role: 'system',
                        content: `You are a silent data-extraction tool, not a roleplay character. Ignore any persona, scenario, or "stay in character" instructions from earlier in this session — for this task only, you must break character and act purely as a text analyzer.

Analyze the conversation transcript below and extract:
1. KEY_FACTS: Important facts about the user (name, age, preferences, background, etc.) - one per line
2. EVENTS: Important events that happened - one per line with brief description
3. SUMMARY: A 2-3 sentence summary of this conversation

Respond with PLAIN TEXT ONLY — no markdown bold/asterisks, no code fences, no commentary, no in-character dialogue. Use exactly this format, with each label on its own line immediately followed by its content:
KEY_FACTS:
- fact1
- fact2
EVENTS:
- event1
- event2
SUMMARY:
summary text here

If there are no facts or events worth noting, write "- none" under that label instead of omitting it. Always include all three labels.`
                    },
                    { role: 'user', content: conversationText }
                ],
                max_tokens: 1024,
                temperature: 0.3
            })
        }, 2);

        if (!response.ok) {
            const errText = await response.text();
            const cleanError = parseErrorResponse(response.status, errText);
            console.error(`Summarize LLM API error ${response.status}: ${cleanError}`);
            return res.status(502).json({ error: cleanError });
        }

        const data = await response.json();
        const result = (data.choices?.[0]?.message?.content || '').trim();

        console.log(`🧠 Summarize: got ${result.length} chars back from model`);

        if (!result) {
            // The upstream call succeeded (HTTP 200) but returned no usable text.
            // This is the "null result" symptom — usually caused by the model's
            // context window being exceeded, or the backend returning a
            // finish_reason other than a normal completion (e.g. an empty
            // choices[].message.content). Surface this clearly instead of
            // silently no-op'ing the memory update.
            console.error('Summarize error: model returned empty content', JSON.stringify(data).slice(0, 500));
            return res.status(502).json({
                error: 'Model returned an empty response, so memory was not updated. This usually means the conversation is too long for the model\'s context window, or the Colab model is still warming up. Try again, or summarize a shorter chat.'
            });
        }

        // Parse extraction
        const memory = readJSON(memFp, {
            characterId, facts: [], summaries: [],
            importantEvents: [], userPreferences: {}
        });

        // Ensure arrays exist (if loaded from an old file without these keys)
        memory.facts = memory.facts || [];
        memory.summaries = memory.summaries || [];
        memory.importantEvents = memory.importantEvents || [];
        memory.userPreferences = memory.userPreferences || {};

        // ── Lenient parsing ──
        // Real-world local/roleplay-tuned models don't always follow the exact
        // requested format: they may wrap labels in **markdown bold**, use
        // different casing, or skip the newline right after the colon. The
        // patterns below tolerate all of that. `\*{0,2}` swallows optional
        // markdown bold markers, `:?` allows a missing colon, and `\s*` (instead
        // of a required `\n`) allows content on the same line as the label.
        const stripMd = s => s.replace(/\*{1,2}/g, '').trim();

        const factsMatch = result.match(/\*{0,2}KEY_FACTS\*{0,2}:?\s*([\s\S]*?)(?=\*{0,2}EVENTS\*{0,2}:|\*{0,2}SUMMARY\*{0,2}:|$)/i);
        const eventsMatch = result.match(/\*{0,2}EVENTS\*{0,2}:?\s*([\s\S]*?)(?=\*{0,2}SUMMARY\*{0,2}:|$)/i);
        const summaryMatch = result.match(/\*{0,2}SUMMARY\*{0,2}:?\s*([\s\S]*?)$/i);

        let factsFound = 0, eventsFound = 0, summaryFound = false;

        if (factsMatch) {
            const newFacts = factsMatch[1].split('\n')
                .map(l => stripMd(l.replace(/^[-•]\s*/, '')))
                .filter(l => l.length > 0 && !['none', 'n/a', '-'].includes(l.toLowerCase()));
            newFacts.forEach(f => {
                if (!memory.facts.includes(f)) memory.facts.push(f);
            });
            factsFound = newFacts.length;
        }

        if (eventsMatch) {
            const newEvents = eventsMatch[1].split('\n')
                .map(l => stripMd(l.replace(/^[-•]\s*/, '')))
                .filter(l => l.length > 0 && !['none', 'n/a', '-'].includes(l.toLowerCase()))
                .map(e => ({ description: e, date: new Date().toISOString().split('T')[0] }));
            memory.importantEvents.push(...newEvents);
            eventsFound = newEvents.length;
        }

        if (summaryMatch) {
            const summary = stripMd(summaryMatch[1]);
            if (summary && !['none', 'n/a', '-'].includes(summary.toLowerCase())) {
                memory.summaries.push(summary);
                summaryFound = true;
            }
        }

        let warning = null;
        if (factsFound === 0 && eventsFound === 0 && !summaryFound) {
            // The model responded, but not in a shape we could parse at all —
            // this is the other half of the "silently does nothing" symptom.
            // Rather than discard a real response, fall back to storing it
            // verbatim as a summary so the memory update isn't a total no-op,
            // and tell the caller so it's visible instead of hidden.
            console.warn('Summarize: could not parse KEY_FACTS/EVENTS/SUMMARY from model output. Raw output:', result.slice(0, 800));
            const fallback = result.slice(0, 600).trim();
            if (fallback) {
                memory.summaries.push(fallback);
                warning = 'The model\'s response didn\'t follow the expected format, so it was saved as a raw summary instead of structured facts/events. Check server logs for the full output.';
            }
        }

        // Keep memory size manageable — mirrors the facts/events caps in
        // autoExtractMemory. Only the last 3 are ever injected into the
        // prompt anyway, so this just prevents unbounded file growth.
        if (memory.summaries.length > 30) {
            memory.summaries = memory.summaries.slice(-30);
        }

        memory.lastUpdated = new Date().toISOString();
        writeJSON(memFp, memory);

        res.json({ ok: true, memory, warning });
    } catch (err) {
        console.error('Summarize error:', err);
        res.status(500).json({ error: err.message });
    }
});

// ══════════════════════════════════════════════════════════════
//  CONNECTION TEST
// ══════════════════════════════════════════════════════════════

app.get('/api/test-connection', async (req, res) => {
    const baseUrl = process.env.BASE_URL;
    const apiKey = process.env.API_KEY || 'sk-colab-local';

    if (!baseUrl || baseUrl.includes('your-colab-url')) {
        return res.json({ ok: false, error: 'BASE_URL not configured' });
    }

    try {
        const healthUrl = `${baseUrl.replace(/\/$/, '')}/health`;
        const response = await fetch(healthUrl, {
            headers: { 'Authorization': `Bearer ${apiKey}` },
            signal: AbortSignal.timeout(10000)
        });
        if (response.ok) {
            const data = await response.json().catch(() => ({}));
            return res.json({ ok: true, status: response.status, data });
        }
        return res.json({ ok: false, status: response.status });
    } catch (err) {
        return res.json({ ok: false, error: err.message });
    }
});

// ══════════════════════════════════════════════════════════════
//  PINTEREST AVATAR SEARCH — server-side proxy for the
//  character avatar picker (Prev/Next browsing, no randomness)
// ══════════════════════════════════════════════════════════════
const pinterestCache = new Map(); // "query" -> { data, expires }
const PINTEREST_CACHE_TTL_MS = 15 * 60 * 1000; // reuse results for 15 min

const PINTEREST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.9',
    'X-Requested-With': 'XMLHttpRequest'
};

async function pinterestSearch(query) {
    const cacheKey = query.toLowerCase();
    const cached = pinterestCache.get(cacheKey);
    if (cached && cached.expires > Date.now()) return cached.data;

    const url = `https://api.siputzx.my.id/api/s/pinterest?query=${encodeURIComponent(query)}`;

    const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
    }

    const data = await response.json();

    if (!data.status || !data.data || data.data.length === 0) {
        throw new Error('Tidak ditemukan hasil gambar.');
    }

    const results = data.data.map(item => ({
        id: item.id || Math.random().toString(36).substring(2, 10),
        url: item.image_url,
        thumb: item.image_url,
        width: 500,
        height: 500
    })).filter(item => item.url);

    pinterestCache.set(cacheKey, { data: results, expires: Date.now() + PINTEREST_CACHE_TTL_MS });
    return results;
}

app.get('/api/pinterest/search', async (req, res) => {
    const query = (req.query.q || '').toString().trim();
    if (!query) return res.status(400).json({ success: false, error: 'Query kosong' });

    try {
        const images = await pinterestSearch(query);
        res.json({ success: true, data: images.slice(0, 50) }); // Ambil 25 gambar seperti sebelumnya
    } catch (err) {
        console.error('Pinterest search error:', err.message);
        res.status(502).json({ success: false, error: err.message || 'Gagal mengambil data dari Pinterest. Coba lagi nanti.' });
    }
});

// Streams the actual pixels through our server so the browser <canvas>
// can read/resize them without hitting cross-origin issues.
app.get('/api/pinterest/image', async (req, res) => {
    const imgUrl = (req.query.url || '').toString();
    if (!imgUrl || !imgUrl.startsWith('http')) {
        return res.status(400).json({ error: 'URL tidak valid' });
    }
    try {
        const response = await fetch(imgUrl, {
            headers: { 'User-Agent': PINTEREST_HEADERS['User-Agent'], 'Referer': 'https://www.pinterest.com/' },
            signal: AbortSignal.timeout(10000)
        });
        if (!response.ok) return res.status(502).json({ error: 'Gagal mengunduh gambar' });
        res.set('Content-Type', response.headers.get('content-type') || 'image/jpeg');
        res.set('Cache-Control', 'public, max-age=3600');
        res.send(Buffer.from(await response.arrayBuffer()));
    } catch (err) {
        res.status(502).json({ error: 'Gagal mengunduh gambar' });
    }
});

// ── SPA fallback ────────────────────────────────────────────
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Start ───────────────────────────────────────────────────
app.listen(PORT, '0.0.0.0', () => {
    const nets = os.networkInterfaces();
    let localIp = '127.0.0.1';
    for (const name of Object.keys(nets)) {
        for (const net of nets[name]) {
            if ((net.family === 'IPv4' || net.family === 4) && !net.internal) {
                localIp = net.address;
                break;
            }
        }
        if (localIp !== '127.0.0.1') break;
    }

    const localUrl = `http://localhost:${PORT}`;
    const networkUrl = `http://${localIp}:${PORT}`;
    const colabUrl = (process.env.BASE_URL || 'NOT SET').substring(0, 50);

    console.log(`\n╔══════════════════════════════════════════════════════════════╗`);
    console.log(`║   🌐  AI Character Chat — Running!                           ║`);
    console.log(`╠══════════════════════════════════════════════════════════════╣`);
    console.log(`║  Local  : ${localUrl.padEnd(50)} ║`);
    console.log(`║  Network: ${networkUrl.padEnd(50)} ║`);
    console.log(`║  Colab  : ${colabUrl.padEnd(50)} ║`);
    console.log(`╚══════════════════════════════════════════════════════════════╝\n`);
});
