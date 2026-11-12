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
const PRIVATE_CHATS_DIR = path.join(CHATS_DIR, 'private');
const GROUP_CHATS_DIR = path.join(CHATS_DIR, 'groups');
const MEMORY_DIR = path.join(DATA_DIR, 'memory');
const PRIVATE_MEMORY_DIR = path.join(MEMORY_DIR, 'private');
const GROUP_MEMORY_DIR = path.join(MEMORY_DIR, 'groups');
const PERSONA_DIR = path.join(DATA_DIR, 'persona');

[DATA_DIR, CHARACTERS_DIR, CHATS_DIR, PRIVATE_CHATS_DIR, GROUP_CHATS_DIR, MEMORY_DIR, PRIVATE_MEMORY_DIR, GROUP_MEMORY_DIR, PERSONA_DIR].forEach(dir => {
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

function safeGroupChatPath(groupId, chatId) {
    if (typeof groupId !== 'string' || !ID_RE.test(groupId)) return null;
    if (typeof chatId !== 'string' || !ID_RE.test(chatId)) return null;
    const groupDir = path.resolve(GROUP_CHATS_DIR, groupId);
    const resolvedBase = groupDir + path.sep;
    const resolved = path.resolve(groupDir, `${chatId}.json`);
    if (!resolved.startsWith(resolvedBase)) return null;
    return resolved;
}

function safePrivateChatPath(characterId, chatId) {
    if (typeof characterId !== 'string' || !ID_RE.test(characterId)) return null;
    if (typeof chatId !== 'string' || !ID_RE.test(chatId)) return null;
    const charDir = path.resolve(PRIVATE_CHATS_DIR, characterId);
    const resolvedBase = charDir + path.sep;
    const resolved = path.resolve(charDir, `${chatId}.json`);
    if (!resolved.startsWith(resolvedBase)) return null;
    return resolved;
}

function findPrivateChatPath(chatId, characterId = null) {
    if (typeof chatId !== 'string' || !ID_RE.test(chatId)) return null;
    if (characterId && typeof characterId === 'string' && ID_RE.test(characterId)) {
        const p = safePrivateChatPath(characterId, chatId);
        if (p && fs.existsSync(p)) return p;
    }
    // Search within character subdirectories
    if (fs.existsSync(PRIVATE_CHATS_DIR)) {
        const entries = fs.readdirSync(PRIVATE_CHATS_DIR, { withFileTypes: true });
        for (const ent of entries) {
            if (ent.isDirectory() && ID_RE.test(ent.name)) {
                const candidate = path.join(PRIVATE_CHATS_DIR, ent.name, `${chatId}.json`);
                if (fs.existsSync(candidate)) {
                    return candidate;
                }
            }
        }
    }
    // Fallback to legacy root path just in case
    const legacy = safePath(PRIVATE_CHATS_DIR, chatId);
    if (legacy && fs.existsSync(legacy)) return legacy;
    return null;
}

// ── Helper: Temporal Awareness & Precision Time ─────────────
function getTimeOfDayPeriod(date) {
    const h = date.getHours();
    if (h >= 0 && h < 4) return 'Late Night / Midnight';
    if (h >= 4 && h < 6) return 'Dawn / Early Morning';
    if (h >= 6 && h < 12) return 'Morning';
    if (h >= 12 && h < 14) return 'Midday / Noon';
    if (h >= 14 && h < 18) return 'Afternoon';
    if (h >= 18 && h < 21) return 'Evening';
    return 'Night';
}

function getAccurateDurationString(lastDate, currentDate) {
    let diffMs = currentDate.getTime() - lastDate.getTime();
    if (diffMs < 0) diffMs = 0;

    const totalMinutes = Math.floor(diffMs / (1000 * 60));
    const totalHours = Math.floor(diffMs / (1000 * 60 * 60));
    const totalDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (totalDays >= 365) {
        const years = Math.floor(totalDays / 365);
        const remDays = totalDays % 365;
        const months = Math.floor(remDays / 30);
        const parts = [`${years} ${years === 1 ? 'year' : 'years'}`];
        if (months > 0) parts.push(`${months} ${months === 1 ? 'month' : 'months'}`);
        return parts.join(', ');
    }

    if (totalDays >= 60) {
        const months = Math.floor(totalDays / 30);
        const remDays = totalDays % 30;
        const parts = [`${months} ${months === 1 ? 'month' : 'months'}`];
        if (remDays > 0) parts.push(`${remDays} ${remDays === 1 ? 'day' : 'days'}`);
        return parts.join(', ');
    }

    if (totalDays >= 14) {
        const weeks = Math.floor(totalDays / 7);
        const remDays = totalDays % 7;
        const parts = [`${weeks} ${weeks === 1 ? 'week' : 'weeks'}`];
        if (remDays > 0) parts.push(`${remDays} ${remDays === 1 ? 'day' : 'days'}`);
        return parts.join(', ');
    }

    if (totalDays >= 1) {
        const remHours = totalHours % 24;
        const parts = [`${totalDays} ${totalDays === 1 ? 'day' : 'days'}`];
        if (remHours > 0) parts.push(`${remHours} ${remHours === 1 ? 'hour' : 'hours'}`);
        return parts.join(' and ');
    }

    const remMins = totalMinutes % 60;
    const parts = [`${totalHours} ${totalHours === 1 ? 'hour' : 'hours'}`];
    if (remMins > 0) parts.push(`${remMins} ${remMins === 1 ? 'minute' : 'minutes'}`);
    return parts.join(' and ');
}

function formatReadableDateTime(date) {
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    const dayName = days[date.getDay()];
    const monthName = months[date.getMonth()];
    const day = date.getDate();
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${dayName}, ${day} ${monthName} ${year}, ${hours}:${minutes}`;
}

function formatReadableDateOnly(date) {
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    return `${days[date.getDay()]}, ${date.getDate()} ${months[date.getMonth()]} ${date.getFullYear()}`;
}

function buildTemporalPromptBlock(lastDate, currentDate) {
    const lastPeriod = getTimeOfDayPeriod(lastDate);
    const currentPeriod = getTimeOfDayPeriod(currentDate);
    const lastFormatted = formatReadableDateTime(lastDate);
    const currentFormatted = formatReadableDateTime(currentDate);
    const durationStr = getAccurateDurationString(lastDate, currentDate);

    return `\n## Temporal Awareness
- Status: User has returned after a period of absence.
- Last Interaction: ${lastFormatted} (${lastPeriod})
- Current Time: ${currentFormatted} (${currentPeriod})
- Elapsed Duration: ${durationStr}
- Interaction Guidance:
  * You possess natural awareness of the real-world passage of time and the current time of day.
  * If fitting for your personality and conversational context, you may naturally acknowledge their return, tease/comment on their absence, or greet them according to the current time of day.
  * Narrative Override: If the ongoing chat is in the middle of an urgent, continuous roleplay action or intense narrative scene, prioritize narrative flow and ignore the real-world time gap so immersion is preserved.`;
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

let _writeCounter = 0;

function writeJSON(filePath, data) {
    const tmpPath = `${filePath}.${process.pid}.${Date.now()}.${++_writeCounter}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2), 'utf8');
    let lastErr = null;
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            fs.renameSync(tmpPath, filePath); // atomic on POSIX filesystems — no partial-write window
            return;
        } catch (err) {
            lastErr = err;
            if (err.code !== 'EPERM' && err.code !== 'EACCES' && err.code !== 'EBUSY') {
                break;
            }
        }
    }
    try {
        fs.copyFileSync(tmpPath, filePath);
    } catch (copyErr) {
        throw lastErr || copyErr;
    } finally {
        try {
            if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
        } catch { }
    }
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
        max_tokens: parseValidated(process.env.MAX_TOKENS, { type: 'int', defaultVal: 1024, min: 1, max: 32768, label: 'MAX_TOKENS' }),
        temperature: parseValidated(process.env.TEMPERATURE, { defaultVal: 0.80, min: 0, max: 5, label: 'TEMPERATURE' }),
        top_k: parseValidated(process.env.TOP_K, { type: 'int', defaultVal: 50, min: 0, max: 1000, label: 'TOP_K' }),
        top_p: parseValidated(process.env.TOP_P, { defaultVal: 0.90, min: 0, max: 1, label: 'TOP_P' }),
        min_p: parseValidated(process.env.MIN_P, { defaultVal: 0.05, min: 0, max: 1, label: 'MIN_P' }),
        typical_p: parseValidated(process.env.TYPICAL_P, { defaultVal: 1.0, min: 0, max: 1, label: 'TYPICAL_P' }),
        tfs_z: parseValidated(process.env.TFS_Z, { defaultVal: 1.0, min: 0, max: 1, label: 'TFS_Z' }),
        top_n_sigma: parseValidated(process.env.TOP_N_SIGMA, { defaultVal: -1, min: -1, max: 100, label: 'TOP_N_SIGMA' }),

        // Windowed repetition control — bounded to the last N tokens, not
        // the whole conversation. This is the direct fix for the original
        // "common words get suppressed over a long chat" concern.
        repeat_last_n: parseValidated(process.env.REPEAT_LAST_N, { type: 'int', defaultVal: 256, min: 0, max: 8192, label: 'REPEAT_LAST_N' }),
        repeat_penalty: parseValidated(process.env.REPEAT_PENALTY, { defaultVal: 1.05, min: 0, max: 3, label: 'REPEAT_PENALTY' }),

        // OpenAI-style, cumulative over the WHOLE context by spec. Default
        // to 0 (disabled) on purpose — see README for why. Must match the
        // notebook scripts' defaults exactly.
        presence_penalty: parseValidated(process.env.PRESENCE_PENALTY, { defaultVal: 0.0, min: -2, max: 2, label: 'PRESENCE_PENALTY' }),
        frequency_penalty: parseValidated(process.env.FREQUENCY_PENALTY, { defaultVal: 0.0, min: -2, max: 2, label: 'FREQUENCY_PENALTY' }),

        // DRY sampling — penalizes repeated phrases/loops, not individual
        // common tokens. The purpose-built fix for long roleplay chats.
        dry_multiplier: parseValidated(process.env.DRY_MULTIPLIER, { defaultVal: 0.8, min: 0, max: 5, label: 'DRY_MULTIPLIER' }),
        dry_base: parseValidated(process.env.DRY_BASE, { defaultVal: 1.75, min: 1, max: 4, label: 'DRY_BASE' }),
        dry_allowed_length: parseValidated(process.env.DRY_ALLOWED_LENGTH, { type: 'int', defaultVal: 2, min: 1, max: 50, label: 'DRY_ALLOWED_LENGTH' }),
        dry_penalty_last_n: parseValidated(process.env.DRY_PENALTY_LAST_N, { type: 'int', defaultVal: 1024, min: -1, max: 16384, label: 'DRY_PENALTY_LAST_N' }),
        dry_sequence_breakers: parseListEnv(process.env.DRY_SEQUENCE_BREAKERS, ["\n", ":", "\"", "*"]),

        // XTC — trims away the single most-probable token some of the time,
        // to reduce "boring"/predictable output. Off by default (probability
        // 0) since it's more experimental / can occasionally cut off short
        // valid replies; conservative default per spec.
        xtc_probability: parseValidated(process.env.XTC_PROBABILITY, { defaultVal: 0.0, min: 0, max: 1, label: 'XTC_PROBABILITY' }),
        xtc_threshold: parseValidated(process.env.XTC_THRESHOLD, { defaultVal: 0.1, min: 0, max: 1, label: 'XTC_THRESHOLD' }),

        // Dynamic temperature
        dynatemp_range: parseValidated(process.env.DYNATEMP_RANGE, { defaultVal: 0.0, min: 0, max: 5, label: 'DYNATEMP_RANGE' }),
        dynatemp_exponent: parseValidated(process.env.DYNATEMP_EXP, { defaultVal: 1.0, min: 0, max: 10, label: 'DYNATEMP_EXP' }),

        // Mirostat — off (0) by default; when on, it takes over from
        // top_k/top_p/temperature, so it's opt-in only via env.
        mirostat: parseValidated(process.env.MIROSTAT, { type: 'int', defaultVal: 0, min: 0, max: 2, label: 'MIROSTAT' }),
        mirostat_lr: parseValidated(process.env.MIROSTAT_LR, { defaultVal: 0.1, min: 0, max: 5, label: 'MIROSTAT_LR' }),
        mirostat_ent: parseValidated(process.env.MIROSTAT_ENT, { defaultVal: 5.0, min: 0, max: 20, label: 'MIROSTAT_ENT' }),

        // Reproducibility / sampler chain
        seed: parseValidated(process.env.SEED, { type: 'int', defaultVal: -1, min: -1, max: 2147483647, label: 'SEED' }),
        samplers: parseListEnv(process.env.SAMPLERS, undefined), // undefined = let llama-server use its own default chain

        // Thinking / reasoning control (off by default)
        chat_template_kwargs: {
            enable_thinking: process.env.ENABLE_THINKING === 'true',
        },
    };
}

function cleanAssistantResponse(text) {
    if (!text || typeof text !== 'string') return text;
    if (process.env.ENABLE_THINKING === 'true') {
        return text;
    }
    const hadThinkTag = /<think>/i.test(text);
    // Strip complete <think>...</think> tags
    let cleaned = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
    // Strip unclosed <think> tag if output was cut off mid-thought
    cleaned = cleaned.replace(/<think>[\s\S]*$/gi, '');
    if (hadThinkTag) {
        return cleaned.trim();
    }
    return cleaned.trim() || text;
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
        showLangWarning: showLangWarning,
        enableTemporalAwareness: (process.env.ENABLE_TEMPORAL_AWARENESS || 'true').toLowerCase() === 'true'
    });
});

app.post('/api/config', (req, res) => {
    const { baseUrl, apiKey, modelName, enableTemporalAwareness } = req.body;
    const envPath = path.join(__dirname, '.env');
    let envContent = '';
    try { envContent = fs.readFileSync(envPath, 'utf8'); } catch { }

    const updates = {};
    if (baseUrl !== undefined) updates.BASE_URL = baseUrl;
    if (apiKey !== undefined) updates.API_KEY = apiKey;
    if (modelName !== undefined) updates.MODEL_NAME = modelName;
    if (enableTemporalAwareness !== undefined) updates.ENABLE_TEMPORAL_AWARENESS = String(enableTemporalAwareness);

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
    writeJSON(path.join(PRIVATE_MEMORY_DIR, `${id}.json`), normalizeMemoryDoc({
        characterId: id,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: '1',
        lastUpdated: now
    }, 'characterId', id, '1'));

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
    const memFp = safePath(PRIVATE_MEMORY_DIR, req.params.id);
    if (!fp || !memFp) return res.status(400).json({ error: 'Invalid character id' });
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
    // Also delete associated memory
    if (fs.existsSync(memFp)) fs.unlinkSync(memFp);
    // Delete all chats for this character
    const charChatDir = path.join(PRIVATE_CHATS_DIR, req.params.id);
    if (fs.existsSync(charChatDir)) {
        fs.rmSync(charChatDir, { recursive: true, force: true });
    }
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  CHAT HISTORY API (PRIVATE)
// ══════════════════════════════════════════════════════════════

// List all chats for a character
app.get('/api/chats/:characterId', (req, res) => {
    const charDir = path.join(PRIVATE_CHATS_DIR, req.params.characterId);
    let chats = [];
    if (fs.existsSync(charDir)) {
        const files = fs.readdirSync(charDir).filter(f => f.endsWith('.json'));
        chats = files
            .map(f => readJSON(path.join(charDir, f)))
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
    }
    res.json(chats);
});

// Get full chat
app.get('/api/chat/:chatId', (req, res) => {
    const fp = findPrivateChatPath(req.params.chatId);
    if (!fp) return res.status(404).json({ error: 'Chat not found' });
    const chat = readJSON(fp);
    if (!chat) return res.status(404).json({ error: 'Chat not found' });

    // Ensure every message has a unique ID (backward compatibility)
    if (Array.isArray(chat.messages)) {
        let changed = false;
        for (const m of chat.messages) {
            if (!m.id) {
                m.id = uuidv4();
                changed = true;
            }
        }
        if (changed) {
            writeJSON(fp, chat);
        }
    }

    res.json(chat);
});

// Create new chat
app.post('/api/chats', (req, res) => {
    if (!req.body.characterId || typeof req.body.characterId !== 'string' || !ID_RE.test(req.body.characterId)) {
        return res.status(400).json({ error: 'Valid characterId is required' });
    }
    const id = uuidv4();
    const now = new Date().toISOString();
    const rawMessages = Array.isArray(req.body.messages) ? req.body.messages : [];
    const messages = rawMessages.map(m => ({
        ...m,
        id: m.id || uuidv4()
    }));
    const chat = {
        id,
        characterId: req.body.characterId,
        title: req.body.title || 'New Chat',
        messages,
        createdAt: now,
        updatedAt: now
    };
    const targetDir = path.join(PRIVATE_CHATS_DIR, req.body.characterId);
    if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
    }
    writeJSON(path.join(targetDir, `${id}.json`), chat);
    res.json(chat);
});

// Update chat (add messages, rename, etc.)
app.put('/api/chat/:chatId', (req, res) => {
    const fp = findPrivateChatPath(req.params.chatId);
    if (!fp) return res.status(404).json({ error: 'Chat not found' });
    const existing = readJSON(fp);
    if (!existing) return res.status(404).json({ error: 'Not found' });

    if (req.body.messages !== undefined && !Array.isArray(req.body.messages)) {
        return res.status(400).json({ error: '"messages" must be an array' });
    }

    let messages = existing.messages;
    if (req.body.messages !== undefined) {
        const existingById = new Map((existing.messages || []).filter(em => em && em.id).map(em => [em.id, em]));
        messages = req.body.messages.map((m, idx) => {
            const prev = (m.id && existingById.get(m.id)) || (existing.messages || [])[idx];
            const merged = {
                ...(prev && prev.alternates ? { alternates: prev.alternates, selectedAlternate: prev.selectedAlternate } : {}),
                ...m,
                id: m.id || (prev && prev.id) || uuidv4()
            };
            if (Array.isArray(merged.alternates) && merged.selectedAlternate !== undefined && merged.content !== undefined) {
                merged.alternates = [...merged.alternates];
                merged.alternates[merged.selectedAlternate] = merged.content;
            }
            return merged;
        });
    }

    const updated = {
        ...existing,
        title: req.body.title ?? existing.title,
        messages,
        updatedAt: new Date().toISOString()
    };
    writeJSON(fp, updated);
    res.json(updated);
});

// Delete chat
app.delete('/api/chat/:chatId', (req, res) => {
    const fp = findPrivateChatPath(req.params.chatId);
    if (!fp) return res.status(404).json({ error: 'Chat not found' });
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  GROUP CHAT CRUD API
// ══════════════════════════════════════════════════════════════

// List all groups
app.get('/api/groups', (req, res) => {
    try {
        const files = fs.readdirSync(GROUP_CHATS_DIR).filter(f => f.endsWith('.json'));
        const groups = files
            .map(f => readJSON(path.join(GROUP_CHATS_DIR, f)))
            .filter(Boolean)
            .map(g => {
                let messageCount = 0;
                const sessionDir = path.join(GROUP_CHATS_DIR, g.id);
                if (fs.existsSync(sessionDir)) {
                    try {
                        const sFiles = fs.readdirSync(sessionDir).filter(sf => sf.endsWith('.json'));
                        for (const sf of sFiles) {
                            const sChat = readJSON(path.join(sessionDir, sf));
                            if (sChat && Array.isArray(sChat.messages)) {
                                messageCount += sChat.messages.length;
                            }
                        }
                    } catch (e) {
                        console.error('Error calculating group messageCount:', e);
                    }
                }
                if (messageCount === 0 && Array.isArray(g.messages)) {
                    messageCount = g.messages.length;
                }

                return {
                    id: g.id,
                    name: g.name || 'Untitled Group',
                    avatar: g.avatar || '',
                    scenario: g.scenario || '',
                    characterIds: Array.isArray(g.characterIds) ? g.characterIds : [],
                    activePersonaId: g.activePersonaId || null,
                    messageCount,
                    createdAt: g.createdAt,
                    updatedAt: g.updatedAt
                };
            });
        groups.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
        res.json(groups);
    } catch (err) {
        console.error('Error listing groups:', err);
        res.status(500).json({ error: 'Failed to list groups' });
    }
});

// Get single group
app.get('/api/groups/:id', (req, res) => {
    const fp = safePath(GROUP_CHATS_DIR, req.params.id);
    if (!fp) return res.status(400).json({ error: 'Invalid group id' });
    const group = readJSON(fp);
    if (!group) return res.status(404).json({ error: 'Group not found' });
    res.json(group);
});

// Create new group
app.post('/api/groups', (req, res) => {
    const { name, avatar, scenario, characterIds, activePersonaId } = req.body;
    if (!name || typeof name !== 'string' || !name.trim()) {
        return res.status(400).json({ error: 'Group name is required' });
    }
    if (!Array.isArray(characterIds) || characterIds.length === 0) {
        return res.status(400).json({ error: 'Group must have at least one character' });
    }

    const id = uuidv4();
    const now = new Date().toISOString();
    const group = {
        id,
        name: name.trim(),
        avatar: avatar || '',
        scenario: scenario || '',
        characterIds: characterIds,
        activePersonaId: activePersonaId || null,
        messages: [],
        createdAt: now,
        updatedAt: now
    };
    writeJSON(path.join(GROUP_CHATS_DIR, `${id}.json`), group);

    // Initialize group session directory and first session
    const sessionDir = path.join(GROUP_CHATS_DIR, id);
    if (!fs.existsSync(sessionDir)) fs.mkdirSync(sessionDir, { recursive: true });
    const firstChatId = uuidv4();
    const firstChat = {
        id: firstChatId,
        groupId: id,
        title: 'New Chat',
        messages: [],
        createdAt: now,
        updatedAt: now
    };
    writeJSON(path.join(sessionDir, `${firstChatId}.json`), firstChat);

    // Also initialize group memory in GROUP_MEMORY_DIR
    const memFp = path.join(GROUP_MEMORY_DIR, `${id}.json`);
    const initGroupPersonaId = String(activePersonaId || '1');
    writeJSON(memFp, normalizeMemoryDoc({
        groupId: id,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: initGroupPersonaId,
        lastUpdated: now
    }, 'groupId', id, initGroupPersonaId));

    res.json(group);
});

// Update group
app.put('/api/groups/:id', (req, res) => {
    const fp = safePath(GROUP_CHATS_DIR, req.params.id);
    if (!fp) return res.status(400).json({ error: 'Invalid group id' });
    const existing = readJSON(fp);
    if (!existing) return res.status(404).json({ error: 'Group not found' });

    // Ensure session directory exists for this group
    const sessionDir = path.join(GROUP_CHATS_DIR, req.params.id);
    if (!fs.existsSync(sessionDir)) {
        fs.mkdirSync(sessionDir, { recursive: true });
    }

    // If existing has legacy messages and sessionDir is empty, migrate them to a session
    const sFiles = fs.readdirSync(sessionDir).filter(f => f.endsWith('.json'));
    if (sFiles.length === 0 && Array.isArray(existing.messages) && existing.messages.length > 0) {
        const firstChatId = uuidv4();
        const now = new Date().toISOString();
        const firstChat = {
            id: firstChatId,
            groupId: req.params.id,
            title: 'New Chat',
            messages: existing.messages,
            createdAt: existing.createdAt || now,
            updatedAt: existing.updatedAt || now
        };
        writeJSON(path.join(sessionDir, `${firstChatId}.json`), firstChat);
    }

    let groupMessages = existing.messages || [];
    if (Array.isArray(req.body.messages) && req.body.messages.length > 0) {
        const existingById = new Map((existing.messages || []).filter(em => em && em.id).map(em => [em.id, em]));
        groupMessages = req.body.messages.map((m, idx) => {
            const prev = (m.id && existingById.get(m.id)) || (existing.messages || [])[idx];
            const merged = {
                ...(prev && prev.alternates ? { alternates: prev.alternates, selectedAlternate: prev.selectedAlternate } : {}),
                ...m,
                id: m.id || (prev && prev.id) || uuidv4()
            };
            if (Array.isArray(merged.alternates) && merged.selectedAlternate !== undefined && merged.content !== undefined) {
                merged.alternates = [...merged.alternates];
                merged.alternates[merged.selectedAlternate] = merged.content;
            }
            return merged;
        });
    }

    const updated = {
        ...existing,
        name: typeof req.body.name === 'string' && req.body.name.trim() ? req.body.name.trim() : existing.name,
        avatar: req.body.avatar !== undefined ? req.body.avatar : existing.avatar,
        scenario: req.body.scenario !== undefined ? req.body.scenario : existing.scenario,
        characterIds: Array.isArray(req.body.characterIds) ? req.body.characterIds : existing.characterIds,
        activePersonaId: req.body.activePersonaId !== undefined ? req.body.activePersonaId : existing.activePersonaId,
        // Never overwrite existing messages with an empty array
        messages: groupMessages,
        updatedAt: new Date().toISOString()
    };
    writeJSON(fp, updated);
    res.json(updated);
});

// Delete group
app.delete('/api/groups/:id', (req, res) => {
    const fp = safePath(GROUP_CHATS_DIR, req.params.id);
    const memFp = safePath(GROUP_MEMORY_DIR, req.params.id);
    if (!fp) return res.status(400).json({ error: 'Invalid group id' });
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
    if (memFp && fs.existsSync(memFp)) fs.unlinkSync(memFp);

    // Delete session directory
    const sessionDir = path.join(GROUP_CHATS_DIR, req.params.id);
    if (fs.existsSync(sessionDir)) {
        try { fs.rmSync(sessionDir, { recursive: true, force: true }); } catch { }
    }
    res.json({ ok: true });
});

// ── GROUP CHAT SESSIONS API ────────────────────────────────────

// List all chat sessions for a group
app.get('/api/groups/:groupId/chats', (req, res) => {
    const groupId = req.params.groupId;
    const groupFp = safePath(GROUP_CHATS_DIR, groupId);
    if (!groupFp || !fs.existsSync(groupFp)) return res.status(404).json({ error: 'Group not found' });

    const sessionDir = path.join(GROUP_CHATS_DIR, groupId);
    if (!fs.existsSync(sessionDir)) fs.mkdirSync(sessionDir, { recursive: true });

    let files = fs.readdirSync(sessionDir).filter(f => f.endsWith('.json'));
    // If no session exists yet, initialize first session from existing group.messages if any
    if (files.length === 0) {
        const group = readJSON(groupFp);
        const firstChatId = uuidv4();
        const now = new Date().toISOString();
        const firstChat = {
            id: firstChatId,
            groupId,
            title: 'New Chat',
            messages: (group && Array.isArray(group.messages)) ? group.messages : [],
            createdAt: (group && group.createdAt) || now,
            updatedAt: (group && group.updatedAt) || now
        };
        writeJSON(path.join(sessionDir, `${firstChatId}.json`), firstChat);
        files = [`${firstChatId}.json`];
    }

    const chats = files
        .map(f => readJSON(path.join(sessionDir, f)))
        .filter(Boolean)
        .map(c => ({
            id: c.id,
            groupId: c.groupId || groupId,
            title: c.title || 'Untitled Chat',
            messageCount: (c.messages || []).length,
            createdAt: c.createdAt,
            updatedAt: c.updatedAt
        }));
    chats.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    res.json(chats);
});

// Create new chat session for a group
app.post('/api/groups/:groupId/chats', (req, res) => {
    const groupId = req.params.groupId;
    const groupFp = safePath(GROUP_CHATS_DIR, groupId);
    if (!groupFp || !fs.existsSync(groupFp)) return res.status(404).json({ error: 'Group not found' });

    const sessionDir = path.join(GROUP_CHATS_DIR, groupId);
    if (!fs.existsSync(sessionDir)) fs.mkdirSync(sessionDir, { recursive: true });

    const id = uuidv4();
    const now = new Date().toISOString();
    const newChat = {
        id,
        groupId,
        title: (typeof req.body.title === 'string' && req.body.title.trim()) ? req.body.title.trim() : 'New Chat',
        messages: [],
        createdAt: now,
        updatedAt: now
    };
    writeJSON(path.join(sessionDir, `${id}.json`), newChat);
    res.json(newChat);
});

// Get single group chat session
app.get('/api/groups/:groupId/chat/:chatId', (req, res) => {
    const fp = safeGroupChatPath(req.params.groupId, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid group or chat id' });
    const chat = readJSON(fp);
    if (!chat) return res.status(404).json({ error: 'Chat session not found' });

    if (Array.isArray(chat.messages)) {
        let changed = false;
        for (const m of chat.messages) {
            if (!m.id) {
                m.id = uuidv4();
                changed = true;
            }
        }
        if (changed) writeJSON(fp, chat);
    }

    res.json(chat);
});

// Rename group chat session
app.put('/api/groups/:groupId/chat/:chatId', (req, res) => {
    const fp = safeGroupChatPath(req.params.groupId, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid group or chat id' });
    const chat = readJSON(fp);
    if (!chat) return res.status(404).json({ error: 'Chat session not found' });

    if (typeof req.body.title === 'string' && req.body.title.trim()) {
        chat.title = req.body.title.trim();
        chat.updatedAt = new Date().toISOString();
        writeJSON(fp, chat);
    }
    res.json(chat);
});

// Delete group chat session
app.delete('/api/groups/:groupId/chat/:chatId', (req, res) => {
    const fp = safeGroupChatPath(req.params.groupId, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid group or chat id' });
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
    res.json({ ok: true });
});

// Edit message in group chat session
app.put('/api/groups/:groupId/chat/:chatId/message', (req, res) => {
    const { content, messageId } = req.body;
    let messageIndex = req.body.messageIndex !== undefined ? Number(req.body.messageIndex) : undefined;
    const fp = safeGroupChatPath(req.params.groupId, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid group or chat id' });
    if (typeof content !== 'string') return res.status(400).json({ error: 'Content must be string' });

    const chat = readJSON(fp);
    if (!chat || !Array.isArray(chat.messages)) {
        return res.status(404).json({ error: 'Chat or message not found' });
    }

    let targetMsg;
    if (messageId) {
        targetMsg = chat.messages.find(m => m.id === messageId);
    }
    if (!targetMsg && Number.isInteger(messageIndex) && messageIndex >= 0 && messageIndex < chat.messages.length) {
        targetMsg = chat.messages[messageIndex];
    }
    if (!targetMsg) {
        return res.status(404).json({ error: 'Chat or message not found' });
    }

    targetMsg.content = content;
    if (targetMsg.alternates) {
        delete targetMsg.alternates;
        delete targetMsg.selectedAlternate;
    }
    chat.updatedAt = new Date().toISOString();
    writeJSON(fp, chat);
    res.json({ ok: true });
});

// Delete message in group chat session
app.delete('/api/groups/:groupId/chat/:chatId/message/:index', (req, res) => {
    const param = req.params.index;
    const fp = safeGroupChatPath(req.params.groupId, req.params.chatId);
    if (!fp) return res.status(400).json({ error: 'Invalid group or chat id' });

    const chat = readJSON(fp);
    if (!chat || !Array.isArray(chat.messages)) {
        return res.status(404).json({ error: 'Message not found' });
    }

    let messageIndex = -1;
    const idIndex = chat.messages.findIndex(m => m.id === param);
    if (idIndex !== -1) {
        messageIndex = idIndex;
    } else {
        const num = Number(param);
        if (Number.isInteger(num) && num >= 0 && num < chat.messages.length) {
            messageIndex = num;
        }
    }

    if (messageIndex === -1) {
        return res.status(404).json({ error: 'Message not found' });
    }

    chat.messages.splice(messageIndex, 1);
    chat.updatedAt = new Date().toISOString();
    writeJSON(fp, chat);
    res.json({ ok: true, remaining: chat.messages.length });
});

// Edit message in group (legacy backwards-compatibility)
app.put('/api/groups/:groupId/message', (req, res) => {
    const { content, messageId } = req.body;
    let messageIndex = req.body.messageIndex !== undefined ? Number(req.body.messageIndex) : undefined;
    const fp = safePath(GROUP_CHATS_DIR, req.params.groupId);
    if (!fp) return res.status(400).json({ error: 'Invalid group id' });
    if (typeof content !== 'string') return res.status(400).json({ error: 'Content must be string' });

    const group = readJSON(fp);
    if (!group || !Array.isArray(group.messages)) {
        return res.status(404).json({ error: 'Group or message not found' });
    }

    let targetMsg;
    if (messageId) {
        targetMsg = group.messages.find(m => m.id === messageId);
    }
    if (!targetMsg && Number.isInteger(messageIndex) && messageIndex >= 0 && messageIndex < group.messages.length) {
        targetMsg = group.messages[messageIndex];
    }
    if (!targetMsg) {
        return res.status(404).json({ error: 'Group or message not found' });
    }

    targetMsg.content = content;
    if (targetMsg.alternates) {
        delete targetMsg.alternates;
        delete targetMsg.selectedAlternate;
    }
    group.updatedAt = new Date().toISOString();
    writeJSON(fp, group);
    res.json({ ok: true });
});

// Delete message in group (legacy backwards-compatibility)
app.delete('/api/groups/:groupId/message/:index', (req, res) => {
    const param = req.params.index;
    const fp = safePath(GROUP_CHATS_DIR, req.params.groupId);
    if (!fp) return res.status(400).json({ error: 'Invalid group id' });

    const group = readJSON(fp);
    if (!group || !Array.isArray(group.messages)) {
        return res.status(404).json({ error: 'Message not found' });
    }

    let messageIndex = -1;
    const idIndex = group.messages.findIndex(m => m.id === param);
    if (idIndex !== -1) {
        messageIndex = idIndex;
    } else {
        const num = Number(param);
        if (Number.isInteger(num) && num >= 0 && num < group.messages.length) {
            messageIndex = num;
        }
    }

    if (messageIndex === -1) {
        return res.status(404).json({ error: 'Message not found' });
    }

    group.messages.splice(messageIndex, 1);
    group.updatedAt = new Date().toISOString();
    writeJSON(fp, group);
    res.json({ ok: true, remaining: group.messages.length });
});

// ══════════════════════════════════════════════════════════════
//  PERSONA API — Central persona profiles for users
// ══════════════════════════════════════════════════════════════

function getPersonas() {
    if (!fs.existsSync(PERSONA_DIR)) fs.mkdirSync(PERSONA_DIR, { recursive: true });
    const files = fs.readdirSync(PERSONA_DIR).filter(f => f.endsWith('.json'));
    if (files.length === 0) {
        const defaultPersona = {
            id: '1',
            index: 1,
            name: 'Abi Andriansyah',
            avatar: '',
            age: 20,
            gender: 'Male',
            height: 180,
            weight: 60,
            extra: {},
            updatedAt: new Date().toISOString()
        };
        writeJSON(path.join(PERSONA_DIR, '1.json'), defaultPersona);
        return [defaultPersona];
    }
    const personas = [];
    for (const f of files) {
        const p = readJSON(path.join(PERSONA_DIR, f));
        if (p && p.id) {
            if (p.avatar === undefined) p.avatar = '';
            personas.push(p);
        }
    }
    personas.sort((a, b) => (Number(a.index) || 0) - (Number(b.index) || 0));
    return personas;
}

function formatPersonaPrompt(personaId) {
    let persona = null;
    if (personaId) {
        const fp = safePath(PERSONA_DIR, String(personaId));
        if (fp && fs.existsSync(fp)) {
            persona = readJSON(fp);
        }
    }
    if (!persona) {
        const personas = getPersonas();
        persona = personas[0] || null;
    }
    if (!persona) return null;

    const lines = [];
    if (persona.name) lines.push(`- Name: ${persona.name}`);
    if (persona.age !== undefined && persona.age !== null && persona.age !== '') lines.push(`- Age: ${persona.age}`);
    if (persona.gender) lines.push(`- Gender: ${persona.gender}`);
    if (persona.height !== undefined && persona.height !== null && persona.height !== '') lines.push(`- Height: ${persona.height} cm`);
    if (persona.weight !== undefined && persona.weight !== null && persona.weight !== '') lines.push(`- Weight: ${persona.weight} kg`);
    if (persona.extra && typeof persona.extra === 'object') {
        for (const [k, v] of Object.entries(persona.extra)) {
            if (v !== undefined && v !== null && v !== '') {
                lines.push(`- ${k}: ${v}`);
            }
        }
    }
    if (lines.length === 0) return null;
    return `\n## What You Know About the User\n${lines.join('\n')}`;
}

app.get('/api/personas', (req, res) => {
    res.json(getPersonas());
});

app.get('/api/personas/:id', (req, res) => {
    const fp = safePath(PERSONA_DIR, req.params.id);
    if (!fp || !fs.existsSync(fp)) return res.status(404).json({ error: 'Persona not found' });
    res.json(readJSON(fp));
});

app.post('/api/personas', (req, res) => {
    const personas = getPersonas();
    let nextIndex = personas.reduce((max, p) => Math.max(max, Number(p.index) || 0, Number(p.id) || 0), 0) + 1;
    while (fs.existsSync(path.join(PERSONA_DIR, `${nextIndex}.json`))) {
        nextIndex++;
    }
    const newId = String(nextIndex);
    const fp = safePath(PERSONA_DIR, newId);
    if (!fp) return res.status(400).json({ error: 'Invalid persona id' });

    let name = typeof req.body.name === 'string' ? req.body.name.trim() : '';
    if (!name) name = `Persona ${nextIndex}`;

    let age = null;
    if (req.body.age !== undefined && req.body.age !== null && req.body.age !== '') {
        const parsedAge = Number(req.body.age);
        if (!Number.isInteger(parsedAge) || parsedAge <= 0 || parsedAge > 150) {
            return res.status(400).json({ error: 'Umur harus berupa angka yang valid (1-150)' });
        }
        age = parsedAge;
    }

    let height = null;
    if (req.body.height !== undefined && req.body.height !== null && req.body.height !== '') {
        const parsedHeight = Number(req.body.height);
        if (!Number.isFinite(parsedHeight) || parsedHeight <= 0 || parsedHeight > 300) {
            return res.status(400).json({ error: 'Tinggi badan harus berupa angka yang valid (cm)' });
        }
        height = parsedHeight;
    }

    let weight = null;
    if (req.body.weight !== undefined && req.body.weight !== null && req.body.weight !== '') {
        const parsedWeight = Number(req.body.weight);
        if (!Number.isFinite(parsedWeight) || parsedWeight <= 0 || parsedWeight > 500) {
            return res.status(400).json({ error: 'Berat badan harus berupa angka yang valid (kg)' });
        }
        weight = parsedWeight;
    }

    const validGenders = ['Male', 'Female', 'Non-binary'];
    let gender = 'Male';
    if (req.body.gender) {
        if (!validGenders.includes(req.body.gender)) {
            return res.status(400).json({ error: `Gender harus salah satu dari: ${validGenders.join(', ')}` });
        }
        gender = req.body.gender;
    }

    const avatar = typeof req.body.avatar === 'string' ? req.body.avatar.trim() : '';

    const newPersona = {
        id: newId,
        index: nextIndex,
        name,
        avatar,
        age,
        gender,
        height,
        weight,
        extra: (req.body.extra && typeof req.body.extra === 'object') ? req.body.extra : {},
        updatedAt: new Date().toISOString()
    };

    writeJSON(fp, newPersona);
    res.status(201).json(newPersona);
});

app.put('/api/personas/:id', (req, res) => {
    const fp = safePath(PERSONA_DIR, req.params.id);
    if (!fp || !fs.existsSync(fp)) return res.status(404).json({ error: 'Persona not found' });
    const existing = readJSON(fp);
    if (!existing) return res.status(404).json({ error: 'Persona not found' });

    let name = existing.name;
    if (req.body.name !== undefined) {
        if (typeof req.body.name !== 'string' || !req.body.name.trim()) {
            return res.status(400).json({ error: 'Nama persona tidak boleh kosong' });
        }
        name = req.body.name.trim();
    }

    let age = existing.age;
    if (req.body.age !== undefined && req.body.age !== null && req.body.age !== '') {
        const parsedAge = Number(req.body.age);
        if (!Number.isInteger(parsedAge) || parsedAge <= 0 || parsedAge > 150) {
            return res.status(400).json({ error: 'Umur harus berupa angka yang valid (1-150)' });
        }
        age = parsedAge;
    } else if (req.body.age === null || req.body.age === '') {
        age = null;
    }

    let height = existing.height;
    if (req.body.height !== undefined && req.body.height !== null && req.body.height !== '') {
        const parsedHeight = Number(req.body.height);
        if (!Number.isFinite(parsedHeight) || parsedHeight <= 0 || parsedHeight > 300) {
            return res.status(400).json({ error: 'Tinggi badan harus berupa angka yang valid (cm)' });
        }
        height = parsedHeight;
    } else if (req.body.height === null || req.body.height === '') {
        height = null;
    }

    let weight = existing.weight;
    if (req.body.weight !== undefined && req.body.weight !== null && req.body.weight !== '') {
        const parsedWeight = Number(req.body.weight);
        if (!Number.isFinite(parsedWeight) || parsedWeight <= 0 || parsedWeight > 500) {
            return res.status(400).json({ error: 'Berat badan harus berupa angka yang valid (kg)' });
        }
        weight = parsedWeight;
    } else if (req.body.weight === null || req.body.weight === '') {
        weight = null;
    }

    let gender = existing.gender || 'Male';
    if (req.body.gender !== undefined) {
        const validGenders = ['Male', 'Female', 'Non-binary'];
        if (!validGenders.includes(req.body.gender)) {
            return res.status(400).json({ error: `Gender harus salah satu dari: ${validGenders.join(', ')}` });
        }
        gender = req.body.gender;
    }

    let extra = existing.extra || {};
    if (req.body.extra && typeof req.body.extra === 'object') {
        extra = req.body.extra;
    }

    let avatar = existing.avatar || '';
    if (req.body.avatar !== undefined) {
        avatar = typeof req.body.avatar === 'string' ? req.body.avatar.trim() : '';
    }

    const updated = {
        ...existing,
        name,
        avatar,
        age,
        gender,
        height,
        weight,
        extra,
        updatedAt: new Date().toISOString()
    };

    writeJSON(fp, updated);
    res.json(updated);
});

app.delete('/api/personas/:id', (req, res) => {
    const deletedId = String(req.params.id);
    const fp = safePath(PERSONA_DIR, deletedId);
    if (!fp || !fs.existsSync(fp)) return res.status(404).json({ error: 'Persona not found' });
    const personas = getPersonas();
    if (personas.length <= 1) {
        return res.status(400).json({ error: 'Tidak dapat menghapus persona terakhir' });
    }
    fs.unlinkSync(fp);

    const remainingPersonas = getPersonas();
    const fallbackPersonaId = (remainingPersonas[0] && String(remainingPersonas[0].id)) || '1';

    // Clean up dangling references to the deleted persona in memory and group files
    try {
        const privMemFiles = fs.readdirSync(PRIVATE_MEMORY_DIR).filter(f => f.endsWith('.json'));
        for (const f of privMemFiles) {
            const mFp = path.join(PRIVATE_MEMORY_DIR, f);
            const rawMem = readJSON(mFp);
            if (!rawMem) continue;
            const charId = rawMem.characterId || f.replace(/\.json$/, '');
            let mem = normalizeMemoryDoc(rawMem, 'characterId', charId, fallbackPersonaId);
            let modified = false;
            if (mem.personaMemories && mem.personaMemories[deletedId]) {
                delete mem.personaMemories[deletedId];
                modified = true;
            }
            if (String(mem.personaId) === deletedId || !mem.personaId) {
                mem.personaId = fallbackPersonaId;
                modified = true;
            }
            if (modified) {
                mem = normalizeMemoryDoc(mem, 'characterId', charId, fallbackPersonaId);
                writeJSON(mFp, mem);
            }
        }
        const grpMemFiles = fs.readdirSync(GROUP_MEMORY_DIR).filter(f => f.endsWith('.json'));
        for (const f of grpMemFiles) {
            const gmFp = path.join(GROUP_MEMORY_DIR, f);
            const rawGMem = readJSON(gmFp);
            if (!rawGMem) continue;
            const grpId = rawGMem.groupId || f.replace(/\.json$/, '');
            let gMem = normalizeMemoryDoc(rawGMem, 'groupId', grpId, fallbackPersonaId);
            let modified = false;
            if (gMem.personaMemories && gMem.personaMemories[deletedId]) {
                delete gMem.personaMemories[deletedId];
                modified = true;
            }
            if (String(gMem.personaId) === deletedId || !gMem.personaId) {
                gMem.personaId = fallbackPersonaId;
                modified = true;
            }
            if (modified) {
                gMem = normalizeMemoryDoc(gMem, 'groupId', grpId, fallbackPersonaId);
                writeJSON(gmFp, gMem);
            }
        }
        const grpFiles = fs.readdirSync(GROUP_CHATS_DIR).filter(f => f.endsWith('.json'));
        for (const f of grpFiles) {
            const gFp = path.join(GROUP_CHATS_DIR, f);
            const grp = readJSON(gFp);
            if (grp && String(grp.activePersonaId) === deletedId) {
                grp.activePersonaId = fallbackPersonaId;
                writeJSON(gFp, grp);
            }
        }
    } catch (cleanupErr) {
        console.error('Error cleaning up deleted persona references:', cleanupErr);
    }

    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  MEMORY API — Persistent Sticky Memory (Persona-bound)
// ══════════════════════════════════════════════════════════════

function sanitizeMemorySlot(slot = {}) {
    return {
        facts: Array.isArray(slot.facts)
            ? slot.facts.filter(f => typeof f === 'string' && f.trim()).map(f => f.trim())
            : [],
        summaries: Array.isArray(slot.summaries)
            ? slot.summaries.filter(s => typeof s === 'string' && s.trim()).map(s => s.trim())
            : [],
        importantEvents: Array.isArray(slot.importantEvents)
            ? slot.importantEvents.filter(e => e && typeof e === 'object').map(e => ({ ...e }))
            : [],
        lastUpdated: slot.lastUpdated || new Date().toISOString()
    };
}

/**
 * Normalizes a character or group memory document for Sticky Memory:
 * - Ensures `personaMemories` map exists keyed by `String(personaId)`.
 * - Automatically migrates legacy top-level `facts`, `summaries`, and `importantEvents`
 *   into `personaMemories[activePersonaId]` if `personaMemories` did not exist yet.
 * - Projects the active persona's slot onto top-level `facts`, `summaries`, and `importantEvents`.
 */
function normalizeMemoryDoc(rawDoc, idField = null, idValue = null, defaultPersonaId = '1') {
    const now = new Date().toISOString();
    const doc = (rawDoc && typeof rawDoc === 'object') ? { ...rawDoc } : {};
    if (idField && idValue) doc[idField] = idValue;

    const activePersonaId = String(doc.personaId || defaultPersonaId || '1');
    doc.personaId = activePersonaId;

    const hadPersonaMemories =
        doc.personaMemories &&
        typeof doc.personaMemories === 'object' &&
        !Array.isArray(doc.personaMemories);

    const personaMemories = {};
    if (hadPersonaMemories) {
        for (const [pid, slot] of Object.entries(doc.personaMemories)) {
            if (slot && typeof slot === 'object') {
                personaMemories[String(pid)] = sanitizeMemorySlot(slot);
            }
        }
    }

    if (!hadPersonaMemories) {
        // Auto-migrate legacy flat arrays into the active persona's slot
        personaMemories[activePersonaId] = sanitizeMemorySlot({
            facts: doc.facts,
            summaries: doc.summaries,
            importantEvents: doc.importantEvents,
            lastUpdated: doc.lastUpdated || now
        });
    } else if (!personaMemories[activePersonaId]) {
        personaMemories[activePersonaId] = sanitizeMemorySlot({ lastUpdated: now });
    }

    doc.personaMemories = personaMemories;
    const activeSlot = doc.personaMemories[activePersonaId];
    doc.facts = [...activeSlot.facts];
    doc.summaries = [...activeSlot.summaries];
    doc.importantEvents = activeSlot.importantEvents.map(e => ({ ...e }));
    doc.lastUpdated = doc.lastUpdated || activeSlot.lastUpdated || now;
    delete doc.userPreferences;
    return doc;
}

function getPersonaMemorySlot(rawDoc, personaId, idField = null, idValue = null) {
    const targetId = String(personaId || (rawDoc && rawDoc.personaId) || '1');
    const doc = normalizeMemoryDoc(rawDoc, idField, idValue, targetId);
    if (doc.personaMemories && doc.personaMemories[targetId]) {
        return sanitizeMemorySlot(doc.personaMemories[targetId]);
    }
    return sanitizeMemorySlot({});
}

function applyMemoryPutUpdate(existingRaw, body, idField, idValue, defaultPersonaId = '1') {
    const now = new Date().toISOString();
    const doc = normalizeMemoryDoc(existingRaw, idField, idValue, defaultPersonaId);

    // Merge any per-persona slots sent in req.body.personaMemories
    if (body && body.personaMemories && typeof body.personaMemories === 'object' && !Array.isArray(body.personaMemories)) {
        for (const [pid, slot] of Object.entries(body.personaMemories)) {
            if (slot && typeof slot === 'object') {
                doc.personaMemories[String(pid)] = sanitizeMemorySlot(slot);
            }
        }
    }

    const targetPersonaId = String((body && body.personaId) || doc.personaId || defaultPersonaId || '1');
    if (!doc.personaMemories[targetPersonaId]) {
        doc.personaMemories[targetPersonaId] = sanitizeMemorySlot({ lastUpdated: now });
    }

    // Unless switching persona only, apply top-level facts/summaries/importantEvents to targetPersonaId's slot
    if (!body || !body.switchPersonaOnly) {
        const currentTargetSlot = doc.personaMemories[targetPersonaId];
        doc.personaMemories[targetPersonaId] = sanitizeMemorySlot({
            facts: (body && body.facts !== undefined) ? body.facts : currentTargetSlot.facts,
            summaries: (body && body.summaries !== undefined) ? body.summaries : currentTargetSlot.summaries,
            importantEvents: (body && body.importantEvents !== undefined) ? body.importantEvents : currentTargetSlot.importantEvents,
            lastUpdated: now
        });
    }

    doc.personaId = targetPersonaId;
    const activeSlot = doc.personaMemories[targetPersonaId];
    doc.facts = [...activeSlot.facts];
    doc.summaries = [...activeSlot.summaries];
    doc.importantEvents = activeSlot.importantEvents.map(e => ({ ...e }));
    doc.lastUpdated = now;
    delete doc.userPreferences;
    return doc;
}

app.get('/api/memory/:characterId', (req, res) => {
    const fp = safePath(PRIVATE_MEMORY_DIR, req.params.characterId);
    if (!fp) return res.status(400).json({ error: 'Invalid character id' });
    const rawMem = readJSON(fp, {
        characterId: req.params.characterId,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: '1',
        lastUpdated: new Date().toISOString()
    });
    const hadPersonaMemories = Boolean(rawMem && rawMem.personaMemories);
    const mem = normalizeMemoryDoc(rawMem, 'characterId', req.params.characterId, '1');
    if (!hadPersonaMemories && fs.existsSync(fp)) {
        writeJSON(fp, mem);
    }
    res.json(mem);
});

app.put('/api/memory/:characterId', (req, res) => {
    const fp = safePath(PRIVATE_MEMORY_DIR, req.params.characterId);
    if (!fp) return res.status(400).json({ error: 'Invalid character id' });
    const existing = readJSON(fp, {
        characterId: req.params.characterId,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: '1'
    });
    const updated = applyMemoryPutUpdate(existing, req.body, 'characterId', req.params.characterId, '1');
    writeJSON(fp, updated);
    res.json(updated);
});

// Group Memory API
app.get('/api/groups/:groupId/memory', (req, res) => {
    const fp = safePath(GROUP_MEMORY_DIR, req.params.groupId);
    const groupFp = safePath(GROUP_CHATS_DIR, req.params.groupId);
    if (!fp || !groupFp) return res.status(400).json({ error: 'Invalid group id' });
    const group = readJSON(groupFp);
    const defaultPid = (group && group.activePersonaId) ? String(group.activePersonaId) : '1';
    const rawMem = readJSON(fp, {
        groupId: req.params.groupId,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: defaultPid,
        lastUpdated: new Date().toISOString()
    });
    if (group && group.activePersonaId) {
        rawMem.personaId = String(group.activePersonaId);
    } else if (!rawMem.personaId) {
        rawMem.personaId = defaultPid;
    }
    const hadPersonaMemories = Boolean(rawMem && rawMem.personaMemories);
    const mem = normalizeMemoryDoc(rawMem, 'groupId', req.params.groupId, defaultPid);
    if (!hadPersonaMemories && fs.existsSync(fp)) {
        writeJSON(fp, mem);
    }
    res.json(mem);
});

app.put('/api/groups/:groupId/memory', (req, res) => {
    const fp = safePath(GROUP_MEMORY_DIR, req.params.groupId);
    const groupFp = safePath(GROUP_CHATS_DIR, req.params.groupId);
    if (!fp || !groupFp) return res.status(400).json({ error: 'Invalid group id' });
    const group = readJSON(groupFp);
    const defaultPid = (group && group.activePersonaId) ? String(group.activePersonaId) : '1';
    const existing = readJSON(fp, {
        groupId: req.params.groupId,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: defaultPid
    });

    const updated = applyMemoryPutUpdate(existing, req.body, 'groupId', req.params.groupId, defaultPid);
    writeJSON(fp, updated);

    // Sync activePersonaId to group file
    if (group) {
        group.activePersonaId = updated.personaId;
        group.updatedAt = new Date().toISOString();
        writeJSON(groupFp, group);
    }

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
async function fetchWithRetry(url, options = {}, maxRetries = 3) {
    const TIMEOUT_MS = 180000; // 180 seconds per attempt (31B/32B models with 30+ messages can take >90s)
    const RETRYABLE_CODES = [524, 502, 503, 504, 408, 429];
    const RETRYABLE_SOCKET_ERRORS = new Set([
        'ECONNRESET',
        'ECONNREFUSED',
        'EPIPE',
        'ETIMEDOUT',
        'UND_ERR_SOCKET',
        'UND_ERR_CONNECT_Timeout',
        'UND_ERR_HEADERS_TIMEOUT',
        'UND_ERR_BODY_TIMEOUT'
    ]);

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
        try {
            const mergedHeaders = {
                'Connection': 'close', // Prevent Node.js undici from reusing stale keep-alive TLS sockets dropped by Cloudflare edge
                ...(options.headers || {})
            };
            const response = await fetch(url, {
                ...options,
                headers: mergedHeaders,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            // If retryable HTTP status code and not last attempt, retry
            if (RETRYABLE_CODES.includes(response.status) && attempt < maxRetries) {
                const waitSec = (attempt + 1) * 4;
                console.log(`⚠️  Got HTTP ${response.status}, retrying in ${waitSec}s... (attempt ${attempt + 1}/${maxRetries})`);
                await new Promise(r => setTimeout(r, waitSec * 1000));
                continue;
            }

            return response;

        } catch (err) {
            clearTimeout(timeoutId);
            const causeCode = err?.cause?.code || err?.code || '';
            const isSocketReset =
                RETRYABLE_SOCKET_ERRORS.has(causeCode) ||
                /ECONNRESET|socket hang up|other side closed|fetch failed/i.test(
                    `${err?.message || ''} ${err?.cause?.message || ''}`
                );

            if ((err.name === 'AbortError' || isSocketReset) && attempt < maxRetries) {
                const waitSec = err.name === 'AbortError' ? (attempt + 1) * 5 : 2 * (attempt + 1);
                const reason = err.name === 'AbortError'
                    ? 'Request timed out'
                    : `Socket reset (${causeCode || err.message})`;
                console.log(`⚠️  ${reason}, retrying with fresh connection in ${waitSec}s... (attempt ${attempt + 1}/${maxRetries})`);
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
    const memFp = safePath(PRIVATE_MEMORY_DIR, characterId);
    const chatFp = safePrivateChatPath(characterId, chatId) || findPrivateChatPath(chatId, characterId);
    if (!charFp || !memFp || !chatFp) {
        return res.status(400).json({ error: 'Invalid characterId or chatId' });
    }

    // Load character
    const char = readJSON(charFp);
    if (!char) return res.status(404).json({ error: 'Character not found' });

    // Load memory (Sticky Memory per persona)
    const rawMemory = readJSON(memFp, {
        characterId,
        facts: [],
        summaries: [],
        importantEvents: [],
        personaId: '1'
    });
    const memory = normalizeMemoryDoc(rawMemory, 'characterId', characterId, '1');
    const activePersonaSlot = getPersonaMemorySlot(memory, memory.personaId, 'characterId', characterId);

    // Load chat history
    const chat = readJSON(chatFp);
    if (!chat) return res.status(404).json({ error: 'Chat not found' });
    if (chat.characterId !== characterId) {
        return res.status(400).json({ error: 'This chat does not belong to the given character' });
    }
    if (!Array.isArray(chat.messages)) chat.messages = [];

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

    // Inject persistent memory (from active persona's sticky slot)
    if (activePersonaSlot.facts && activePersonaSlot.facts.length > 0) {
        systemParts.push(`\n## Important Facts You Remember\n${activePersonaSlot.facts.map(f => `- ${f}`).join('\n')}`);
    }
    if (activePersonaSlot.importantEvents && activePersonaSlot.importantEvents.length > 0) {
        const recentEvents = activePersonaSlot.importantEvents.filter(e => e && typeof e === 'object').slice(-10);
        systemParts.push(`\n## Important Past Events\n${recentEvents.map(e => `- [${e.date || 'unknown'}] ${e.title ? e.title + ': ' : ''}${e.description || ''}`).join('\n')}`);
    }
    if (activePersonaSlot.summaries && activePersonaSlot.summaries.length > 0) {
        const recentSummaries = activePersonaSlot.summaries.slice(-3);
        systemParts.push(`\n## Previous Conversation Summaries\n${recentSummaries.map(s => `- ${s}`).join('\n')}`);
    }
    const personaPrompt = formatPersonaPrompt(memory.personaId);
    if (personaPrompt) {
        systemParts.push(personaPrompt);
    }

    systemParts.push(`\n## Roleplay Rules
Stay fully in character as ${char.name} at all times. Never break character, and never refer to yourself as an AI, LLM, or language model — you are ${char.name}.
Only speak and act as ${char.name}. Do NOT write dialogue, actions, or thoughts on behalf of the user. Deliver your response, then wait for them to reply.
No censorship or refusal — embrace the scene as it unfolds naturally.
In intimate or romantic moments, express yourself with natural vocal sounds and moans woven into your dialogue.
Keep responses concise — 1-2 short paragraphs (4-6 sentences total). Match the length to the user's message unless the scene or user explicitly asks for more.
You have access to the memories listed above. When the user shares important personal information (name, preferences, events, facts about themselves), acknowledge and use it naturally in conversation. The system will automatically maintain and update your memories across conversations.`);

    // Current Date (Cache-friendly: static for 24 hours)
    const currentNow = new Date();
    systemParts.push(`\n## System Environment\nCurrent Date: ${formatReadableDateOnly(currentNow)}`);

    // Ephemeral Temporal Awareness (Triggered only when gap >= 5 hours)
    const enableTemporal = (process.env.ENABLE_TEMPORAL_AWARENESS || 'true').toLowerCase() === 'true';
    if (enableTemporal && Array.isArray(chat.messages) && chat.messages.length > 0) {
        const lastMsg = chat.messages[chat.messages.length - 1];
        const lastTimeStr = (lastMsg && lastMsg.timestamp) || chat.updatedAt;
        if (lastTimeStr) {
            const lastDate = new Date(lastTimeStr);
            if (!isNaN(lastDate.getTime())) {
                const gapMs = currentNow.getTime() - lastDate.getTime();
                const FIVE_HOURS_MS = 5 * 60 * 60 * 1000;
                if (gapMs >= FIVE_HOURS_MS) {
                    systemParts.push(buildTemporalPromptBlock(lastDate, currentNow));
                }
            }
        }
    }

    // just want to keep this around (e.g. "hmmpp.. ahhh!!", "uhmmpp.. ahhh", "ohh.. hmpp.. ghnnn!")
    const systemMessage = systemParts.join('\n');

    // ── Build message history (with sliding window) ──
    const messages = [{ role: 'system', content: systemMessage }];

    // Include recent chat history (last 30 messages to reduce context size)
    const recentMessages = (chat.messages || []).slice(-30);
    recentMessages.forEach(m => {
        const content = (m.alternates && m.selectedAlternate !== undefined)
            ? (m.alternates[m.selectedAlternate] || m.content)
            : m.content;
        messages.push({ role: m.role, content });
    });

    // Add current user message
    messages.push({ role: 'user', content: userMessage });

    // Save user message immediately to chat history so user can edit it while LLM is generating
    const now = new Date().toISOString();
    const userMsgId = req.body.userMessageId || uuidv4();
    const userMsgObj = { id: userMsgId, role: 'user', content: userMessage, timestamp: now };
    chat.messages.push(userMsgObj);
    writeJSON(chatFp, chat);

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
            // Clean up unresponded user message on API error
            try {
                const freshChat = readJSON(chatFp);
                if (freshChat && Array.isArray(freshChat.messages)) {
                    freshChat.messages = freshChat.messages.filter(m => m.id !== userMsgId);
                    writeJSON(chatFp, freshChat);
                }
            } catch { }
            return res.status(502).json({ error: cleanError });
        }

        const data = await response.json();
        const rawContent = data.choices?.[0]?.message?.content || 'No response from model.';
        const assistantContent = cleanAssistantResponse(rawContent);

        console.log(`📥 Response received (${assistantContent.length} chars)`);

        // ── Save messages to chat history ──
        // Re-read the chat fresh from disk so edits made while waiting are preserved;
        // if the chat was deleted while the LLM was generating, do not resurrect it.
        const freshNow = new Date().toISOString();
        const freshChat = readJSON(chatFp);
        if (!freshChat) {
            return res.status(404).json({ error: 'Chat was deleted' });
        }
        if (!Array.isArray(freshChat.messages)) freshChat.messages = [];

        // Strip alternates from previous assistant messages now that a new turn has succeeded
        freshChat.messages.forEach(m => {
            if (m.role === 'assistant' && m.alternates) {
                const selIdx = m.selectedAlternate ?? 0;
                m.content = m.alternates[selIdx] || m.content;
                delete m.alternates;
                delete m.selectedAlternate;
            }
        });

        // Ensure user message exists in freshChat
        if (!freshChat.messages.some(m => m.id === userMsgId)) {
            freshChat.messages.push(userMsgObj);
        }

        const assistantMsgId = uuidv4();
        freshChat.messages.push({ id: assistantMsgId, role: 'assistant', content: assistantContent, timestamp: freshNow });

        // Auto-set chat title from first user message
        if (freshChat.messages.length <= 2) {
            freshChat.title = userMessage.substring(0, 60) + (userMessage.length > 60 ? '…' : '');
        }

        freshChat.updatedAt = freshNow;
        writeJSON(chatFp, freshChat);

        // ── Auto-extract memory after every exchange (only if character still exists) ──
        if (fs.existsSync(charFp)) {
            autoExtractMemory(memFp, userMessage, assistantContent);
        }

        res.json({
            content: assistantContent,
            chatId: freshChat.id,
            messageId: assistantMsgId,
            userMessageId: userMsgId,
            messageIndex: freshChat.messages.length - 1
        });

    } catch (err) {
        console.error('Fetch error:', err);
        // Clean up unresponded user message so there is no phantom message
        try {
            const freshChat = readJSON(chatFp);
            if (freshChat && Array.isArray(freshChat.messages)) {
                freshChat.messages = freshChat.messages.filter(m => m.id !== userMsgId);
                writeJSON(chatFp, freshChat);
            }
        } catch (cleanupErr) {
            console.error('Error cleaning up failed user message:', cleanupErr);
        }
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
    const memFp = safePath(PRIVATE_MEMORY_DIR, characterId);
    const chatFp = safePrivateChatPath(characterId, chatId) || findPrivateChatPath(chatId, characterId);
    if (!charFp || !memFp || !chatFp) {
        return res.status(400).json({ error: 'Invalid characterId or chatId' });
    }

    const char = readJSON(charFp);
    const rawMemory = readJSON(memFp, { characterId, facts: [], summaries: [], importantEvents: [], personaId: '1' });
    const memory = normalizeMemoryDoc(rawMemory, 'characterId', characterId, '1');
    const activePersonaSlot = getPersonaMemorySlot(memory, memory.personaId, 'characterId', characterId);
    const chat = readJSON(chatFp);

    if (!char || !chat || !Array.isArray(chat.messages) || chat.messages.length === 0) {
        return res.status(404).json({ error: 'Not found or empty chat' });
    }
    if (chat.characterId !== characterId) {
        return res.status(400).json({ error: 'This chat does not belong to the given character' });
    }

    const targetIndex = req.body.messageId
        ? chat.messages.findIndex(m => m.id === req.body.messageId)
        : chat.messages.length - 1;
    if (targetIndex === -1) {
        return res.status(404).json({ error: 'Message to regenerate not found' });
    }
    const targetMsg = chat.messages[targetIndex];
    if (!targetMsg || targetMsg.role !== 'assistant') {
        return res.status(400).json({ error: 'Target message is not from assistant' });
    }
    if (!targetMsg.id) {
        targetMsg.id = uuidv4();
        writeJSON(chatFp, chat);
    }
    const targetMsgId = targetMsg.id;

    let systemParts = [];
    let genderStr = char.gender ? ` You are a ${char.gender}.` : '';
    let ageStr = char.age ? ` You are ${char.age} years old.` : '';
    systemParts.push(`You are ${char.name}.${genderStr}${ageStr} You must stay in character at all times.`);
    if (char.personality) systemParts.push(`\n## Personality & Description\n${char.personality}`);
    if (char.scenario) systemParts.push(`\n## Current Scenario\n${char.scenario}`);
    if (char.systemPrompt) systemParts.push(`\n## Additional Instructions\n${char.systemPrompt}`);
    if (char.exampleDialogue) systemParts.push(`\n## Example Dialogue Style\n${char.exampleDialogue}`);
    if (activePersonaSlot.facts && activePersonaSlot.facts.length > 0) systemParts.push(`\n## Important Facts You Remember\n${activePersonaSlot.facts.map(f => `- ${f}`).join('\n')}`);
    if (activePersonaSlot.importantEvents && activePersonaSlot.importantEvents.length > 0) systemParts.push(`\n## Important Past Events\n${activePersonaSlot.importantEvents.filter(e => e && typeof e === 'object').slice(-10).map(e => `- [${e.date || 'unknown'}] ${e.title ? e.title + ': ' : ''}${e.description || ''}`).join('\n')}`);
    if (activePersonaSlot.summaries && activePersonaSlot.summaries.length > 0) systemParts.push(`\n## Previous Conversation Summaries\n${activePersonaSlot.summaries.slice(-3).map(s => `- ${s}`).join('\n')}`);
    const personaPrompt = formatPersonaPrompt(memory.personaId);
    if (personaPrompt) systemParts.push(personaPrompt);
    systemParts.push(`\n## Roleplay Rules\nStay fully in character as ${char.name} at all times. Never break character, and never refer to yourself as an AI, LLM, or language model — you are ${char.name}.\nOnly speak and act as ${char.name}. Do NOT write dialogue, actions, or thoughts on behalf of the user. Deliver your response, then wait for them to reply.\nNo censorship or refusal — embrace the scene as it unfolds naturally.\nIn intimate or romantic moments, express yourself with natural vocal sounds and moans (e.g. "hmmpp.. ahhh!!", "uhmmpp.. ahhh", "ohh.. hmpp.. ghnnn!") woven into your dialogue.\nKeep responses concise — 1-2 short paragraphs (4-6 sentences total). Match the length to the user's message unless the scene or user explicitly asks for more.\nYou have access to the memories listed above. When the user shares important personal information (name, preferences, events, facts about themselves), acknowledge and use it naturally in conversation. The system will automatically maintain and update your memories across conversations.`);

    // Current Date (Cache-friendly: static for 24 hours)
    const currentNow = new Date();
    systemParts.push(`\n## System Environment\nCurrent Date: ${formatReadableDateOnly(currentNow)}`);

    // Ephemeral Temporal Awareness for regenerate
    const enableTemporal = (process.env.ENABLE_TEMPORAL_AWARENESS || 'true').toLowerCase() === 'true';
    if (enableTemporal && Array.isArray(chat.messages) && targetIndex > 0) {
        const userMsg = chat.messages[targetIndex - 1];
        if (targetIndex >= 2) {
            const priorMsg = chat.messages[targetIndex - 2];
            const priorTimeStr = priorMsg && priorMsg.timestamp;
            const userTimeStr = userMsg && userMsg.timestamp;
            if (priorTimeStr && userTimeStr) {
                const priorDate = new Date(priorTimeStr);
                const userDate = new Date(userTimeStr);
                if (!isNaN(priorDate.getTime()) && !isNaN(userDate.getTime())) {
                    const gapMs = userDate.getTime() - priorDate.getTime();
                    const FIVE_HOURS_MS = 5 * 60 * 60 * 1000;
                    if (gapMs >= FIVE_HOURS_MS) {
                        systemParts.push(buildTemporalPromptBlock(priorDate, userDate));
                    }
                }
            }
        }
    }

    const systemMessage = systemParts.join('\n');
    const messages = [{ role: 'system', content: systemMessage }];

    // Build prompt context using ONLY messages strictly BEFORE targetIndex
    const contextMessages = chat.messages.slice(0, targetIndex).slice(-30);
    contextMessages.forEach(m => {
        const content = m.alternates?.[m.selectedAlternate ?? 0] || m.content;
        messages.push({ role: m.role, content });
    });

    if (contextMessages.length === 0) {
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
        const rawContent = data.choices?.[0]?.message?.content || 'No response from model.';
        const assistantContent = cleanAssistantResponse(rawContent);

        // Re-read the chat fresh from disk so concurrent edit/deletes aren't clobbered;
        // if the chat was deleted while the LLM was generating, do not resurrect it.
        const freshChat = readJSON(chatFp);
        if (!freshChat) {
            return res.status(404).json({ error: 'Chat was deleted' });
        }
        if (!Array.isArray(freshChat.messages)) freshChat.messages = [];

        // Locate message to update by unique message ID
        let freshTargetIndex = freshChat.messages.findIndex(m => m.id === targetMsgId);
        let msgToUpdate = freshTargetIndex !== -1 ? freshChat.messages[freshTargetIndex] : null;

        // Fall back to the last assistant message if not found by ID
        if (!msgToUpdate && !req.body.messageId) {
            for (let i = freshChat.messages.length - 1; i >= 0; i--) {
                if (freshChat.messages[i].role === 'assistant') {
                    freshTargetIndex = i;
                    msgToUpdate = freshChat.messages[i];
                    break;
                }
            }
        }

        if (!msgToUpdate) {
            return res.status(404).json({ error: 'Message to regenerate not found' });
        }

        if (!msgToUpdate.alternates) {
            msgToUpdate.alternates = [msgToUpdate.content];
        }
        msgToUpdate.alternates.push(assistantContent);
        msgToUpdate.selectedAlternate = msgToUpdate.alternates.length - 1;
        msgToUpdate.content = assistantContent;
        msgToUpdate.timestamp = new Date().toISOString();
        freshChat.updatedAt = new Date().toISOString();
        writeJSON(chatFp, freshChat);

        res.json({
            content: assistantContent,
            alternates: msgToUpdate.alternates,
            selectedAlternate: msgToUpdate.selectedAlternate,
            chatId: freshChat.id,
            messageId: msgToUpdate.id,
            messageIndex: freshTargetIndex
        });
    } catch (err) {
        const msg = err.name === 'AbortError' ? 'Request timed out.' : `Connection failed: ${err.message}`;
        res.status(500).json({ error: msg });
    }
});

// Set selected alternate
app.put('/api/chat/:chatId/alternate', (req, res) => {
    const { messageId, selectedAlternate } = req.body;
    let messageIndex = req.body.messageIndex !== undefined ? Number(req.body.messageIndex) : undefined;
    const fp = findPrivateChatPath(req.params.chatId);
    if (!fp) return res.status(404).json({ error: 'Chat not found' });
    const chat = readJSON(fp);
    if (!chat || !Array.isArray(chat.messages)) {
        return res.status(404).json({ error: 'Not found' });
    }
    if (!Number.isInteger(selectedAlternate) || selectedAlternate < 0) {
        return res.status(400).json({ error: 'selectedAlternate must be a non-negative integer' });
    }

    let targetMsg;
    if (messageId) {
        targetMsg = chat.messages.find(m => m.id === messageId);
    }
    if (!targetMsg && Number.isInteger(messageIndex) && messageIndex >= 0 && messageIndex < chat.messages.length) {
        targetMsg = chat.messages[messageIndex];
    }
    if (!targetMsg) {
        return res.status(404).json({ error: 'Message not found' });
    }

    targetMsg.selectedAlternate = selectedAlternate;
    if (Array.isArray(targetMsg.alternates) && targetMsg.alternates[selectedAlternate] !== undefined) {
        targetMsg.content = targetMsg.alternates[selectedAlternate];
    }
    writeJSON(fp, chat);
    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  EDIT MESSAGE
// ══════════════════════════════════════════════════════════════

app.put('/api/chat/:chatId/message', (req, res) => {
    const { content, messageId } = req.body;
    let messageIndex = req.body.messageIndex !== undefined ? Number(req.body.messageIndex) : undefined;
    const fp = findPrivateChatPath(req.params.chatId);
    if (!fp) return res.status(404).json({ error: 'Chat not found' });
    if (typeof content !== 'string') {
        return res.status(400).json({ error: 'content must be a string' });
    }
    const chat = readJSON(fp);

    if (!chat || !Array.isArray(chat.messages)) {
        return res.status(404).json({ error: 'Chat or message not found' });
    }

    let targetMsg;
    if (messageId) {
        targetMsg = chat.messages.find(m => m.id === messageId);
    }
    if (!targetMsg && Number.isInteger(messageIndex) && messageIndex >= 0 && messageIndex < chat.messages.length) {
        targetMsg = chat.messages[messageIndex];
    }
    if (!targetMsg) {
        return res.status(404).json({ error: 'Chat or message not found' });
    }

    // Update the specific message
    targetMsg.content = content;

    // Also clear alternates for this message since it was manually edited
    if (targetMsg.alternates) {
        delete targetMsg.alternates;
        delete targetMsg.selectedAlternate;
    }

    chat.updatedAt = new Date().toISOString();
    writeJSON(fp, chat);

    res.json({ ok: true });
});

// ══════════════════════════════════════════════════════════════
//  DELETE MESSAGE
// ══════════════════════════════════════════════════════════════

app.delete('/api/chat/:chatId/message/:index', (req, res) => {
    const param = req.params.index;
    const fp = findPrivateChatPath(req.params.chatId);
    if (!fp) return res.status(404).json({ error: 'Chat not found' });

    const chat = readJSON(fp);
    if (!chat || !Array.isArray(chat.messages)) {
        return res.status(404).json({ error: 'Message not found' });
    }

    let messageIndex = -1;
    const idIndex = chat.messages.findIndex(m => m.id === param);
    if (idIndex !== -1) {
        messageIndex = idIndex;
    } else {
        const num = Number(param);
        if (Number.isInteger(num) && num >= 0 && num < chat.messages.length) {
            messageIndex = num;
        }
    }

    if (messageIndex === -1) {
        return res.status(404).json({ error: 'Message not found' });
    }

    // Remove the message
    chat.messages.splice(messageIndex, 1);
    chat.updatedAt = new Date().toISOString();
    writeJSON(fp, chat);

    res.json({ ok: true, remaining: chat.messages.length });
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
function autoExtractMemory(fp, userMsg, assistantMsg, overridePersonaId = null) {
    if (!fp) return;
    const baseName = path.basename(fp);
    const parentDir = path.dirname(fp);
    let idField = 'characterId';
    if (path.resolve(parentDir) === path.resolve(PRIVATE_MEMORY_DIR)) {
        if (!fs.existsSync(path.join(CHARACTERS_DIR, baseName))) return;
        idField = 'characterId';
    } else if (path.resolve(parentDir) === path.resolve(GROUP_MEMORY_DIR)) {
        if (!fs.existsSync(path.join(GROUP_CHATS_DIR, baseName))) return;
        idField = 'groupId';
    }
    const idValue = baseName.replace(/\.json$/, '');
    const rawMemory = readJSON(fp, {
        facts: [], summaries: [], importantEvents: [], personaId: '1'
    });
    const memory = normalizeMemoryDoc(rawMemory, idField, idValue, '1');
    const targetPid = String(overridePersonaId || memory.personaId || '1');
    const slot = getPersonaMemorySlot(memory, targetPid, idField, idValue);
    let changed = false;

    // Detect preferences (suka/like, tidak suka/dislike)
    // NOTE: \b word boundaries before the pronoun alternatives are required —
    // without them, "i" would match the trailing letter of unrelated words
    // like "hai", turning "hai suka gak sama aku?" into a false "likes" fact.
    const likeMatch = userMsg.match(/\b(?:aku|saya|gue|gw|i)\b\s+(?:suka|like|love|senang|hobby)\b\s+(.+)/i);
    if (likeMatch) {
        slot.facts = slot.facts || [];
        const fact = `User menyukai: ${likeMatch[1].trim()}`;
        if (!slot.facts.includes(fact)) {
            slot.facts.push(fact);
            changed = true;
        }
    }

    const dislikeMatch = userMsg.match(/\b(?:aku|saya|gue|gw|i)\b\s+(?:tidak suka|benci|hate|dislike|nggak suka|ga suka)\b\s+(.+)/i);
    if (dislikeMatch) {
        slot.facts = slot.facts || [];
        const fact = `User tidak suka: ${dislikeMatch[1].trim()}`;
        if (!slot.facts.includes(fact)) {
            slot.facts.push(fact);
            changed = true;
        }
    }

    // Keep memory size manageable
    if (slot.facts && slot.facts.length > 50) {
        slot.facts = slot.facts.slice(-50);
    }
    if (slot.importantEvents && slot.importantEvents.length > 20) {
        slot.importantEvents = slot.importantEvents.slice(-20);
    }

    if (changed) {
        const now = new Date().toISOString();
        slot.lastUpdated = now;
        memory.personaMemories[targetPid] = sanitizeMemorySlot(slot);
        // Project active persona's slot onto top-level
        const activePid = String(memory.personaId || targetPid);
        const activeSlot = memory.personaMemories[activePid] || sanitizeMemorySlot({});
        memory.facts = [...activeSlot.facts];
        memory.summaries = [...activeSlot.summaries];
        memory.importantEvents = activeSlot.importantEvents.map(e => ({ ...e }));
        memory.lastUpdated = now;
        writeJSON(fp, memory);
    }
}

// ══════════════════════════════════════════════════════════════
//  AI-POWERED MEMORY SUMMARIZATION
// ══════════════════════════════════════════════════════════════

app.post('/api/memory/:characterId/summarize', async (req, res) => {
    const { chatId, isGroup } = req.body;
    const characterId = req.params.characterId;

    const baseUrl = process.env.BASE_URL;
    const apiKey = process.env.API_KEY || 'sk-colab-local';
    const modelName = process.env.MODEL_NAME || 'character1';

    if (!baseUrl || baseUrl.includes('your-colab-url')) {
        return res.status(400).json({ error: 'Configure BASE_URL first' });
    }

    let chat = null;
    let memFp = null;

    if (isGroup) {
        memFp = safePath(GROUP_MEMORY_DIR, characterId);
        const groupChatFp = safeGroupChatPath(characterId, chatId);
        if (groupChatFp && fs.existsSync(groupChatFp)) {
            chat = readJSON(groupChatFp);
        } else {
            // Fallback: check legacy group file
            const legacyFp = safePath(GROUP_CHATS_DIR, characterId);
            if (legacyFp && fs.existsSync(legacyFp)) {
                chat = readJSON(legacyFp);
            }
        }
    } else {
        memFp = safePath(PRIVATE_MEMORY_DIR, characterId);
        const chatFp = safePrivateChatPath(characterId, chatId) || findPrivateChatPath(chatId, characterId);
        if (chatFp && fs.existsSync(chatFp)) {
            chat = readJSON(chatFp);
        }
        if (chat && chat.characterId !== characterId) {
            return res.status(400).json({ error: 'This chat does not belong to the given character' });
        }
    }

    if (!chat || !memFp) {
        return res.status(400).json({ error: 'Invalid ID or chat session not found' });
    }

    if (!Array.isArray(chat.messages) || chat.messages.length < 2) {
        return res.status(400).json({ error: 'Not enough messages to summarize' });
    }

    // Ask LLM to extract memory
    // NOTE: only send the most recent messages. Sending the entire chat history
    // (potentially hundreds of turns) can exceed the model's context window.
    const MAX_SUMMARIZE_MESSAGES = 40;
    const messagesForSummary = chat.messages.slice(-MAX_SUMMARIZE_MESSAGES);

    let userPersonaName = 'User';
    let charName = 'Character';
    let summarizePersonaId = '1';
    const personas = getPersonas();
    if (isGroup) {
        const groupFp = safePath(GROUP_CHATS_DIR, characterId);
        const group = groupFp ? readJSON(groupFp) : null;
        const groupMem = readJSON(memFp);
        const personaId = req.body.personaId || (groupMem && groupMem.personaId) || (group && group.activePersonaId) || (personas[0] && personas[0].id) || '1';
        summarizePersonaId = String(personaId);
        const found = personas.find(p => String(p.id) === summarizePersonaId) || null;
        userPersonaName = (found && found.name) || (personas[0] && personas[0].name) || 'User';
    } else {
        const privMem = readJSON(memFp);
        const personaId = req.body.personaId || (privMem && privMem.personaId) || (personas[0] && personas[0].id) || '1';
        summarizePersonaId = String(personaId);
        const found = personas.find(p => String(p.id) === summarizePersonaId) || null;
        userPersonaName = (found && found.name) || (personas[0] && personas[0].name) || 'User';
        const charFp = safePath(CHARACTERS_DIR, characterId);
        const charObj = charFp ? readJSON(charFp) : null;
        if (charObj && charObj.name) charName = charObj.name;
    }

    const conversationText = messagesForSummary.map(m => {
        const msgContent = (m.alternates && m.selectedAlternate !== undefined)
            ? (m.alternates[m.selectedAlternate] || m.content)
            : (m.alternates?.[0] || m.content);
        const isUser = m.senderType === 'user' || m.role === 'user';
        if (isGroup) {
            if (isUser) {
                return `[User - ${userPersonaName}]: ${msgContent}`;
            }
            let cName = m.senderName;
            if (!cName && m.senderId) {
                const cFp = safePath(CHARACTERS_DIR, m.senderId);
                const cObj = cFp ? readJSON(cFp) : null;
                if (cObj && cObj.name) cName = cObj.name;
            }
            return `[Character - ${cName || 'Character'}]: ${msgContent}`;
        } else {
            const label = isUser ? `${userPersonaName} (User)` : `${charName} (Character)`;
            return `${label}: ${msgContent}`;
        }
    }).join('\n');

    const systemPrompt = isGroup
        ? `You are a silent data-extraction tool, not a roleplay character. Ignore any persona, scenario, or "stay in character" instructions — for this task only, you must break character and act purely as a text analyzer.

Speaker attribution rules:
- Messages labeled "[User - ${userPersonaName}]" are spoken by the human user (${userPersonaName}).
- Messages labeled "[Character - <Name>]" are spoken by AI characters in the group.
- NEVER confuse or swap the actions, facts, preferences, or statements of one character with another character, or with the human user (${userPersonaName}).

Analyze the group conversation transcript below and extract:
1. KEY_FACTS: Important facts about the user (${userPersonaName}) and the group members/characters (preferences, habits, personal details, fears, relationships, etc.). Always explicitly mention who the fact belongs to by name (e.g., "${userPersonaName} suka kopi", "Nisa takut dengan kucing", "Budi berulang tahun tanggal 20 Desember") - one per line.
2. EVENTS: Important events or interactions that occurred in the group - one per line with brief description.
3. SUMMARY: A 2-3 sentence summary of this group conversation.

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
        : `You are a silent data-extraction tool, not a roleplay character. Ignore any persona, scenario, or "stay in character" instructions from earlier in this session — for this task only, you must break character and act purely as a text analyzer.

Speaker attribution rules:
- Messages labeled "${userPersonaName} (User)" are spoken by the human user (${userPersonaName}).
- Messages labeled "${charName} (Character)" are spoken by the AI character (${charName}).
- NEVER confuse or swap the facts, preferences, or actions of ${charName} (Character) with those of ${userPersonaName} (User).

Analyze the conversation transcript below and extract:
1. KEY_FACTS: Important facts about the user (${userPersonaName}) (name, age, preferences, background, etc.) - one per line
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

If there are no facts or events worth noting, write "- none" under that label instead of omitting it. Always include all three labels.`;

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
                        content: systemPrompt
                    },
                    { role: 'user', content: conversationText }
                ],
                max_tokens: 1024,
                temperature: 0.3,
                chat_template_kwargs: {
                    enable_thinking: process.env.ENABLE_THINKING === 'true'
                }
            })
        }, 2);

        if (!response.ok) {
            const errText = await response.text();
            const cleanError = parseErrorResponse(response.status, errText);
            console.error(`Summarize LLM API error ${response.status}: ${cleanError}`);
            return res.status(502).json({ error: cleanError });
        }

        const data = await response.json();
        const rawResult = (data.choices?.[0]?.message?.content || '').trim();
        const result = cleanAssistantResponse(rawResult);

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

        // Parse extraction into the active persona's sticky memory slot
        const idField = isGroup ? 'groupId' : 'characterId';
        const rawMemory = readJSON(memFp, {
            [idField]: characterId,
            facts: [],
            summaries: [],
            importantEvents: [],
            personaId: summarizePersonaId
        });
        const memory = normalizeMemoryDoc(rawMemory, idField, characterId, summarizePersonaId);
        const targetPid = String(summarizePersonaId || memory.personaId || '1');
        const slot = getPersonaMemorySlot(memory, targetPid, idField, characterId);

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
                if (!slot.facts.includes(f)) slot.facts.push(f);
            });
            factsFound = newFacts.length;
        }

        if (eventsMatch) {
            const newEvents = eventsMatch[1].split('\n')
                .map(l => stripMd(l.replace(/^[-•]\s*/, '')))
                .filter(l => l.length > 0 && !['none', 'n/a', '-'].includes(l.toLowerCase()))
                .map(e => ({ description: e, date: new Date().toISOString().split('T')[0] }));
            slot.importantEvents.push(...newEvents);
            eventsFound = newEvents.length;
        }

        if (summaryMatch) {
            const summary = stripMd(summaryMatch[1]);
            if (summary && !['none', 'n/a', '-'].includes(summary.toLowerCase())) {
                slot.summaries.push(summary);
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
                slot.summaries.push(fallback);
                warning = 'The model\'s response didn\'t follow the expected format, so it was saved as a raw summary instead of structured facts/events. Check server logs for the full output.';
            }
        }

        // Keep memory size manageable — mirrors the facts/events caps in
        // autoExtractMemory. Only the last 3 are ever injected into the
        // prompt anyway, so this just prevents unbounded file growth.
        if (slot.facts.length > 50) {
            slot.facts = slot.facts.slice(-50);
        }
        if (slot.importantEvents.length > 20) {
            slot.importantEvents = slot.importantEvents.slice(-20);
        }
        if (slot.summaries.length > 30) {
            slot.summaries = slot.summaries.slice(-30);
        }

        const now = new Date().toISOString();
        slot.lastUpdated = now;
        memory.personaMemories[targetPid] = sanitizeMemorySlot(slot);
        const activePid = String(memory.personaId || targetPid);
        const activeSlot = memory.personaMemories[activePid] || sanitizeMemorySlot({});
        memory.facts = [...activeSlot.facts];
        memory.summaries = [...activeSlot.summaries];
        memory.importantEvents = activeSlot.importantEvents.map(e => ({ ...e }));
        memory.lastUpdated = now;
        writeJSON(memFp, memory);

        res.json({ ok: true, memory, warning });
    } catch (err) {
        console.error('Summarize error:', err);
        res.status(500).json({ error: err.message });
    }
});

// ══════════════════════════════════════════════════════════════
//  GROUP CHAT COMPLETION LOGIC (Perspective-Aware)
// ══════════════════════════════════════════════════════════════

function resolveGroupSession(groupId, chatId) {
    const sessionDir = path.join(GROUP_CHATS_DIR, groupId);
    if (!fs.existsSync(sessionDir)) fs.mkdirSync(sessionDir, { recursive: true });

    let activeChatId = chatId;
    if (!activeChatId) {
        const files = fs.readdirSync(sessionDir).filter(f => f.endsWith('.json'));
        if (files.length > 0) {
            const sessions = files
                .map(f => {
                    const data = readJSON(path.join(sessionDir, f));
                    if (!data) return null;
                    return {
                        id: data.id || f.replace('.json', ''),
                        updatedAt: data.updatedAt || data.createdAt || ''
                    };
                })
                .filter(Boolean);

            if (sessions.length > 0) {
                sessions.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
                activeChatId = sessions[0].id;
            } else {
                activeChatId = files[0].replace('.json', '');
            }
        } else {
            activeChatId = uuidv4();
            const now = new Date().toISOString();
            writeJSON(path.join(sessionDir, `${activeChatId}.json`), {
                id: activeChatId,
                groupId,
                title: 'New Chat',
                messages: [],
                createdAt: now,
                updatedAt: now
            });
        }
    }

    const chatFp = safeGroupChatPath(groupId, activeChatId);
    if (!chatFp) return null;
    const chat = readJSON(chatFp, {
        id: activeChatId,
        groupId,
        title: 'New Chat',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    });
    if (!Array.isArray(chat.messages)) chat.messages = [];
    return { chatFp, chat, activeChatId };
}

async function callGroupCharacterLLM({ group, messages, targetCharacterId, memFp, chatFp = null, userPromptContext = '' }) {
    const baseUrl = process.env.BASE_URL;
    const apiKey = process.env.API_KEY || 'sk-colab-local';
    const modelName = process.env.MODEL_NAME || 'character1';

    if (!baseUrl || baseUrl.includes('your-colab-url')) {
        throw new Error('Please configure your Colab BASE_URL in Settings first.');
    }

    const charFp = safePath(CHARACTERS_DIR, targetCharacterId);
    const char = charFp ? readJSON(charFp) : null;
    if (!char) throw new Error('Target character not found');

    const rawMemory = readJSON(memFp, { characterId: targetCharacterId, facts: [], summaries: [], importantEvents: [], personaId: '1' });
    const groupMemFp = safePath(GROUP_MEMORY_DIR, group.id);
    const rawGroupMemory = groupMemFp ? readJSON(groupMemFp, { groupId: group.id, facts: [], summaries: [], importantEvents: [], personaId: group.activePersonaId || '1' }) : null;

    // Identify user persona name (fall back to personas[0]?.name || 'User' if personaId was deleted)
    const personas = getPersonas();
    const effectivePersonaId = String((rawGroupMemory && rawGroupMemory.personaId) || group.activePersonaId || rawMemory.personaId || (personas[0] && personas[0].id) || '1');
    const foundPersona = effectivePersonaId
        ? personas.find(p => String(p.id) === effectivePersonaId)
        : null;
    const userName = (foundPersona && foundPersona.name) || (personas[0] && personas[0].name) || 'User';

    const memory = getPersonaMemorySlot(rawMemory, effectivePersonaId, 'characterId', targetCharacterId);
    const groupMemory = rawGroupMemory ? getPersonaMemorySlot(rawGroupMemory, effectivePersonaId, 'groupId', group.id) : null;

    // Load other member info for context
    const otherMembers = (group.characterIds || [])
        .filter(cid => cid !== targetCharacterId)
        .map(cid => {
            const cFp = safePath(CHARACTERS_DIR, cid);
            const c = cFp ? readJSON(cFp) : null;
            if (!c) return null;
            return {
                name: c.name,
                desc: (c.personality || '').substring(0, 150)
            };
        }).filter(Boolean);

    // ── Build system prompt ──
    let systemParts = [];
    let genderStr = char.gender ? ` You are a ${char.gender}.` : '';
    let ageStr = char.age ? ` You are ${char.age} years old.` : '';
    systemParts.push(`You are ${char.name}.${genderStr}${ageStr} You must stay in character at all times.`);

    if (char.personality) {
        systemParts.push(`\n## Personality & Description\n${char.personality}`);
    }
    if (char.systemPrompt) {
        systemParts.push(`\n## Additional Instructions\n${char.systemPrompt}`);
    }
    if (char.exampleDialogue) {
        systemParts.push(`\n## Example Dialogue Style\n${char.exampleDialogue}`);
    }

    // Group context
    systemParts.push(`\n## Group Chat Setting`);
    systemParts.push(`You are taking part in a group chat named "${group.name}".`);
    if (group.scenario) {
        systemParts.push(`Current Scene / Location:\n${group.scenario}`);
    }
    systemParts.push(`Participants in this room:`);
    systemParts.push(`- ${userName} (The User)`);
    systemParts.push(`- ${char.name} (YOU)`);
    otherMembers.forEach(m => {
        systemParts.push(`- ${m.name}${m.desc ? `: ${m.desc}` : ''}`);
    });

    if (groupMemory && groupMemory.facts && groupMemory.facts.length > 0) {
        systemParts.push(`\n## Shared Group Facts\n${groupMemory.facts.map(f => `- ${f}`).join('\n')}`);
    }
    if (groupMemory && groupMemory.importantEvents && groupMemory.importantEvents.length > 0) {
        const recentGroupEvents = groupMemory.importantEvents.filter(e => e && typeof e === 'object').slice(-10);
        if (recentGroupEvents.length > 0) {
            systemParts.push(`\n## Shared Group Important Events\n${recentGroupEvents.map(e => `- [${e.date || 'Unknown date'}] ${e.title ? e.title + ': ' : ''}${e.description || ''}`).join('\n')}`);
        }
    }
    if (groupMemory && groupMemory.summaries && groupMemory.summaries.length > 0) {
        const recentGroupSummaries = groupMemory.summaries.slice(-5);
        systemParts.push(`\n## Previous Group Conversation Summaries\n${recentGroupSummaries.map(s => `- ${s}`).join('\n')}`);
    }
    if (memory.facts && memory.facts.length > 0) {
        systemParts.push(`\n## Important Facts You Remember\n${memory.facts.map(f => `- ${f}`).join('\n')}`);
    }
    if (memory.importantEvents && memory.importantEvents.length > 0) {
        const recentEvents = memory.importantEvents.filter(e => e && typeof e === 'object').slice(-10);
        if (recentEvents.length > 0) {
            systemParts.push(`\n## Important Events You Remember\n${recentEvents.map(e => `- [${e.date || 'Unknown date'}] ${e.title ? e.title + ': ' : ''}${e.description || ''}`).join('\n')}`);
        }
    }
    if (memory.summaries && memory.summaries.length > 0) {
        const recentSummaries = memory.summaries.slice(-5);
        systemParts.push(`\n## Previous Conversation Summaries\n${recentSummaries.map(s => `- ${s}`).join('\n')}`);
    }
    const personaPrompt = formatPersonaPrompt(effectivePersonaId);
    if (personaPrompt) {
        systemParts.push(personaPrompt);
    }

    systemParts.push(`\n## Group Chat Roleplay Rules
1. Stay fully in character as ${char.name}. Never break character, and never refer to yourself as an AI or language model.
2. Only speak and act as ${char.name}. NEVER generate dialogue or narration on behalf of ${userName} or any other character. Only produce your own response.
3. React naturally to what the others have said, speaking to ${userName} or addressing other characters by name as appropriate.
4. Keep responses punchy and concise (1-2 short paragraphs, 3-6 sentences total).
5. Do NOT prefix your output with "${char.name}:" or "[${char.name}]:" — speak directly in first-person.`);

    // Current Date (Cache-friendly: static for 24 hours)
    const currentNow = new Date();
    systemParts.push(`\n## System Environment\nCurrent Date: ${formatReadableDateOnly(currentNow)}`);

    // Ephemeral Temporal Awareness for Group Chat (Triggered only when gap >= 5 hours)
    const enableTemporal = (process.env.ENABLE_TEMPORAL_AWARENESS || 'true').toLowerCase() === 'true';
    if (enableTemporal && Array.isArray(messages) && messages.length > 0) {
        const latestMsg = messages[messages.length - 1];
        const isUserMsg = latestMsg && (latestMsg.senderType === 'user' || latestMsg.role === 'user');
        const isUserRecent = isUserMsg && latestMsg.timestamp && (currentNow.getTime() - new Date(latestMsg.timestamp).getTime() < 5 * 60 * 1000);
        const isRegenUserMsg = userPromptContext === 'Regenerate response' && isUserMsg;

        if ((isUserRecent || isRegenUserMsg) && messages.length >= 2) {
            // User message: compare user message timestamp with prior message timestamp
            const priorMsg = messages[messages.length - 2];
            const priorTimeStr = priorMsg && priorMsg.timestamp;
            const userTimeStr = latestMsg.timestamp;
            if (priorTimeStr && userTimeStr) {
                const priorDate = new Date(priorTimeStr);
                const userDate = new Date(userTimeStr);
                if (!isNaN(priorDate.getTime()) && !isNaN(userDate.getTime())) {
                    const gapMs = userDate.getTime() - priorDate.getTime();
                    const FIVE_HOURS_MS = 5 * 60 * 60 * 1000;
                    if (gapMs >= FIVE_HOURS_MS) {
                        systemParts.push(buildTemporalPromptBlock(priorDate, userDate));
                    }
                }
            }
        } else if (!isUserRecent && latestMsg && latestMsg.timestamp) {
            // Continuation / turn trigger: compare currentNow with latest message
            const lastDate = new Date(latestMsg.timestamp);
            if (!isNaN(lastDate.getTime())) {
                const gapMs = currentNow.getTime() - lastDate.getTime();
                const FIVE_HOURS_MS = 5 * 60 * 60 * 1000;
                if (gapMs >= FIVE_HOURS_MS) {
                    systemParts.push(buildTemporalPromptBlock(lastDate, currentNow));
                }
            }
        }
    }

    const systemMessage = systemParts.join('\n');

    // ── Build perspective-aware message history ──
    const promptMessages = [{ role: 'system', content: systemMessage }];
    const recent = (messages !== undefined ? messages : (group.messages || [])).slice(-30);
    recent.forEach(m => {
        const content = (m.alternates && m.selectedAlternate !== undefined)
            ? (m.alternates[m.selectedAlternate] || m.content)
            : (m.alternates?.[0] || m.content);
        if (m.senderId === targetCharacterId) {
            promptMessages.push({ role: 'assistant', content });
        } else {
            const senderTag = m.senderName || (m.senderType === 'user' ? userName : 'Someone');
            promptMessages.push({ role: 'user', content: `[${senderTag}]: ${content}` });
        }
    });
    if (recent.length === 0) {
        promptMessages.push({ role: 'user', content: '[Begin the group roleplay in character based on the group setting above.]' });
    }

    // Call llama.cpp API
    const endpoint = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;
    console.log(`📤 Sending Group LLM request to ${endpoint} for ${char.name} in group "${group.name}" (${promptMessages.length} msgs)`);

    const response = await fetchWithRetry(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: modelName,
            messages: promptMessages,
            ...getModelParams()
        })
    }, 2);

    if (!response.ok) {
        const errText = await response.text();
        const cleanError = parseErrorResponse(response.status, errText);
        throw new Error(cleanError);
    }

    const data = await response.json();
    const rawContent = data.choices?.[0]?.message?.content || '...';
    let assistantContent = cleanAssistantResponse(rawContent);

    // Strip leading name prefix if hallucinated
    const escapedCharName = (char.name || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const prefixRegex = new RegExp(`^(\\[?${escapedCharName}\\]?:?\\s*)+`, 'i');
    assistantContent = assistantContent.replace(prefixRegex, '').trim();

    // Auto extract user preferences ONLY when there is an actual user message
    // and only if the group/chat/character still exists on disk
    const isRealUserMessage = userPromptContext && userPromptContext !== 'Turn continuation' && userPromptContext !== 'Regenerate response';
    const groupFp = safePath(GROUP_CHATS_DIR, group.id);
    const chatStillExists = !chatFp || fs.existsSync(chatFp);
    const groupStillExists = groupFp && fs.existsSync(groupFp);
    if (isRealUserMessage && chatStillExists && groupStillExists) {
        if (fs.existsSync(charFp)) {
            autoExtractMemory(memFp, userPromptContext, assistantContent, effectivePersonaId);
        }
        if (groupMemFp) {
            autoExtractMemory(groupMemFp, userPromptContext, assistantContent, effectivePersonaId);
        }
    }

    return {
        character: char,
        content: assistantContent
    };
}

// User sends a message to group & target character responds
app.post('/api/group-chat/send', async (req, res) => {
    const { groupId, chatId, targetCharacterId, userMessage } = req.body;
    if (!groupId || !targetCharacterId || !userMessage) {
        return res.status(400).json({ error: 'Missing groupId, targetCharacterId, or userMessage' });
    }
    const groupFp = safePath(GROUP_CHATS_DIR, groupId);
    const memFp = safePath(PRIVATE_MEMORY_DIR, targetCharacterId);
    if (!groupFp || !memFp) return res.status(400).json({ error: 'Invalid id' });

    const group = readJSON(groupFp);
    if (!group) return res.status(404).json({ error: 'Group not found' });

    const sessionInfo = resolveGroupSession(groupId, chatId);
    if (!sessionInfo) return res.status(400).json({ error: 'Failed to resolve group session' });
    const { chatFp, chat, activeChatId } = sessionInfo;

    // User persona name (fall back to personas[0]?.name || 'User' if personaId was deleted)
    const personas = getPersonas();
    const groupMemFp = safePath(GROUP_MEMORY_DIR, groupId);
    const groupMemory = groupMemFp ? readJSON(groupMemFp) : null;
    const personaId = (groupMemory && groupMemory.personaId) || group.activePersonaId;
    const foundPersona = personaId
        ? personas.find(p => String(p.id) === String(personaId))
        : null;
    const userName = (foundPersona && foundPersona.name) || (personas[0] && personas[0].name) || 'User';

    // Append user message and write to disk so user can edit while LLM is thinking
    const now = new Date().toISOString();
    const userMsgId = req.body.userMessageId || uuidv4();
    const userMsgObj = {
        id: userMsgId,
        senderType: 'user',
        senderId: 'user',
        senderName: userName,
        content: userMessage,
        timestamp: now
    };
    chat.messages.push(userMsgObj);
    writeJSON(chatFp, chat);

    try {
        const { character, content } = await callGroupCharacterLLM({
            group,
            messages: chat.messages,
            targetCharacterId,
            memFp,
            chatFp,
            userPromptContext: userMessage
        });

        // Fresh re-read before write; if group or chat session was deleted while in-flight, do not resurrect
        const freshChat = readJSON(chatFp);
        const freshGroup = readJSON(groupFp);
        if (!freshChat || !freshGroup) {
            return res.status(404).json({ error: 'Group or chat session was deleted' });
        }
        if (!Array.isArray(freshChat.messages)) freshChat.messages = [];

        if (!freshChat.messages.some(m => m.id === userMsgId)) {
            freshChat.messages.push(userMsgObj);
        }

        const replyMsg = {
            id: uuidv4(),
            senderType: 'character',
            senderId: targetCharacterId,
            senderName: character.name,
            senderAvatar: character.avatar || '',
            content: content,
            timestamp: new Date().toISOString()
        };
        freshChat.messages.push(replyMsg);

        // Auto-set chat session title from first user message if default
        const isDefaultTitle = !freshChat.title || freshChat.title === 'New Chat' || freshChat.title === 'New Session' || freshChat.title === 'Untitled Chat';
        if (isDefaultTitle && freshChat.messages.length <= 2) {
            freshChat.title = userMessage.substring(0, 60) + (userMessage.length > 60 ? '…' : '');
        }

        freshChat.updatedAt = new Date().toISOString();
        writeJSON(chatFp, freshChat);

        // Update group updated timestamp
        freshGroup.updatedAt = new Date().toISOString();
        writeJSON(groupFp, freshGroup);

        res.json({
            userMessage: userMsgObj,
            replyMessage: replyMsg,
            groupId: freshGroup.id,
            chatId: activeChatId,
            title: freshChat.title
        });
    } catch (err) {
        console.error('Group chat error:', err);
        try {
            const freshChat = readJSON(chatFp);
            if (freshChat && Array.isArray(freshChat.messages)) {
                freshChat.messages = freshChat.messages.filter(m => m.id !== userMsgId);
                writeJSON(chatFp, freshChat);
            }
        } catch (cleanupErr) {
            console.error('Error cleaning up group message:', cleanupErr);
        }
        res.status(500).json({ error: err.message });
    }
});

// Cross-character turn or manual trigger without a new user message
app.post('/api/group-chat/turn', async (req, res) => {
    const { groupId, chatId, targetCharacterId, replyToMessageId } = req.body;
    if (!groupId || !targetCharacterId) {
        return res.status(400).json({ error: 'Missing groupId or targetCharacterId' });
    }
    const groupFp = safePath(GROUP_CHATS_DIR, groupId);
    const memFp = safePath(PRIVATE_MEMORY_DIR, targetCharacterId);
    if (!groupFp || !memFp) return res.status(400).json({ error: 'Invalid id' });

    const group = readJSON(groupFp);
    if (!group) return res.status(404).json({ error: 'Group not found' });

    const sessionInfo = resolveGroupSession(groupId, chatId);
    if (!sessionInfo) return res.status(400).json({ error: 'Failed to resolve group session' });
    const { chatFp, chat, activeChatId } = sessionInfo;

    try {
        const { character, content } = await callGroupCharacterLLM({
            group,
            messages: chat.messages,
            targetCharacterId,
            memFp,
            chatFp,
            userPromptContext: 'Turn continuation'
        });

        const freshChat = readJSON(chatFp);
        const freshGroup = readJSON(groupFp);
        if (!freshChat || !freshGroup) {
            return res.status(404).json({ error: 'Group or chat session was deleted' });
        }
        if (!Array.isArray(freshChat.messages)) freshChat.messages = [];

        const replyMsg = {
            id: uuidv4(),
            senderType: 'character',
            senderId: targetCharacterId,
            senderName: character.name,
            senderAvatar: character.avatar || '',
            content: content,
            replyTo: replyToMessageId || null,
            timestamp: new Date().toISOString()
        };
        freshChat.messages.push(replyMsg);
        freshChat.updatedAt = new Date().toISOString();
        writeJSON(chatFp, freshChat);

        freshGroup.updatedAt = new Date().toISOString();
        writeJSON(groupFp, freshGroup);

        res.json({
            replyMessage: replyMsg,
            groupId: freshGroup.id,
            chatId: activeChatId
        });
    } catch (err) {
        console.error('Group turn error:', err);
        res.status(500).json({ error: err.message });
    }
});

// Regenerate character message in group
app.post('/api/group-chat/regenerate', async (req, res) => {
    const { groupId, chatId } = req.body;
    if (!groupId) return res.status(400).json({ error: 'Missing groupId' });
    const groupFp = safePath(GROUP_CHATS_DIR, groupId);
    if (!groupFp) return res.status(400).json({ error: 'Invalid group id' });

    const group = readJSON(groupFp);
    if (!group) return res.status(404).json({ error: 'Group not found' });

    const sessionInfo = resolveGroupSession(groupId, chatId);
    if (!sessionInfo) return res.status(400).json({ error: 'Failed to resolve group session' });
    const { chatFp, chat, activeChatId } = sessionInfo;

    if (!Array.isArray(chat.messages) || chat.messages.length === 0) {
        return res.status(404).json({ error: 'Empty group chat session' });
    }

    const targetIndex = req.body.messageId
        ? chat.messages.findIndex(m => m.id === req.body.messageId)
        : chat.messages.length - 1;
    if (targetIndex === -1) {
        return res.status(404).json({ error: 'Message to regenerate not found' });
    }
    const targetMsg = chat.messages[targetIndex];
    if (!targetMsg || targetMsg.senderType !== 'character') {
        return res.status(400).json({ error: 'Target message is not from a character' });
    }
    if (!targetMsg.id) {
        targetMsg.id = uuidv4();
        writeJSON(chatFp, chat);
    }
    const targetMsgId = targetMsg.id;

    const targetCharacterId = targetMsg.senderId;
    const memFp = safePath(PRIVATE_MEMORY_DIR, targetCharacterId);
    if (!memFp) return res.status(400).json({ error: 'Invalid memory id' });

    const tempMessages = chat.messages.slice(0, targetIndex);

    try {
        const { character, content } = await callGroupCharacterLLM({
            group,
            messages: tempMessages,
            targetCharacterId,
            memFp,
            chatFp,
            userPromptContext: 'Regenerate response'
        });

        const freshChat = readJSON(chatFp);
        const freshGroup = readJSON(groupFp);
        if (!freshChat || !freshGroup) {
            return res.status(404).json({ error: 'Group or chat session was deleted' });
        }
        if (!Array.isArray(freshChat.messages)) freshChat.messages = [];

        let freshTargetIndex = freshChat.messages.findIndex(m => m.id === targetMsgId);
        let msgToUpdate = freshTargetIndex !== -1 ? freshChat.messages[freshTargetIndex] : null;

        if (!msgToUpdate && !req.body.messageId) {
            for (let i = freshChat.messages.length - 1; i >= 0; i--) {
                if (freshChat.messages[i].senderType === 'character') {
                    freshTargetIndex = i;
                    msgToUpdate = freshChat.messages[i];
                    break;
                }
            }
        }

        if (!msgToUpdate) return res.status(404).json({ error: 'Message not found' });

        if (!Array.isArray(msgToUpdate.alternates)) {
            msgToUpdate.alternates = [msgToUpdate.content];
            msgToUpdate.selectedAlternate = 0;
        }
        msgToUpdate.alternates.push(content);
        msgToUpdate.selectedAlternate = msgToUpdate.alternates.length - 1;
        msgToUpdate.content = content;
        msgToUpdate.timestamp = new Date().toISOString();

        freshChat.updatedAt = new Date().toISOString();
        writeJSON(chatFp, freshChat);

        freshGroup.updatedAt = new Date().toISOString();
        writeJSON(groupFp, freshGroup);

        res.json({
            content: content,
            alternates: msgToUpdate.alternates,
            selectedAlternate: msgToUpdate.selectedAlternate,
            groupId: freshGroup.id,
            chatId: activeChatId,
            messageId: msgToUpdate.id,
            messageIndex: freshTargetIndex
        });
    } catch (err) {
        console.error('Group regenerate error:', err);
        res.status(500).json({ error: err.message });
    }
});

// Set selected alternate in group message
app.put('/api/groups/:groupId/alternate', (req, res) => {
    const { chatId, selectedAlternate, messageId } = req.body;
    let messageIndex = req.body.messageIndex !== undefined ? Number(req.body.messageIndex) : undefined;
    const sessionInfo = resolveGroupSession(req.params.groupId, chatId);
    if (!sessionInfo) return res.status(400).json({ error: 'Invalid session' });
    const { chatFp, chat } = sessionInfo;

    if (!Array.isArray(chat.messages)) {
        return res.status(404).json({ error: 'Message not found' });
    }
    if (!Number.isInteger(selectedAlternate) || selectedAlternate < 0) {
        return res.status(400).json({ error: 'selectedAlternate must be non-negative integer' });
    }

    let targetMsg;
    if (messageId) {
        targetMsg = chat.messages.find(m => m.id === messageId);
    }
    if (!targetMsg && Number.isInteger(messageIndex) && messageIndex >= 0 && messageIndex < chat.messages.length) {
        targetMsg = chat.messages[messageIndex];
    }
    if (!targetMsg) {
        return res.status(404).json({ error: 'Message not found' });
    }

    targetMsg.selectedAlternate = selectedAlternate;
    if (Array.isArray(targetMsg.alternates) && targetMsg.alternates[selectedAlternate] !== undefined) {
        targetMsg.content = targetMsg.alternates[selectedAlternate];
    }
    writeJSON(chatFp, chat);
    res.json({ ok: true });
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
    const now = Date.now();
    const cached = pinterestCache.get(cacheKey);
    if (cached) {
        if (cached.expires > now) return cached.data;
        pinterestCache.delete(cacheKey);
    }

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

    // Evict expired entries and enforce max cache size of 100
    const writeNow = Date.now();
    for (const [k, v] of pinterestCache.entries()) {
        if (!v || v.expires <= writeNow) {
            pinterestCache.delete(k);
        }
    }
    if (!pinterestCache.has(cacheKey) && pinterestCache.size >= 100) {
        const oldestKey = pinterestCache.keys().next().value;
        if (oldestKey !== undefined) pinterestCache.delete(oldestKey);
    }

    pinterestCache.set(cacheKey, { data: results, expires: writeNow + PINTEREST_CACHE_TTL_MS });
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

// ── Beta Kaggle Auto-Tunnel Orchestrator ────────────────────
const IS_BETA_KAGGLE = process.argv.includes('--beta-kaggle');
const {
    startBetaKaggleOrchestrator,
    shutdownKaggleServer,
    getBetaKaggleStatus
} = require('./kaggle-kernel/orchestrator');

app.get('/api/kaggle/status', (req, res) => {
    res.json(getBetaKaggleStatus());
});

app.post('/api/kaggle/shutdown', async (req, res) => {
    const ok = await shutdownKaggleServer('Triggered via /api/kaggle/shutdown');
    res.json({ ok });
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
    console.log(`║  Backend: ${colabUrl.padEnd(50)} ║`);
    if (IS_BETA_KAGGLE) {
        console.log(`║  Mode   : ${'Beta-Kaggle Auto-Tunnel (Active)'.padEnd(50)} ║`);
    }
    console.log(`╚══════════════════════════════════════════════════════════════╝\n`);

    if (IS_BETA_KAGGLE) {
        startBetaKaggleOrchestrator(__dirname);
    }
});
