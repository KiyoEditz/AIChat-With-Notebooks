/**
 * settings.js — controller for the unified Settings page.
 * Self-contained: this page does not load app.js, so it has its own
 * small `api()` helper (same contract as app.js's) and its own DOM
 * wiring. Relies on window.AIChatTheme from theme.js for the Appearance
 * tab's storage/validation/derivation logic.
 */
(function () {
    'use strict';

    // ── tiny api() helper, same contract as app.js's ──
    async function api(method, url, body = null) {
        const opts = { method, headers: { 'Content-Type': 'application/json' } };
        if (body) opts.body = JSON.stringify(body);

        let res;
        try {
            res = await fetch(url, opts);
        } catch (err) {
            return { error: `Network error: ${err.message}` };
        }

        let data;
        try {
            data = await res.json();
        } catch {
            data = { error: `Server returned an unexpected response (HTTP ${res.status})` };
        }

        if (!res.ok && !data.error) {
            data.error = `Request failed (HTTP ${res.status})`;
        }
        return data;
    }

    // ── Tab switching ──
    const TABS = ['configuration', 'persona', 'appearance', 'notifications', 'data-backup'];

    function activateTab(tabName) {
        if (!TABS.includes(tabName)) tabName = TABS[0];

        document.querySelectorAll('.settings-nav-item').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        TABS.forEach(name => {
            const panel = document.getElementById(`tab-${name}`);
            if (panel) panel.hidden = name !== tabName;
        });

        try {
            const url = new URL(window.location.href);
            url.searchParams.set('tab', tabName);
            window.history.replaceState(null, '', url);
        } catch (err) { /* ignore (e.g. file:// URLs) */ }
    }

    document.querySelectorAll('.settings-nav-item').forEach(btn => {
        if (btn.disabled) return;
        btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    });

    // Deep-link support: settings.html?tab=appearance
    const initialTab = new URLSearchParams(window.location.search).get('tab');
    activateTab(initialTab || 'configuration');

    // ══════════════════════════════════════════════════════
    //  Configuration tab
    // ══════════════════════════════════════════════════════
    async function loadConfigTab() {
        const config = await api('GET', '/api/config');
        document.getElementById('cfg-base-url').value = config.baseUrl || '';
        document.getElementById('cfg-api-key').value = config.apiKey || 'sk-colab-local';
        document.getElementById('cfg-model').value = config.modelName || 'character1';
    }

    function showConnectionResult(ok, message) {
        const el = document.getElementById('settings-connection-result');
        el.style.display = 'block';
        el.className = 'connection-result ' + (ok ? 'success' : 'error');
        el.textContent = message;
    }

    async function saveConfig() {
        const baseUrl = document.getElementById('cfg-base-url').value.trim();
        const apiKey = document.getElementById('cfg-api-key').value.trim();
        const modelName = document.getElementById('cfg-model').value;

        await api('POST', '/api/config', { baseUrl, apiKey, modelName });
        showConnectionResult(true, '✅ Settings saved.');
    }

    document.getElementById('btn-settings-save').addEventListener('click', saveConfig);

    document.getElementById('btn-settings-test-connection').addEventListener('click', async () => {
        const btn = document.getElementById('btn-settings-test-connection');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Testing...';

        try {
            // Save current field values first so the test uses the latest ones.
            const baseUrl = document.getElementById('cfg-base-url').value.trim();
            const apiKey = document.getElementById('cfg-api-key').value.trim();
            const modelName = document.getElementById('cfg-model').value;
            await api('POST', '/api/config', { baseUrl, apiKey, modelName });

            const result = await api('GET', '/api/test-connection');
            if (result.ok) {
                showConnectionResult(true, '✅ Connection successful! Server is healthy.');
            } else {
                showConnectionResult(false, `❌ ${result.error || 'Connection failed'}`);
            }
        } finally {
            btn.disabled = false;
            btn.innerHTML = '🔌 Test Connection';
        }
    });

    loadConfigTab();

    // ══════════════════════════════════════════════════════
    //  Appearance tab (theme)
    // ══════════════════════════════════════════════════════
    var accentPicker = document.getElementById('accent-picker');
    var bgPicker = document.getElementById('bg-picker');
    var accentValue = document.getElementById('accent-value');
    var bgValue = document.getElementById('bg-value');
    var saveIndicator = document.getElementById('save-indicator');
    var themeSaveTimer = null;

    function reflectThemeInputs(palette) {
        accentPicker.value = palette.accent;
        bgPicker.value = palette.background;
        accentValue.textContent = palette.accent.toUpperCase();
        bgValue.textContent = palette.background.toUpperCase();
    }

    function flashSaved() {
        saveIndicator.classList.add('show');
        clearTimeout(flashSaved._t);
        flashSaved._t = setTimeout(() => saveIndicator.classList.remove('show'), 1600);
    }

    function applyAndPersistTheme(palette) {
        window.AIChatTheme.set(palette);
        reflectThemeInputs(palette);
        clearTimeout(themeSaveTimer);
        themeSaveTimer = setTimeout(flashSaved, 120);
    }

    reflectThemeInputs(window.AIChatTheme.get());

    accentPicker.addEventListener('input', () => {
        applyAndPersistTheme({ accent: accentPicker.value, background: bgPicker.value });
    });
    bgPicker.addEventListener('input', () => {
        applyAndPersistTheme({ accent: accentPicker.value, background: bgPicker.value });
    });

    document.querySelectorAll('.preset-swatch').forEach(btn => {
        btn.addEventListener('click', () => {
            applyAndPersistTheme({ accent: btn.dataset.accent, background: btn.dataset.bg });
        });
    });

    document.getElementById('btn-apply-theme').addEventListener('click', () => {
        applyAndPersistTheme({ accent: accentPicker.value, background: bgPicker.value });
    });

    document.getElementById('btn-reset-theme').addEventListener('click', () => {
        window.AIChatTheme.reset();
        reflectThemeInputs(window.AIChatTheme.DEFAULTS);
        flashSaved();
    });

    // ══════════════════════════════════════════════════════
    //  Persona tab
    // ══════════════════════════════════════════════════════
    let personaState = {
        personas: [],
        activeId: null,
        extra: {}
    };

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    const personaNameInput = document.getElementById('persona-name');
    const personaAgeInput = document.getElementById('persona-age');
    const personaGenderInput = document.getElementById('persona-gender');
    const personaHeightInput = document.getElementById('persona-height');
    const personaWeightInput = document.getElementById('persona-weight');
    const personaBadgeIndex = document.getElementById('persona-badge-index');
    const btnDeletePersona = document.getElementById('btn-delete-persona');
    const personaSaveIndicator = document.getElementById('persona-save-indicator');

    // Strict number-only restriction on typing / pasting for Age, Height, Weight
    [personaAgeInput, personaHeightInput, personaWeightInput].forEach(inp => {
        if (!inp) return;
        inp.addEventListener('input', () => {
            inp.value = inp.value.replace(/[^0-9]/g, '');
        });
        inp.addEventListener('keydown', (e) => {
            // Block 'e', 'E', '+', '-', '.'
            if (['e', 'E', '+', '-', '.'].includes(e.key)) {
                e.preventDefault();
            }
        });
    });

    // Extra attributes key selector
    const extraKeySelect = document.getElementById('persona-extra-key-select');
    const extraCustomKey = document.getElementById('persona-extra-custom-key');
    const extraValInput = document.getElementById('persona-extra-val');
    const btnAddExtra = document.getElementById('btn-add-extra-attr');
    const extraItemsContainer = document.getElementById('persona-extra-items');

    if (extraKeySelect) {
        extraKeySelect.addEventListener('change', () => {
            if (extraKeySelect.value === 'custom') {
                extraCustomKey.style.display = 'inline-block';
                extraCustomKey.focus();
            } else {
                extraCustomKey.style.display = 'none';
            }
        });
    }

    function renderExtraAttributes() {
        if (!extraItemsContainer) return;
        const entries = Object.entries(personaState.extra || {});
        if (entries.length === 0) {
            extraItemsContainer.innerHTML = '<div style="color:var(--text-muted); font-size:0.8rem; padding:4px 0">Belum ada atribut tambahan.</div>';
            return;
        }
        extraItemsContainer.innerHTML = entries.map(([k, v]) => `
            <div class="persona-extra-item">
                <span class="persona-extra-key">${escapeHtml(k)}:</span>
                <span class="persona-extra-val">${escapeHtml(v)}</span>
                <button type="button" class="btn-remove-extra" data-key="${escapeHtml(k)}" title="Hapus atribut">✕</button>
            </div>
        `).join('');

        extraItemsContainer.querySelectorAll('.btn-remove-extra').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.key;
                delete personaState.extra[key];
                renderExtraAttributes();
            });
        });
    }

    if (btnAddExtra) {
        btnAddExtra.addEventListener('click', () => {
            let key = extraKeySelect.value;
            if (key === 'custom') {
                key = extraCustomKey.value.trim();
            }
            const val = extraValInput.value.trim();
            if (!key) {
                alert('Nama atribut tidak boleh kosong');
                return;
            }
            if (!val) {
                alert('Nilai atribut tidak boleh kosong');
                return;
            }
            personaState.extra[key] = val;
            extraValInput.value = '';
            if (extraKeySelect.value === 'custom') {
                extraCustomKey.value = '';
            }
            renderExtraAttributes();
        });
    }

    function renderPersonaCards() {
        const container = document.getElementById('persona-list-cards');
        if (!container) return;
        container.innerHTML = personaState.personas.map(p => {
            const isActive = String(p.id) === String(personaState.activeId);
            return `
                <div class="persona-card ${isActive ? 'active' : ''}" data-id="${escapeHtml(p.id)}">
                    <div class="persona-card-header">
                        <span class="persona-card-num">#${escapeHtml(p.index || p.id)}</span>
                        ${isActive ? '<span class="persona-active-badge">Aktif Diedit</span>' : ''}
                    </div>
                    <div class="persona-card-name">${escapeHtml(p.name || 'Tanpa Nama')}</div>
                    <div class="persona-card-meta">
                        ${p.gender ? escapeHtml(p.gender) : 'Male'} • ${p.age ? escapeHtml(p.age) + ' th' : '-'}
                    </div>
                </div>
            `;
        }).join('');

        container.querySelectorAll('.persona-card').forEach(card => {
            card.addEventListener('click', () => {
                selectPersona(card.dataset.id);
            });
        });
    }

    function selectPersona(id) {
        personaState.activeId = String(id);
        const p = personaState.personas.find(item => String(item.id) === String(id));
        if (!p) return;

        personaBadgeIndex.textContent = `Persona #${p.index || p.id}`;
        personaNameInput.value = p.name || '';
        personaAgeInput.value = (p.age !== undefined && p.age !== null) ? p.age : '';
        personaGenderInput.value = p.gender || 'Male';
        personaHeightInput.value = (p.height !== undefined && p.height !== null) ? p.height : '';
        personaWeightInput.value = (p.weight !== undefined && p.weight !== null) ? p.weight : '';
        personaState.extra = Object.assign({}, p.extra || {});
        renderExtraAttributes();

        // Show delete button only if there is more than 1 persona
        btnDeletePersona.style.display = personaState.personas.length > 1 ? 'inline-flex' : 'none';

        renderPersonaCards();
    }

    async function loadPersonaTab() {
        const personas = await api('GET', '/api/personas');
        if (Array.isArray(personas)) {
            personaState.personas = personas;
            if (!personaState.activeId || !personas.find(p => String(p.id) === String(personaState.activeId))) {
                personaState.activeId = personas[0] ? String(personas[0].id) : null;
            }
            if (personaState.activeId) {
                selectPersona(personaState.activeId);
            }
        }
    }

    document.getElementById('btn-create-persona').addEventListener('click', async () => {
        const nextNum = personaState.personas.length + 1;
        const newPersona = await api('POST', '/api/personas', {
            name: `Persona ${nextNum}`,
            gender: 'Male'
        });
        if (newPersona && newPersona.id) {
            personaState.personas.push(newPersona);
            selectPersona(newPersona.id);
        }
    });

    document.getElementById('btn-delete-persona').addEventListener('click', async () => {
        if (personaState.personas.length <= 1) {
            alert('Tidak dapat menghapus persona terakhir.');
            return;
        }
        if (!confirm('Hapus persona ini?')) return;
        const res = await api('DELETE', `/api/personas/${personaState.activeId}`);
        if (res && res.ok) {
            personaState.personas = personaState.personas.filter(p => String(p.id) !== String(personaState.activeId));
            personaState.activeId = personaState.personas[0]?.id || null;
            if (personaState.activeId) {
                selectPersona(personaState.activeId);
            }
        } else {
            alert(res.error || 'Gagal menghapus persona');
        }
    });

    document.getElementById('btn-save-persona').addEventListener('click', async () => {
        const name = personaNameInput.value.trim();
        if (!name) {
            alert('Nama persona tidak boleh kosong');
            personaNameInput.focus();
            return;
        }

        const ageStr = personaAgeInput.value.trim();
        const heightStr = personaHeightInput.value.trim();
        const weightStr = personaWeightInput.value.trim();
        const gender = personaGenderInput.value;

        const body = {
            name,
            gender,
            age: ageStr ? parseInt(ageStr, 10) : null,
            height: heightStr ? parseInt(heightStr, 10) : null,
            weight: weightStr ? parseInt(weightStr, 10) : null,
            extra: personaState.extra
        };

        const updated = await api('PUT', `/api/personas/${personaState.activeId}`, body);
        if (updated && updated.id) {
            const idx = personaState.personas.findIndex(p => String(p.id) === String(personaState.activeId));
            if (idx !== -1) personaState.personas[idx] = updated;
            renderPersonaCards();

            personaSaveIndicator.classList.add('show');
            clearTimeout(personaSaveIndicator._t);
            personaSaveIndicator._t = setTimeout(() => personaSaveIndicator.classList.remove('show'), 1600);
        } else {
            alert(updated.error || 'Gagal menyimpan persona');
        }
    });

    loadPersonaTab();
})();
