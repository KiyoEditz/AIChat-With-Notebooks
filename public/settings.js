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
        if (!config || config.error) {
            if (config && config.error) {
                showConnectionResult(false, config.error);
            }
            return;
        }
        document.getElementById('cfg-base-url').value = config.baseUrl || '';
        document.getElementById('cfg-api-key').value = config.apiKey || 'sk-colab-local';

        const modelSelect = document.getElementById('cfg-model');
        const modelVal = config.modelName || 'character1';
        if (modelSelect) {
            const hasOption = Array.from(modelSelect.options).some(opt => opt.value === modelVal);
            if (!hasOption) {
                const customOpt = document.createElement('option');
                customOpt.value = modelVal;
                customOpt.textContent = modelVal;
                modelSelect.appendChild(customOpt);
            }
            modelSelect.value = modelVal;
        }

        if (config.enableTemporalAwareness !== undefined) {
            reflectTemporalAwareness(config.enableTemporalAwareness);
        }
    }

    const apiKeyInput = document.getElementById('cfg-api-key');
    const toggleApiKeyBtn = document.getElementById('btn-toggle-api-key');
    if (toggleApiKeyBtn && apiKeyInput) {
        toggleApiKeyBtn.addEventListener('click', () => {
            const isMasked = apiKeyInput.type === 'password';
            apiKeyInput.type = isMasked ? 'text' : 'password';
            toggleApiKeyBtn.textContent = isMasked ? 'Hide' : 'Reveal';
            toggleApiKeyBtn.setAttribute('aria-label', isMasked ? 'Hide API key' : 'Reveal API key');
        });
    }

    function showConnectionResult(ok, message, techDetail = '') {
        const el = document.getElementById('settings-connection-result');
        if (!el) return;
        el.style.display = 'block';
        el.className = 'connection-result ' + (ok ? 'success' : 'error');
        if (!ok && techDetail) {
            el.innerHTML = `
                <div class="conn-result-header">
                    <span>${escapeHtml(message)}</span>
                    <button type="button" class="conn-copy-btn" id="btn-copy-conn-detail">Copy technical details</button>
                </div>
            `;
            const copyBtn = document.getElementById('btn-copy-conn-detail');
            if (copyBtn) {
                copyBtn.addEventListener('click', async () => {
                    let copied = false;
                    if (navigator.clipboard && window.isSecureContext) {
                        try {
                            await navigator.clipboard.writeText(techDetail);
                            copied = true;
                        } catch { /* fallback below */ }
                    }
                    if (!copied) {
                        const ta = document.createElement('textarea');
                        ta.value = techDetail;
                        ta.setAttribute('readonly', '');
                        ta.style.position = 'fixed';
                        ta.style.top = '-9999px';
                        ta.style.opacity = '0';
                        document.body.appendChild(ta);
                        ta.select();
                        try { copied = document.execCommand('copy'); } catch { /* ignore */ }
                        document.body.removeChild(ta);
                    }
                    if (copied) {
                        copyBtn.textContent = 'Copied';
                        setTimeout(() => { copyBtn.textContent = 'Copy technical details'; }, 1600);
                    }
                });
            }
        } else {
            el.textContent = message;
        }
    }

    async function saveConfig() {
        const baseUrl = document.getElementById('cfg-base-url').value.trim();
        const apiKey = document.getElementById('cfg-api-key').value.trim();
        const modelName = document.getElementById('cfg-model').value;

        const res = await api('POST', '/api/config', { baseUrl, apiKey, modelName });
        if (!res || res.error) {
            showConnectionResult(false, (res && res.error) || 'Failed to save configuration.');
            return;
        }
        showConnectionResult(true, 'Configuration saved locally to .env.');
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
            const saveRes = await api('POST', '/api/config', { baseUrl, apiKey, modelName });
            if (saveRes && saveRes.error) {
                showConnectionResult(false, saveRes.error);
                return;
            }

            if (!baseUrl) {
                showConnectionResult(false, 'Not configured — enter your Colab Cloudflare URL or local server endpoint first.');
                return;
            }

            const result = await api('GET', '/api/test-connection');
            if (result && result.ok) {
                showConnectionResult(true, `Connected to ${baseUrl} · Model: ${modelName}`);
            } else {
                const detail = `Endpoint: ${baseUrl}\nModel: ${modelName}\nError: ${(result && result.error) || 'Connection failed'}`;
                showConnectionResult(false, (result && result.error) || 'Connection failed — check that your server is running.', detail);
            }
        } finally {
            btn.disabled = false;
            btn.textContent = 'Test Connection';
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
        const p = (palette && typeof palette === 'object' && palette.accent && palette.background)
            ? palette
            : window.AIChatTheme.get();
        accentPicker.value = String(p.accent).toLowerCase();
        bgPicker.value = String(p.background).toLowerCase();
        accentValue.textContent = String(p.accent).toUpperCase();
        bgValue.textContent = String(p.background).toUpperCase();
        const mode = p.mode || 'system';
        document.querySelectorAll('.theme-mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        const fontSize = p.fontSize || 'medium';
        document.querySelectorAll('.font-size-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.size === fontSize);
        });
    }

    function flashSaved() {
        saveIndicator.classList.add('show');
        clearTimeout(flashSaved._t);
        flashSaved._t = setTimeout(() => saveIndicator.classList.remove('show'), 1600);
    }

    function applyAndPersistTheme(palette) {
        const saved = window.AIChatTheme.set(palette);
        reflectThemeInputs(saved);
        clearTimeout(themeSaveTimer);
        themeSaveTimer = setTimeout(flashSaved, 120);
    }

    reflectThemeInputs(window.AIChatTheme.get());

    document.querySelectorAll('.theme-mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            if (window.AIChatTheme.setMode) {
                const updated = window.AIChatTheme.setMode(mode);
                reflectThemeInputs(updated);
                flashSaved();
            }
        });
    });

    document.querySelectorAll('.font-size-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const size = btn.dataset.size;
            if (window.AIChatTheme.setFontSize) {
                const updated = window.AIChatTheme.setFontSize(size);
                reflectThemeInputs(updated);
                flashSaved();
            }
        });
    });

    function reflectTemporalAwareness(enabled) {
        const isTrue = enabled === true || enabled === 'true';
        document.querySelectorAll('.temporal-btn').forEach(btn => {
            btn.classList.toggle('active', (btn.dataset.enabled === 'true') === isTrue);
        });
    }

    document.querySelectorAll('.temporal-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const enabled = btn.dataset.enabled === 'true';
            reflectTemporalAwareness(enabled);
            flashSaved();
            try {
                await api('POST', '/api/config', { enableTemporalAwareness: enabled });
            } catch (err) {
                console.error('Failed to update temporal awareness config:', err);
            }
        });
    });

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
        const currentMode = document.querySelector('.theme-mode-btn.active')?.dataset.mode || window.AIChatTheme.get().mode;
        const currentFontSize = document.querySelector('.font-size-btn.active')?.dataset.size || window.AIChatTheme.get().fontSize || 'medium';
        applyAndPersistTheme({ accent: accentPicker.value, background: bgPicker.value, mode: currentMode, fontSize: currentFontSize });
    });

    document.getElementById('btn-reset-theme').addEventListener('click', () => {
        const resetPalette = window.AIChatTheme.reset();
        reflectThemeInputs(resetPalette || window.AIChatTheme.get());
        flashSaved();
    });

    window.addEventListener('storage', (e) => {
        if (e.key === 'aichat-theme-palette' && window.AIChatTheme) {
            const current = window.AIChatTheme.get();
            window.AIChatTheme.apply(current);
            reflectThemeInputs(current);
        }
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

    function getAvatarInnerHtml(avatar, name) {
        if (avatar && (avatar.startsWith('data:') || avatar.startsWith('http://') || avatar.startsWith('https://') || avatar.startsWith('/'))) {
            return `<img src="${escapeHtml(avatar)}" alt="${escapeHtml(name || 'avatar')}">`;
        }
        return escapeHtml(avatar || (name ? name.trim().charAt(0).toUpperCase() : 'U'));
    }

    const personaNameInput = document.getElementById('persona-name');
    const personaAgeInput = document.getElementById('persona-age');
    const personaGenderInput = document.getElementById('persona-gender');
    const personaHeightInput = document.getElementById('persona-height');
    const personaWeightInput = document.getElementById('persona-weight');
    const personaBadgeIndex = document.getElementById('persona-badge-index');
    const btnDeletePersona = document.getElementById('btn-delete-persona');
    const personaSaveIndicator = document.getElementById('persona-save-indicator');

    // Avatar & Pinterest elements
    const personaAvatarInput = document.getElementById('persona-avatar');
    const personaAvatarFile = document.getElementById('persona-avatar-file');
    const personaAvatarPreview = document.getElementById('persona-avatar-preview');
    const personaAvatarRemoveBtn = document.getElementById('persona-avatar-remove-btn');
    const personaPinterestToggle = document.getElementById('persona-pinterest-toggle');
    const personaPinterestPanel = document.getElementById('persona-pinterest-panel');
    const personaPinterestQuery = document.getElementById('persona-pinterest-query');
    const personaPinterestSearchBtn = document.getElementById('persona-pinterest-search-btn');
    const personaPinterestResults = document.getElementById('persona-pinterest-results');
    const personaPinterestThumb = document.getElementById('persona-pinterest-thumb');
    const personaPinterestCounter = document.getElementById('persona-pinterest-counter');
    const personaPinterestPrevBtn = document.getElementById('persona-pinterest-prev-btn');
    const personaPinterestNextBtn = document.getElementById('persona-pinterest-next-btn');
    const personaPinterestUseBtn = document.getElementById('persona-pinterest-use-btn');
    const personaPinterestStatus = document.getElementById('persona-pinterest-status');

    function updatePersonaAvatarPreview(avatarData) {
        if (!personaAvatarPreview || !personaAvatarRemoveBtn) return;
        if (avatarData && (avatarData.startsWith('data:') || avatarData.startsWith('http://') || avatarData.startsWith('https://') || avatarData.startsWith('/'))) {
            personaAvatarPreview.innerHTML = `<img src="${escapeHtml(avatarData)}" alt="avatar">`;
            personaAvatarPreview.classList.add('has-image');
            personaAvatarRemoveBtn.style.display = 'flex';
        } else {
            personaAvatarPreview.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <span>Upload</span>`;
            personaAvatarPreview.classList.remove('has-image');
            personaAvatarRemoveBtn.style.display = 'none';
        }
    }

    function handlePersonaAvatarUpload(e) {
        const file = e.target.files && e.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            alert('Pilih file gambar (JPG, PNG, GIF, WebP)');
            if (personaAvatarFile) personaAvatarFile.value = '';
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('Ukuran file maksimal 5MB');
            if (personaAvatarFile) personaAvatarFile.value = '';
            return;
        }

        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                const canvas = document.createElement('canvas');
                const MAX_SIZE = 256;
                let width = img.width;
                let height = img.height;

                if (width > height) {
                    if (width > MAX_SIZE) {
                        height *= MAX_SIZE / width;
                        width = MAX_SIZE;
                    }
                } else {
                    if (height > MAX_SIZE) {
                        width *= MAX_SIZE / height;
                        height = MAX_SIZE;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                const dataUrl = canvas.toDataURL('image/webp', 0.85);
                if (personaAvatarInput) personaAvatarInput.value = dataUrl;
                updatePersonaAvatarPreview(dataUrl);
            };
            img.onerror = function() {
                alert('Gagal memuat file gambar.');
            };
            img.src = event.target.result;
        };
        reader.onerror = function() {
            alert('Gagal membaca file gambar.');
        };
        reader.readAsDataURL(file);
        if (personaAvatarFile) personaAvatarFile.value = '';
    }

    function removePersonaAvatar() {
        if (personaAvatarInput) personaAvatarInput.value = '';
        if (personaAvatarFile) personaAvatarFile.value = '';
        updatePersonaAvatarPreview('');
    }

    let personaPinterest = { results: [], index: 0, loading: false };

    function togglePersonaPinterest() {
        if (!personaPinterestPanel) return;
        const isVisible = personaPinterestPanel.style.display !== 'none';
        personaPinterestPanel.style.display = isVisible ? 'none' : 'block';
        if (personaPinterestToggle) personaPinterestToggle.classList.toggle('active', !isVisible);
        if (!isVisible && personaPinterestQuery && !personaPinterestQuery.value) {
            personaPinterestQuery.value = personaNameInput.value.trim() || 'anime avatar';
            searchPersonaPinterest();
        }
    }

    async function searchPersonaPinterest() {
        const query = personaPinterestQuery ? personaPinterestQuery.value.trim() : '';
        if (!query || personaPinterest.loading) return;
        personaPinterest.loading = true;
        if (personaPinterestStatus) personaPinterestStatus.textContent = 'Searching Pinterest...';
        if (personaPinterestResults) personaPinterestResults.style.display = 'none';

        try {
            const data = await api('GET', `/api/pinterest/search?q=${encodeURIComponent(query)}`);
            const items = (data && Array.isArray(data.data)) ? data.data : ((data && Array.isArray(data.images)) ? data.images : []);
            if (data && data.success && items.length > 0) {
                personaPinterest.results = items;
                personaPinterest.index = 0;
                showPersonaPinterestResult();
                if (personaPinterestStatus) personaPinterestStatus.textContent = `Found ${items.length} images.`;
            } else {
                if (personaPinterestStatus) personaPinterestStatus.textContent = data.error || 'No images found. Try another keyword.';
            }
        } catch (err) {
            if (personaPinterestStatus) personaPinterestStatus.textContent = 'Failed to search images.';
        } finally {
            personaPinterest.loading = false;
        }
    }

    function showPersonaPinterestResult() {
        const { results, index } = personaPinterest;
        if (!results.length || !personaPinterestThumb || !personaPinterestResults) return;
        const item = results[index];
        const thumbUrl = typeof item === 'string' ? item : (item.thumb || item.url);
        personaPinterestThumb.src = `/api/pinterest/image?url=${encodeURIComponent(thumbUrl)}`;
        if (personaPinterestCounter) personaPinterestCounter.textContent = `${index + 1} / ${results.length}`;
        personaPinterestResults.style.display = 'flex';
    }

    async function usePersonaPinterestImage() {
        const { results, index } = personaPinterest;
        if (!results.length) return;
        const item = results[index];
        const fullUrl = typeof item === 'string' ? item : (item.url || item.thumb);
        const proxyUrl = `/api/pinterest/image?url=${encodeURIComponent(fullUrl)}`;
        if (personaPinterestUseBtn) personaPinterestUseBtn.disabled = true;
        if (personaPinterestStatus) personaPinterestStatus.textContent = 'Processing avatar...';

        try {
            const res = await fetch(proxyUrl);
            if (!res.ok) throw new Error('Failed to download image');
            const blob = await res.blob();
            const dataUrl = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error('Failed to read image data'));
                reader.readAsDataURL(blob);
            });
            const resizedDataUrl = await new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = function() {
                    const canvas = document.createElement('canvas');
                    const MAX_SIZE = 256;
                    let width = img.width;
                    let height = img.height;
                    if (width > height) {
                        if (width > MAX_SIZE) {
                            height *= MAX_SIZE / width;
                            width = MAX_SIZE;
                        }
                    } else {
                        if (height > MAX_SIZE) {
                            width *= MAX_SIZE / height;
                            height = MAX_SIZE;
                        }
                    }
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);
                    resolve(canvas.toDataURL('image/webp', 0.85));
                };
                img.onerror = () => reject(new Error('Failed to decode image'));
                img.src = dataUrl;
            });
            if (personaAvatarInput) personaAvatarInput.value = resizedDataUrl;
            updatePersonaAvatarPreview(resizedDataUrl);
            if (personaPinterestStatus) personaPinterestStatus.textContent = 'Avatar applied from Pinterest.';
            if (personaPinterestPanel) personaPinterestPanel.style.display = 'none';
            if (personaPinterestToggle) personaPinterestToggle.classList.remove('active');
        } catch (err) {
            if (personaPinterestStatus) personaPinterestStatus.textContent = 'Failed to process image.';
        } finally {
            if (personaPinterestUseBtn) personaPinterestUseBtn.disabled = false;
        }
    }

    if (personaAvatarPreview && personaAvatarFile) {
        personaAvatarPreview.addEventListener('click', () => personaAvatarFile.click());
        personaAvatarFile.addEventListener('change', handlePersonaAvatarUpload);
    }
    if (personaAvatarRemoveBtn) {
        personaAvatarRemoveBtn.addEventListener('click', removePersonaAvatar);
    }
    if (personaPinterestToggle) {
        personaPinterestToggle.addEventListener('click', togglePersonaPinterest);
    }
    if (personaPinterestSearchBtn) {
        personaPinterestSearchBtn.addEventListener('click', searchPersonaPinterest);
    }
    if (personaPinterestQuery) {
        personaPinterestQuery.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                searchPersonaPinterest();
            }
        });
    }
    if (personaPinterestPrevBtn) {
        personaPinterestPrevBtn.addEventListener('click', () => {
            if (!personaPinterest.results.length) return;
            personaPinterest.index = (personaPinterest.index - 1 + personaPinterest.results.length) % personaPinterest.results.length;
            showPersonaPinterestResult();
        });
    }
    if (personaPinterestNextBtn) {
        personaPinterestNextBtn.addEventListener('click', () => {
            if (!personaPinterest.results.length) return;
            personaPinterest.index = (personaPinterest.index + 1) % personaPinterest.results.length;
            showPersonaPinterestResult();
        });
    }
    if (personaPinterestUseBtn) {
        personaPinterestUseBtn.addEventListener('click', usePersonaPinterestImage);
    }

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
                    <div class="persona-card-body">
                        <div class="persona-card-avatar">${getAvatarInnerHtml(p.avatar, p.name || 'User')}</div>
                        <div class="persona-card-info">
                            <div class="persona-card-name">${escapeHtml(p.name || 'Tanpa Nama')}</div>
                            <div class="persona-card-meta">
                                ${p.gender ? escapeHtml(p.gender) : 'Male'} • ${p.age ? escapeHtml(p.age) + ' th' : '-'}
                            </div>
                        </div>
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

        if (personaBadgeIndex) personaBadgeIndex.textContent = `Persona #${p.index || p.id}`;
        personaNameInput.value = p.name || '';
        if (personaAvatarInput) personaAvatarInput.value = p.avatar || '';
        if (personaAvatarFile) personaAvatarFile.value = '';
        updatePersonaAvatarPreview(p.avatar || '');
        if (personaPinterestPanel) personaPinterestPanel.style.display = 'none';
        if (personaPinterestToggle) personaPinterestToggle.classList.remove('active');

        personaAgeInput.value = (p.age !== undefined && p.age !== null) ? p.age : '';
        personaGenderInput.value = p.gender || 'Male';
        personaHeightInput.value = (p.height !== undefined && p.height !== null) ? p.height : '';
        personaWeightInput.value = (p.weight !== undefined && p.weight !== null) ? p.weight : '';
        personaState.extra = Object.assign({}, p.extra || {});
        renderExtraAttributes();

        // Show delete button and header only if there is more than 1 persona
        const canDelete = personaState.personas.length > 1;
        const editorHeader = document.getElementById('persona-editor-header');
        if (editorHeader) {
            editorHeader.style.display = canDelete ? 'flex' : 'none';
        }
        if (btnDeletePersona) {
            btnDeletePersona.style.display = canDelete ? 'inline-flex' : 'none';
        }

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
            gender: 'Male',
            avatar: ''
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
        const avatar = personaAvatarInput ? personaAvatarInput.value.trim() : '';

        const body = {
            name,
            gender,
            avatar,
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
