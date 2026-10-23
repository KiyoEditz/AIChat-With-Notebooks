/**
 * ═══════════════════════════════════════════════════════════
 *  AI Character Chat — Frontend Application
 * ═══════════════════════════════════════════════════════════
 */

// ── State ────────────────────────────────────────────────
let state = {
    characters: [],
    currentCharacterId: null,
    currentChatId: null,
    currentMessages: [],
    chats: [],
    memory: null,
    personas: [],
    sendingChats: new Set(),
    config: {},
    characterSearchQuery: '',
    pinterest: { results: [], index: 0, loading: false },
    groupPinterest: { results: [], index: 0, loading: false },
    unreadChats: new Set(),
    unreadCharacters: new Set(),
    unreadGroups: new Set(),
    _chatNavToken: 0, // monotonic guard against out-of-order chat navigation (C14/15/16)
    currentTab: 'characters',
    groups: [],
    currentGroupId: null,
    currentGroupChatId: null,
    groupChats: [],
    _groupChatNavToken: 0,
    _deletingGroupChats: new Set(),
    activeSpeakerId: null,
    groupSearchQuery: '',
    groupModalSelectedIds: new Set()
};

function getActiveChatId() {
    return state.currentGroupId ? state.currentGroupChatId : state.currentChatId;
}

function isChatSending(chatId) {
    if (!chatId) return false;
    return state.sendingChats && state.sendingChats.has(chatId);
}

function isCurrentChatSending() {
    return isChatSending(getActiveChatId());
}

function updateChatSendingUI() {
    const isBusy = isCurrentChatSending();
    const typingIndicator = $('#typing-indicator');
    const sendBtn = $('#btn-send');
    const takeTurnBtn = $('#btn-take-turn');
    if (typingIndicator) {
        typingIndicator.style.display = isBusy ? 'flex' : 'none';
    }
    if (sendBtn) {
        sendBtn.disabled = isBusy;
    }
    if (takeTurnBtn) {
        takeTurnBtn.disabled = isBusy;
    }
    $$('.message-action-btn').forEach(btn => {
        btn.disabled = isBusy;
        btn.style.opacity = isBusy ? '0.25' : '';
        btn.style.pointerEvents = isBusy ? 'none' : '';
    });
}

// Backward-compatibility getter/setter for state.sending
Object.defineProperty(state, 'sending', {
    get() {
        return isCurrentChatSending();
    },
    set(val) {
        const id = getActiveChatId();
        if (!id) return;
        if (val) state.sendingChats.add(id);
        else state.sendingChats.delete(id);
        updateChatSendingUI();
    }
});

window.langData = {};
window.t = (key) => window.langData[key] || key;

// ── DOM Refs ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Avatar Helper ────────────────────────────────────────
function getAvatarInnerHtml(avatar, name) {
    if (avatar && (avatar.startsWith('data:') || avatar.startsWith('http://') || avatar.startsWith('https://') || avatar.startsWith('/'))) {
        return `<img src="${escapeHtml(avatar)}" alt="${escapeHtml(name || 'avatar')}">`;
    }
    return escapeHtml(avatar || (name ? name.charAt(0).toUpperCase() : 'A'));
}

function getActivePersona() {
    if (!state.personas || state.personas.length === 0) return null;
    let personaId = state.memory?.personaId;
    if (state.currentGroupId) {
        const group = state.groups.find(g => g.id === state.currentGroupId);
        if (group?.activePersonaId) {
            personaId = group.activePersonaId;
        }
    }
    if (personaId) {
        const found = state.personas.find(p => String(p.id) === String(personaId));
        if (found) return found;
    }
    return state.personas[0] || null;
}

function getUserAvatarInnerHtml() {
    const persona = getActivePersona();
    if (persona && persona.avatar) {
        return getAvatarInnerHtml(persona.avatar, persona.name || 'User');
    }
    return '👤';
}

function updateAvatarPreview(avatarData) {
    const preview = $('#avatar-preview');
    const removeBtn = $('#avatar-remove-btn');
    if (!preview || !removeBtn) return;
    if (avatarData && (avatarData.startsWith('data:') || avatarData.startsWith('http://') || avatarData.startsWith('https://') || avatarData.startsWith('/'))) {
        preview.innerHTML = `<img src="${escapeHtml(avatarData)}" alt="avatar">`;
        preview.classList.add('has-image');
        removeBtn.style.display = 'flex';
    } else {
        preview.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span>Upload</span>`;
        preview.classList.remove('has-image');
        removeBtn.style.display = 'none';
    }
}

// ── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadConfig();
    } catch (err) {
        console.error('Failed to load config', err);
    }
    try {
        await Promise.all([loadCharacters(), loadPersonas(), loadGroups()]);
    } catch (err) {
        console.error('Failed to load characters, personas, or groups', err);
        if (typeof toast === 'function') toast('Failed to load data from server. Check that the server is running.', 'error');
    }
    testConnection();
    bindEvents(); // must always run so the UI stays interactive even if the above failed
});

async function loadPersonas() {
    try {
        const personas = await api('GET', '/api/personas');
        if (Array.isArray(personas)) {
            state.personas = personas;
        }
    } catch (err) {
        console.error('Failed to load personas', err);
    }
}

// ══════════════════════════════════════════════════════════
//  API Helpers
// ══════════════════════════════════════════════════════════

async function api(method, url, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    if (body) opts.body = JSON.stringify(body);

    let res;
    try {
        res = await fetch(url, opts);
    } catch (err) {
        // Network-level failure (server down, DNS, offline, etc.)
        return { error: `Network error: ${err.message}` };
    }

    let data;
    try {
        data = await res.json();
    } catch {
        // Response wasn't valid JSON (e.g. an HTML error page from a 500)
        data = { error: `Server returned an unexpected response (HTTP ${res.status})` };
    }

    if (!res.ok && !data.error) {
        data.error = `Request failed (HTTP ${res.status})`;
    }
    return data;
}

// ══════════════════════════════════════════════════════════
//  CONFIG
// ══════════════════════════════════════════════════════════

async function loadConfig() {
    state.config = await api('GET', '/api/config');

    if (state.config.uiLang) {
        await loadLanguage(state.config.uiLang, state.config.showLangWarning);
    }
}

const RTL_LANGS = ['AR', 'HE', 'FA', 'UR'];

async function loadLanguage(langCode, showWarning) {
    try {
        const res = await fetch(`/Lang/${langCode}.json`);
        if (res.ok) {
            window.langData = await res.json();

            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.getAttribute('data-i18n');
                if (window.langData[key]) el.innerHTML = window.langData[key];
            });

            document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
                const key = el.getAttribute('data-i18n-placeholder');
                if (window.langData[key]) el.placeholder = window.langData[key];
            });

            document.querySelectorAll('[data-i18n-title]').forEach(el => {
                const key = el.getAttribute('data-i18n-title');
                if (window.langData[key]) el.title = window.langData[key];
            });

            document.documentElement.lang = langCode.toLowerCase();
            document.documentElement.dir = RTL_LANGS.includes(langCode.toUpperCase()) ? 'rtl' : 'ltr';

            if (showWarning && typeof openModal === 'function') {
                openModal('modal-lang-warning');
            }
        } else {
            console.error(`Language file not found: /Lang/${langCode}.json (HTTP ${res.status})`);
            if (typeof toast === 'function') {
                toast(`Language "${langCode}" not found — falling back to default text`, 'error');
            }
        }
    } catch (err) {
        console.error('Failed to load language', err);
        if (typeof toast === 'function') toast('Failed to load language file', 'error');
    }
}

async function testConnection() {
    const result = await api('GET', '/api/test-connection');
    const statusDot = $('.status-dot');
    const statusText = $('.status-text');

    if (result.ok) {
        statusDot.className = 'status-dot online';
        statusText.textContent = t('sidebar.connected') || 'Connected to Backend';
    } else {
        statusDot.className = 'status-dot offline';
        statusText.textContent = result.error || t('sidebar.disconnected') || 'Disconnected';
    }
}

// ══════════════════════════════════════════════════════════
//  CHARACTERS
// ══════════════════════════════════════════════════════════

async function loadCharacters() {
    state.characters = await api('GET', '/api/characters');
    renderCharacterList();
}

function renderCharacterList() {
    const list = $('#character-list');

    if (state.characters.length === 0) {
        list.innerHTML = `
            <div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.82rem;">
                ${t('app.no_chars')}
            </div>`;
        return;
    }

    const query = (state.characterSearchQuery || '').trim().toLowerCase();
    const filtered = query
        ? state.characters.filter(c => (c.name || '').toLowerCase().includes(query))
        : state.characters;

    if (filtered.length === 0) {
        list.innerHTML = `
            <div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.82rem;">
                ${t('app.no_search_results')}
            </div>`;
        return;
    }

    list.innerHTML = filtered.map(c => `
        <div class="character-item ${c.id === state.currentCharacterId ? 'active' : ''}" style="position: relative;"
             data-id="${c.id}" onclick="selectCharacter('${c.id}')">
            <div class="character-item-avatar">${getAvatarInnerHtml(c.avatar, c.name)}</div>
            <div class="character-item-info">
                <div class="character-item-name">${escapeHtml(c.name)}</div>
                <div class="character-item-desc">${escapeHtml((c.personality || '').substring(0, 50))}</div>
            </div>
            ${(state.unreadCharacters && state.unreadCharacters.has(c.id)) ? '<span class="chat-item-unread-dot" title="New reply" style="position: absolute; right: 12px; top: 14px;"></span>' : ''}
        </div>
    `).join('');
}

function searchCharacters(query) {
    state.characterSearchQuery = query;
    renderCharacterList();

    const clearBtn = $('#btn-clear-char-search');
    if (clearBtn) clearBtn.style.display = query ? 'flex' : 'none';
}

function clearCharacterSearch() {
    const input = $('#character-search-input');
    if (input) input.value = '';
    searchCharacters('');
}

function clearCharacterSearchAndFocus() {
    clearCharacterSearch();
    const input = $('#character-search-input');
    if (input) input.focus();
}

async function selectCharacter(id) {
    state.currentGroupId = null;
    state.currentGroupChatId = null;
    if (state.unreadCharacters) state.unreadCharacters.delete(id);
    const speakerContainer = $('#group-speaker-container');
    if (speakerContainer) speakerContainer.style.display = 'none';
    const groupChatTitle = $('#group-chat-list-title');
    if (groupChatTitle) groupChatTitle.style.display = 'none';
    const groupChatList = $('#group-chat-list');
    if (groupChatList) groupChatList.style.display = 'none';

    state.currentCharacterId = id;
    state.currentChatId = null;
    state.currentMessages = [];
    renderCharacterList();
    renderGroupList();

    // Reset status to Online for single character
    $('#chat-header-status').textContent = t('chat.status.online') || 'Online';

    // Load chats and memory for this character in parallel
    const [chats, memory] = await Promise.all([
        api('GET', `/api/chats/${id}`),
        api('GET', `/api/memory/${id}`)
    ]);
    // The user may have already clicked a different character while this
    // GET was in flight — don't let a slower, stale response overwrite
    // whatever character they've since switched to.
    if (state.currentCharacterId !== id) return;
    state.chats = Array.isArray(chats) ? chats : [];
    if (memory && !memory.error) {
        state.memory = memory;
    }
    renderChatList();

    const char = state.characters.find(c => c.id === id);
    if (!char) return;

    // Update chat header
    $('#chat-header-name').textContent = char.name;
    $('#chat-header-avatar').innerHTML = getAvatarInnerHtml(char.avatar, char.name);

    // If there are existing chats, load the most recent one
    if (state.chats.length > 0) {
        await loadChat(state.chats[0].id);
    } else {
        // Start a new chat automatically
        await startNewChat();
    }

    if (state.currentCharacterId !== id) return; // switched away during loadChat/startNewChat

    // Switch to chat view
    showView('view-chat');
    closeSidebarOnMobile();
}

// ── Character CRUD ──

function openCharacterEditor(editId = null) {
    const char = editId ? state.characters.find(c => c.id === editId) : null;

    $('#char-modal-title').textContent = char ? t('char_edit.title_edit') : t('char_edit.title_new');
    $('#char-edit-id').value = editId || '';
    $('#char-name').value = char ? char.name : '';
    $('#char-gender').value = char ? (char.gender || '') : '';
    $('#char-age').value = char ? (char.age || '') : '';
    $('#char-avatar').value = char ? (char.avatar || '') : '';
    // Update avatar preview after modal opens
    setTimeout(() => updateAvatarPreview($('#char-avatar').value), 50);
    $('#char-personality').value = char ? (char.personality || '') : '';
    $('#char-scenario').value = char ? (char.scenario || '') : '';
    $('#char-first-message').value = char ? (char.firstMessage || '') : '';
    $('#char-example').value = char ? (char.exampleDialogue || '') : '';
    $('#char-system').value = char ? (char.systemPrompt || '') : '';
    $('#btn-delete-character').style.display = char ? 'flex' : 'none';

    // Reset the Pinterest picker for this editing session
    state.pinterest = { results: [], index: 0, loading: false };
    $('#pinterest-panel').style.display = 'none';
    $('#pinterest-results').style.display = 'none';
    $('#pinterest-status').textContent = '';
    $('#pinterest-query').value = char ? (char.name || '') : '';
    $('#avatar-pinterest-toggle').classList.remove('active');

    openModal('modal-character');
    closeSidebarOnMobile();
}

async function saveCharacter() {
    const editId = $('#char-edit-id').value;
    const data = {
        name: $('#char-name').value.trim() || 'Unnamed',
        gender: $('#char-gender').value,
        age: $('#char-age').value.trim(),
        avatar: $('#char-avatar').value.trim(),
        personality: $('#char-personality').value.trim(),
        scenario: $('#char-scenario').value.trim(),
        firstMessage: $('#char-first-message').value.trim(),
        exampleDialogue: $('#char-example').value.trim(),
        systemPrompt: $('#char-system').value.trim()
    };

    if (editId) {
        await api('PUT', `/api/characters/${editId}`, data);
        toast(t('toast.char_updated'), 'success');
    } else {
        const newChar = await api('POST', '/api/characters', data);
        toast(t('toast.char_created'), 'success');
        state.currentCharacterId = newChar.id;
    }

    closeModal('modal-character');
    clearCharacterSearch();
    await loadCharacters();

    if (state.currentCharacterId) {
        selectCharacter(state.currentCharacterId);
    }
}

async function deleteCharacter() {
    const editId = $('#char-edit-id').value;
    if (!editId) return;
    if (!confirm(t('app.confirm_delete_char'))) return;

    await api('DELETE', `/api/characters/${editId}`);
    toast(t('toast.char_deleted'), 'success');
    closeModal('modal-character');

    state.currentCharacterId = null;
    state.currentChatId = null;
    await loadCharacters();
    showView('view-welcome');
}

// ══════════════════════════════════════════════════════════
//  GROUPS & GROUP CHAT
// ══════════════════════════════════════════════════════════

function switchSidebarTab(tab) {
    state.currentTab = tab;
    const charTabBtn = $('#tab-btn-characters');
    const groupTabBtn = $('#tab-btn-groups');
    const charPanel = $('#sidebar-panel-characters');
    const groupPanel = $('#sidebar-panel-groups');

    if (tab === 'characters') {
        if (charTabBtn) charTabBtn.classList.add('active');
        if (groupTabBtn) groupTabBtn.classList.remove('active');
        if (charPanel) charPanel.style.display = 'flex';
        if (groupPanel) groupPanel.style.display = 'none';
        const groupChatTitle = $('#group-chat-list-title');
        if (groupChatTitle) groupChatTitle.style.display = 'none';
        const groupChatList = $('#group-chat-list');
        if (groupChatList) groupChatList.style.display = 'none';
        renderChatList();
    } else {
        if (groupTabBtn) groupTabBtn.classList.add('active');
        if (charTabBtn) charTabBtn.classList.remove('active');
        if (groupPanel) groupPanel.style.display = 'flex';
        if (charPanel) charPanel.style.display = 'none';
        const charChatTitle = $('#chat-list-title');
        if (charChatTitle) charChatTitle.style.display = 'none';
        const charChatList = $('#chat-list');
        if (charChatList) charChatList.innerHTML = '';
        renderGroupList();
        if (state.currentGroupId) {
            renderGroupChatList();
        }
    }
}
window.switchSidebarTab = switchSidebarTab;

async function loadGroups() {
    try {
        const groups = await api('GET', '/api/groups');
        state.groups = Array.isArray(groups) ? groups : [];
        renderGroupList();
    } catch (err) {
        console.error('Failed to load groups:', err);
    }
}

function renderGroupList() {
    const list = $('#group-list');
    if (!list) return;

    if (!state.groups || state.groups.length === 0) {
        list.innerHTML = `
            <div style="padding: 24px 16px; text-align: center; color: var(--text-muted); font-size: 0.82rem; line-height: 1.5;">
                Belum ada grup chat.<br>Klik tombol <strong>+ Grup Baru</strong> untuk membuat obrolan multi-karakter!
            </div>`;
        return;
    }

    const query = (state.groupSearchQuery || '').trim().toLowerCase();
    const filtered = query
        ? state.groups.filter(g => (g.name || '').toLowerCase().includes(query))
        : state.groups;

    if (filtered.length === 0) {
        list.innerHTML = `
            <div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.82rem;">
                Tidak ada grup yang cocok dengan pencarian.
            </div>`;
        return;
    }

    list.innerHTML = filtered.map(g => {
        const memberCount = (g.characterIds || []).length;
        const avatarHtml = g.avatar
            ? `<img src="${escapeHtml(g.avatar)}" alt="${escapeHtml(g.name)}">`
            : '👥';
        const hasUnread = state.unreadGroups && state.unreadGroups.has(g.id);
        return `
            <div class="group-item ${g.id === state.currentGroupId ? 'active' : ''}" style="position: relative;"
                 data-id="${g.id}" onclick="selectGroup('${g.id}')">
                <div class="group-avatar-stack">${avatarHtml}</div>
                <div class="group-item-info">
                    <div class="group-item-name">${escapeHtml(g.name)}</div>
                    <div class="group-item-meta">${memberCount} karakter</div>
                </div>
                ${hasUnread ? '<span class="chat-item-unread-dot" title="New reply" style="position: absolute; right: 12px; top: 14px;"></span>' : ''}
            </div>
        `;
    }).join('');
}

function searchGroups(query) {
    state.groupSearchQuery = query;
    renderGroupList();
    const clearBtn = $('#btn-clear-group-search');
    if (clearBtn) clearBtn.style.display = query ? 'flex' : 'none';
}
window.searchGroups = searchGroups;

function clearGroupSearch() {
    const input = $('#group-search-input');
    if (input) input.value = '';
    searchGroups('');
}
window.clearGroupSearch = clearGroupSearch;

async function selectGroup(groupId, preserveChatId = null) {
    state.currentGroupId = groupId;
    state.currentGroupChatId = preserveChatId;
    state.currentCharacterId = null;
    state.currentChatId = null;
    if (state.unreadGroups) state.unreadGroups.delete(groupId);
    renderCharacterList();
    renderGroupList();

    // Hide single character chat list
    const singleChatTitle = $('#chat-list-title');
    if (singleChatTitle) singleChatTitle.style.display = 'none';
    const singleChatList = $('#chat-list');
    if (singleChatList) singleChatList.innerHTML = '';

    // Show chat view
    $('#view-welcome').classList.remove('active');
    $('#view-chat').classList.add('active');

    try {
        const group = await api('GET', `/api/groups/${groupId}`);
        if (!group || group.error) {
            toast(t('app.error_load_group') || 'Grup tidak ditemukan', 'error');
            return;
        }

        // Update local group cache
        const idx = state.groups.findIndex(g => g.id === groupId);
        if (idx !== -1) {
            state.groups[idx] = { ...state.groups[idx], ...group };
        } else {
            state.groups.push(group);
        }

        // Default active speaker
        if (!state.activeSpeakerId || !group.characterIds.includes(state.activeSpeakerId)) {
            state.activeSpeakerId = (group.characterIds && group.characterIds.length > 0) ? group.characterIds[0] : null;
        }

        // Update Header
        const avatarHtml = group.avatar
            ? `<img src="${escapeHtml(group.avatar)}" alt="${escapeHtml(group.name)}" style="width:100%;height:100%;object-fit:cover;border-radius:var(--radius-sm)">`
            : '👥';
        $('#chat-header-avatar').innerHTML = avatarHtml;
        $('#chat-header-name').textContent = group.name;

        // Requirement 1: Summarize header status to "<count> Anggota"
        const memberCount = (group.characterIds || []).length;
        $('#chat-header-status').textContent = `${memberCount} ${t('group.members_suffix') || 'Anggota'}`;

        // Render speaker bar
        renderGroupSpeakerBar(group);

        // Load sessions for this group
        await loadGroupChats(groupId);

        closeSidebarOnMobile();
    } catch (err) {
        console.error('Error selecting group:', err);
    }
}
window.selectGroup = selectGroup;

// ── GROUP CHAT SESSIONS ──

async function loadGroupChats(groupId) {
    const listTitle = $('#group-chat-list-title');
    const list = $('#group-chat-list');
    if (!listTitle || !list) return;

    listTitle.style.display = 'block';
    list.style.display = 'block';

    const myToken = ++state._groupChatNavToken;
    try {
        const chats = await api('GET', `/api/groups/${groupId}/chats`);
        if (myToken !== state._groupChatNavToken) return;
        state.groupChats = Array.isArray(chats) ? chats : [];
        renderGroupChatList();

        if (state.groupChats.length > 0) {
            const exists = state.groupChats.some(c => c.id === state.currentGroupChatId);
            const targetChatId = exists ? state.currentGroupChatId : state.groupChats[0].id;
            await loadGroupChat(targetChatId);
        } else {
            await startNewGroupChat();
        }
    } catch (err) {
        console.error('Failed to load group chats:', err);
    }
}

function renderGroupChatList() {
    const listTitle = $('#group-chat-list-title');
    const list = $('#group-chat-list');
    if (!listTitle || !list) return;

    if (!state.currentGroupId || state.groupChats.length === 0) {
        listTitle.style.display = 'none';
        list.style.display = 'none';
        list.innerHTML = '';
        return;
    }

    listTitle.style.display = 'block';
    list.style.display = 'block';

    list.innerHTML = state.groupChats.map(c => `
        <div class="chat-item ${c.id === state.currentGroupChatId ? 'active' : ''}"
             data-id="${c.id}" onclick="loadGroupChat('${c.id}')">
            <span class="chat-item-icon">💬</span>
            <span class="chat-item-title">${escapeHtml(c.title || 'New Session')}</span>
            ${state.unreadChats.has(c.id) ? '<span class="chat-item-unread-dot" title="New reply"></span>' : ''}
            <div class="chat-item-actions">
                <button class="chat-item-edit" onclick="event.stopPropagation(); startRenameGroupChat('${c.id}')" title="Rename">✏️</button>
                <button class="chat-item-delete" onclick="event.stopPropagation(); deleteGroupChat('${c.id}')" title="Delete">✕</button>
            </div>
        </div>
    `).join('');
}

async function loadGroupChat(chatId) {
    if (!state.currentGroupId) return;
    const myToken = ++state._groupChatNavToken;

    const chat = await api('GET', `/api/groups/${state.currentGroupId}/chat/${chatId}`);
    if (!chat || chat.error) return;
    if (myToken !== state._groupChatNavToken) return;

    state.currentGroupChatId = chatId;
    state.unreadChats.delete(chatId);
    renderGroupChatList();
    renderGroupMessages(chat.messages || []);
    closeSidebarOnMobile();
    updateChatSendingUI();
}

async function startNewGroupChat() {
    if (!state.currentGroupId) return;
    try {
        const newChat = await api('POST', `/api/groups/${state.currentGroupId}/chats`, {
            title: t('group.new_session') || 'New Session'
        });
        state.groupChats.unshift(newChat);
        state.currentGroupChatId = newChat.id;
        renderGroupChatList();
        renderGroupMessages([]);
        updateChatSendingUI();
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    }
}
window.startNewGroupChat = startNewGroupChat;

window.startRenameGroupChat = function (chatId) {
    const chatItem = document.querySelector(`#group-chat-list .chat-item[data-id="${chatId}"]`);
    if (!chatItem) return;

    const chat = state.groupChats.find(c => c.id === chatId);
    if (!chat) return;

    const currentTitle = chat.title || 'New Session';

    chatItem.removeAttribute('onclick');
    chatItem.innerHTML = `
        <span class="chat-item-icon">💬</span>
        <div class="chat-item-rename-container">
            <input type="text" class="chat-item-rename-input" value="${escapeHtml(currentTitle)}"
                   data-chat-id="${chatId}" autocomplete="off">
            <div class="chat-item-rename-actions">
                <button class="chat-rename-save" onclick="event.stopPropagation(); saveRenameGroupChat('${chatId}')" title="Simpan">✓</button>
                <button class="chat-rename-cancel" onclick="event.stopPropagation(); cancelRenameGroupChat()" title="Batal">✕</button>
            </div>
        </div>
    `;

    const input = chatItem.querySelector('.chat-item-rename-input');
    input.focus();
    input.select();

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            saveRenameGroupChat(chatId);
        } else if (e.key === 'Escape') {
            cancelRenameGroupChat();
        }
    });
};

window.saveRenameGroupChat = async function (chatId) {
    const input = document.querySelector(`#group-chat-list .chat-item-rename-input[data-chat-id="${chatId}"]`);
    if (!input) return;

    const newTitle = input.value.trim();
    if (!newTitle) return cancelRenameGroupChat();

    const requestGroupId = state.currentGroupId;

    try {
        await api('PUT', `/api/groups/${requestGroupId}/chat/${chatId}`, { title: newTitle });
        if (state.currentGroupId !== requestGroupId) return;

        const chat = state.groupChats.find(c => c.id === chatId);
        if (chat) chat.title = newTitle;

        renderGroupChatList();
        toast('Session renamed!', 'success');
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
        if (state.currentGroupId === requestGroupId) renderGroupChatList();
    }
};

window.cancelRenameGroupChat = function () {
    renderGroupChatList();
};

async function deleteGroupChat(chatId) {
    if (!confirm(t('app.confirm_delete_group_chat') || 'Hapus sesi chat grup ini?')) return;

    state._deletingGroupChats = state._deletingGroupChats || new Set();
    if (state._deletingGroupChats.has(chatId)) return;
    state._deletingGroupChats.add(chatId);

    const myToken = ++state._groupChatNavToken;
    const requestGroupId = state.currentGroupId;

    try {
        await api('DELETE', `/api/groups/${requestGroupId}/chat/${chatId}`);

        if (state.currentGroupChatId === chatId) {
            state.currentGroupChatId = null;
        }

        const freshChats = await api('GET', `/api/groups/${requestGroupId}/chats`);
        if (myToken !== state._groupChatNavToken) return;
        if (state.currentGroupId !== requestGroupId) return;

        state.groupChats = Array.isArray(freshChats) ? freshChats : [];

        if (state.groupChats.length > 0) {
            await loadGroupChat(state.groupChats[0].id);
        } else {
            await startNewGroupChat();
        }

        toast(t('toast.chat_deleted') || 'Chat deleted', 'info');
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    } finally {
        state._deletingGroupChats.delete(chatId);
    }
}
window.deleteGroupChat = deleteGroupChat;

function renderGroupSpeakerBar(group) {
    const container = $('#group-speaker-container');
    const chipsDiv = $('#group-speaker-chips');
    if (!container || !chipsDiv) return;

    if (!group || !state.currentGroupId) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'flex';

    const members = (group.characterIds || []).map(cid => {
        return state.characters.find(c => c.id === cid);
    }).filter(Boolean);

    if (!state.activeSpeakerId && members.length > 0) {
        state.activeSpeakerId = members[0].id;
    }

    chipsDiv.innerHTML = members.map(m => {
        const isActive = m.id === state.activeSpeakerId;
        const avatarHtml = m.avatar
            ? `<img src="${escapeHtml(m.avatar)}" alt="${escapeHtml(m.name)}">`
            : (m.name ? m.name.charAt(0).toUpperCase() : 'C');
        return `
            <button type="button" class="speaker-chip ${isActive ? 'active' : ''}"
                    onclick="setActiveGroupSpeaker('${m.id}')"
                    title="Pilih ${escapeHtml(m.name)} untuk menjawab pesan berikutnya">
                <div class="speaker-chip-avatar">${avatarHtml}</div>
                <span>${escapeHtml(m.name)}</span>
            </button>
        `;
    }).join('');
}

function setActiveGroupSpeaker(charId) {
    state.activeSpeakerId = charId;
    const group = state.groups.find(g => g.id === state.currentGroupId);
    if (group) renderGroupSpeakerBar(group);
}
window.setActiveGroupSpeaker = setActiveGroupSpeaker;

function renderGroupMessages(messages) {
    const container = $('#chat-messages');
    if (!container) return;

    if (!messages || messages.length === 0) {
        container.innerHTML = `
            <div class="empty-chat">
                <div class="empty-chat-icon">👥</div>
                <p>Mulai percakapan di sesi grup ini!</p>
            </div>`;
        return;
    }

    const currentGroup = state.groups.find(g => g.id === state.currentGroupId);

    container.innerHTML = messages.map((m, index) => {
        const time = m.timestamp ? formatTime(m.timestamp) : '';
        const isUser = m.senderType === 'user' || m.senderId === 'user';
        const isLast = index === messages.length - 1;
        const isLastCharacter = !isUser && isLast;

        let avatarHtml = '👤';
        let senderName = m.senderName || (isUser ? 'User' : 'Character');

        if (isUser) {
            avatarHtml = getUserAvatarInnerHtml();
        } else {
            const char = state.characters.find(c => c.id === m.senderId);
            if (char) {
                senderName = char.name;
                avatarHtml = getAvatarInnerHtml(char.avatar, char.name);
            } else if (m.senderAvatar) {
                avatarHtml = getAvatarInnerHtml(m.senderAvatar, senderName);
            }
        }

        let contentHtml = formatMessageContent(m.content);

        // Alternates
        let alternatesHtml = '';
        if (m.alternates && m.alternates.length > 1) {
            const total = m.alternates.length;
            const current = (m.selectedAlternate || 0) + 1;
            alternatesHtml = `
                <div class="alternate-nav">
                    <button onclick="changeGroupAlternate(${index}, -1, '${m.id || ''}')">&lt;</button>
                    <span>${current}/${total}</span>
                    <button onclick="changeGroupAlternate(${index}, 1, '${m.id || ''}')">&gt;</button>
                </div>
            `;
            const selIdx = m.selectedAlternate || 0;
            contentHtml = formatMessageContent(m.alternates[selIdx] || m.content);
        }

        const regenerateBtn = isLastCharacter
            ? `<button class="regenerate-btn" onclick="regenerateGroupLastMessage()" title="${t('chat.action.regenerate') || 'Regenerate'}">🔄</button>`
            : '';
        const touchEvents = isLastCharacter ? 'ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, this)"' : '';
        const msgId = m.id || '';

        return `
            <div class="message ${isUser ? 'user' : 'assistant'}" data-index="${index}" data-id="${msgId}">
                <div class="message-avatar">${avatarHtml}</div>
                <div style="flex:1; min-width:0;">
                    <div class="group-message-sender">
                        <span>${escapeHtml(senderName)}</span>
                    </div>
                    <div class="message-bubble-container" ${touchEvents}>
                        <div class="message-bubble" style="width:100%">${contentHtml}</div>
                        ${regenerateBtn}
                    </div>
                    ${alternatesHtml}
                    <div class="message-meta">
                        ${time ? `<span class="message-time">${time}</span>` : ''}
                        <button class="message-action-btn" onclick="editGroupMessage(${index}, '${msgId}')" title="${t('chat.action.edit') || t('btn.edit') || 'Edit'}">✏️</button>
                        <button class="message-action-btn message-delete-btn" onclick="deleteGroupMessage(${index}, '${msgId}')" title="${t('chat.action.delete') || 'Hapus'}">🗑️</button>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    scrollToBottom();
}

function appendGroupMessage(senderType, content, senderName, avatarData, isLastCharacter = false, senderId = null, messageId = null) {
    const container = $('#chat-messages');
    const empty = container.querySelector('.empty-chat');
    if (empty) empty.remove();

    const isUser = senderType === 'user';
    const avatarHtml = isUser ? getUserAvatarInnerHtml() : getAvatarInnerHtml(avatarData, senderName);
    const now = new Date();
    const time = formatTime(now.toISOString());

    if (isLastCharacter) {
        $$('.regenerate-btn').forEach(btn => btn.remove());
        $$('.message-bubble-container').forEach(c => {
            c.removeAttribute('ontouchstart');
            c.removeAttribute('ontouchend');
            c.classList.remove('swiped-left');
        });
    }

    const index = document.querySelectorAll('#chat-messages .message').length;
    const regenerateBtn = isLastCharacter
        ? `<button class="regenerate-btn" onclick="regenerateGroupLastMessage()" title="${t('chat.action.regenerate') || 'Regenerate'}">🔄</button>`
        : '';
    const touchEvents = isLastCharacter ? 'ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, this)"' : '';
    const msgId = messageId || '';

    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user' : 'assistant'}`;
    div.dataset.index = index;
    div.dataset.id = msgId;
    div.innerHTML = `
        <div class="message-avatar">${avatarHtml}</div>
        <div style="flex:1; min-width:0;">
            <div class="group-message-sender">
                <span>${escapeHtml(senderName || 'Member')}</span>
            </div>
            <div class="message-bubble-container" ${touchEvents}>
                <div class="message-bubble" style="width:100%">${formatMessageContent(content)}</div>
                ${regenerateBtn}
            </div>
            <div class="message-meta">
                <span class="message-time">${time}</span>
                <button class="message-action-btn" onclick="editGroupMessage(${index}, '${msgId}')" title="${t('chat.action.edit') || t('btn.edit') || 'Edit'}">✏️</button>
                <button class="message-action-btn message-delete-btn" onclick="deleteGroupMessage(${index}, '${msgId}')" title="${t('chat.action.delete') || 'Hapus'}">🗑️</button>
            </div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

async function sendGroupMessage(text) {
    const input = $('#chat-input');
    const group = state.groups.find(g => g.id === state.currentGroupId);
    if (!group) return;

    if (!state.currentGroupChatId) {
        await startNewGroupChat();
    }

    const requestGroupId = state.currentGroupId;
    const requestChatId = state.currentGroupChatId;

    if (isChatSending(requestChatId)) {
        toast(t('toast.please_wait') || 'Harap tunggu balasan selesai', 'info');
        return;
    }

    // Auto-detect @Mention
    let targetId = state.activeSpeakerId;
    for (const cid of (group.characterIds || [])) {
        const c = state.characters.find(x => x.id === cid);
        if (c && c.name) {
            const escapedName = c.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`@${escapedName}\\b`, 'i');
            if (regex.test(text)) {
                targetId = cid;
                state.activeSpeakerId = cid;
                renderGroupSpeakerBar(group);
                break;
            }
        }
    }

    if (!targetId && group.characterIds && group.characterIds.length > 0) {
        targetId = group.characterIds[0];
        state.activeSpeakerId = targetId;
        renderGroupSpeakerBar(group);
    }

    state.sendingChats.add(requestChatId);
    input.value = '';
    autoResizeInput();

    // Show user message immediately
    // Show user message immediately with unique ID
    const userPersona = getActivePersona();
    const userName = (userPersona && userPersona.name) ? userPersona.name : 'User';
    const userMsgId = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : ('grp-msg-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9));
    const userBubble = appendGroupMessage('user', text, userName, userPersona?.avatar, false, null, userMsgId);

    updateChatSendingUI();

    try {
        const result = await api('POST', '/api/group-chat/send', {
            groupId: requestGroupId,
            chatId: requestChatId,
            targetCharacterId: targetId,
            userMessage: text,
            userMessageId: userMsgId
        });

        if (result.error) {
            if (state.currentGroupId === requestGroupId && state.currentGroupChatId === requestChatId) {
                if (userBubble) userBubble.remove();
                input.value = text;
                autoResizeInput();
            }
            toast(result.error, 'error');
            return;
        }

        if (state.currentGroupId !== requestGroupId || state.currentGroupChatId !== requestChatId) {
            state.unreadChats.add(requestChatId);
            state.unreadGroups.add(requestGroupId);
            renderGroupList();
            if (state.currentGroupId === requestGroupId) {
                renderGroupChatList();
            }
            return;
        }

        // Fetch fresh group chat session messages and re-render
        const fresh = await api('GET', `/api/groups/${requestGroupId}/chat/${requestChatId}`);
        if (fresh && Array.isArray(fresh.messages)) {
            renderGroupMessages(fresh.messages);
            const sChat = state.groupChats.find(c => c.id === requestChatId);
            if (sChat) {
                if (fresh.title) sChat.title = fresh.title;
                sChat.messageCount = fresh.messages.length;
                renderGroupChatList();
            }
        } else if (result.replyMessage) {
            appendGroupMessage('character', result.replyMessage.content, result.replyMessage.senderName, result.replyMessage.senderAvatar, true, result.replyMessage.senderId, result.replyMessage.id);
        }
    } catch (err) {
        if (state.currentGroupId === requestGroupId && state.currentGroupChatId === requestChatId) {
            if (userBubble) userBubble.remove();
            input.value = text;
            autoResizeInput();
        }
        toast(`Error: ${err.message}`, 'error');
    } finally {
        state.sendingChats.delete(requestChatId);
        if (state.currentGroupId === requestGroupId && state.currentGroupChatId === requestChatId) {
            updateChatSendingUI();
        }
    }
}

async function triggerActiveSpeakerTurn() {
    if (!state.currentGroupId) return;
    const group = state.groups.find(g => g.id === state.currentGroupId);
    if (!group) return;

    if (!state.activeSpeakerId && group.characterIds && group.characterIds.length > 0) {
        state.activeSpeakerId = group.characterIds[0];
    }
    if (!state.activeSpeakerId) {
        toast('Pilih karakter yang akan berbicara', 'info');
        return;
    }

    await triggerGroupTurn(state.activeSpeakerId);
}
window.triggerActiveSpeakerTurn = triggerActiveSpeakerTurn;

async function triggerGroupTurn(targetCharacterId, replyToMessageId = null) {
    if (!state.currentGroupId) return;

    if (!state.currentGroupChatId) {
        await startNewGroupChat();
    }

    const requestGroupId = state.currentGroupId;
    const requestChatId = state.currentGroupChatId;

    if (isChatSending(requestChatId)) {
        toast(t('toast.please_wait') || 'Harap tunggu balasan selesai', 'info');
        return;
    }

    state.sendingChats.add(requestChatId);
    updateChatSendingUI();

    try {
        const result = await api('POST', '/api/group-chat/turn', {
            groupId: requestGroupId,
            chatId: requestChatId,
            targetCharacterId: targetCharacterId,
            replyToMessageId: replyToMessageId
        });

        if (result.error) {
            toast(result.error, 'error');
            return;
        }

        if (state.currentGroupId !== requestGroupId || state.currentGroupChatId !== requestChatId) {
            state.unreadChats.add(requestChatId);
            state.unreadGroups.add(requestGroupId);
            renderGroupList();
            if (state.currentGroupId === requestGroupId) {
                renderGroupChatList();
            }
            return;
        }

        const fresh = await api('GET', `/api/groups/${requestGroupId}/chat/${requestChatId}`);
        if (fresh && Array.isArray(fresh.messages)) {
            renderGroupMessages(fresh.messages);
            const sChat = state.groupChats.find(c => c.id === requestChatId);
            if (sChat) {
                sChat.messageCount = fresh.messages.length;
                renderGroupChatList();
            }
        } else if (result.replyMessage) {
            appendGroupMessage('character', result.replyMessage.content, result.replyMessage.senderName, result.replyMessage.senderAvatar, true, result.replyMessage.senderId);
        }
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    } finally {
        state.sendingChats.delete(requestChatId);
        if (state.currentGroupId === requestGroupId && state.currentGroupChatId === requestChatId) {
            updateChatSendingUI();
        }
    }
}

async function regenerateGroupLastMessage() {
    if (!state.currentGroupId || !state.currentGroupChatId) return;

    const requestGroupId = state.currentGroupId;
    const requestChatId = state.currentGroupChatId;

    if (isChatSending(requestChatId)) return;

    const lastMsgEl = document.querySelector('#chat-messages .message:last-child');
    const messageId = lastMsgEl?.dataset?.id;

    state.sendingChats.add(requestChatId);
    const btn = document.querySelector('.regenerate-btn');
    if (btn) btn.innerHTML = '<span class="spinner"></span>';
    updateChatSendingUI();

    try {
        const result = await api('POST', '/api/group-chat/regenerate', {
            groupId: requestGroupId,
            chatId: requestChatId,
            messageId: messageId || undefined
        });

        if (result.error) {
            toast(result.error, 'error');
            return;
        }

        if (state.currentGroupId !== requestGroupId || state.currentGroupChatId !== requestChatId) {
            state.unreadChats.add(requestChatId);
            state.unreadGroups.add(requestGroupId);
            renderGroupList();
            if (state.currentGroupId === requestGroupId) {
                renderGroupChatList();
            }
            return;
        }

        const fresh = await api('GET', `/api/groups/${requestGroupId}/chat/${requestChatId}`);
        if (fresh && Array.isArray(fresh.messages)) {
            renderGroupMessages(fresh.messages);
        }
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    } finally {
        state.sendingChats.delete(requestChatId);
        if (state.currentGroupId === requestGroupId && state.currentGroupChatId === requestChatId) {
            if (btn) btn.innerHTML = '🔄';
            updateChatSendingUI();
        }
    }
}
window.regenerateGroupLastMessage = regenerateGroupLastMessage;

async function changeGroupAlternate(index, delta, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    if (!state.currentGroupId || !state.currentGroupChatId) return;
    const group = await api('GET', `/api/groups/${state.currentGroupId}/chat/${state.currentGroupChatId}`);
    if (!group || !Array.isArray(group.messages)) return;

    let msg = null;
    let actualIndex = index;
    if (messageId) {
        const fIdx = group.messages.findIndex(m => m.id === messageId);
        if (fIdx !== -1) {
            msg = group.messages[fIdx];
            actualIndex = fIdx;
        }
    }
    if (!msg && group.messages[index]) {
        msg = group.messages[index];
    }
    if (!msg || !msg.alternates || msg.alternates.length <= 1) return;

    let newIndex = (msg.selectedAlternate || 0) + delta;
    if (newIndex < 0) newIndex = msg.alternates.length - 1;
    if (newIndex >= msg.alternates.length) newIndex = 0;

    await api('PUT', `/api/groups/${state.currentGroupId}/alternate`, {
        chatId: state.currentGroupChatId,
        messageIndex: actualIndex,
        messageId: messageId || undefined,
        selectedAlternate: newIndex
    });

    msg.selectedAlternate = newIndex;
    renderGroupMessages(group.messages);
}
window.changeGroupAlternate = changeGroupAlternate;

window.editGroupMessage = async function (index, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    if (!state.currentGroupId || !state.currentGroupChatId) return;

    const requestGroupId = state.currentGroupId;
    const requestChatId = state.currentGroupChatId;

    const groupChat = await api('GET', `/api/groups/${requestGroupId}/chat/${requestChatId}`);
    if (!groupChat || !Array.isArray(groupChat.messages)) return;
    if (state.currentGroupId !== requestGroupId || state.currentGroupChatId !== requestChatId) return;

    let msg = null;
    let actualIndex = index;
    if (messageId) {
        const fIdx = groupChat.messages.findIndex(m => m.id === messageId);
        if (fIdx !== -1) {
            msg = groupChat.messages[fIdx];
            actualIndex = fIdx;
        }
    }
    if (!msg && groupChat.messages[index]) {
        msg = groupChat.messages[index];
    }
    if (!msg) return;

    const messageDiv = (messageId && document.querySelector(`.message[data-id="${messageId}"]`))
        || document.querySelector(`.message[data-index="${index}"]`);
    if (!messageDiv) return;

    const msgEl = messageDiv.querySelector('.message-bubble-container');
    const metaEl = messageDiv.querySelector('.message-meta');
    const alternatesEl = messageDiv.querySelector('.alternate-nav');

    const originalContent = msg.alternates ? (msg.alternates[msg.selectedAlternate || 0] || msg.content) : msg.content;

    if (msgEl) msgEl.style.display = 'none';
    if (metaEl) metaEl.style.display = 'none';
    if (alternatesEl) alternatesEl.style.display = 'none';

    const existing = document.getElementById(`group-edit-container-${actualIndex}`);
    if (existing) existing.remove();

    const editContainer = document.createElement('div');
    editContainer.className = 'message-edit-container';
    editContainer.id = `group-edit-container-${actualIndex}`;
    editContainer.dataset.msgId = messageId || (msg.id || '');
    editContainer.innerHTML = `
        <textarea class="message-edit-textarea" id="group-edit-textarea-${actualIndex}">${escapeHtml(originalContent)}</textarea>
        <div class="message-edit-actions">
            <button class="btn-secondary" style="padding: 6px 12px; font-size: 0.85rem;" onclick="cancelEditGroupMessage(${actualIndex}, '${messageId || ''}')">${t('btn.cancel') || 'Cancel'}</button>
            <button class="btn-primary" style="padding: 6px 12px; font-size: 0.85rem;" onclick="saveEditGroupMessage(${actualIndex}, '${messageId || ''}')">${t('btn.save') || 'Save'}</button>
        </div>
    `;

    msgEl.parentNode.insertBefore(editContainer, msgEl);
    messageDiv.classList.add('is-editing');

    const textarea = document.getElementById(`group-edit-textarea-${actualIndex}`);
    const resizeTextarea = () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.max(120, textarea.scrollHeight) + 'px';
    };
    textarea.addEventListener('input', resizeTextarea);
    setTimeout(() => {
        resizeTextarea();
        textarea.focus();
    }, 0);
};

window.cancelEditGroupMessage = function (index, messageId) {
    const messageDiv = (messageId && document.querySelector(`.message[data-id="${messageId}"]`))
        || document.querySelector(`.message[data-index="${index}"]`);

    const editContainer = document.getElementById(`group-edit-container-${index}`)
        || (messageDiv && messageDiv.querySelector('.message-edit-container'));
    if (editContainer) editContainer.remove();

    if (!messageDiv) return;

    const msgEl = messageDiv.querySelector('.message-bubble-container');
    const metaEl = messageDiv.querySelector('.message-meta');
    const alternatesEl = messageDiv.querySelector('.alternate-nav');

    if (msgEl) msgEl.style.display = '';
    if (metaEl) metaEl.style.display = 'flex';
    if (alternatesEl) alternatesEl.style.display = 'flex';

    messageDiv.classList.remove('is-editing');
};

window.saveEditGroupMessage = async function (index, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    const editContainer = document.getElementById(`group-edit-container-${index}`)
        || (messageId ? document.querySelector(`.message-edit-container[data-msg-id="${messageId}"]`) : null);
    const textarea = editContainer ? editContainer.querySelector('.message-edit-textarea') : document.getElementById(`group-edit-textarea-${index}`);
    if (!textarea) return;

    const newContent = textarea.value.trim();
    if (!newContent) return cancelEditGroupMessage(index, messageId);

    const requestGroupId = state.currentGroupId;
    const requestChatId = state.currentGroupChatId;

    const btn = editContainer ? editContainer.querySelector('.btn-primary') : textarea.nextElementSibling.querySelector('.btn-primary');
    if (btn) {
        btn.disabled = true;
        btn.textContent = t('btn.saving') || 'Saving...';
    }

    try {
        const payload = {
            messageIndex: index,
            content: newContent
        };
        if (messageId) payload.messageId = messageId;

        const result = await api('PUT', `/api/groups/${requestGroupId}/chat/${requestChatId}/message`, payload);

        if (result && result.error) {
            toast(result.error, 'error');
            if (btn) {
                btn.disabled = false;
                btn.textContent = t('btn.save') || 'Save';
            }
            return;
        }

        if (state.currentGroupId === requestGroupId && state.currentGroupChatId === requestChatId) {
            const fresh = await api('GET', `/api/groups/${requestGroupId}/chat/${requestChatId}`);
            if (fresh && Array.isArray(fresh.messages)) {
                renderGroupMessages(fresh.messages);
            }
        }
    } catch (err) {
        toast(`Error saving edit: ${err.message}`, 'error');
        if (btn) {
            btn.disabled = false;
            btn.textContent = t('btn.save') || 'Save';
        }
    }
};

async function deleteGroupMessage(index, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    if (!state.currentGroupChatId) return;
    if (!confirm(t('chat.confirm_delete_msg') || 'Hapus pesan ini?')) return;
    const target = messageId || index;
    try {
        await api('DELETE', `/api/groups/${state.currentGroupId}/chat/${state.currentGroupChatId}/message/${target}`);
        const fresh = await api('GET', `/api/groups/${state.currentGroupId}/chat/${state.currentGroupChatId}`);
        if (fresh && Array.isArray(fresh.messages)) {
            renderGroupMessages(fresh.messages);
            const sChat = state.groupChats.find(c => c.id === state.currentGroupChatId);
            if (sChat) {
                sChat.messageCount = fresh.messages.length;
                renderGroupChatList();
            }
        }
    } catch (err) {
        toast(`Error deleting message: ${err.message}`, 'error');
    }
}
window.deleteGroupMessage = deleteGroupMessage;


// ── Modal Group Editor ──
function openNewGroupModal() {
    $('#group-edit-id').value = '';
    $('#group-modal-title').textContent = t('group_edit.title_new') || '👥 Buat Grup Baru';
    $('#group-name').value = '';
    $('#group-scenario').value = '';
    $('#group-avatar').value = '';
    $('#group-avatar-file').value = '';
    updateGroupAvatarPreview('');
    $('#btn-delete-group').style.display = 'none';

    // Reset Pinterest for group
    state.groupPinterest = { results: [], index: 0, loading: false };
    const pPanel = $('#group-pinterest-panel');
    if (pPanel) pPanel.style.display = 'none';
    const pResults = $('#group-pinterest-results');
    if (pResults) pResults.style.display = 'none';
    const pStatus = $('#group-pinterest-status');
    if (pStatus) pStatus.textContent = '';
    const pQuery = $('#group-pinterest-query');
    if (pQuery) pQuery.value = '';
    const pToggle = $('#group-avatar-pinterest-toggle');
    if (pToggle) pToggle.classList.remove('active');

    state.groupModalSelectedIds = new Set();
    renderGroupModalMemberPicker();
    openModal('modal-group');
}
window.openNewGroupModal = openNewGroupModal;

function openEditGroupModal(groupId) {
    const group = state.groups.find(g => g.id === groupId);
    if (!group) return;

    $('#group-edit-id').value = group.id;
    $('#group-modal-title').textContent = t('group_edit.title_edit') || '✏️ Edit Grup';
    $('#group-name').value = group.name || '';
    $('#group-scenario').value = group.scenario || '';
    $('#group-avatar').value = group.avatar || '';
    $('#group-avatar-file').value = '';
    updateGroupAvatarPreview(group.avatar || '');
    $('#btn-delete-group').style.display = 'block';

    // Reset Pinterest for group
    state.groupPinterest = { results: [], index: 0, loading: false };
    const pPanel = $('#group-pinterest-panel');
    if (pPanel) pPanel.style.display = 'none';
    const pResults = $('#group-pinterest-results');
    if (pResults) pResults.style.display = 'none';
    const pStatus = $('#group-pinterest-status');
    if (pStatus) pStatus.textContent = '';
    const pQuery = $('#group-pinterest-query');
    if (pQuery) pQuery.value = group.name || '';
    const pToggle = $('#group-avatar-pinterest-toggle');
    if (pToggle) pToggle.classList.remove('active');

    state.groupModalSelectedIds = new Set(group.characterIds || []);
    renderGroupModalMemberPicker();
    openModal('modal-group');
}
window.openEditGroupModal = openEditGroupModal;

function renderGroupModalMemberPicker(query = '') {
    const picker = $('#group-character-picker');
    const countEl = $('#selected-member-count');
    if (!picker) return;

    const q = (query || '').trim().toLowerCase();
    const filtered = q
        ? state.characters.filter(c => (c.name || '').toLowerCase().includes(q))
        : state.characters;

    if (countEl) {
        countEl.textContent = `${state.groupModalSelectedIds.size} terpilih`;
    }

    if (filtered.length === 0) {
        picker.innerHTML = `<div style="grid-column:1/-1;padding:12px;text-align:center;color:var(--text-muted);font-size:0.8rem;">Tidak ada karakter yang cocok.</div>`;
        return;
    }

    picker.innerHTML = filtered.map(c => {
        const isSelected = state.groupModalSelectedIds.has(c.id);
        const avatarHtml = c.avatar
            ? `<img src="${escapeHtml(c.avatar)}" alt="${escapeHtml(c.name)}">`
            : (c.name ? c.name.charAt(0).toUpperCase() : 'C');
        return `
            <div class="picker-char-card ${isSelected ? 'selected' : ''}" onclick="toggleGroupMemberSelection('${c.id}')">
                <input type="checkbox" class="picker-char-check" ${isSelected ? 'checked' : ''} onclick="event.stopPropagation(); toggleGroupMemberSelection('${c.id}')">
                <div class="picker-char-avatar">${avatarHtml}</div>
                <div class="picker-char-name">${escapeHtml(c.name)}</div>
            </div>
        `;
    }).join('');
}

function filterGroupModalMembers(query) {
    renderGroupModalMemberPicker(query);
}
window.filterGroupModalMembers = filterGroupModalMembers;

function toggleGroupMemberSelection(charId) {
    if (state.groupModalSelectedIds.has(charId)) {
        state.groupModalSelectedIds.delete(charId);
    } else {
        state.groupModalSelectedIds.add(charId);
    }
    const searchInput = $('#group-modal-member-search');
    renderGroupModalMemberPicker(searchInput ? searchInput.value : '');
}
window.toggleGroupMemberSelection = toggleGroupMemberSelection;

async function saveGroup() {
    let editId = $('#group-edit-id').value;
    const name = ($('#group-name').value || '').trim();
    const scenario = ($('#group-scenario').value || '').trim();
    const avatar = $('#group-avatar').value;

    if (!name) {
        toast('Nama grup harus diisi', 'error');
        $('#group-name').focus();
        return;
    }

    const characterIds = Array.from(state.groupModalSelectedIds);
    if (characterIds.length === 0) {
        toast('Pilih minimal 1 karakter untuk anggota grup', 'error');
        return;
    }

    // Preserve existing activePersonaId if already set on group
    const existingGroup = state.groups.find(g => g.id === editId);
    const activePersonaId = existingGroup?.activePersonaId || null;

    const payload = {
        name,
        scenario,
        avatar,
        activePersonaId,
        characterIds
    };

    try {
        const wasCurrentGroup = (state.currentGroupId === editId);
        const preservedChatId = wasCurrentGroup ? state.currentGroupChatId : null;

        if (editId) {
            await api('PUT', `/api/groups/${editId}`, payload);
            toast(t('toast.group_updated') || 'Grup berhasil diperbarui!', 'success');
        } else {
            const created = await api('POST', '/api/groups', payload);
            toast(t('toast.group_created') || 'Grup berhasil dibuat!', 'success');
            editId = created.id;
        }

        await loadGroups();
        closeModal('modal-group');

        const targetId = editId || (state.groups[0] ? state.groups[0].id : null);
        if (targetId) {
            await selectGroup(targetId, preservedChatId);
        }
    } catch (err) {
        toast(`Error saving group: ${err.message}`, 'error');
    }
}
window.saveGroup = saveGroup;

async function confirmDeleteGroup() {
    const editId = $('#group-edit-id').value;
    if (!editId) return;

    if (!confirm('Hapus grup ini beserta seluruh riwayat obrolannya?')) return;

    try {
        await api('DELETE', `/api/groups/${editId}`);
        toast(t('toast.group_deleted') || 'Grup berhasil dihapus', 'info');
        closeModal('modal-group');
        state.currentGroupId = null;
        state.currentGroupChatId = null;
        state.groupChats = [];
        const groupChatTitle = $('#group-chat-list-title');
        if (groupChatTitle) groupChatTitle.style.display = 'none';
        const groupChatList = $('#group-chat-list');
        if (groupChatList) groupChatList.style.display = 'none';
        await loadGroups();
        showView('view-welcome');
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    }
}
window.confirmDeleteGroup = confirmDeleteGroup;

function updateGroupAvatarPreview(avatarData) {
    const preview = $('#group-avatar-preview');
    const removeBtn = $('#group-avatar-remove-btn');
    if (!preview || !removeBtn) return;
    if (avatarData) {
        preview.innerHTML = `<img src="${escapeHtml(avatarData)}" alt="avatar">`;
        preview.classList.add('has-image');
        removeBtn.style.display = 'flex';
    } else {
        preview.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span>Upload</span>`;
        preview.classList.remove('has-image');
        removeBtn.style.display = 'none';
    }
}

function removeGroupAvatar() {
    $('#group-avatar').value = '';
    $('#group-avatar-file').value = '';
    updateGroupAvatarPreview('');
}
window.removeGroupAvatar = removeGroupAvatar;

// ══════════════════════════════════════════════════════════
//  CHATS
// ══════════════════════════════════════════════════════════

function renderChatList() {
    const listTitle = $('#chat-list-title');
    const list = $('#chat-list');

    if (!state.currentCharacterId || state.chats.length === 0) {
        listTitle.style.display = 'none';
        list.innerHTML = '';
        return;
    }

    listTitle.style.display = 'block';
    list.innerHTML = state.chats.map(c => `
        <div class="chat-item ${c.id === state.currentChatId ? 'active' : ''}"
             data-id="${c.id}" onclick="loadChat('${c.id}')">
            <span class="chat-item-icon">💬</span>
            <span class="chat-item-title">${escapeHtml(c.title)}</span>
            ${state.unreadChats.has(c.id) ? '<span class="chat-item-unread-dot" title="New reply"></span>' : ''}
            <div class="chat-item-actions">
                <button class="chat-item-edit" onclick="event.stopPropagation(); startRenameChat('${c.id}')" title="Rename">✏️</button>
                <button class="chat-item-delete" onclick="event.stopPropagation(); deleteChat('${c.id}')" title="Delete">✕</button>
            </div>
        </div>
    `).join('');
}

async function startNewChat() {
    if (!state.currentCharacterId) return;

    const myToken = ++state._chatNavToken;
    const characterIdAtStart = state.currentCharacterId;

    const char = state.characters.find(c => c.id === state.currentCharacterId);
    const initialMessages = [];

    // Add first message if character has one
    if (char && char.firstMessage) {
        initialMessages.push({
            role: 'assistant',
            content: char.firstMessage,
            timestamp: new Date().toISOString()
        });
    }

    const chat = await api('POST', '/api/chats', {
        characterId: characterIdAtStart,
        title: 'New Chat',
        messages: initialMessages
    });

    // Reload chat list for whichever character this was created for
    const chats = await api('GET', `/api/chats/${characterIdAtStart}`);

    if (myToken !== state._chatNavToken) return; // superseded by a newer navigation

    state.currentChatId = chat.id;
    state.chats = chats;
    renderChatList();
    renderMessages(chat.messages || []);
    showView('view-chat');
    closeSidebarOnMobile();
    updateChatSendingUI();
}

async function loadChat(chatId) {
    state.currentGroupId = null;
    state.currentGroupChatId = null;
    const speakerContainer = $('#group-speaker-container');
    if (speakerContainer) speakerContainer.style.display = 'none';
    const groupChatTitle = $('#group-chat-list-title');
    if (groupChatTitle) groupChatTitle.style.display = 'none';
    const groupChatList = $('#group-chat-list');
    if (groupChatList) groupChatList.style.display = 'none';
    renderGroupList();

    // Reset status to Online for single character
    $('#chat-header-status').textContent = t('chat.status.online') || 'Online';

    // A monotonic token shared with startNewChat/deleteChat: whichever
    // call is the most recent one wins, regardless of which of several
    // in-flight requests happens to resolve last (classic out-of-order
    // resolution when a user clicks between chats quickly).
    const myToken = ++state._chatNavToken;

    const chat = await api('GET', `/api/chat/${chatId}`);
    if (!chat || chat.error) return;
    if (myToken !== state._chatNavToken) return; // superseded by a newer navigation

    state.currentChatId = chatId;
    state.unreadChats.delete(chatId);
    renderChatList();
    renderMessages(chat.messages || []);
    closeSidebarOnMobile();
    updateChatSendingUI();
}

async function deleteChat(chatId) {
    if (!confirm(t('app.confirm_delete_chat'))) return;

    // Guard against double-clicking the delete button firing two
    // overlapping DELETE requests for the same chat.
    state._deletingChats = state._deletingChats || new Set();
    if (state._deletingChats.has(chatId)) return;
    state._deletingChats.add(chatId);

    const myToken = ++state._chatNavToken;

    try {
        await api('DELETE', `/api/chat/${chatId}`);

        if (state.currentChatId === chatId) {
            state.currentChatId = null;
            state.currentMessages = [];
        }

        state.chats = await api('GET', `/api/chats/${state.currentCharacterId}`);
        if (myToken !== state._chatNavToken) return; // a newer navigation took over meanwhile
        renderChatList();

        if (state.chats.length > 0) {
            await loadChat(state.chats[0].id);
        } else {
            await startNewChat();
        }
    } finally {
        state._deletingChats.delete(chatId);
    }
}

// ══════════════════════════════════════════════════════════
//  MESSAGES
// ══════════════════════════════════════════════════════════

function renderMessages(messages) {
    state.currentMessages = Array.isArray(messages) ? messages : [];
    const currentChat = state.chats.find(c => c.id === state.currentChatId);
    if (currentChat) {
        currentChat.messages = state.currentMessages;
    }
    const container = $('#chat-messages');
    const char = state.characters.find(c => c.id === state.currentCharacterId);
    const charAvatar = char ? getAvatarInnerHtml(char.avatar, char.name) : 'A';

    if (!messages || messages.length === 0) {
        container.innerHTML = `
            <div class="empty-chat">
                <div class="empty-chat-icon">💬</div>
                <p>${t('chat.empty')}</p>
            </div>`;
        return;
    }

    container.innerHTML = messages.map((m, index) => {
        const time = m.timestamp ? formatTime(m.timestamp) : '';
        const userAvatar = getUserAvatarInnerHtml();
        const avatar = m.role === 'user' ? userAvatar : charAvatar;
        const isLastAssistant = m.role === 'assistant' && index === messages.length - 1;

        let contentHtml = formatMessageContent(m.content);
        let alternatesHtml = '';

        if (m.alternates && m.alternates.length > 1) {
            const total = m.alternates.length;
            const current = (m.selectedAlternate || 0) + 1;
            alternatesHtml = `
                <div class="alternate-nav">
                    <button onclick="changeAlternate(${index}, -1, '${m.id || ''}')">&lt;</button>
                    <span>${current}/${total}</span>
                    <button onclick="changeAlternate(${index}, 1, '${m.id || ''}')">&gt;</button>
                </div>
            `;
            const selIdx = m.selectedAlternate || 0;
            contentHtml = formatMessageContent(m.alternates[selIdx] || m.content);
        }

        const regenerateBtn = isLastAssistant ? `<button class="regenerate-btn" onclick="regenerateLastMessage()" title="${t('chat.action.regenerate') || 'Regenerate'}">🔄</button>` : '';
        const touchEvents = isLastAssistant ? 'ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, this)"' : '';
        const msgId = m.id || '';

        return `
            <div class="message ${m.role}" data-index="${index}" data-id="${msgId}">
                <div class="message-avatar">${avatar}</div>
                <div style="flex:1; min-width:0;">
                    <div class="message-bubble-container" ${touchEvents}>
                        <div class="message-bubble" style="width:100%">${contentHtml}</div>
                        ${regenerateBtn}
                    </div>
                    ${alternatesHtml}
                    <div class="message-meta">
                        ${time ? `<span class="message-time">${time}</span>` : ''}
                        <button class="message-action-btn" onclick="editMessage(${index}, '${msgId}')" title="${t('chat.action.edit') || t('btn.edit') || 'Edit'}">✏️</button>
                        <button class="message-action-btn message-delete-btn" onclick="deleteMessage(${index}, '${msgId}')" title="${t('chat.action.delete') || 'Hapus'}">🗑️</button>
                    </div>
                </div>
            </div>`;
    }).join('');

    scrollToBottom();
}

function appendMessage(role, content, isLastAssistant = false, messageId = null) {
    if (!Array.isArray(state.currentMessages)) {
        state.currentMessages = [];
    }
    state.currentMessages.push({
        role,
        content,
        timestamp: new Date().toISOString(),
        id: messageId || ''
    });
    const currentChat = state.chats.find(c => c.id === state.currentChatId);
    if (currentChat) {
        currentChat.messages = state.currentMessages;
    }

    const container = $('#chat-messages');
    const char = state.characters.find(c => c.id === state.currentCharacterId);
    const charAvatar = char ? getAvatarInnerHtml(char.avatar, char.name) : 'A';

    const empty = container.querySelector('.empty-chat');
    if (empty) empty.remove();

    if (isLastAssistant) {
        $$('.regenerate-btn').forEach(btn => btn.remove());
        $$('.message-bubble-container').forEach(c => {
            c.removeAttribute('ontouchstart');
            c.removeAttribute('ontouchend');
            c.classList.remove('swiped-left');
        });
    }

    const userAvatar = getUserAvatarInnerHtml();
    const avatar = role === 'user' ? userAvatar : charAvatar;
    const time = formatTime(new Date().toISOString());
    const index = document.querySelectorAll('#chat-messages .message').length;
    const regenerateBtn = isLastAssistant ? `<button class="regenerate-btn" onclick="regenerateLastMessage()" title="${t('chat.action.regenerate') || 'Regenerate'}">🔄</button>` : '';
    const touchEvents = isLastAssistant ? 'ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, this)"' : '';
    const msgId = messageId || '';

    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.dataset.index = index;
    div.dataset.id = msgId;
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div style="flex:1; min-width:0;">
            <div class="message-bubble-container" ${touchEvents}>
                <div class="message-bubble" style="width:100%">${formatMessageContent(content)}</div>
                ${regenerateBtn}
            </div>
            <div class="message-meta">
                <span class="message-time">${time}</span>
                <button class="message-action-btn" onclick="editMessage(${index}, '${msgId}')" title="${t('chat.action.edit') || t('btn.edit') || 'Edit'}">✏️</button>
                <button class="message-action-btn message-delete-btn" onclick="deleteMessage(${index}, '${msgId}')" title="${t('chat.action.delete') || 'Hapus'}">🗑️</button>
            </div>
        </div>`;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    const container = $('#chat-messages');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

// ══════════════════════════════════════════════════════════
//  SEND MESSAGE
// ══════════════════════════════════════════════════════════

async function sendMessage() {
    const input = $('#chat-input');
    const text = input.value.trim();
    if (!text) return;

    if (state.currentGroupId) {
        return sendGroupMessage(text);
    }

    if (!state.currentCharacterId || !state.currentChatId) return;

    const requestChatId = state.currentChatId; // capture before the await
    const requestCharacterId = state.currentCharacterId;

    if (isChatSending(requestChatId)) {
        toast(t('toast.please_wait') || 'Please wait for the current reply to finish', 'info');
        return;
    }

    state.sendingChats.add(requestChatId);
    input.value = '';
    autoResizeInput();

    // Show user message immediately with a unique ID
    const userMsgId = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : ('msg-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9));
    const userBubble = appendMessage('user', text, false, userMsgId);

    // Update typing indicator & send button for this chat
    updateChatSendingUI();

    try {
        const result = await api('POST', '/api/chat/send', {
            characterId: requestCharacterId,
            chatId: requestChatId,
            userMessage: text,
            userMessageId: userMsgId
        });

        if (result.error) {
            // The message was never saved server-side — remove the phantom
            // bubble and give the text back so the user can retry.
            if (state.currentChatId === requestChatId && !state.currentGroupId) {
                if (userBubble) userBubble.remove();
                if (Array.isArray(state.currentMessages) && state.currentMessages.length > 0) {
                    state.currentMessages.pop();
                }
                input.value = text;
                autoResizeInput();
            }
            toast(result.error, 'error');
            return;
        }

        if (state.currentChatId !== requestChatId || state.currentGroupId) {
            // User switched to a different chat while this was in flight —
            // the reply was saved server-side, just don't render it here (H6).
            state.unreadChats.add(requestChatId);
            state.unreadCharacters.add(requestCharacterId);
            renderCharacterList();
            if (state.currentCharacterId === requestCharacterId && !state.currentGroupId) {
                renderChatList();
            }
            return;
        }

        // Show assistant response
        appendMessage('assistant', result.content, true, result.messageId);

        // Refresh chat list (title might have changed)
        state.chats = await api('GET', `/api/chats/${requestCharacterId}`);
        const currentChatInList = state.chats.find(c => c.id === state.currentChatId);
        if (currentChatInList && state.currentMessages) {
            currentChatInList.messages = state.currentMessages;
        }
        if (state.currentCharacterId === requestCharacterId && !state.currentGroupId) {
            renderChatList();
        }

    } catch (err) {
        if (state.currentChatId === requestChatId && !state.currentGroupId) {
            if (userBubble) userBubble.remove();
            if (Array.isArray(state.currentMessages) && state.currentMessages.length > 0) {
                state.currentMessages.pop();
            }
            input.value = text;
            autoResizeInput();
        }
        toast(`Error: ${err.message}`, 'error');
    } finally {
        state.sendingChats.delete(requestChatId);
        if (state.currentChatId === requestChatId && !state.currentGroupId) {
            updateChatSendingUI();
        }
    }
}

// ══════════════════════════════════════════════════════════
//  REGENERATE & ALTERNATES
// ══════════════════════════════════════════════════════════

let touchStartX = 0;
window.handleTouchStart = function (e) {
    touchStartX = e.changedTouches[0].screenX;
};
window.handleTouchEnd = function (e, element) {
    let touchEndX = e.changedTouches[0].screenX;
    if (touchStartX - touchEndX > 40) {
        element.classList.add('swiped-left');
    } else if (touchEndX - touchStartX > 40) {
        element.classList.remove('swiped-left');
    }
};

window.regenerateLastMessage = async function () {
    if (!state.currentCharacterId || !state.currentChatId) return;
    const requestChatId = state.currentChatId; // capture before the await
    const requestCharacterId = state.currentCharacterId;

    if (isChatSending(requestChatId)) return;

    const lastMsgEl = document.querySelector('#chat-messages .message:last-child');
    const messageId = lastMsgEl?.dataset?.id;

    state.sendingChats.add(requestChatId);
    const btn = document.querySelector('.regenerate-btn');
    if (btn) btn.innerHTML = '<span class="spinner"></span>';
    updateChatSendingUI();

    try {
        const result = await api('POST', '/api/chat/regenerate', {
            characterId: requestCharacterId,
            chatId: requestChatId,
            messageId: messageId || undefined
        });

        if (result.error) {
            toast(result.error, 'error');
            return;
        }

        if (state.currentChatId !== requestChatId || state.currentGroupId) {
            state.unreadChats.add(requestChatId);
            state.unreadCharacters.add(requestCharacterId);
            renderCharacterList();
            if (state.currentCharacterId === requestCharacterId && !state.currentGroupId) {
                renderChatList();
            }
            return; // user switched chats meanwhile
        }

        const chat = await api('GET', `/api/chat/${requestChatId}`);
        if (chat && !chat.error && state.currentChatId === requestChatId && !state.currentGroupId) {
            renderMessages(chat.messages || []);
        }
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    } finally {
        state.sendingChats.delete(requestChatId);
        if (state.currentChatId === requestChatId && !state.currentGroupId) {
            if (btn) btn.innerHTML = '🔄';
            updateChatSendingUI();
        }
    }
};

window.changeAlternate = async function (msgIndex, delta, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    state._alternateNavBusy = state._alternateNavBusy || new Set();
    if (state._alternateNavBusy.has(msgIndex)) return; // ignore rapid double-clicks mid-request
    state._alternateNavBusy.add(msgIndex);

    const requestChatId = state.currentChatId; // capture before the await

    try {
        const chat = await api('GET', `/api/chat/${requestChatId}`);
        if (!chat || chat.error) return;
        // The user may have switched to a different chat while this GET was
        // in flight — don't render a stale chat's messages into (or write
        // this alternate-selection onto) whatever chat is showing now.
        if (state.currentChatId !== requestChatId) return;

        let msg = null;
        let actualIndex = msgIndex;
        if (messageId && Array.isArray(chat.messages)) {
            const fIdx = chat.messages.findIndex(m => m.id === messageId);
            if (fIdx !== -1) {
                msg = chat.messages[fIdx];
                actualIndex = fIdx;
            }
        }
        if (!msg && chat.messages && chat.messages[msgIndex]) {
            msg = chat.messages[msgIndex];
        }
        if (!msg || !msg.alternates) return;

        let sel = msg.selectedAlternate || 0;
        sel += delta;
        if (sel < 0) sel = 0;
        if (sel >= msg.alternates.length) sel = msg.alternates.length - 1;

        msg.selectedAlternate = sel;

        // Optimistic re-render
        renderMessages(chat.messages);

        await api('PUT', `/api/chat/${requestChatId}/alternate`, {
            messageIndex: actualIndex,
            messageId: messageId || undefined,
            selectedAlternate: sel
        });
    } finally {
        state._alternateNavBusy.delete(msgIndex);
    }
};

window.editMessage = async function (index, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    const requestChatId = state.currentChatId; // capture before the await
    const chat = await api('GET', `/api/chat/${requestChatId}`);
    if (!chat || chat.error) return;
    if (state.currentChatId !== requestChatId) return;

    let msg = null;
    let actualIndex = index;
    if (messageId && Array.isArray(chat.messages)) {
        const fIdx = chat.messages.findIndex(m => m.id === messageId);
        if (fIdx !== -1) {
            msg = chat.messages[fIdx];
            actualIndex = fIdx;
        }
    }
    if (!msg && chat.messages && chat.messages[index]) {
        msg = chat.messages[index];
    }
    if (!msg) return;

    const messageDiv = (messageId && document.querySelector(`.message[data-id="${messageId}"]`))
        || document.querySelector(`.message[data-index="${index}"]`);
    if (!messageDiv) return;

    const msgEl = messageDiv.querySelector('.message-bubble-container');
    const metaEl = messageDiv.querySelector('.message-meta');
    const alternatesEl = messageDiv.querySelector('.alternate-nav');

    const originalContent = msg.alternates ? (msg.alternates[msg.selectedAlternate || 0] || msg.content) : msg.content;

    if (msgEl) msgEl.style.display = 'none';
    if (metaEl) metaEl.style.display = 'none';
    if (alternatesEl) alternatesEl.style.display = 'none';

    const existingEdit = messageDiv.querySelector('.message-edit-container') || document.getElementById(`edit-container-${actualIndex}`);
    if (existingEdit) existingEdit.remove();

    const editContainer = document.createElement('div');
    editContainer.className = 'message-edit-container';
    editContainer.id = `edit-container-${actualIndex}`;
    editContainer.dataset.msgId = messageId || (msg.id || '');
    editContainer.innerHTML = `
        <textarea class="message-edit-textarea" id="edit-textarea-${actualIndex}">${escapeHtml(originalContent)}</textarea>
        <div class="message-edit-actions">
            <button class="btn-secondary" style="padding: 6px 12px; font-size: 0.85rem;" onclick="cancelEdit(${actualIndex}, '${messageId || ''}')">${t('btn.cancel')}</button>
            <button class="btn-primary" style="padding: 6px 12px; font-size: 0.85rem;" onclick="saveEdit(${actualIndex}, '${messageId || ''}')">${t('btn.save')}</button>
        </div>
    `;

    msgEl.parentNode.insertBefore(editContainer, msgEl);

    messageDiv.classList.add('is-editing');

    const textarea = document.getElementById(`edit-textarea-${actualIndex}`);
    const resizeTextarea = () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.max(120, textarea.scrollHeight) + 'px';
    };
    textarea.addEventListener('input', resizeTextarea);
    setTimeout(resizeTextarea, 0);
};

window.cancelEdit = function (index, messageId) {
    const messageDiv = (messageId && document.querySelector(`.message[data-id="${messageId}"]`))
        || document.querySelector(`.message[data-index="${index}"]`);

    const editContainer = document.getElementById(`edit-container-${index}`)
        || (messageDiv && messageDiv.querySelector('.message-edit-container'));
    if (editContainer) editContainer.remove();

    if (!messageDiv) return;

    const msgEl = messageDiv.querySelector('.message-bubble-container');
    const metaEl = messageDiv.querySelector('.message-meta');
    const alternatesEl = messageDiv.querySelector('.alternate-nav');

    if (msgEl) msgEl.style.display = '';
    if (metaEl) metaEl.style.display = 'flex';
    if (alternatesEl) alternatesEl.style.display = 'flex';

    messageDiv.classList.remove('is-editing');
};

window.saveEdit = async function (index, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    const editContainer = document.getElementById(`edit-container-${index}`)
        || (messageId ? document.querySelector(`.message-edit-container[data-msg-id="${messageId}"]`) : null);
    const textarea = editContainer ? editContainer.querySelector('.message-edit-textarea') : document.getElementById(`edit-textarea-${index}`);
    if (!textarea) return;

    const newContent = textarea.value.trim();
    if (!newContent) return cancelEdit(index, messageId);

    const requestChatId = state.currentChatId; // capture before any await
    const btn = editContainer ? editContainer.querySelector('.btn-primary') : textarea.nextElementSibling.querySelector('.btn-primary');
    if (btn) {
        btn.disabled = true;
        btn.textContent = t('btn.saving');
    }

    try {
        const payload = {
            messageIndex: index,
            content: newContent
        };
        if (messageId) payload.messageId = messageId;

        const result = await api('PUT', `/api/chat/${requestChatId}/message`, payload);

        if (result.error) {
            toast(result.error, 'error');
            if (btn) {
                btn.disabled = false;
                btn.textContent = t('btn.save');
            }
            return;
        }

        // Only refresh the rendered messages if the user is still on the chat that was just edited.
        if (state.currentChatId === requestChatId) {
            await loadChat(requestChatId);
        }
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
        if (btn) {
            btn.disabled = false;
            btn.textContent = t('btn.save');
        }
    }
};

window.deleteMessage = async function (index, messageId) {
    if (isCurrentChatSending()) return toast(t('toast.please_wait') || 'Tunggu balasan selesai dulu ya', 'info');
    const confirmText = t('chat.confirm_delete_msg') || 'Hapus pesan ini dari percakapan?';
    if (!confirm(confirmText)) return;

    const requestChatId = state.currentChatId;
    const target = messageId || index;
    try {
        const result = await api('DELETE', `/api/chat/${requestChatId}/message/${target}`);
        if (result.error) {
            toast(result.error, 'error');
            return;
        }

        if (state.currentChatId === requestChatId) {
            await loadChat(requestChatId);
            toast(t('chat.msg_deleted') || 'Pesan berhasil dihapus');
        }
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
    }
};

// ══════════════════════════════════════════════════════════
//  MEMORY
// ══════════════════════════════════════════════════════════

async function openMemoryViewer() {
    if (!state.currentCharacterId && !state.currentGroupId) return;

    if (state.currentGroupId) {
        const requestGroupId = state.currentGroupId;
        const group = state.groups.find(g => g.id === requestGroupId);
        $('#memory-char-name').textContent = group ? group.name : 'Group';

        const [memory, personas] = await Promise.all([
            api('GET', `/api/groups/${requestGroupId}/memory`),
            api('GET', '/api/personas')
        ]);
        if (state.currentGroupId !== requestGroupId) return;

        if (Array.isArray(personas)) {
            state.personas = personas;
        }
        state.memory = memory || { facts: [], summaries: [], importantEvents: [], personaId: null };
        if (group && group.activePersonaId) {
            state.memory.personaId = group.activePersonaId;
        } else if (!state.memory.personaId && state.personas.length > 0) {
            state.memory.personaId = state.personas[0].id;
        }
        state.memoryModalDraftPersonaId = state.memory.personaId;
        renderMemory();
        openModal('modal-memory');
        closeSidebarOnMobile();
        return;
    }

    const requestCharacterId = state.currentCharacterId; // capture before the await
    const char = state.characters.find(c => c.id === state.currentCharacterId);
    $('#memory-char-name').textContent = char ? char.name : 'Character';

    const [memory, personas] = await Promise.all([
        api('GET', `/api/memory/${requestCharacterId}`),
        api('GET', '/api/personas')
    ]);
    // Bail out if the user switched characters while this GET was in
    // flight — don't open the memory modal with the WRONG character's data.
    if (state.currentCharacterId !== requestCharacterId) return;

    if (Array.isArray(personas)) {
        state.personas = personas;
    }
    state.memory = memory;
    if (!state.memory.personaId && state.personas.length > 0) {
        state.memory.personaId = state.personas[0].id;
    }
    state.memoryModalDraftPersonaId = state.memory.personaId;
    renderMemory();
    openModal('modal-memory');
    closeSidebarOnMobile();
}

function renderMemory() {
    const mem = state.memory;
    if (!mem) return;

    // Facts
    const factsEl = $('#memory-facts');
    factsEl.innerHTML = (mem.facts || []).map((f, i) => `
        <div class="memory-item">
            <span class="memory-item-text">${escapeHtml(f)}</span>
            <button class="memory-item-delete" onclick="removeMemoryItem('facts', ${i})">✕</button>
        </div>
    `).join('') || `<div style="color:var(--text-muted); font-size:0.8rem; padding:8px">${t('memory.empty_facts')}</div>`;

    // Persona selector
    renderPersonaSelector();

    // Events
    const eventsEl = $('#memory-events');
    eventsEl.innerHTML = (mem.importantEvents || []).map((e, i) => `
        <div class="memory-item">
            <span class="memory-item-key">${escapeHtml(e.date || '')}</span>
            <span class="memory-item-text">${escapeHtml(e.description)}</span>
            <button class="memory-item-delete" onclick="removeMemoryItem('importantEvents', ${i})">✕</button>
        </div>
    `).join('') || `<div style="color:var(--text-muted); font-size:0.8rem; padding:8px">${t('memory.empty_events')}</div>`;

    // Summaries
    const summariesEl = $('#memory-summaries');
    summariesEl.innerHTML = (mem.summaries || []).map((s, i) => `
        <div class="memory-item">
            <span class="memory-item-text">${escapeHtml(s)}</span>
            <button class="memory-item-delete" onclick="removeMemoryItem('summaries', ${i})">✕</button>
        </div>
    `).join('') || `<div style="color:var(--text-muted); font-size:0.8rem; padding:8px">${t('memory.empty_summary')}</div>`;
}

function renderPersonaSelector() {
    const container = $('#memory-persona-selector');
    const summaryEl = $('#memory-persona-summary');
    if (!container || !summaryEl) return;

    const personas = state.personas || [];
    const currentId = String(state.memoryModalDraftPersonaId || state.memory?.personaId || (personas[0]?.id || '1'));

    if (personas.length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted); font-size:0.8rem">Belum ada persona tersimpan.</div>';
        summaryEl.innerHTML = '';
        return;
    }

    container.innerHTML = personas.map(p => {
        const isSelected = String(p.id) === currentId;
        return `
            <button type="button" class="persona-chip-btn ${isSelected ? 'active' : ''}" data-id="${escapeHtml(p.id)}">
                <span class="persona-chip-avatar">${getAvatarInnerHtml(p.avatar, p.name || 'User')}</span>
                <span class="persona-chip-index">#${escapeHtml(p.index || p.id)}</span>
                <span class="persona-chip-name">${escapeHtml(p.name || 'User')}</span>
            </button>
        `;
    }).join('');

    container.querySelectorAll('.persona-chip-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = btn.dataset.id;
            state.memoryModalDraftPersonaId = id;
            renderPersonaSelector();
        });
    });

    const activePersona = personas.find(p => String(p.id) === currentId) || personas[0];
    if (activePersona) {
        const details = [];
        if (activePersona.gender) details.push(`Gender: ${escapeHtml(activePersona.gender)}`);
        if (activePersona.age !== undefined && activePersona.age !== null && activePersona.age !== '') details.push(`Umur: ${escapeHtml(activePersona.age)} th`);
        if (activePersona.height) details.push(`TB: ${escapeHtml(activePersona.height)} cm`);
        if (activePersona.weight) details.push(`BB: ${escapeHtml(activePersona.weight)} kg`);
        if (activePersona.extra && typeof activePersona.extra === 'object') {
            for (const [k, v] of Object.entries(activePersona.extra)) {
                if (v) details.push(`${escapeHtml(k)}: ${escapeHtml(v)}`);
            }
        }
        summaryEl.innerHTML = `
            <div class="persona-active-card">
                <div class="persona-active-avatar">${getAvatarInnerHtml(activePersona.avatar, activePersona.name || 'User')}</div>
                <div class="persona-active-info">
                    <div class="persona-active-title">${escapeHtml(activePersona.name || 'User')}</div>
                    <div class="persona-active-meta">${details.length > 0 ? details.join(' • ') : 'Tidak ada detail tambahan'}</div>
                </div>
            </div>
        `;
    } else {
        summaryEl.innerHTML = '';
    }
}

function removeMemoryItem(key, index) {
    if (!state.memory) return;
    state.memory[key].splice(index, 1);
    renderMemory();
}

function addMemoryFact() {
    const input = $('#memory-new-fact');
    const val = input.value.trim();
    if (!val || !state.memory) return;
    state.memory.facts = state.memory.facts || [];
    state.memory.facts.push(val);
    input.value = '';
    renderMemory();
}

function addMemoryEvent() {
    const input = $('#memory-new-event');
    const val = input.value.trim();
    if (!val || !state.memory) return;
    state.memory.importantEvents = state.memory.importantEvents || [];
    state.memory.importantEvents.push({
        description: val,
        date: new Date().toISOString().split('T')[0]
    });
    input.value = '';
    renderMemory();
}

async function saveMemory() {
    if (state.currentGroupId) {
        if (!state.memory) return;
        const requestGroupId = state.currentGroupId;

        if (state.memoryModalDraftPersonaId) {
            state.memory.personaId = state.memoryModalDraftPersonaId;
        }

        await api('PUT', `/api/groups/${requestGroupId}/memory`, state.memory);
        if (state.currentGroupId !== requestGroupId) return;
        const group = state.groups.find(g => g.id === requestGroupId);
        if (group && state.memory.personaId) {
            group.activePersonaId = state.memory.personaId;
        }

        if (state.currentGroupChatId) {
            const freshChat = await api('GET', `/api/groups/${requestGroupId}/chat/${state.currentGroupChatId}`);
            if (freshChat && Array.isArray(freshChat.messages)) {
                renderGroupMessages(freshChat.messages);
            }
        }

        toast(t('toast.group_memory_saved') || 'Memori grup disimpan!', 'success');
        closeModal('modal-memory');
        return;
    }

    if (!state.currentCharacterId || !state.memory) return;
    const requestCharacterId = state.currentCharacterId; // capture before the await

    if (state.memoryModalDraftPersonaId) {
        state.memory.personaId = state.memoryModalDraftPersonaId;
    }

    await api('PUT', `/api/memory/${requestCharacterId}`, state.memory);
    if (state.currentCharacterId !== requestCharacterId) return;

    if (state.currentChatId) {
        const freshChat = await api('GET', `/api/chat/${state.currentChatId}`);
        if (freshChat && Array.isArray(freshChat.messages) && state.currentCharacterId === requestCharacterId) {
            state.currentMessages = freshChat.messages;
            const currentChat = state.chats.find(c => c.id === state.currentChatId);
            if (currentChat) currentChat.messages = freshChat.messages;
            renderMessages(freshChat.messages);
        } else if (Array.isArray(state.currentMessages) && state.currentMessages.length > 0) {
            renderMessages(state.currentMessages);
        }
    }

    toast(t('toast.memory_saved'), 'success');
    closeModal('modal-memory');
}

async function summarizeMemory() {
    if (state.currentGroupId) {
        if (!state.currentGroupChatId) {
            toast(t('toast.select_chat_first') || 'Pilih sesi chat terlebih dahulu', 'error');
            return;
        }
        const requestGroupId = state.currentGroupId;
        const btn = $('#btn-summarize-memory');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> ' + (t('btn.saving') || 'Menyimpan...');

        try {
            const result = await api('POST', `/api/memory/${requestGroupId}/summarize`, {
                chatId: state.currentGroupChatId,
                isGroup: true
            });
            if (state.currentGroupId !== requestGroupId) return;

            if (result.error) {
                toast(result.error, 'error');
            } else {
                state.memory = result.memory;
                renderMemory();
                if (result.warning) {
                    toast(result.warning, 'warning');
                } else {
                    toast(t('toast.memory_summarized') || 'Memori berhasil diringkas!', 'success');
                }
            }
        } catch (err) {
            if (state.currentGroupId === requestGroupId) {
                toast(`Error: ${err.message}`, 'error');
            }
        } finally {
            btn.disabled = false;
            btn.innerHTML = t('memory.btn_ai_summary') || '🤖 AI Summarize Chat';
        }
        return;
    }

    if (!state.currentCharacterId || !state.currentChatId) {
        toast(t('toast.select_chat_first'), 'error');
        return;
    }

    const requestCharacterId = state.currentCharacterId; // capture before the (5-120s) await
    const btn = $('#btn-summarize-memory');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> ' + t('btn.saving');

    try {
        const result = await api('POST', `/api/memory/${requestCharacterId}/summarize`, {
            chatId: state.currentChatId
        });

        // The LLM call above can take anywhere from 5 to 120 seconds. If
        // the user switched to a different character while waiting, DO
        // NOT overwrite state.memory / re-render — that would show (and
        // let the user unknowingly Save) the WRONG character's memory
        // on top of whatever character is now actually being viewed.
        if (state.currentCharacterId !== requestCharacterId) return;

        if (result.error) {
            toast(result.error, 'error');
        } else {
            state.memory = result.memory;
            renderMemory();
            if (result.warning) {
                toast(result.warning, 'warning');
            } else {
                toast(t('toast.memory_summarized'), 'success');
            }
        }
    } catch (err) {
        if (state.currentCharacterId === requestCharacterId) {
            toast(`Error: ${err.message}`, 'error');
        }
    } finally {
        // Always safe to reset regardless of which character is active —
        // #btn-summarize-memory is a single fixed element in the shared
        // memory modal, not per-character.
        btn.disabled = false;
        btn.innerHTML = t('memory.btn_ai_summary') || '🤖 AI Summarize Chat';
    }
}

// ══════════════════════════════════════════════════════════
//  UI HELPERS
// ══════════════════════════════════════════════════════════

function showView(viewId) {
    $$('.view').forEach(v => v.classList.remove('active'));
    $(`#${viewId}`).classList.add('active');
}

function openModal(id) {
    $(`#${id}`).classList.add('show');
}

function closeModal(id) {
    if (id === 'modal-memory') {
        state.memoryModalDraftPersonaId = null;
    }
    $(`#${id}`).classList.remove('show');
}

function toast(message, type = 'info') {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    document.body.appendChild(el);

    const duration = Math.min(7000, Math.max(3000, message.length * 60));
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'all 0.3s ease';
        setTimeout(() => el.remove(), 300);
    }, duration);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    // div.innerHTML escapes & < > for text nodes, but NOT quotes (quotes only
    // matter inside attribute values, which is exactly where several callers
    // use this output — e.g. value="${escapeHtml(x)}"). Escape them explicitly.
    return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function formatMessageContent(content) {
    if (!content) return '';
    let html = escapeHtml(content);

    // Protect code blocks and inline code from markdown formatting
    const codeBlocks = [];
    html = html.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g, (match, lang, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push(`<pre><code class="language-${lang}">${code}</code></pre>`);
        return `@@CODE_BLOCK_${idx}@@`;
    });
    html = html.replace(/`([^`\n]+)`/g, (match, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push(`<code>${code}</code>`);
        return `@@CODE_BLOCK_${idx}@@`;
    });

    // Bold-italic: ***text***
    html = html.replace(/(?<!\*)\*\*\*(?!\s|\*)((?:(?!\*\*\*)[^\n]|\n(?!\n))+?)(?<!\s|\*)\*\*\*(?!\*)/g, '<strong><em class="action-text">$1</em></strong>');

    // Bold: **text** (and __text__)
    html = html.replace(/(?<!\*)\*\*(?!\s|\*)((?:(?!\*\*)[^\n]|\n(?!\n))+?)(?<!\s|\*)\*\*(?!\*)/g, '<strong>$1</strong>');
    html = html.replace(/\b__([^\n_]+)__\b/g, '<strong>$1</strong>');

    // Italic / Action text: *action*
    html = html.replace(/(?<!\*)\*(?!\s|\*)((?:[^\*\n]|\n(?!\n))+?)(?<!\s|\*)\*(?!\*)/g, '<em class="action-text">$1</em>');

    // Dialogue text in quotes: "dialog"
    html = html.replace(/(?:&quot;|“)((?:(?!(?:&quot;|”|“)).|\n(?!\n))+?)(?:&quot;|”)/g, '<span class="dialog-text">&quot;$1&quot;</span>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    // Restore protected code blocks
    html = html.replace(/@@CODE_BLOCK_(\d+)@@/g, (_, idx) => codeBlocks[Number(idx)] || '');

    return html;
}

function formatTime(isoString) {
    try {
        const d = new Date(isoString);
        return d.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
    } catch {
        return '';
    }
}

function autoResizeInput() {
    const input = $('#chat-input');
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
}

// ══════════════════════════════════════════════════════════
//  EVENT BINDINGS
// ══════════════════════════════════════════════════════════

function bindEvents() {
    // Settings — dedicated page (Configuration, Appearance, Notifications,
    // Data & backup all live there now), not a popup.
    $('#btn-settings').addEventListener('click', () => {
        window.location.href = 'settings.html';
    });

    // Character
    $('#btn-new-character').addEventListener('click', () => openCharacterEditor());
    $('#btn-save-character').addEventListener('click', saveCharacter);
    $('#btn-delete-character').addEventListener('click', deleteCharacter);

    // Avatar upload
    $('#avatar-preview').addEventListener('click', () => {
        $('#char-avatar-file').click();
    });
    $('#char-avatar-file').addEventListener('change', handleAvatarUpload);
    $('#avatar-remove-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        removeAvatar();
    });

    // Pinterest avatar picker
    $('#avatar-pinterest-toggle').addEventListener('click', togglePinterestPanel);
    $('#pinterest-search-btn').addEventListener('click', searchPinterest);
    $('#pinterest-query').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); searchPinterest(); }
    });
    $('#pinterest-prev-btn').addEventListener('click', () => stepPinterestResult(-1));
    $('#pinterest-next-btn').addEventListener('click', () => stepPinterestResult(1));
    $('#pinterest-use-btn').addEventListener('click', usePinterestImage);
    $('#btn-edit-character').addEventListener('click', () => {
        if (state.currentGroupId) {
            openEditGroupModal(state.currentGroupId);
        } else if (state.currentCharacterId) {
            openCharacterEditor(state.currentCharacterId);
        }
    });

    // Chat
    $('#btn-new-chat').addEventListener('click', () => {
        if (state.currentGroupId) {
            startNewGroupChat();
        } else {
            startNewChat();
        }
    });
    $('#chat-form').addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
    });

    // Group Search & Avatar
    const groupSearchInput = $('#group-search-input');
    if (groupSearchInput) {
        groupSearchInput.addEventListener('input', (e) => searchGroups(e.target.value));
    }
    const groupAvatarPreview = $('#group-avatar-preview');
    if (groupAvatarPreview) {
        groupAvatarPreview.addEventListener('click', () => {
            $('#group-avatar-file').click();
        });
    }
    const groupAvatarInput = $('#group-avatar-file');
    if (groupAvatarInput) {
        groupAvatarInput.addEventListener('change', handleGroupAvatarUpload);
    }
    const groupAvatarRemoveBtn = $('#group-avatar-remove-btn');
    if (groupAvatarRemoveBtn) {
        groupAvatarRemoveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeGroupAvatar();
        });
    }

    // Group Pinterest avatar picker
    const groupPinterestToggle = $('#group-avatar-pinterest-toggle');
    if (groupPinterestToggle) {
        groupPinterestToggle.addEventListener('click', toggleGroupPinterestPanel);
    }
    const groupPinterestSearchBtn = $('#group-pinterest-search-btn');
    if (groupPinterestSearchBtn) {
        groupPinterestSearchBtn.addEventListener('click', searchGroupPinterest);
    }
    const groupPinterestQuery = $('#group-pinterest-query');
    if (groupPinterestQuery) {
        groupPinterestQuery.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                searchGroupPinterest();
            }
        });
    }
    const groupPinterestPrevBtn = $('#group-pinterest-prev-btn');
    if (groupPinterestPrevBtn) {
        groupPinterestPrevBtn.addEventListener('click', () => stepGroupPinterestResult(-1));
    }
    const groupPinterestNextBtn = $('#group-pinterest-next-btn');
    if (groupPinterestNextBtn) {
        groupPinterestNextBtn.addEventListener('click', () => stepGroupPinterestResult(1));
    }
    const groupPinterestUseBtn = $('#group-pinterest-use-btn');
    if (groupPinterestUseBtn) {
        groupPinterestUseBtn.addEventListener('click', useGroupPinterestImage);
    }

    // Auto-resize textarea
    $('#chat-input').addEventListener('input', autoResizeInput);

    // Enter to send (Shift+Enter for newline, or mobile always newline)
    $('#chat-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            if (window.innerWidth <= 768) {
                // On mobile, Enter just creates a newline, don't send
                return;
            }
            e.preventDefault();
            sendMessage();
        }
    });

    // Memory
    $('#btn-memory-view').addEventListener('click', openMemoryViewer);
    $('#btn-save-memory').addEventListener('click', saveMemory);
    $('#btn-summarize-memory').addEventListener('click', summarizeMemory);
    $('#btn-add-fact').addEventListener('click', addMemoryFact);
    $('#btn-add-event').addEventListener('click', addMemoryEvent);

    // Character search
    $('#character-search-input').addEventListener('input', (e) => searchCharacters(e.target.value));
    $('#btn-clear-char-search').addEventListener('click', clearCharacterSearchAndFocus);

    // Enter key in memory inputs
    $('#memory-new-fact').addEventListener('keydown', (e) => { if (e.key === 'Enter') addMemoryFact(); });
    $('#memory-new-event').addEventListener('keydown', (e) => { if (e.key === 'Enter') addMemoryEvent(); });

    // Sidebar toggle
    $$('.mobile-menu-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                $('#sidebar').classList.toggle('open');
            } else {
                $('#sidebar').classList.toggle('closed');
            }
        });
    });

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        const sidebar = $('#sidebar');
        if (sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) &&
            !e.target.closest('.mobile-menu-btn')) {
            sidebar.classList.remove('open');
        }
    });

    // Close modals on overlay click
    $$('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.classList.remove('show');
            }
        });
    });

    // Escape key to close modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            $$('.modal-overlay.show').forEach(m => m.classList.remove('show'));
        }
    });
}

function closeSidebarOnMobile() {
    if (window.innerWidth <= 768) {
        $('#sidebar').classList.remove('open');
    }
}

// ══════════════════════════════════════════════════════════
//  AVATAR UPLOAD HANDLERS
// ══════════════════════════════════════════════════════════

function handleAvatarUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        toast('Pilih file gambar (JPG, PNG, GIF, WebP)', 'error');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        toast('Ukuran file maksimal 5MB', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        img.onload = function () {
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
            $('#char-avatar').value = dataUrl;
            updateAvatarPreview(dataUrl);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function removeAvatar() {
    $('#char-avatar').value = '';
    $('#char-avatar-file').value = '';
    updateAvatarPreview('');
}

function handleGroupAvatarUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        toast('Pilih file gambar (JPG, PNG, GIF, WebP)', 'error');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        toast('Ukuran file maksimal 5MB', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = async function (event) {
        try {
            const resized = await resizeImageDataUrl(event.target.result, 256);
            $('#group-avatar').value = resized;
            updateGroupAvatarPreview(resized);
        } catch (err) {
            toast('Gagal memproses gambar avatar grup', 'error');
        }
    };
    reader.readAsDataURL(file);
}

// ══════════════════════════════════════════════════════════
//  PINTEREST AVATAR PICKER
//  Sequential Prev/Next browsing through search results —
//  never random, so a missed image can always be revisited.
// ══════════════════════════════════════════════════════════

function togglePinterestPanel() {
    const panel = $('#pinterest-panel');
    const btn = $('#avatar-pinterest-toggle');
    const isOpen = panel.style.display !== 'none';
    panel.style.display = isOpen ? 'none' : 'block';
    btn.classList.toggle('active', !isOpen);
    if (!isOpen) {
        setTimeout(() => $('#pinterest-query').focus(), 50);
    }
}

async function searchPinterest() {
    const query = $('#pinterest-query').value.trim();
    const statusEl = $('#pinterest-status');
    const resultsEl = $('#pinterest-results');

    if (!query) {
        statusEl.textContent = 'Masukkan kata kunci pencarian dulu ya.';
        return;
    }
    if (state.pinterest.loading) return;

    state.pinterest.loading = true;
    resultsEl.style.display = 'none';
    statusEl.textContent = 'Mencari gambar...';

    try {
        const res = await api('GET', `/api/pinterest/search?q=${encodeURIComponent(query)}`);
        if (!res.success) {
            statusEl.textContent = res.error || 'Gagal mencari gambar.';
            state.pinterest.results = [];
            return;
        }
        if (!res.data || res.data.length === 0) {
            statusEl.textContent = `Tidak ada gambar ditemukan untuk "${query}".`;
            state.pinterest.results = [];
            return;
        }
        state.pinterest.results = res.data;
        state.pinterest.index = 0;
        statusEl.textContent = '';
        resultsEl.style.display = 'flex';
        renderPinterestResult();
    } catch (err) {
        statusEl.textContent = 'Terjadi kesalahan saat menghubungi server.';
    } finally {
        state.pinterest.loading = false;
    }
}

function renderPinterestResult() {
    const { results, index } = state.pinterest;
    if (!results.length) return;
    const item = results[index];
    $('#pinterest-thumb').src = `/api/pinterest/image?url=${encodeURIComponent(item.thumb)}`;
    $('#pinterest-counter').textContent = `${index + 1} / ${results.length}`;
}

function stepPinterestResult(direction) {
    const { results } = state.pinterest;
    if (!results.length) return;
    // Clamp instead of wrapping so Prev/Next always moves one-by-one
    // and stops cleanly at the first/last image.
    state.pinterest.index = Math.max(0, Math.min(results.length - 1, state.pinterest.index + direction));
    renderPinterestResult();
}

async function usePinterestImage() {
    const { results, index } = state.pinterest;
    if (!results.length) return;
    const item = results[index];
    const statusEl = $('#pinterest-status');
    const useBtn = $('#pinterest-use-btn');

    useBtn.disabled = true;
    statusEl.textContent = 'Mengunduh gambar...';

    try {
        const proxyUrl = `/api/pinterest/image?url=${encodeURIComponent(item.url)}`;
        const response = await fetch(proxyUrl);
        if (!response.ok) throw new Error('download failed');
        const blob = await response.blob();
        const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });

        const resizedDataUrl = await resizeImageDataUrl(dataUrl, 256);
        $('#char-avatar').value = resizedDataUrl;
        updateAvatarPreview(resizedDataUrl);
        statusEl.textContent = '✅ Avatar terpasang dari Pinterest.';
        togglePinterestPanel();
    } catch (err) {
        statusEl.textContent = 'Gagal menggunakan gambar ini, coba gambar lain.';
    } finally {
        useBtn.disabled = false;
    }
}

// ══════════════════════════════════════════════════════════
//  GROUP PINTEREST AVATAR PICKER
// ══════════════════════════════════════════════════════════

function toggleGroupPinterestPanel() {
    const panel = $('#group-pinterest-panel');
    const btn = $('#group-avatar-pinterest-toggle');
    if (!panel || !btn) return;
    const isOpen = panel.style.display !== 'none';
    panel.style.display = isOpen ? 'none' : 'block';
    btn.classList.toggle('active', !isOpen);
    if (!isOpen) {
        const queryInput = $('#group-pinterest-query');
        if (queryInput && !queryInput.value) {
            queryInput.value = ($('#group-name').value || '').trim();
        }
        setTimeout(() => queryInput && queryInput.focus(), 50);
    }
}

async function searchGroupPinterest() {
    const queryEl = $('#group-pinterest-query');
    const statusEl = $('#group-pinterest-status');
    const resultsEl = $('#group-pinterest-results');
    if (!queryEl || !statusEl || !resultsEl) return;

    const query = queryEl.value.trim();
    if (!query) {
        statusEl.textContent = 'Masukkan kata kunci pencarian dulu ya.';
        return;
    }
    if (state.groupPinterest.loading) return;

    state.groupPinterest.loading = true;
    resultsEl.style.display = 'none';
    statusEl.textContent = 'Mencari gambar...';

    try {
        const res = await api('GET', `/api/pinterest/search?q=${encodeURIComponent(query)}`);
        if (!res.success) {
            statusEl.textContent = res.error || 'Gagal mencari gambar.';
            state.groupPinterest.results = [];
            return;
        }
        if (!res.data || res.data.length === 0) {
            statusEl.textContent = `Tidak ada gambar ditemukan untuk "${query}".`;
            state.groupPinterest.results = [];
            return;
        }
        state.groupPinterest.results = res.data;
        state.groupPinterest.index = 0;
        statusEl.textContent = '';
        resultsEl.style.display = 'flex';
        renderGroupPinterestResult();
    } catch (err) {
        statusEl.textContent = 'Terjadi kesalahan saat menghubungi server.';
    } finally {
        state.groupPinterest.loading = false;
    }
}

function renderGroupPinterestResult() {
    const { results, index } = state.groupPinterest;
    if (!results.length) return;
    const item = results[index];
    const thumbEl = $('#group-pinterest-thumb');
    const counterEl = $('#group-pinterest-counter');
    if (thumbEl) thumbEl.src = `/api/pinterest/image?url=${encodeURIComponent(item.thumb)}`;
    if (counterEl) counterEl.textContent = `${index + 1} / ${results.length}`;
}

function stepGroupPinterestResult(direction) {
    const { results } = state.groupPinterest;
    if (!results.length) return;
    state.groupPinterest.index = Math.max(0, Math.min(results.length - 1, state.groupPinterest.index + direction));
    renderGroupPinterestResult();
}

async function useGroupPinterestImage() {
    const { results, index } = state.groupPinterest;
    if (!results.length) return;
    const item = results[index];
    const statusEl = $('#group-pinterest-status');
    const useBtn = $('#group-pinterest-use-btn');

    if (useBtn) useBtn.disabled = true;
    if (statusEl) statusEl.textContent = 'Mengunduh gambar...';

    try {
        const proxyUrl = `/api/pinterest/image?url=${encodeURIComponent(item.url)}`;
        const response = await fetch(proxyUrl);
        if (!response.ok) throw new Error('download failed');
        const blob = await response.blob();
        const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });

        const resizedDataUrl = await resizeImageDataUrl(dataUrl, 256);
        $('#group-avatar').value = resizedDataUrl;
        updateGroupAvatarPreview(resizedDataUrl);
        if (statusEl) statusEl.textContent = '✅ Avatar terpasang dari Pinterest.';
        toggleGroupPinterestPanel();
    } catch (err) {
        if (statusEl) statusEl.textContent = 'Gagal menggunakan gambar ini, coba gambar lain.';
    } finally {
        if (useBtn) useBtn.disabled = false;
    }
}

// Shared resize helper: mirrors the sizing/quality used by manual upload
function resizeImageDataUrl(srcDataUrl, maxSize) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            let width = img.width;
            let height = img.height;
            if (width > height) {
                if (width > maxSize) { height *= maxSize / width; width = maxSize; }
            } else {
                if (height > maxSize) { width *= maxSize / height; height = maxSize; }
            }
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            canvas.getContext('2d').drawImage(img, 0, 0, width, height);
            resolve(canvas.toDataURL('image/webp', 0.85));
        };
        img.onerror = reject;
        img.src = srcDataUrl;
    });
}

// ══════════════════════════════════════════════════════════
//  CHAT RENAME
// ══════════════════════════════════════════════════════════

window.startRenameChat = function (chatId) {
    const chatItem = document.querySelector(`.chat-item[data-id="${chatId}"]`);
    if (!chatItem) return;

    const chat = state.chats.find(c => c.id === chatId);
    if (!chat) return;

    const currentTitle = chat.title || 'New Chat';

    chatItem.removeAttribute('onclick');
    chatItem.innerHTML = `
        <span class="chat-item-icon">💬</span>
        <div class="chat-item-rename-container">
            <input type="text" class="chat-item-rename-input" value="${escapeHtml(currentTitle)}"
                   data-chat-id="${chatId}" autocomplete="off">
            <div class="chat-item-rename-actions">
                <button class="chat-rename-save" onclick="event.stopPropagation(); saveRenameChat('${chatId}')" title="Simpan">✓</button>
                <button class="chat-rename-cancel" onclick="event.stopPropagation(); cancelRenameChat()" title="Batal">✕</button>
            </div>
        </div>
    `;

    const input = chatItem.querySelector('.chat-item-rename-input');
    input.focus();
    input.select();

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            saveRenameChat(chatId);
        } else if (e.key === 'Escape') {
            cancelRenameChat();
        }
    });
};

window.saveRenameChat = async function (chatId) {
    const input = document.querySelector(`.chat-item-rename-input[data-chat-id="${chatId}"]`);
    if (!input) return;

    const newTitle = input.value.trim();
    if (!newTitle) return cancelRenameChat();

    const requestCharacterId = state.currentCharacterId; // capture before the await

    try {
        await api('PUT', `/api/chat/${chatId}`, { title: newTitle });

        // If the user switched characters while this PUT was in flight,
        // state.chats now belongs to a different character and won't
        // contain this chat — the rename still succeeded server-side,
        // but there's nothing to update in the currently-visible list.
        if (state.currentCharacterId !== requestCharacterId) return;

        const chat = state.chats.find(c => c.id === chatId);
        if (chat) chat.title = newTitle;

        renderChatList();
        toast('Chat renamed!', 'success');
    } catch (err) {
        toast(`Error: ${err.message}`, 'error');
        if (state.currentCharacterId === requestCharacterId) renderChatList();
    }
};

window.cancelRenameChat = function () {
    renderChatList();
};
