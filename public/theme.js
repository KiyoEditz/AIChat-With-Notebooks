/**
 * theme.js — shared theme engine
 *
 * Loaded synchronously in <head> right after styles.css on every page so
 * the saved palette or theme mode is applied to :root before first paint.
 *
 * Supports:
 * - System preference ('system', default): tracks prefers-color-scheme
 * - Dark theme: Obsidian Ink (#0E1013) + Verdigris Sage (#5E9E8B)
 * - Light theme: Alabaster Paper (#F6F7F8) + Deep Pine (#2F6F5E)
 * - Custom user-picked accent & background colors via color-mix()
 * - Chat bubble font size ('small': 14px, 'medium': 16px, 'large': 18px)
 */
(function () {
    'use strict';

    var STORAGE_KEY = 'aichat-theme-palette';
    var DARK_DEFAULTS = { accent: '#5E9E8B', background: '#0E1013', mode: 'dark', fontSize: 'medium' };
    var LIGHT_DEFAULTS = { accent: '#2F6F5E', background: '#F6F7F8', mode: 'light', fontSize: 'medium' };
    var FONT_SIZES = {
        small: '14px',
        medium: '16px',
        large: '18px'
    };

    function normalizeFontSize(size) {
        if (size === 'small' || size === 'medium' || size === 'large') return size;
        return 'medium';
    }

    function normalizeHex(hex, fallback) {
        if (typeof hex !== 'string') return fallback;
        var clean = hex.trim();
        if (/^#[0-9a-fA-F]{6}$/.test(clean)) return clean.toUpperCase();
        if (/^#[0-9a-fA-F]{3}$/.test(clean)) {
            var r = clean[1], g = clean[2], b = clean[3];
            return ('#' + r + r + g + g + b + b).toUpperCase();
        }
        return fallback;
    }

    function hexToRgb(hex) {
        var clean = (hex || '').replace('#', '');
        if (clean.length === 3) {
            clean = clean.split('').map(function (c) { return c + c; }).join('');
        }
        var n = parseInt(clean, 16);
        if (isNaN(n) || clean.length !== 6) return { r: 14, g: 16, b: 19 };
        return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
    }

    // WCAG relative luminance — used to decide whether text and surface mixes
    // on top of the chosen background should be light-mode or dark-mode.
    function relativeLuminance(hex) {
        var rgb = hexToRgb(hex);
        var chans = [rgb.r, rgb.g, rgb.b].map(function (v) {
            var c = v / 255;
            return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * chans[0] + 0.7152 * chans[1] + 0.0722 * chans[2];
    }

    function getSystemPalette() {
        var prefersLight = false;
        try {
            prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
        } catch (err) { /* ignore */ }
        return prefersLight
            ? { accent: LIGHT_DEFAULTS.accent, background: LIGHT_DEFAULTS.background, mode: 'system', fontSize: 'medium' }
            : { accent: DARK_DEFAULTS.accent, background: DARK_DEFAULTS.background, mode: 'system', fontSize: 'medium' };
    }

    var DEFAULTS = getSystemPalette();

    function loadPalette() {
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return null;
            var parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== 'object') return null;

            var acc = typeof parsed.accent === 'string' ? parsed.accent.toUpperCase() : '';
            var bg = typeof parsed.background === 'string' ? parsed.background.toUpperCase() : '';
            var fontSize = normalizeFontSize(parsed.fontSize);

            // Migrate legacy neon-purple default (#A855F7 / #121212) or recover from full-black (#000000 / #000000) bug
            if ((acc === '#A855F7' && bg === '#121212') || (acc === '#000000' && bg === '#000000')) {
                try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
                return null;
            }

            if (parsed.mode === 'system') {
                var sys = getSystemPalette();
                sys.fontSize = fontSize;
                return sys;
            }

            if (/^#[0-9a-fA-F]{6}$/.test(acc) && /^#[0-9a-fA-F]{6}$/.test(bg)) {
                return {
                    accent: acc,
                    background: bg,
                    mode: parsed.mode || (relativeLuminance(bg) > 0.45 ? 'light' : 'dark'),
                    fontSize: fontSize
                };
            }
            return null;
        } catch (err) {
            console.warn('[theme] could not read saved palette:', err);
            return null;
        }
    }

    function applyPalette(palette) {
        var effective = (palette && palette.mode === 'system') ? getSystemPalette() : (palette || getSystemPalette());
        var fontSize = normalizeFontSize(palette && palette.fontSize);
        var root = document.documentElement;
        var bgLum = relativeLuminance(effective.background);
        var accentLum = relativeLuminance(effective.accent);
        var isLight = bgLum > 0.45;

        root.setAttribute('data-theme', isLight ? 'light' : 'dark');
        root.setAttribute('data-theme-mode', (palette && palette.mode) || (isLight ? 'light' : 'dark'));
        root.setAttribute('data-font-size', fontSize);
        root.style.setProperty('--accent-base', effective.accent);
        root.style.setProperty('--bg-base', effective.background);
        root.style.setProperty('--surface-mix-target', isLight ? '#000000' : '#FFFFFF');
        root.style.setProperty('--text-primary', isLight ? '#24272B' : '#DEDAD2');
        root.style.setProperty('--text-on-accent', accentLum > 0.42 ? '#111418' : '#FFFFFF');
        root.style.setProperty('--chat-font-size', FONT_SIZES[fontSize] || '16px');
    }

    function savePalette(palette) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(palette));
            return true;
        } catch (err) {
            console.warn('[theme] could not persist palette:', err);
            return false;
        }
    }

    var current = loadPalette() || getSystemPalette();
    applyPalette(current);

    // Listen for OS theme changes when in 'system' mode (or when no custom palette is saved)
    try {
        if (window.matchMedia) {
            var mq = window.matchMedia('(prefers-color-scheme: light)');
            var onSystemChange = function () {
                var saved = loadPalette();
                if (!saved || saved.mode === 'system') {
                    var sys = getSystemPalette();
                    if (saved && saved.fontSize) sys.fontSize = saved.fontSize;
                    applyPalette(sys);
                }
            };
            if (mq.addEventListener) mq.addEventListener('change', onSystemChange);
            else if (mq.addListener) mq.addListener(onSystemChange);
        }
    } catch (err) { /* ignore */ }

    window.AIChatTheme = {
        DEFAULTS: DEFAULTS,
        DARK_DEFAULTS: DARK_DEFAULTS,
        LIGHT_DEFAULTS: LIGHT_DEFAULTS,
        FONT_SIZES: FONT_SIZES,
        load: loadPalette,
        apply: applyPalette,
        get: function () { return loadPalette() || getSystemPalette(); },
        set: function (palette) {
            var sys = getSystemPalette();
            var saved = loadPalette();
            var accent = normalizeHex(palette && palette.accent, (saved && saved.accent) || sys.accent);
            var background = normalizeHex(palette && palette.background, (saved && saved.background) || sys.background);
            var mode = (palette && palette.mode) || (saved && saved.mode) || (relativeLuminance(background) > 0.45 ? 'light' : 'dark');
            var fontSize = normalizeFontSize((palette && palette.fontSize) || (saved && saved.fontSize) || 'medium');
            var next = {
                accent: accent,
                background: background,
                mode: mode,
                fontSize: fontSize
            };
            applyPalette(next);
            savePalette(next);
            return next;
        },
        setMode: function (mode) {
            var saved = loadPalette();
            var fontSize = normalizeFontSize(saved && saved.fontSize);
            var next;
            if (mode === 'light') {
                next = { accent: LIGHT_DEFAULTS.accent, background: LIGHT_DEFAULTS.background, mode: 'light', fontSize: fontSize };
            } else if (mode === 'dark') {
                next = { accent: DARK_DEFAULTS.accent, background: DARK_DEFAULTS.background, mode: 'dark', fontSize: fontSize };
            } else {
                var sys = getSystemPalette();
                next = { accent: sys.accent, background: sys.background, mode: 'system', fontSize: fontSize };
            }
            applyPalette(next);
            savePalette(next);
            return next;
        },
        setFontSize: function (size) {
            var currentPalette = loadPalette() || getSystemPalette();
            var next = {
                accent: currentPalette.accent,
                background: currentPalette.background,
                mode: currentPalette.mode || 'system',
                fontSize: normalizeFontSize(size)
            };
            applyPalette(next);
            savePalette(next);
            return next;
        },
        reset: function () {
            try { localStorage.removeItem(STORAGE_KEY); } catch (err) { /* ignore */ }
            var sys = getSystemPalette();
            applyPalette(sys);
            return sys;
        }
    };
})();
