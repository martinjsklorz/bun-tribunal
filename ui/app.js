/* Bun Tribunal: vanilla JS, no build step. Loaded after config.js.
 *
 * Sections: models · DOM references · helpers · storage · network · image prep · health · cards ·
 * rounds · summary · retry · scoreboard · inputs · samples · confetti · phone clock · boot
 */
(() => {
  'use strict';

  const CFG = window.HOTDOG_CONFIG;
  const ORDER = ['llm', 'clef', 'cnn']; // card order
  const SCORE_KEY = 'bun-tribunal:score';
  const UPLOAD_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']); // what the servers accept as-is
  const SMALL_UPLOAD_BYTES = 1.5 * 1024 * 1024; // supported files up to this size skip the decode/downscale

  const MODELS = {
    llm: {
      name: 'LLM', url: CFG.llm,
      sub: 'General-purpose vision LLM',
      loading: 'Asking a chatbot to squint…',
      slowQuip: 'had to phone a friend in the cloud',
      latencyLabel: 'API', latencyTitle: 'latency_ms: API round trip (network + generation)',
      deviceLabel: 'Host', deviceTitle: 'API host reported by /health',
      fix: './run.sh llm', fixWhat: 'Sets the provider, model and API key. Then restart ./run.sh.',
    },
    clef: {
      name: 'CLEF-Flash', url: CFG.clef,
      sub: 'Cloudflare · 9B multimodal decision model',
      loading: 'Deliberating with 9B parameters…',
      slowQuip: 'brought more parameters to the table',
      fix: './run.sh download', fixWhat: 'Downloads the weights (4-bit MLX, ~6 GB, on an Apple Silicon Mac; ~19 GB elsewhere).',
    },
    cnn: {
      name: 'CNN', url: CFG.cnn,
      sub: 'ConvNeXt-Tiny · fine-tuned on a laptop',
      loading: 'Squinting at pixels…',
      slowQuip: 'tripped over its own pooling layers',
      fix: './run.sh train', fixWhat: 'Fine-tunes the CNN on your machine.',
    },
  };
  const CNN_ARCH_NAMES = { convnext_tiny: 'ConvNeXt-Tiny', efficientnet_b0: 'EfficientNet-B0', mobilenet_v3_large: 'MobileNetV3-Large' };

  const media = {
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)'),
    singleColumn: window.matchMedia('(max-width: 760px)'), // keep in sync with styles.css
    coarsePointer: window.matchMedia('(pointer: coarse)'),
  };

  // ---------- DOM references (scripts run at the end of <body>) ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const dom = {
    health: Object.fromEntries(ORDER.map((id) => [id, document.getElementById('health-' + id)])),
    offlineHelp: $('#offline-help'), offlineNames: $('#offline-names'),
    viewfinder: $('#viewfinder'), preview: $('#preview'), emptyState: $('#empty-state'),
    scanline: $('#scanline'), phoneBanner: $('#phone-banner'), phoneClock: $('#phone-clock'),
    upload: $('#btn-upload'), shutter: $('#btn-shutter'), rerun: $('#btn-rerun'),
    fileInput: $('#file-input'), cameraInput: $('#camera-input'), samples: $('#samples'),
    summaryVerdict: $('#summary-verdict'), summaryHeadline: $('#summary-headline'), summaryDetail: $('#summary-detail'),
    votes: $('#votes'), retryFailed: $('#retry-failed'), retryFailedText: $('#retry-failed-text'),
    summarySpeed: $('#summary-speed'), speedHeadline: $('#speed-headline'), speedDetail: $('#speed-detail'),
    summaryTruth: $('#summary-truth'), truthGroup: $('#summary-truth .truth-buttons'),
    truthButtons: document.querySelectorAll('.truth-btn'), truthDetail: $('#truth-detail'),
    cards: $('.cards'), scoreBody: $('#score-body'), scoreFoot: $('#score-foot'), scoreReset: $('#score-reset'),
    dropOverlay: $('#drop-overlay'), confetti: $('#confetti'), toast: $('#toast'),
  };

  // ---------- DOM + format helpers ----------
  /** Tiny element builder: el('p', { class: 'x', text: 'hi' }, [child, 'text', null]). */
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, val] of Object.entries(attrs)) {
      if (key === 'class') node.className = val;
      else if (key === 'text') node.textContent = val;
      else if (key === 'html') node.innerHTML = val; // only ever used for the static ICONS below
      else if (val === true) node.setAttribute(key, '');
      else if (val !== false && val != null) node.setAttribute(key, val);
    }
    for (const child of children) if (child) node.append(child);
    return node;
  }

  /** Write text only when it changes (no needless layout/paint on every health poll). */
  function setText(node, text) {
    if (node.textContent !== text) node.textContent = text;
  }

  function fmtMs(ms) {
    if (ms == null || Number.isNaN(ms)) return '—';
    if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
    if (ms >= 100) return Math.round(ms) + ' ms';
    return ms.toFixed(1) + ' ms';
  }
  const fmtPct = (p) => (p * 100).toFixed(1) + '%';
  const verdictOf = (res) => (res.is_hotdog ? 'hotdog' : 'not_hotdog');
  const namesOf = (ids) => ids.map((id) => MODELS[id].name);
  const nonEmpty = (v) => (typeof v === 'string' && v.trim() ? v.trim() : '');

  function joinNames(list) {
    if (list.length <= 1) return list.join('');
    return list.slice(0, -1).join(', ') + ' and ' + list[list.length - 1];
  }

  let toastTimer = 0;
  function toast(msg) {
    dom.toast.textContent = msg;
    dom.toast.classList.add('is-on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => dom.toast.classList.remove('is-on'), 3800);
  }

  const ICONS = {
    hot: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    not: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 6.5l11 11m0-11-11 11" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round"/></svg>',
    split: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v16M5 8h14M5 8l-3 6h6zM19 8l-3 6h6z" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/></svg>',
    err: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v5m0 3.5v.5" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>',
  };

  /** kind: hot | not | split | err */
  function bannerEl(kind, text, extraClass = '') {
    return el('div', { class: `banner banner--${kind} ${extraClass}`.trim() }, [
      el('span', { class: 'banner__icon', html: ICONS[kind] }),
      el('span', { class: 'banner__text', text }),
    ]);
  }

  // ---------- storage (localStorage can throw: private mode, quota, disabled) ----------
  function loadJSON(key) {
    try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch { return null; }
  }
  function saveJSON(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch { /* best effort */ }
  }

  // ---------- network ----------
  /**
   * fetch() that gives up after `ms` and can also be cancelled by a parent signal (e.g. the round).
   * Rejects with err.kind = 'timeout' | 'aborted' | 'network'.
   */
  async function fetchWithTimeout(url, opts, ms, parentSignal) {
    const ctrl = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, ms);
    const onParentAbort = () => ctrl.abort();
    if (parentSignal?.aborted) ctrl.abort();
    else parentSignal?.addEventListener('abort', onParentAbort, { once: true });
    try {
      return await fetch(url, { ...opts, signal: ctrl.signal });
    } catch (e) {
      const kind = timedOut ? 'timeout' : ctrl.signal.aborted ? 'aborted' : 'network';
      const message = timedOut ? `No answer after ${Math.round(ms / 1000)} s (browser timeout)` : e.message || 'Network error';
      throw Object.assign(new Error(message), { kind });
    } finally {
      clearTimeout(timer);
      parentSignal?.removeEventListener('abort', onParentAbort);
    }
  }

  /** POST the image to one model. Resolves to a normalized result; rejects with err.kind/status. */
  async function requestClassify(id, upload, signal) {
    const form = new FormData();
    form.append('file', upload.blob, upload.name);
    const t0 = performance.now();
    const r = await fetchWithTimeout(MODELS[id].url + '/classify', { method: 'POST', body: form }, CFG.classifyTimeoutMs, signal);
    const body = await r.json().catch(() => null);
    if (!r.ok) {
      throw Object.assign(new Error(body?.detail || r.statusText || 'Request failed'), { kind: 'http', status: r.status });
    }
    if (!body) throw Object.assign(new Error('Invalid JSON response'), { kind: 'http' });
    return normalizeResult(body, performance.now() - t0);
  }

  function normalizeResult(body, rttMs) {
    const p = body.probabilities || {};
    const isHot = typeof body.is_hotdog === 'boolean' ? body.is_hotdog : body.label === 'hotdog';
    const pHot = typeof p.hotdog === 'number' ? p.hotdog : (isHot ? body.confidence : 1 - body.confidence);
    return {
      is_hotdog: isHot,
      confidence: typeof body.confidence === 'number' ? body.confidence : (isHot ? pHot : 1 - pHot),
      probabilities: { hotdog: pHot, not_hotdog: 1 - pHot },
      latency_ms: body.latency_ms,
      total_ms: typeof body.total_ms === 'number' ? body.total_ms : body.latency_ms,
      rtt_ms: rttMs,
      // LLM extras (absent for the classifiers)
      reason: nonEmpty(body.reason) || null,
      confidence_source: body.confidence_source || null,
      attempts: typeof body.attempts === 'number' ? body.attempts : 1,
    };
  }

  // ---------- image prep ----------
  // The same image goes to three servers, so shrink big photos once here (EXIF orientation applied)
  // instead of uploading a 5–12 MB original three times. Small JPEG/PNG/WebP files are sent untouched.

  function canvasToBlob(source, w, h, type, quality) {
    const canvas = el('canvas', { width: w, height: h });
    const ctx = canvas.getContext('2d');
    if (type === 'image/jpeg') { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h); } // no alpha in JPEG
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, w, h);
    return new Promise((resolve, reject) => {
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode image'))), type, quality);
    });
  }

  /** Decode via <img> (also handles SVG and formats createImageBitmap rejects) and encode as PNG. */
  function rasterize(src, size) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const w = size || img.naturalWidth || 512;
        const h = size || img.naturalHeight || 512;
        canvasToBlob(img, w, h, 'image/png').then(resolve, reject);
      };
      img.onerror = () => reject(new Error('Could not decode image'));
      img.src = src;
    });
  }

  async function prepareUpload(file, objectUrl) {
    const supported = UPLOAD_TYPES.has(file.type);
    if (supported && file.size <= SMALL_UPLOAD_BYTES) return file; // servers apply EXIF orientation themselves
    let bitmap = null;
    try { bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch { /* fall back below */ }
    if (!bitmap) return supported ? file : rasterize(objectUrl);
    try {
      const scale = Math.min(1, CFG.uploadMaxSide / Math.max(bitmap.width, bitmap.height));
      if (scale === 1 && supported) return file;
      const w = Math.max(1, Math.round(bitmap.width * scale));
      const h = Math.max(1, Math.round(bitmap.height * scale));
      const blob = scale < 1
        ? await canvasToBlob(bitmap, w, h, 'image/jpeg', CFG.uploadJpegQuality)
        : await canvasToBlob(bitmap, w, h, 'image/png');
      return supported && blob.size >= file.size ? file : blob;
    } catch {
      return supported ? file : rasterize(objectUrl);
    } finally {
      bitmap.close();
    }
  }

  function uploadName(file, blob) {
    if (blob === file) return file.name || 'upload.' + (file.type.split('/')[1] || 'png');
    const base = (file.name || 'upload').replace(/\.[^.]+$/, '');
    return base + (blob.type === 'image/jpeg' ? '.jpg' : '.png');
  }

  // ---------- health ----------
  // status: loading (checking / still loading weights) | ready | unavailable (not set up) | offline
  const health = Object.fromEntries(ORDER.map((id) => [id, { status: 'loading', data: null }]));

  function healthDetail(id) {
    const d = health[id].data;
    return d ? nonEmpty(d.detail) || nonEmpty(d.error) : '';
  }
  // ready:false with an explanation = not set up / misconfigured; without one = still loading
  function statusOf(data) {
    if (data.ready !== false) return 'ready';
    return nonEmpty(data.detail) || nonEmpty(data.error) ? 'unavailable' : 'loading';
  }
  function healthLabel(id) {
    const h = health[id];
    if (h.status === 'loading') return h.data ? (id === 'llm' ? 'warming up' : 'loading weights') : 'checking…';
    return { ready: 'ready', unavailable: 'not set up', offline: 'offline' }[h.status];
  }
  function healthDevice(id) {
    const d = health[id].data;
    if (!d || health[id].status === 'unavailable') return '';
    return (id === 'llm' ? d.base_url_host || d.device : d.device) || '';
  }
  function cardSub(id) {
    const d = health[id].data;
    if (id === 'llm' && d?.model_id) return d.model_id + (d.base_url_host ? ' via ' + d.base_url_host : '');
    if (id === 'cnn' && d?.arch) return (CNN_ARCH_NAMES[d.arch] || d.arch) + ' · fine-tuned on a laptop';
    return MODELS[id].sub;
  }
  /** A 503 from /classify while /health says the model isn't set up. */
  function isNotSetUp(id, err) {
    return err?.kind === 'http' && err.status === 503 && health[id].status === 'unavailable';
  }
  /** What the card's idle/error slot depends on; re-render only when this changes (keeps focus). */
  function slotKey(id) {
    return `${health[id].status}|${!!health[id].data}|${healthDetail(id)}`;
  }

  function renderHealth(id) {
    const m = MODELS[id];
    const { status } = health[id];
    const label = healthLabel(id);
    const device = healthDevice(id);
    const detail = healthDetail(id);

    const li = dom.health[id];
    li.dataset.status = status;
    setText($('.health__text', li), `${m.name} · ${label}${device ? ' · ' + device : ''}`);
    li.title = m.url + (detail ? ' — ' + detail : '') + (status === 'unavailable' ? ' — fix: ' + m.fix : '');

    const card = cards[id];
    card.pill.dataset.status = status;
    card.pill.title = detail;
    setText(card.pillText, label);
    setText(card.sub, cardSub(id));
    setText(card.device, device || '—');
    card.device.title = device;

    if (card.shown !== slotKey(id)) {
      const state = card.root.dataset.state;
      if (state === 'idle') setCardIdle(id);
      else if (state === 'error' && card.err.status === 503) {
        setCardError(id, card.err);
        // the head-to-head wording depends on it too ("isn't set up" vs "forfeited")
        if (current?.errors[id]) {
          renderVotes(current);
          if (!tally(current).pending.length) onRoundProgress(current);
        }
      }
    }

    const offline = ORDER.filter((k) => health[k].status === 'offline');
    dom.offlineHelp.hidden = offline.length === 0;
    setText(dom.offlineNames, joinNames(offline.map((k) => `${MODELS[k].name} (${MODELS[k].url})`)));
  }

  async function checkHealth(id) {
    try {
      const r = await fetchWithTimeout(MODELS[id].url + '/health', { cache: 'no-store' }, CFG.healthTimeoutMs);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      health[id] = { status: statusOf(data), data };
    } catch {
      health[id] = { status: 'offline', data: null };
    }
    renderHealth(id);
  }

  // Polls all three; slower once everything is ready. Paused while the tab is hidden.
  const healthPoll = { timer: 0, inFlight: false };
  async function pollHealth() {
    clearTimeout(healthPoll.timer);
    if (healthPoll.inFlight) return;
    healthPoll.inFlight = true;
    await Promise.all(ORDER.map(checkHealth));
    healthPoll.inFlight = false;
    if (document.hidden) return;
    const allReady = ORDER.every((id) => health[id].status === 'ready');
    healthPoll.timer = setTimeout(pollHealth, allReady ? CFG.healthIntervalMs : CFG.healthRetryMs);
  }
  function onVisibilityChange() {
    if (document.hidden) clearTimeout(healthPoll.timer);
    else pollHealth(); // refresh right away; states may be stale
  }

  // ---------- cards ----------
  const cards = {};

  function buildCard(id) {
    const m = MODELS[id];
    const root = document.getElementById('card-' + id);
    const probRow = (key, label) => el('div', { class: 'prob prob--' + key }, [
      el('span', { class: 'prob__label', text: label }),
      el('span', { class: 'prob__track' }, [el('span', { class: 'prob__fill' })]),
      el('span', { class: 'prob__val', text: '—' }),
    ]);
    const stat = (key, label, title) => el('div', { class: 'stat', title }, [
      el('dt', { text: label }), el('dd', { class: 'stat--' + key, text: '—' }),
    ]);

    const head = el('header', { class: 'card__head' }, [
      el('div', {}, [
        el('h3', { class: 'card__title', id: `card-${id}-title` }, [el('span', { class: 'swatch swatch--' + id, 'aria-hidden': 'true' }), m.name]),
        el('p', { class: 'card__sub', text: m.sub }),
      ]),
      el('div', { class: 'card__tags' }, [
        el('span', { class: 'pill pill--status', 'data-status': 'loading' }, [
          el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { class: 'pill__text', text: 'checking' }),
        ]),
      ]),
    ]);
    const slot = el('div', { class: 'card__slot' });
    const probs = el('div', { class: 'probs', role: 'group', 'aria-label': m.name + ' probabilities' }, [
      probRow('hot', 'Hotdog'), probRow('not', 'Not hotdog'),
    ]);
    const stats = el('dl', { class: 'stats' }, [
      stat('latency', m.latencyLabel || 'Model', m.latencyTitle || 'latency_ms: model forward pass only'),
      stat('total', 'Total', 'total_ms: incl. decode / preprocess'),
      stat('rtt', 'Round trip', 'Measured in the browser, incl. network + upload'),
      stat('device', m.deviceLabel || 'Device', m.deviceTitle || 'Reported by /health'),
    ]);
    root.replaceChildren(head, slot, probs, stats);

    const probParts = (key) => ({ fill: $(`.prob--${key} .prob__fill`, probs), val: $(`.prob--${key} .prob__val`, probs) });
    cards[id] = {
      root, slot,
      sub: $('.card__sub', head),
      pill: $('.pill--status', head),
      pillText: $('.pill__text', head),
      prob: { hot: probParts('hot'), not: probParts('not') },
      stat: { latency: $('.stat--latency', stats), total: $('.stat--total', stats), rtt: $('.stat--rtt', stats) },
      device: $('.stat--device', stats),
      shown: '', err: null,
    };
    setCardIdle(id);
  }

  /** Swap the card's state and slot content in one go. */
  function setCardSlot(id, state, ...children) {
    const c = cards[id];
    stopLoadingClock(id);
    c.root.dataset.state = state;
    c.root.setAttribute('aria-busy', String(state === 'loading'));
    c.slot.replaceChildren(...children.filter(Boolean));
  }

  function setProbs(id, pHot) {
    const { prob } = cards[id];
    for (const [key, p] of [['hot', pHot], ['not', pHot == null ? null : 1 - pHot]]) {
      prob[key].fill.style.width = p == null ? '0%' : (p * 100).toFixed(1) + '%';
      prob[key].val.textContent = p == null ? '—' : fmtPct(p);
    }
  }
  function setStats(id, res) {
    const { stat } = cards[id];
    stat.latency.textContent = res ? fmtMs(res.latency_ms) : '—';
    stat.total.textContent = res ? fmtMs(res.total_ms) : '—';
    stat.rtt.textContent = res ? fmtMs(res.rtt_ms) : '—';
  }
  function clearNumbers(id) {
    setProbs(id, null);
    setStats(id, null);
  }

  function setCardIdle(id) {
    const h = health[id];
    cards[id].shown = slotKey(id);
    let box;
    if (h.status === 'offline') box = offlineBox(id);
    else if (h.status === 'unavailable') box = notSetUpBox(id);
    else {
      box = el('div', { class: 'slot-idle' }, [
        el('span', { class: 'slot-idle__plate', 'aria-hidden': 'true' }),
        el('span', { text: h.status === 'loading' && h.data ? 'Warming up…' : 'Awaiting a snack…' }),
      ]);
    }
    setCardSlot(id, 'idle', box);
    clearNumbers(id);
  }

  function offlineBox(id) {
    const m = MODELS[id];
    return el('div', { class: 'slot-offline' }, [
      el('strong', { text: m.name + ' is offline' }),
      el('span', { text: `Couldn’t reach ${m.url}. Start everything with:` }),
      cmdLine('./run.sh'),
      el('span', { class: 'muted', text: 'Running already? Check CORS.' }),
    ]);
  }

  // The server is up but answered ready:false with a reason: missing weights, no API key, …
  function notSetUpBox(id) {
    const m = MODELS[id];
    const detail = healthDetail(id);
    return el('div', { class: 'slot-setup' }, [
      el('strong', { class: 'slot-setup__title', text: m.name + ' isn’t set up yet' }),
      detail && el('p', { class: 'slot-setup__detail', title: detail }, [el('span', { text: detail })]),
      el('span', { text: 'To fix it, run:' }),
      cmdLine(m.fix),
      el('span', { class: 'slot-setup__hint', text: m.fixWhat }),
    ]);
  }

  /** A shell command with a copy button (click handled by delegation in wireCards). */
  function cmdLine(cmd) {
    return el('div', { class: 'cmd' }, [
      el('span', { class: 'cmd__prompt', 'aria-hidden': 'true', text: '$' }),
      el('code', { class: 'cmd__text', text: cmd }),
      el('button', { type: 'button', class: 'cmd__copy', 'aria-label': 'Copy command: ' + cmd, text: 'Copy' }),
    ]);
  }

  async function copyCommand(btn) {
    const code = $('.cmd__text', btn.closest('.cmd'));
    const done = () => {
      btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
    };
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(code.textContent); done(); } catch { toast('Couldn’t copy. Select the command instead.'); }
      return;
    }
    // http:// on a LAN IP is not a secure context: select the text and use the legacy copy
    const range = document.createRange();
    range.selectNodeContents(code);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    try { document.execCommand('copy'); done(); } catch { toast('Press Ctrl/Cmd+C to copy.'); }
  }

  // One rAF loop drives the stopwatch of every loading card.
  const loadingClocks = new Map(); // id -> { node, t0 }
  let loadingRaf = 0;
  function tickLoadingClocks() {
    const now = performance.now();
    loadingClocks.forEach(({ node, t0 }) => { node.textContent = fmtMs(now - t0); });
    loadingRaf = loadingClocks.size ? requestAnimationFrame(tickLoadingClocks) : 0;
  }
  function startLoadingClock(id, node) {
    loadingClocks.set(id, { node, t0: performance.now() });
    if (!loadingRaf) loadingRaf = requestAnimationFrame(tickLoadingClocks);
  }
  function stopLoadingClock(id) {
    loadingClocks.delete(id);
  }

  function setCardLoading(id) {
    const clock = el('span', { class: 'slot-loading__clock', text: '0 ms' });
    setCardSlot(id, 'loading', el('div', { class: 'slot-loading' }, [
      el('span', { class: 'spinner', 'aria-hidden': 'true' }),
      el('span', { class: 'slot-loading__text', text: MODELS[id].loading }),
      clock,
    ]));
    clearNumbers(id);
    startLoadingClock(id, clock);
  }

  function setCardResult(id, res) {
    const kind = res.is_hotdog ? 'hot' : 'not';
    const banner = bannerEl(kind, res.is_hotdog ? 'Hotdog!' : 'Not hotdog!', 'banner--card');
    banner.append(el('span', { class: 'banner__conf', text: fmtPct(res.confidence) + ' sure' }));
    // Classifiers don't explain themselves; the placeholder only shows in the 3-up layout, opposite the LLM's reason.
    const extras = resultExtras(res) || el('p', { class: 'says-none', 'aria-hidden': 'true' }, [
      el('span', { class: 'says__kicker', text: 'The model says' }), el('span', { text: 'No comment. Just the numbers.' }),
    ]);
    setCardSlot(id, kind, banner, extras);
    requestAnimationFrame(() => setProbs(id, res.probabilities.hotdog)); // let the bars animate from 0
    setStats(id, res);
  }

  // LLM extras: the one-sentence reason, where the confidence number came from, automatic retries
  const CONF_SOURCES = {
    logprobs: {
      text: 'token logprobs', cls: 'pill--logprobs',
      title: 'Confidence computed from the API’s token probabilities for the answer.',
    },
    'self-reported': {
      text: 'self-reported', cls: 'pill--self', hint: 'grain of salt',
      title: 'The LLM picked this number itself. Chatbots are famously confident; take it with a grain of salt.',
    },
  };
  function resultExtras(res) {
    const retried = res.attempts > 1;
    if (!res.reason && !res.confidence_source && !retried) return null;
    const box = el('div', { class: 'says' });
    if (res.reason) {
      box.append(el('figure', { class: 'says__bubble' }, [
        el('figcaption', { class: 'says__kicker', text: 'The model says' }),
        el('blockquote', { class: 'says__quote', text: `“${res.reason}”` }),
      ]));
    }
    const meta = [];
    if (res.confidence_source) {
      const src = CONF_SOURCES[res.confidence_source]
        || { text: res.confidence_source, cls: '', title: 'confidence_source: ' + res.confidence_source };
      meta.push(
        el('span', { text: 'Confidence:' }),
        el('span', { class: `pill pill--src ${src.cls}`.trim(), title: src.title, tabindex: '0', 'aria-label': `Confidence source: ${src.text}. ${src.title}`, text: src.text }),
        src.hint && el('span', { class: 'says__hint', 'aria-hidden': 'true', title: src.title, text: src.hint }),
      );
    }
    if (retried) {
      const title = `The provider was busy or rate-limited, so the server retried automatically (${res.attempts} attempts).`;
      meta.push(el('span', { class: 'pill pill--src pill--attempts', title, tabindex: '0', 'aria-label': title, text: `${res.attempts} attempts` }));
    }
    if (meta.length) box.append(el('p', { class: 'says__meta' }, meta));
    return box;
  }

  const ERROR_TITLES = { 502: 'Upstream hiccup', 504: 'Timed out' };
  function errorLead(id, code) {
    const m = MODELS[id];
    switch (code) {
      case 503: return health[id].status === 'ready'
        ? 'Hit Retry to get its verdict on this image.'
        : `${m.name} is still loading${id === 'llm' ? '' : ' weights'}. Give it a moment, then retry.`;
      case 502: return (id === 'llm' ? 'The vision API' : 'The upstream service') + ' reported a problem.';
      case 504: return (id === 'llm' ? 'The chatbot' : m.name) + ' took too long to answer.';
      default: return '';
    }
  }

  function setCardError(id, err) {
    const c = cards[id];
    c.err = err;
    c.shown = slotKey(id);
    let content;
    if (err.kind === 'network') content = [offlineBox(id)];
    else if (isNotSetUp(id, err)) content = [notSetUpBox(id)];
    else {
      const code = err.kind === 'timeout' ? 504 : err.status;
      const title = code === 503 ? (health[id].status === 'ready' ? 'Ready now' : 'Still warming up') : ERROR_TITLES[code] || 'Error';
      const lead = errorLead(id, code);
      const showDetail = !lead || (code !== 503 && err.message);
      content = [
        bannerEl('err', title, 'banner--card'),
        el('div', { class: 'slot-error' }, [
          lead && el('p', { text: lead }),
          showDetail && el('p', { class: lead ? 'slot-error__detail' : '', text: (err.status ? `HTTP ${err.status}: ` : '') + err.message }),
        ]),
      ];
    }
    const retryBtn = el('button', { type: 'button', class: 'retry-btn', 'data-retry': id, 'aria-label': `Retry ${MODELS[id].name} on this image` }, [
      el('span', { class: 'retry-btn__icon', 'aria-hidden': 'true', text: '↻' }),
      el('span', { text: 'Retry ' + MODELS[id].name }),
    ]);
    setCardSlot(id, 'error', ...content, retryBtn);
    clearNumbers(id);
  }

  function wireCards() {
    dom.cards.addEventListener('click', (e) => {
      const retryBtn = e.target.closest('[data-retry]');
      if (retryBtn) { retry([retryBtn.dataset.retry]); return; }
      const copyBtn = e.target.closest('.cmd__copy');
      if (copyBtn) copyCommand(copyBtn);
    });
  }

  // ---------- rounds ----------
  // A round = one image judged by all three models. Results of a superseded round are ignored and
  // its requests aborted, so they don't hog the servers.
  let current = null;
  let lastUpload = null;

  function newRound(upload) {
    if (current) { current.ctrl.abort(); current.settle(); }
    let settle;
    const done = new Promise((resolve) => { settle = resolve; });
    current = { upload, ctrl: new AbortController(), done, settle, results: {}, errors: {}, truth: null, pendingTruth: null, scored: false, scoreSnap: null, confettiDone: false };
    return current;
  }

  /** Clears the "round finished" UI before (re-)asking models. */
  function resetRoundUI() {
    dom.scanline.hidden = false;
    dom.phoneBanner.hidden = true;
    dom.summarySpeed.hidden = true;
    dom.summaryTruth.hidden = true;
    dom.retryFailed.hidden = true;
  }

  /** Starts a round; the returned promise resolves once every model answered or failed (or a newer round took over). */
  function runRound(upload) {
    lastUpload = upload;
    const round = newRound(upload);
    dom.rerun.disabled = false;
    resetRoundUI();
    dom.phoneBanner.replaceChildren();
    setSummary('pending', 'Judging…', `All three models are looking at ${upload.label}. The CNN usually answers first.`);
    ORDER.forEach(setCardLoading);
    renderVotes(round);
    ORDER.forEach((id) => classifyOne(id, round));
    return round.done;
  }

  async function classifyOne(id, round) {
    let res = null;
    let err = null;
    try { res = await requestClassify(id, round.upload, round.ctrl.signal); } catch (e) { err = e; }
    if (round !== current) return; // superseded (or aborted) round

    if (res) {
      round.results[id] = res;
      setCardResult(id, res);
      if (res.is_hotdog && !round.confettiDone) { round.confettiDone = true; confetti(); }
    } else {
      round.errors[id] = err;
      setCardError(id, err);
      if (err.kind === 'network') { health[id] = { status: 'offline', data: null }; renderHealth(id); }
      else if (err.status === 503) checkHealth(id); // "not set up" or still loading? /health knows
    }
    renderVotes(round);
    onRoundProgress(round);
  }

  function tally(round) {
    const ok = ORDER.filter((k) => round.results[k]);
    return {
      ok,
      hot: ok.filter((k) => round.results[k].is_hotdog),
      not: ok.filter((k) => !round.results[k].is_hotdog),
      failed: ORDER.filter((k) => round.errors[k]),
      pending: ORDER.filter((k) => !round.results[k] && !round.errors[k]),
    };
  }

  // ---------- summary ----------
  const says = (ids, verdict) => namesOf(ids).join(' + ') + (ids.length > 1 ? ' say ' : ' says ') + verdict;

  /** Majority first; the minority "says not" / "says hotdog". */
  function splitPhrase(t) {
    const hotFirst = t.hot.length >= t.not.length;
    const [maj, min] = hotFirst ? [t.hot, t.not] : [t.not, t.hot];
    return says(maj, hotFirst ? 'hotdog' : 'not hotdog') + ', ' + says(min, hotFirst ? 'not' : 'hotdog');
  }

  /** " CLEF-Flash isn't set up (forfeit). CNN forfeited (error)." (leading space, or '') */
  function forfeitNote(round, t) {
    const unset = t.failed.filter((k) => isNotSetUp(k, round.errors[k]));
    const other = t.failed.filter((k) => !unset.includes(k));
    const parts = [];
    if (unset.length) parts.push(joinNames(namesOf(unset)) + (unset.length > 1 ? ' aren’t' : ' isn’t') + ' set up (forfeit).');
    if (other.length) parts.push(joinNames(namesOf(other)) + ' forfeited (error).');
    return parts.length ? ' ' + parts.join(' ') : '';
  }

  function setSummary(kind, headline, detail) {
    dom.summaryVerdict.dataset.kind = kind;
    dom.summaryHeadline.textContent = headline;
    dom.summaryDetail.textContent = detail;
  }

  function buildVotes() {
    dom.votes.replaceChildren(...ORDER.map((id) => el('li', { class: 'vote', 'data-model': id, 'data-vote': 'pending' }, [
      el('span', { class: 'swatch swatch--' + id, 'aria-hidden': 'true' }),
      el('span', { class: 'vote__name', text: MODELS[id].name }),
      el('span', { class: 'vote__val', text: 'Thinking…' }),
    ])));
  }

  const VOTE_TEXT = { hot: 'Hotdog', not: 'Not hotdog', err: 'Forfeit', pending: 'Thinking…' };
  /** Updates the vote chips in place (rebuilding them would re-announce the live region). */
  function renderVotes(round) {
    for (const li of dom.votes.children) {
      const id = li.dataset.model;
      const res = round.results[id];
      const err = round.errors[id];
      const vote = res ? (res.is_hotdog ? 'hot' : 'not') : err ? 'err' : 'pending';
      const unset = isNotSetUp(id, err);
      li.dataset.vote = vote;
      li.classList.toggle('vote--unset', unset);
      setText($('.vote__val', li), unset ? 'Not set up' : VOTE_TEXT[vote]);
    }
    dom.votes.hidden = false;
  }

  function onRoundProgress(round) {
    const t = tally(round);
    if (t.pending.length) renderPendingSummary(round, t);
    else finishRound(round, t);
  }

  function renderPendingSummary(round, t) {
    const pendingNames = joinNames(namesOf(t.pending));
    const waiting = `Waiting for ${pendingNames} to finish thinking…`;
    if (!t.ok.length) {
      if (t.failed.length) setSummary('pending', 'Judging…', forfeitNote(round, t).trim() + ' ' + waiting);
    } else if (t.hot.length && t.not.length) {
      setSummary('pending', `${t.hot.length}–${t.not.length} so far`,
        `${splitPhrase(t)}. ${pendingNames}${t.pending.length > 1 ? ' hold' : ' holds'} the deciding vote…`);
    } else {
      setSummary('pending', says(t.ok, t.hot.length ? 'hotdog' : 'not hotdog'), waiting);
    }
  }

  function finishRound(round, t) {
    round.settle();
    dom.scanline.hidden = true;
    scoreRoundStats(round);
    renderRetryFailed(t);
    scrollToVerdictOnMobile();
    dom.phoneBanner.replaceChildren(verdictBanner(round, t));
    dom.phoneBanner.hidden = false;
    if (!t.ok.length) return;

    renderSpeed(round, t);
    dom.summaryTruth.hidden = false;
    if (round.pendingTruth) {
      const truth = round.pendingTruth;
      round.pendingTruth = null;
      applyTruth(round, truth);
    }
    renderTruth(round);
  }

  /** Sets the head-to-head summary and returns the matching banner for the phone. */
  function verdictBanner(round, t) {
    const forfeits = forfeitNote(round, t);
    if (!t.ok.length) {
      if (t.failed.every((k) => isNotSetUp(k, round.errors[k]))) {
        setSummary('error', 'Nobody’s set up yet', 'Each card shows the command that fixes it.');
      } else {
        setSummary('error', 'Nobody showed up', 'All three backends failed. See the cards for details, or retry them all.');
      }
      return bannerEl('err', 'No verdict');
    }
    const isHot = t.hot.length > 0;
    const kind = isHot ? 'hot' : 'not';
    const verdict = isHot ? 'Hotdog' : 'Not hotdog';
    if (t.ok.length === 1) {
      setSummary(kind, `${MODELS[t.ok[0]].name} rules alone: ${verdict}`, forfeits.trim() + ' A walkover is still a win.');
      return bannerEl(kind, verdict + '!');
    }
    if (!(t.hot.length && t.not.length)) { // unanimous among those who answered
      const sure = t.ok.map((k) => `${MODELS[k].name} ${fmtPct(round.results[k].confidence)}`).join(', ');
      setSummary(kind, (t.ok.length === ORDER.length ? 'Unanimous: ' : 'Models agree: ') + verdict, `${sure} sure.${forfeits}`);
      return bannerEl(kind, verdict + '!');
    }
    if (t.hot.length !== t.not.length) {
      const majHot = t.hot.length > t.not.length;
      const headline = majHot ? `${t.hot.length}–${t.not.length} split: Hotdog` : `${t.not.length}–${t.hot.length} split: Not hotdog`;
      setSummary('split', headline, `${splitPhrase(t)}. Majority says ${majHot ? 'hotdog' : 'not hotdog'}, but you be the judge.${forfeits}`);
    } else {
      setSummary('split', 'Split decision!', `${splitPhrase(t)}.${forfeits} You’re the tie-breaker.`);
    }
    return bannerEl('split', 'Split decision!');
  }

  function renderSpeed(round, t) {
    if (t.ok.length < 2) return;
    const totalOf = (k) => round.results[k].total_ms;
    const byTime = [...t.ok].sort((a, b) => totalOf(a) - totalOf(b));
    const fast = byTime[0];
    const slow = byTime[byTime.length - 1];
    const factor = totalOf(fast) > 0 ? totalOf(slow) / totalOf(fast) : Infinity;
    dom.speedHeadline.textContent = `${MODELS[fast].name} was ${factor >= 10 ? Math.round(factor) : factor.toFixed(1)}× faster than ${MODELS[slow].name}`;
    dom.speedDetail.textContent = byTime.map((k) => `${MODELS[k].name} ${fmtMs(totalOf(k))}`).join(' · ')
      + ` total. (${MODELS[slow].name} ${MODELS[slow].slowQuip})`;
    dom.summarySpeed.hidden = false;
  }

  function renderRetryFailed(t) {
    if (!t.failed.length) { dom.retryFailed.hidden = true; return; }
    dom.retryFailedText.textContent = t.failed.length === 1 ? 'Retry ' + MODELS[t.failed[0]].name : `Retry ${t.failed.length} failed`;
    dom.retryFailed.hidden = false;
  }

  // Single-column (phone) layout: the cards stack far below the photo, so once every model has
  // answered, bring the user back up to the verdict banner on the phone mockup.
  function scrollToVerdictOnMobile() {
    if (!media.singleColumn.matches || window.scrollY < 4) return;
    window.scrollTo({ top: 0, behavior: media.reducedMotion.matches ? 'auto' : 'smooth' });
  }

  // ---------- retry ----------
  // Re-asks only the failed model(s) about the same image, inside the same round. If the round was
  // already scored (or graded), those stats are rolled back first and re-applied once the retry lands,
  // so the scoreboard never double-counts.
  function retry(ids) {
    const round = current;
    if (!round) return;
    const failed = ids.filter((id) => round.errors[id] && !round.results[id]);
    if (!failed.length) return;
    if (round.scored) unscoreRoundStats(round);
    if (round.truth) { round.pendingTruth = round.truth; gradeRound(round, round.truth, -1); round.truth = null; saveScore(); }
    for (const id of failed) { delete round.errors[id]; setCardLoading(id); }
    resetRoundUI();
    renderVotes(round);
    onRoundProgress(round);
    for (const id of failed) classifyOne(id, round);
  }

  // ---------- scoreboard ----------
  const nonNegative = (v) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : 0);
  const emptyModelScore = () => ({ calls: 0, sumMs: 0, graded: 0, correct: 0, speedWins: 0 });
  const emptyScore = () => ({ rounds: 0, contested: 0, unanimous: 0, models: Object.fromEntries(ORDER.map((k) => [k, emptyModelScore()])) });

  /** Trust nothing from localStorage: unknown/negative/inconsistent numbers become 0 or get clamped. */
  function sanitizeScore(raw) {
    const s = emptyScore();
    if (!raw || typeof raw !== 'object') return s;
    s.rounds = nonNegative(raw.rounds);
    s.contested = nonNegative(raw.contested);
    s.unanimous = Math.min(nonNegative(raw.unanimous), s.contested);
    const models = raw.models && typeof raw.models === 'object' ? raw.models : {};
    for (const k of ORDER) {
      const m = models[k];
      if (!m || typeof m !== 'object') continue;
      for (const field of Object.keys(s.models[k])) s.models[k][field] = nonNegative(m[field]);
      s.models[k].correct = Math.min(s.models[k].correct, s.models[k].graded);
    }
    return s;
  }
  let score = sanitizeScore(loadJSON(SCORE_KEY));
  const dec = (n, by = 1) => Math.max(0, n - by);

  /** Speed + participation stats, counted once per round when every model has answered or failed. */
  function scoreRoundStats(round) {
    if (round.scored) return;
    round.scored = true;
    const t = tally(round);
    const snap = { ms: {}, contested: false, unanimous: false, fast: null };
    score.rounds++;
    for (const k of t.ok) {
      snap.ms[k] = round.results[k].total_ms || 0;
      score.models[k].calls++;
      score.models[k].sumMs += snap.ms[k];
    }
    if (t.ok.length >= 2) {
      score.contested++; snap.contested = true;
      if (!(t.hot.length && t.not.length)) { score.unanimous++; snap.unanimous = true; }
      const fast = t.ok.reduce((a, b) => (round.results[b].total_ms < round.results[a].total_ms ? b : a));
      score.models[fast].speedWins++; snap.fast = fast;
    }
    round.scoreSnap = snap;
    saveScore();
  }

  function unscoreRoundStats(round) {
    const snap = round.scoreSnap;
    if (!round.scored || !snap) return;
    score.rounds = dec(score.rounds);
    for (const [k, ms] of Object.entries(snap.ms)) {
      const m = score.models[k];
      m.calls = dec(m.calls);
      m.sumMs = dec(m.sumMs, ms);
    }
    if (snap.contested) score.contested = dec(score.contested);
    if (snap.unanimous) score.unanimous = dec(score.unanimous);
    if (snap.fast) score.models[snap.fast].speedWins = dec(score.models[snap.fast].speedWins);
    round.scored = false;
    round.scoreSnap = null;
    saveScore();
  }

  /** Adds (+1) or removes (-1) one grade per model that answered. */
  function gradeRound(round, truth, sign) {
    for (const k of ORDER) {
      const res = round.results[k];
      if (!res) continue;
      const m = score.models[k];
      const right = verdictOf(res) === truth;
      if (sign > 0) { m.graded++; if (right) m.correct++; } else { m.graded = dec(m.graded); if (right) m.correct = dec(m.correct); }
    }
  }

  function applyTruth(round, truth) {
    if (round.truth === truth) return;
    if (round.truth) gradeRound(round, round.truth, -1);
    gradeRound(round, truth, +1);
    round.truth = truth;
    saveScore();
    renderTruth(round);
  }

  function renderTruth(round) {
    for (const btn of dom.truthButtons) btn.setAttribute('aria-pressed', String(btn.dataset.truth === round.truth));
    if (!round.truth) { dom.truthDetail.textContent = 'Tell us the truth to keep score.'; return; }
    const parts = ORDER.filter((k) => round.results[k])
      .map((k) => `${MODELS[k].name} ${verdictOf(round.results[k]) === round.truth ? 'got it right' : 'got it wrong'}`);
    dom.truthDetail.textContent = `Recorded. ${parts.join(', ')}.`;
  }

  function saveScore() {
    saveJSON(SCORE_KEY, score);
    renderScore();
  }

  function renderScore() {
    const accs = ORDER.map((k) => { const m = score.models[k]; return m.graded ? m.correct / m.graded : null; });
    const best = Math.max(...accs.map((a) => (a == null ? -1 : a)));
    const soleLeader = accs.filter((a) => a === best).length === 1;
    dom.scoreBody.replaceChildren(...ORDER.map((k, i) => {
      const m = score.models[k];
      const acc = accs[i];
      const lead = acc != null && acc === best && soleLeader;
      return el('tr', { class: lead ? 'is-leader' : '' }, [
        el('th', { scope: 'row' }, [
          el('span', { class: 'swatch swatch--' + k, 'aria-hidden': 'true' }), MODELS[k].name,
          lead && el('span', { class: 'crown', text: 'leads' }),
        ]),
        el('td', {}, [
          el('span', { class: 'acc' }, [
            el('span', { class: 'acc__track' }, [el('span', { class: 'acc__fill', style: `width:${acc == null ? 0 : acc * 100}%` })]),
            el('span', { text: acc == null ? '—' : Math.round(acc * 100) + '%' }),
          ]),
        ]),
        el('td', { text: `${m.correct} / ${m.graded}` }),
        el('td', { text: m.calls ? fmtMs(m.sumMs / m.calls) : '—' }),
        el('td', { text: String(m.speedWins) }),
      ]);
    }));
    dom.scoreFoot.textContent = score.rounds
      ? `${score.rounds} round${score.rounds === 1 ? '' : 's'} played`
        + (score.contested ? ` · unanimous verdict in ${score.unanimous} of ${score.contested}` : '')
      : 'No rounds yet. Judge a photo, then record the truth to track accuracy.';
  }

  function resetScore() {
    score = emptyScore();
    if (current) current.truth = null;
    saveScore();
    if (current) renderTruth(current);
    toast('Scoreboard reset. Fresh buns.');
  }

  // ---------- inputs ----------
  // Every input (pick, drop, paste, camera, sample) takes a ticket; preparing an image is async, so a
  // slow decode must not override something the user chose afterwards. Latest input wins.
  let inputSeq = 0;
  let previewUrl = null;
  function showPreview(url, alt) {
    if (previewUrl?.startsWith('blob:') && previewUrl !== url) URL.revokeObjectURL(previewUrl);
    previewUrl = url;
    dom.preview.src = url;
    dom.preview.alt = 'Photo being judged: ' + alt;
    dom.preview.hidden = false;
    dom.viewfinder.style.setProperty('--bg-img', `url("${url}")`);
    dom.emptyState.classList.add('is-hidden');
    dom.emptyState.inert = true; // invisible now: keep it out of the tab order
  }

  async function handleFile(file, label = 'your photo') {
    if (!file) return;
    if (!file.type.startsWith('image/')) { toast('That is not an image. Even a hotdog has standards.'); return; }
    const seq = ++inputSeq;
    const url = URL.createObjectURL(file);
    let blob = null;
    try { blob = await prepareUpload(file, url); } catch { /* reported below unless superseded */ }
    if (seq !== inputSeq || !blob) {
      URL.revokeObjectURL(url);
      if (seq === inputSeq) toast('Couldn’t read that image format. Try JPEG, PNG or WebP.');
      return;
    }
    showPreview(url, label === 'your photo' && file.name ? file.name : label);
    runRound({ blob, name: uploadName(file, blob), label });
  }

  function openPicker(preferCamera) {
    (preferCamera && media.coarsePointer.matches ? dom.cameraInput : dom.fileInput).click();
  }

  function onPaste(e) {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === 'file' && i.type.startsWith('image/'));
    if (!item) return;
    e.preventDefault();
    handleFile(item.getAsFile(), 'a pasted image');
  }

  function wireDragAndDrop() {
    let depth = 0; // dragenter/dragleave fire for every child element
    const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
    const hide = () => { depth = 0; dom.dropOverlay.classList.remove('is-on'); };
    window.addEventListener('dragenter', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth++;
      dom.dropOverlay.classList.add('is-on');
    });
    window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) hide(); });
    window.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      hide();
      handleFile(e.dataTransfer.files[0]);
    });
  }

  function wireInputs() {
    const onPick = (label) => (e) => { handleFile(e.target.files[0], label); e.target.value = ''; };
    dom.fileInput.addEventListener('change', onPick());
    dom.cameraInput.addEventListener('change', onPick('a fresh snap'));
    dom.upload.addEventListener('click', () => openPicker(false));
    dom.emptyState.addEventListener('click', () => openPicker(false));
    dom.shutter.addEventListener('click', () => { dom.shutter.classList.add('is-firing'); openPicker(true); });
    dom.shutter.addEventListener('animationend', () => dom.shutter.classList.remove('is-firing'));
    dom.rerun.addEventListener('click', () => { if (lastUpload) runRound(lastUpload); });
    dom.retryFailed.addEventListener('click', () => { if (current) retry(tally(current).failed); });
    dom.truthGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-truth]');
      if (btn && current) applyTruth(current, btn.dataset.truth);
    });
    dom.scoreReset.addEventListener('click', resetScore);
    document.addEventListener('paste', onPaste);
    wireDragAndDrop();
  }

  // ---------- samples (original, hand-drawn SVG) ----------
  const SAMPLES = [
    { id: 'hotdog', label: 'a cartoon hotdog', title: 'Hotdog', svg:
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><rect width="200" height="200" fill="#ffe9a8"/>' +
      '<circle cx="100" cy="100" r="80" fill="#fff6d6"/>' +
      '<ellipse cx="100" cy="140" rx="78" ry="10" fill="#000" opacity=".08"/>' +
      '<g transform="rotate(-18 100 105)">' +
      '<rect x="22" y="100" width="156" height="34" rx="17" fill="#d99a52"/>' +
      '<rect x="12" y="86" width="176" height="28" rx="14" fill="#c4442c"/>' +
      '<rect x="20" y="90" width="150" height="7" rx="3.5" fill="#e0715a" opacity=".7"/>' +
      '<path d="M30 100q8-10 16 0t16 0 16 0 16 0 16 0 16 0 16 0 16 0 16 0" stroke="#ffd21f" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>' +
      '<rect x="22" y="70" width="156" height="30" rx="15" fill="#e8ae63"/>' +
      '<rect x="34" y="74" width="120" height="6" rx="3" fill="#f6cf8e"/>' +
      '</g></svg>' },
    { id: 'pizza', label: 'a pizza slice', title: 'Pizza', svg:
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><rect width="200" height="200" fill="#d8ecff"/>' +
      '<circle cx="100" cy="100" r="80" fill="#eef7ff"/>' +
      '<path d="M100 172 34 46q66-30 132 0z" fill="#ffcf4d"/>' +
      '<path d="M100 172 34 46q66-30 132 0z" fill="none" stroke="#f2b134" stroke-width="4" stroke-linejoin="round"/>' +
      '<path d="M30 44q70-34 140 0" stroke="#c9822e" stroke-width="16" fill="none" stroke-linecap="round"/>' +
      '<circle cx="82" cy="78" r="12" fill="#c8362b"/><circle cx="120" cy="74" r="11" fill="#c8362b"/>' +
      '<circle cx="102" cy="112" r="11" fill="#c8362b"/><circle cx="100" cy="148" r="7" fill="#c8362b"/>' +
      '<path d="M66 70q4 6 0 12M136 96q-4 6 0 12M90 132q4 4 0 8" stroke="#4f9a3a" stroke-width="5" fill="none" stroke-linecap="round"/>' +
      '</svg>' },
    { id: 'shoe', label: 'a sneaker', title: 'Shoe', svg:
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><rect width="200" height="200" fill="#e7e2ff"/>' +
      '<circle cx="100" cy="100" r="80" fill="#f3f0ff"/>' +
      '<ellipse cx="100" cy="146" rx="80" ry="8" fill="#000" opacity=".08"/>' +
      '<path d="M22 132V96q0-12 12-14l30-4q10 0 16 10l16 18q18 6 46 8 30 4 34 18v6z" fill="#3f6df2"/>' +
      '<path d="M18 128h166q4 0 4 6v4q0 6-6 6H22q-6 0-6-6v-6q0-4 2-4z" fill="#fff" stroke="#d7d3e8" stroke-width="2"/>' +
      '<path d="M64 84 78 102M74 80l14 18M84 78l14 18" stroke="#fff" stroke-width="4" stroke-linecap="round"/>' +
      '<path d="M120 116q30 2 48 14" stroke="#ffcc29" stroke-width="7" fill="none" stroke-linecap="round"/>' +
      '<circle cx="36" cy="100" r="5" fill="#fff" opacity=".7"/>' +
      '</svg>' },
    { id: 'dachshund', label: 'a very long dog', title: 'Long dog', svg:
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><rect width="200" height="200" fill="#d6f5df"/>' +
      '<circle cx="100" cy="100" r="80" fill="#ecfbf0"/>' +
      '<ellipse cx="100" cy="150" rx="76" ry="7" fill="#000" opacity=".08"/>' +
      '<path d="M26 98q-14-10-8-22" stroke="#9c4f22" stroke-width="7" fill="none" stroke-linecap="round"/>' +
      '<rect x="44" y="128" width="11" height="22" rx="5" fill="#86411b"/><rect x="62" y="128" width="11" height="22" rx="5" fill="#9c4f22"/>' +
      '<rect x="126" y="128" width="11" height="22" rx="5" fill="#86411b"/><rect x="144" y="128" width="11" height="22" rx="5" fill="#9c4f22"/>' +
      '<rect x="22" y="94" width="148" height="42" rx="21" fill="#b05a27"/>' +
      '<rect x="34" y="100" width="110" height="8" rx="4" fill="#c8733c"/>' +
      '<path d="M150 104q2-30 22-34 18-2 22 12 2 8-6 12l-16 4q-6 10-18 10z" fill="#b05a27"/>' +
      '<path d="M162 74q-14 6-10 32 10-4 14-18z" fill="#7a3614"/>' +
      '<circle cx="178" cy="82" r="3.4" fill="#1d1d1f"/><circle cx="194" cy="90" r="4" fill="#1d1d1f"/>' +
      '<rect x="40" y="96" width="14" height="38" fill="#e2342b"/><rect x="40" y="96" width="14" height="38" fill="none" stroke="#a81f19" stroke-width="2"/>' +
      '</svg>' },
  ].map((s) => ({ ...s, url: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(s.svg) }));

  const sampleBlobs = new Map(); // id -> Promise<Blob>, rasterized once

  /** Resolves when the sample's round is complete, or right away if a newer input superseded it. */
  async function playSample(id) {
    const s = SAMPLES.find((x) => x.id === id);
    if (!s) return;
    const seq = ++inputSeq;
    if (!sampleBlobs.has(id)) sampleBlobs.set(id, rasterize(s.url, 512));
    let blob;
    try {
      blob = await sampleBlobs.get(id);
    } catch {
      sampleBlobs.delete(id);
      if (seq === inputSeq) toast('Couldn’t render that sample.');
      return;
    }
    if (seq !== inputSeq) return;
    showPreview(s.url, s.label);
    await runRound({ blob, name: `sample-${s.id}.png`, label: s.label });
  }

  function buildSamples() {
    dom.samples.replaceChildren(...SAMPLES.map((s) => el('li', {}, [
      el('button', { type: 'button', class: 'sample', 'aria-label': 'Try sample: ' + s.label, 'data-sample': s.id }, [
        el('img', { src: s.url, alt: '', width: '72', height: '72' }),
        el('span', { text: s.title }),
      ]),
    ])));
    dom.samples.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-sample]');
      if (btn) playSample(btn.dataset.sample);
    });
  }

  // ---------- confetti ----------
  const CONFETTI_MS = 2600;
  const CONFETTI_COLORS = ['#ffd21f', '#e2342b', '#1fae54', '#ff8a1f', '#ffffff', '#c4442c'];

  function confetti() {
    if (media.reducedMotion.matches) return;
    const cv = dom.confetti;
    const ctx = cv.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = window.innerWidth;
    const H = window.innerHeight;
    cv.width = W * dpr;
    cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const parts = Array.from({ length: 160 }, (_, i) => ({
      x: W * (0.2 + Math.random() * 0.6), y: H * 0.25 + Math.random() * 40,
      vx: (Math.random() - 0.5) * 12, vy: -Math.random() * 12 - 4,
      w: 6 + Math.random() * 8, h: 4 + Math.random() * 5,
      r: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.4,
      color: CONFETTI_COLORS[i % CONFETTI_COLORS.length], sausage: Math.random() < 0.15,
    }));
    const start = performance.now();
    cv.classList.add('is-on');
    const frame = (now) => {
      const t = now - start;
      ctx.clearRect(0, 0, W, H);
      ctx.globalAlpha = Math.max(0, 1 - t / CONFETTI_MS);
      for (const p of parts) {
        p.vy += 0.35; p.vx *= 0.99; p.x += p.vx; p.y += p.vy; p.r += p.vr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.translate(p.x, p.y);
        ctx.rotate(p.r);
        if (p.sausage) {
          ctx.fillStyle = '#e8ae63'; fillRoundRect(ctx, -11, -5, 22, 10, 5);
          ctx.fillStyle = '#c4442c'; fillRoundRect(ctx, -13, -3, 26, 6, 3);
        } else {
          ctx.fillStyle = p.color;
          ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        }
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (t < CONFETTI_MS) requestAnimationFrame(frame);
      else { ctx.clearRect(0, 0, W, H); cv.classList.remove('is-on'); }
    };
    requestAnimationFrame(frame);
  }
  function fillRoundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.fill();
  }

  // ---------- phone status bar clock ----------
  // Real local time in the browser's own hour format, without AM/PM (like a phone status bar).
  const clockFormat = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' });
  function tickClock() {
    const text = clockFormat.formatToParts(new Date())
      .filter((p) => p.type !== 'dayPeriod')
      .map((p) => p.value).join('').trim();
    setText(dom.phoneClock, text);
  }

  // ---------- boot ----------
  ORDER.forEach(buildCard);
  ORDER.forEach(renderHealth);
  buildVotes();
  buildSamples();
  wireCards();
  wireInputs();
  renderScore();

  $('#origin').textContent = window.location.origin;
  $('#footer-urls').textContent = ORDER.map((k) => `${MODELS[k].name} → ${MODELS[k].url}`).join(' · ') + ' · override with ?llm=…&clef=…&cnn=…';

  tickClock();
  setInterval(tickClock, 1000);

  document.addEventListener('visibilitychange', onVisibilityChange);
  pollHealth();

  // Test hook for scripted browser checks: runSample resolves when that round is complete.
  window.__bunTribunal = Object.freeze({ runSample: (id) => playSample(id) });
})();
