/* Bun Tribunal — vanilla JS, no build step. */
(function () {
  'use strict';

  var CFG = window.HOTDOG_CONFIG;
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var ORDER = ['llm', 'clef', 'cnn'];
  var MODELS = {
    clef: {
      id: 'clef', name: 'CLEF-Flash', url: CFG.clef,
      sub: 'Cloudflare · 9B multimodal decision model',
      loading: 'Deliberating with 9B parameters…',
      slowQuip: 'brought more parameters to the table',
      fix: './run.sh download', fixWhat: 'Downloads the weights (~19 GB).',
    },
    cnn: {
      id: 'cnn', name: 'CNN', url: CFG.cnn,
      sub: 'ConvNeXt-Tiny · fine-tuned on a laptop',
      loading: 'Squinting at pixels…',
      slowQuip: 'tripped over its own pooling layers',
      fix: './run.sh train', fixWhat: 'Fine-tunes the CNN on your machine.',
    },
    llm: {
      id: 'llm', name: 'LLM', url: CFG.llm,
      sub: 'General-purpose vision LLM',
      loading: 'Asking a chatbot to squint…',
      slowQuip: 'had to phone a friend in the cloud',
      latencyLabel: 'API', latencyTitle: 'latency_ms: API round trip (network + generation)',
      deviceLabel: 'Host', deviceTitle: 'API host reported by /health',
      fix: './run.sh llm', fixWhat: 'Sets the provider, model and API key. Then restart ./run.sh.',
    },
  };

  // ---------- tiny helpers ----------
  function el(tag, attrs, children) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (attrs[k] === true) n.setAttribute(k, '');
      else if (attrs[k] !== false && attrs[k] != null) n.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return n;
  }
  function fmtMs(ms) {
    if (ms == null || isNaN(ms)) return '—';
    if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
    if (ms >= 100) return Math.round(ms) + ' ms';
    return ms.toFixed(1) + ' ms';
  }
  function fmtPct(p) { return (p * 100).toFixed(1) + '%'; }
  function store(key, val) {
    try {
      if (val === undefined) return JSON.parse(window.localStorage.getItem(key) || 'null');
      window.localStorage.setItem(key, JSON.stringify(val));
    } catch (e) { return null; }
  }
  var toastTimer;
  function toast(msg) {
    var t = $('#toast');
    t.textContent = msg;
    t.classList.add('is-on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove('is-on'); }, 3800);
  }
  function fetchWithTimeout(url, opts, ms) {
    var ctrl = new AbortController();
    var id = setTimeout(function () { ctrl.abort(); }, ms);
    opts = Object.assign({}, opts, { signal: ctrl.signal });
    return fetch(url, opts).finally(function () { clearTimeout(id); });
  }

  var ICONS = {
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    cross: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 6.5l11 11m0-11-11 11" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round"/></svg>',
    split: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v16M5 8h14M5 8l-3 6h6zM19 8l-3 6h6z" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/></svg>',
    warn: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v5m0 3.5v.5" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>',
  };

  function bannerEl(kind, text, extraClass) {
    var cls = { hot: 'banner--hot', not: 'banner--not', split: 'banner--split', err: 'banner--err' }[kind];
    var icon = { hot: ICONS.check, not: ICONS.cross, split: ICONS.split, err: ICONS.warn }[kind];
    var b = el('div', { class: 'banner ' + cls + (extraClass ? ' ' + extraClass : '') });
    b.appendChild(el('span', { class: 'banner__icon', html: icon }));
    b.appendChild(el('span', { class: 'banner__text', text: text }));
    return b;
  }

  // ---------- cards ----------
  var cards = {};
  function buildCard(id) {
    var m = MODELS[id];
    var root = document.getElementById('card-' + id);
    root.innerHTML = '';
    var head = el('header', { class: 'card__head' }, [
      el('div', null, [
        el('h3', { class: 'card__title', id: 'card-' + id + '-title' }, [
          el('span', { class: 'swatch swatch--' + id, 'aria-hidden': 'true' }), m.name,
        ]),
        el('p', { class: 'card__sub', text: m.sub }),
      ]),
      el('div', { class: 'card__tags' }, [
        el('span', { class: 'pill pill--status', 'data-status': 'loading' }, [
          el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { class: 'pill__text', text: 'checking' }),
        ]),
      ]),
    ]);
    var slot = el('div', { class: 'card__slot' });
    function probRow(key, label) {
      return el('div', { class: 'prob prob--' + key }, [
        el('span', { class: 'prob__label', text: label }),
        el('span', { class: 'prob__track' }, [el('span', { class: 'prob__fill' })]),
        el('span', { class: 'prob__val', text: '—' }),
      ]);
    }
    var probs = el('div', { class: 'probs', role: 'group', 'aria-label': m.name + ' probabilities' }, [
      probRow('hot', 'Hotdog'), probRow('not', 'Not hotdog'),
    ]);
    function stat(key, label, title) {
      return el('div', { class: 'stat', title: title }, [
        el('dt', { text: label }), el('dd', { class: 'stat--' + key, text: '—' }),
      ]);
    }
    var stats = el('dl', { class: 'stats' }, [
      stat('latency', m.latencyLabel || 'Model', m.latencyTitle || 'latency_ms: model forward pass only'),
      stat('total', 'Total', 'total_ms: incl. decode / preprocess'),
      stat('rtt', 'Round trip', 'Measured in the browser, incl. network + upload'),
      stat('device', m.deviceLabel || 'Device', m.deviceTitle || 'Reported by /health'),
    ]);
    root.appendChild(head);
    root.appendChild(slot);
    root.appendChild(probs);
    root.appendChild(stats);
    cards[id] = { root: root, slot: slot, probs: probs, stats: stats, timer: null };
    setCardIdle(id);
  }

  function setProbs(id, pHot) {
    var c = cards[id];
    var rows = [['hot', pHot], ['not', pHot == null ? null : 1 - pHot]];
    rows.forEach(function (r) {
      var row = $('.prob--' + r[0], c.probs);
      $('.prob__fill', row).style.width = r[1] == null ? '0%' : (r[1] * 100).toFixed(1) + '%';
      $('.prob__val', row).textContent = r[1] == null ? '—' : fmtPct(r[1]);
    });
  }
  function setStats(id, res) {
    var c = cards[id];
    $('.stat--latency', c.stats).textContent = res ? fmtMs(res.latency_ms) : '—';
    $('.stat--total', c.stats).textContent = res ? fmtMs(res.total_ms) : '—';
    $('.stat--rtt', c.stats).textContent = res ? fmtMs(res.rtt_ms) : '—';
  }
  function stopTimer(id) {
    if (cards[id].timer) { cancelAnimationFrame(cards[id].timer); cards[id].timer = null; }
  }

  function setCardIdle(id) {
    var c = cards[id];
    stopTimer(id);
    c.root.dataset.state = 'idle';
    c.slot.innerHTML = '';
    var h = health[id];
    c.shown = slotKey(id);
    if (h.status === 'offline') {
      c.slot.appendChild(offlineBox(id));
    } else if (h.status === 'unavailable') {
      c.slot.appendChild(notSetUpBox(id));
    } else {
      c.slot.appendChild(el('div', { class: 'slot-idle' }, [
        el('span', { class: 'slot-idle__plate', 'aria-hidden': 'true' }),
        el('span', { text: h.status === 'loading' && h.data ? 'Warming up…' : 'Awaiting a snack…' }),
      ]));
    }
    setProbs(id, null);
    setStats(id, null);
  }

  function offlineBox(id) {
    var m = MODELS[id];
    return el('div', { class: 'slot-offline' }, [
      el('strong', { text: m.name + ' is offline' }),
      el('span', { text: 'Couldn’t reach ' + m.url + '. Start everything with:' }),
      cmdLine('./run.sh'),
      el('span', { class: 'muted', text: 'Running already? Check CORS.' }),
    ]);
  }

  // The server is up but answered ready:false with a reason: missing weights, no API key, …
  function notSetUpBox(id) {
    var m = MODELS[id];
    var detail = healthDetail(id);
    return el('div', { class: 'slot-setup' }, [
      el('strong', { class: 'slot-setup__title', text: m.name + ' isn’t set up yet' }),
      detail ? el('p', { class: 'slot-setup__detail', title: detail }, [el('span', { text: detail })]) : null,
      el('span', { text: 'To fix it, run:' }),
      cmdLine(m.fix),
      el('span', { class: 'slot-setup__hint', text: m.fixWhat }),
    ]);
  }

  // A shell command with a copy button.
  function cmdLine(cmd) {
    var btn = el('button', { type: 'button', class: 'cmd__copy', 'aria-label': 'Copy command: ' + cmd, text: 'Copy' });
    btn.addEventListener('click', function () {
      var done = function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(cmd).then(done, function () { toast('Couldn’t copy. Select the command instead.'); });
      } else {
        var sel = window.getSelection(), range = document.createRange();
        range.selectNodeContents(code);
        sel.removeAllRanges(); sel.addRange(range);
        try { document.execCommand('copy'); done(); } catch (e) { toast('Press Ctrl/Cmd+C to copy.'); }
      }
    });
    var code = el('code', { class: 'cmd__text', text: cmd });
    return el('div', { class: 'cmd' }, [el('span', { class: 'cmd__prompt', 'aria-hidden': 'true', text: '$' }), code, btn]);
  }

  function setCardLoading(id) {
    var c = cards[id];
    stopTimer(id);
    c.root.dataset.state = 'loading';
    c.slot.innerHTML = '';
    var clock = el('span', { class: 'slot-loading__clock', text: '0 ms' });
    c.slot.appendChild(el('div', { class: 'slot-loading' }, [
      el('span', { class: 'spinner', 'aria-hidden': 'true' }),
      el('span', { class: 'slot-loading__text', text: MODELS[id].loading }),
      clock,
    ]));
    setProbs(id, null);
    setStats(id, null);
    var t0 = performance.now();
    (function tick() {
      clock.textContent = fmtMs(performance.now() - t0);
      c.timer = requestAnimationFrame(tick);
    })();
  }

  function setCardResult(id, res) {
    var c = cards[id];
    stopTimer(id);
    c.root.dataset.state = res.is_hotdog ? 'hot' : 'not';
    c.slot.innerHTML = '';
    var b = bannerEl(res.is_hotdog ? 'hot' : 'not', res.is_hotdog ? 'Hotdog!' : 'Not hotdog!', 'banner--card');
    b.appendChild(el('span', { class: 'banner__conf', text: fmtPct(res.confidence) + ' sure' }));
    c.slot.appendChild(b);
    var extras = resultExtras(res);
    // classifiers don't explain themselves; shown only in the 3-up layout, opposite the LLM's reason
    c.slot.appendChild(extras || el('p', { class: 'says-none', 'aria-hidden': 'true' }, [
      el('span', { class: 'says__kicker', text: 'The model says' }), el('span', { text: 'No comment. Just the numbers.' }),
    ]));
    // let the bars animate from 0
    requestAnimationFrame(function () { setProbs(id, res.probabilities.hotdog); });
    setStats(id, res);
  }

  // LLM extras: the one-sentence reason and where the confidence number came from
  var CONF_SOURCES = {
    logprobs: { text: 'token logprobs', cls: 'pill--logprobs',
      title: 'Confidence computed from the API\u2019s token probabilities for the answer.' },
    'self-reported': { text: 'self-reported', cls: 'pill--self', hint: 'grain of salt',
      title: 'The LLM picked this number itself. Chatbots are famously confident; take it with a grain of salt.' },
  };
  function resultExtras(res) {
    if (!res.reason && !res.confidence_source) return null;
    var box = el('div', { class: 'says' });
    if (res.reason) {
      box.appendChild(el('figure', { class: 'says__bubble' }, [
        el('figcaption', { class: 'says__kicker', text: 'The model says' }),
        el('blockquote', { class: 'says__quote', text: '\u201c' + res.reason + '\u201d' }),
      ]));
    }
    if (res.confidence_source) {
      var src = CONF_SOURCES[res.confidence_source] || { text: res.confidence_source, cls: '', title: 'confidence_source: ' + res.confidence_source };
      box.appendChild(el('p', { class: 'says__meta' }, [
        el('span', { text: 'Confidence:' }),
        el('span', { class: 'pill pill--src ' + src.cls, title: src.title, tabindex: '0', 'aria-label': 'Confidence source: ' + src.text + '. ' + src.title, text: src.text }),
        src.hint ? el('span', { class: 'says__hint', 'aria-hidden': 'true', title: src.title, text: src.hint }) : null,
      ]));
    }
    return box;
  }

  function setCardError(id, err) {
    var c = cards[id];
    stopTimer(id);
    c.root.dataset.state = 'error';
    c.err = err;
    c.shown = slotKey(id);
    c.slot.innerHTML = '';
    if (err.kind === 'network') {
      c.slot.appendChild(offlineBox(id));
    } else if (isNotSetUp(id, err)) {
      c.slot.appendChild(notSetUpBox(id));
    } else {
      var title = { 503: health[id].status === 'ready' ? 'Ready now' : 'Still warming up', 502: 'Upstream hiccup', 504: 'Timed out' }[err.status] || 'Error';
      var lead = {
        503: health[id].status === 'ready'
          ? 'Hit Retry to get its verdict on this image.'
          : MODELS[id].name + ' is still loading' + (id === 'llm' ? '' : ' weights') + '. Give it a moment, then retry.',
        502: (id === 'llm' ? 'The vision API' : 'The upstream service') + ' reported a problem.',
        504: (id === 'llm' ? 'The chatbot' : MODELS[id].name) + ' took too long to answer.',
      }[err.status];
      c.slot.appendChild(bannerEl('err', title, 'banner--card'));
      var msg = el('div', { class: 'slot-error' });
      if (lead) msg.appendChild(el('p', { text: lead }));
      if (!lead || (err.status !== 503 && err.message)) {
        msg.appendChild(el('p', { class: lead ? 'slot-error__detail' : '', text: (err.status ? 'HTTP ' + err.status + ': ' : '') + err.message }));
      }
      c.slot.appendChild(msg);
    }
    c.slot.appendChild(retryButton(id));
    setProbs(id, null);
    setStats(id, null);
  }

  function retryButton(id) {
    var b = el('button', { type: 'button', class: 'retry-btn', 'aria-label': 'Retry ' + MODELS[id].name + ' on this image' }, [
      el('span', { class: 'retry-btn__icon', 'aria-hidden': 'true', text: '↻' }),
      el('span', { text: 'Retry ' + MODELS[id].name }),
    ]);
    b.addEventListener('click', function () { retry([id]); });
    return b;
  }

  // ---------- health ----------
  // status: loading (checking / still loading weights) | ready | unavailable (not set up) | offline
  var health = {};
  ORDER.forEach(function (k) { health[k] = { status: 'loading' }; });
  function healthDetail(id) {
    var d = health[id].data;
    if (!d) return '';
    var msg = typeof d.detail === 'string' && d.detail.trim() ? d.detail : typeof d.error === 'string' ? d.error : '';
    return msg.trim();
  }
  function statusOf(data) {
    if (data.ready !== false) return 'ready';
    // ready:false with an explanation = not set up / misconfigured; without one = still loading
    return (typeof data.detail === 'string' && data.detail.trim()) || (typeof data.error === 'string' && data.error.trim())
      ? 'unavailable' : 'loading';
  }
  // a 503 from /classify while /health says the model isn't set up
  function slotKey(id) { var h = health[id]; return h.status + '|' + !!h.data + '|' + healthDetail(id); }
  function isNotSetUp(id, err) {
    return err && err.kind === 'http' && err.status === 503 && health[id].status === 'unavailable';
  }
  function joinNames(list) {
    if (list.length <= 1) return list.join('');
    return list.slice(0, -1).join(', ') + ' and ' + list[list.length - 1];
  }
  function cardSub(id) {
    var d = health[id] && health[id].data;
    if (id === 'llm' && d && d.model_id) return d.model_id + (d.base_url_host ? ' via ' + d.base_url_host : '');
    if (id === 'cnn' && d && d.arch) {
      var pretty = { convnext_tiny: 'ConvNeXt-Tiny', efficientnet_b0: 'EfficientNet-B0', mobilenet_v3_large: 'MobileNetV3-Large' };
      return (pretty[d.arch] || d.arch) + ' · fine-tuned on a laptop';
    }
    return MODELS[id].sub;
  }
  function renderHealth(id) {
    var h = health[id];
    var m = MODELS[id];
    var label = {
      loading: h.data ? (id === 'llm' ? 'warming up' : 'loading weights') : 'checking…',
      ready: 'ready', unavailable: 'not set up', offline: 'offline',
    }[h.status];
    var device = h.data && h.status !== 'unavailable' ? (id === 'llm' ? h.data.base_url_host || h.data.device : h.data.device) : null;
    var detail = healthDetail(id);
    var li = document.getElementById('health-' + id);
    li.dataset.status = h.status;
    $('.health__text', li).textContent = m.name + ' · ' + label + (device ? ' · ' + device : '');
    li.title = m.url + (detail ? ' — ' + detail : '') + (h.status === 'unavailable' ? ' — fix: ' + m.fix : '');
    var card = cards[id];
    var pill = $('.pill--status', card.root);
    pill.dataset.status = h.status;
    pill.title = detail;
    $('.pill__text', pill).textContent = label;
    $('.card__sub', card.root).textContent = cardSub(id);
    var devEl = $('.stat--device', card.stats);
    devEl.textContent = device || '—';
    devEl.title = device || '';
    // re-render only when what the card says would change (keeps focus on its buttons)
    var state = card.root.dataset.state;
    if (card.shown !== slotKey(id)) {
      if (state === 'idle') setCardIdle(id);
      else if (state === 'error' && card.err.status === 503) {
        setCardError(id, card.err);
        // the head-to-head wording depends on it too ("isn't set up" vs "forfeited")
        if (current && current.errors[id]) { renderVotes(current); if (!tally(current).pending.length) onRoundProgress(current); }
      }
    }

    var offline = ORDER.filter(function (k) { return health[k].status === 'offline'; });
    $('#offline-help').hidden = offline.length === 0;
    $('#offline-names').textContent = joinNames(offline.map(function (k) { return MODELS[k].name + ' (' + MODELS[k].url + ')'; }));
  }
  function checkHealth(id) {
    return fetchWithTimeout(MODELS[id].url + '/health', { cache: 'no-store' }, CFG.healthTimeoutMs)
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (data) { health[id] = { status: statusOf(data), data: data }; })
      .catch(function () { health[id] = { status: 'offline', data: null }; })
      .then(function () { renderHealth(id); });
  }
  function pollHealth() {
    Promise.all(ORDER.map(checkHealth)).then(function () {
      var allReady = ORDER.every(function (k) { return health[k].status === 'ready'; });
      setTimeout(pollHealth, allReady ? CFG.healthIntervalMs : CFG.healthRetryMs);
    });
  }

  // ---------- classification rounds ----------
  var current = null;
  var roundSeq = 0;
  var lastInput = null;

  function normalize(body) {
    var p = body.probabilities || {};
    var pHot = typeof p.hotdog === 'number' ? p.hotdog
      : (body.is_hotdog ? body.confidence : 1 - body.confidence);
    var isHot = typeof body.is_hotdog === 'boolean' ? body.is_hotdog : body.label === 'hotdog';
    return {
      model: body.model, is_hotdog: isHot,
      confidence: typeof body.confidence === 'number' ? body.confidence : (isHot ? pHot : 1 - pHot),
      probabilities: { hotdog: pHot, not_hotdog: 1 - pHot },
      latency_ms: body.latency_ms, total_ms: typeof body.total_ms === 'number' ? body.total_ms : body.latency_ms, rtt_ms: body.rtt_ms,
      reason: typeof body.reason === 'string' && body.reason.trim() ? body.reason.trim() : null,
      confidence_source: body.confidence_source || null,
      model_id: body.model_id || null,
    };
  }

  function classifyOne(id, input, round) {
    var fd = new FormData();
    fd.append('file', input.blob, input.name);
    var t0 = performance.now();
    return fetch(MODELS[id].url + '/classify', { method: 'POST', body: fd })
      .then(function (r) {
        return r.json().catch(function () { return null; }).then(function (body) {
          if (!r.ok) {
            var e = new Error((body && body.detail) || r.statusText || 'Request failed');
            e.status = r.status; e.kind = 'http';
            throw e;
          }
          if (!body) throw Object.assign(new Error('Invalid JSON response'), { kind: 'http' });
          body.rtt_ms = performance.now() - t0;
          return normalize(body);
        });
      }, function (netErr) {
        throw Object.assign(new Error(netErr.message || 'Network error'), { kind: 'network' });
      })
      .then(function (res) {
        if (round !== current) return;
        round.results[id] = res;
        setCardResult(id, res);
        renderVotes(round);
        if (res.is_hotdog && !round.confettiDone) { round.confettiDone = true; confetti(); }
      }, function (err) {
        if (round !== current) return;
        round.errors[id] = err;
        setCardError(id, err);
        renderVotes(round);
        if (err.kind === 'network') { health[id] = { status: 'offline', data: null }; renderHealth(id); }
        else if (err.status === 503) checkHealth(id); // "not set up" or still loading? /health knows
      })
      .then(function () { if (round === current) onRoundProgress(round); });
  }

  function runRound(input) {
    lastInput = input;
    var round = { id: ++roundSeq, input: input, results: {}, errors: {}, truth: null, scored: false, confettiDone: false };
    current = round;
    $('#btn-rerun').disabled = false;
    $('#phone-banner').hidden = true;
    $('#phone-banner').innerHTML = '';
    $('#scanline').hidden = false;
    $('#summary-speed').hidden = true;
    $('#summary-truth').hidden = true;
    $('#retry-failed').hidden = true;
    setSummary('pending', 'Judging…', 'All three models are looking at ' + input.label + '. The CNN usually answers first.');
    ORDER.forEach(function (id) { setCardLoading(id); });
    renderVotes(round);
    ORDER.forEach(function (id) { classifyOne(id, input, round); });
  }

  // ---------- retry ----------
  // Re-asks only the failed model(s) about the same image, inside the same round. If the round was
  // already scored (or graded), those stats are rolled back first and re-applied once the retry lands,
  // so the scoreboard never double-counts.
  function retry(ids) {
    var round = current;
    if (!round || !round.input) return;
    ids = ids.filter(function (id) { return round.errors[id] && !round.results[id]; });
    if (!ids.length) return;
    if (round.scored) unscoreRoundStats(round);
    if (round.truth) { round.pendingTruth = round.truth; ungradeRound(round); }
    ids.forEach(function (id) { delete round.errors[id]; setCardLoading(id); });
    $('#scanline').hidden = false;
    $('#phone-banner').hidden = true;
    $('#summary-speed').hidden = true;
    $('#summary-truth').hidden = true;
    $('#retry-failed').hidden = true;
    renderVotes(round);
    onRoundProgress(round);
    ids.forEach(function (id) { classifyOne(id, round.input, round); });
  }

  function namesOf(list) { return list.map(function (k) { return MODELS[k].name; }); }
  function says(list, verdict) { return namesOf(list).join(' + ') + (list.length > 1 ? ' say ' : ' says ') + verdict; }
  function tally(round) {
    var ok = ORDER.filter(function (k) { return round.results[k]; });
    return {
      ok: ok,
      hot: ok.filter(function (k) { return round.results[k].is_hotdog; }),
      not: ok.filter(function (k) { return !round.results[k].is_hotdog; }),
      failed: ORDER.filter(function (k) { return round.errors[k]; }),
      pending: ORDER.filter(function (k) { return !round.results[k] && !round.errors[k]; }),
    };
  }
  function splitPhrase(t) {
    // majority first; the minority "says not" / "says hotdog"
    var hotFirst = t.hot.length >= t.not.length;
    var maj = hotFirst ? t.hot : t.not, min = hotFirst ? t.not : t.hot;
    return says(maj, hotFirst ? 'hotdog' : 'not hotdog') + ', ' + says(min, hotFirst ? 'not' : 'hotdog');
  }
  // "CLEF-Flash isn't set up (forfeit). CNN forfeited (error)."
  function forfeitNote(round, t) {
    var unset = t.failed.filter(function (k) { return isNotSetUp(k, round.errors[k]); });
    var other = t.failed.filter(function (k) { return unset.indexOf(k) < 0; });
    var parts = [];
    if (unset.length) parts.push(joinNames(namesOf(unset)) + (unset.length > 1 ? ' aren’t' : ' isn’t') + ' set up (forfeit).');
    if (other.length) parts.push(joinNames(namesOf(other)) + ' forfeited (error).');
    return parts.length ? ' ' + parts.join(' ') : '';
  }

  function renderVotes(round) {
    var ul = $('#votes');
    ul.innerHTML = '';
    ORDER.forEach(function (k) {
      var r = round.results[k], e = round.errors[k];
      var vote = r ? (r.is_hotdog ? 'hot' : 'not') : e ? 'err' : 'pending';
      var unset = isNotSetUp(k, e);
      var text = { hot: 'Hotdog', not: 'Not hotdog', err: unset ? 'Not set up' : 'Forfeit', pending: 'Thinking…' }[vote];
      ul.appendChild(el('li', { class: 'vote' + (unset ? ' vote--unset' : ''), 'data-vote': vote }, [
        el('span', { class: 'swatch swatch--' + k, 'aria-hidden': 'true' }),
        el('span', { class: 'vote__name', text: MODELS[k].name }),
        el('span', { class: 'vote__val', text: text }),
      ]));
    });
    ul.hidden = false;
  }

  // Single-column (phone) layout: the cards stack far below the photo, so once every model has
  // answered, bring the user back up to the verdict banner on the phone mockup.
  var singleColumn = window.matchMedia('(max-width: 760px)');
  function scrollToVerdictOnMobile() {
    if (!singleColumn.matches || window.scrollY < 4) return;
    window.scrollTo({ top: 0, behavior: reduceMotion.matches ? 'auto' : 'smooth' });
  }

  function onRoundProgress(round) {
    var t = tally(round);
    if (t.pending.length) {
      var waiting = 'Waiting for ' + joinNames(namesOf(t.pending)) + ' to finish thinking…';
      if (!t.ok.length) {
        if (t.failed.length) setSummary('pending', 'Judging…', forfeitNote(round, t).trim() + ' ' + waiting);
        return;
      }
      if (t.hot.length && t.not.length) {
        setSummary('pending', t.hot.length + '–' + t.not.length + ' so far',
          splitPhrase(t) + '. ' + joinNames(namesOf(t.pending)) + (t.pending.length > 1 ? ' hold' : ' holds') + ' the deciding vote…');
      } else {
        setSummary('pending', says(t.ok, t.hot.length ? 'hotdog' : 'not hotdog'), waiting);
      }
      return;
    }
    $('#scanline').hidden = true;
    scoreRoundStats(round);
    showRetryFailed(t);
    scrollToVerdictOnMobile();
    var pb = $('#phone-banner');
    pb.innerHTML = '';
    if (t.ok.length === 0) {
      if (t.failed.every(function (k) { return isNotSetUp(k, round.errors[k]); })) {
        setSummary('error', 'Nobody’s set up yet', 'Each card shows the command that fixes it.');
      } else {
        setSummary('error', 'Nobody showed up', 'All three backends failed. See the cards for details, or retry them all.');
      }
      pb.appendChild(bannerEl('err', 'No verdict'));
      pb.hidden = false;
      return;
    }
    var unanimous = !(t.hot.length && t.not.length);
    var isHot = t.hot.length > 0;
    if (t.ok.length === 1) {
      setSummary(isHot ? 'hot' : 'not', MODELS[t.ok[0]].name + ' rules alone: ' + (isHot ? 'Hotdog' : 'Not hotdog'),
        forfeitNote(round, t).trim() + ' A walkover is still a win.');
      pb.appendChild(bannerEl(isHot ? 'hot' : 'not', isHot ? 'Hotdog!' : 'Not hotdog!'));
    } else if (unanimous) {
      setSummary(isHot ? 'hot' : 'not', (t.ok.length === ORDER.length ? 'Unanimous: ' : 'Models agree: ') + (isHot ? 'Hotdog' : 'Not hotdog'),
        t.ok.map(function (k) { return MODELS[k].name + ' ' + fmtPct(round.results[k].confidence); }).join(', ') + ' sure.' + forfeitNote(round, t));
      pb.appendChild(bannerEl(isHot ? 'hot' : 'not', isHot ? 'Hotdog!' : 'Not hotdog!'));
    } else if (t.hot.length !== t.not.length) {
      var majHot = t.hot.length > t.not.length;
      setSummary('split', t.hot.length > t.not.length ? t.hot.length + '–' + t.not.length + ' split: Hotdog' : t.not.length + '–' + t.hot.length + ' split: Not hotdog',
        splitPhrase(t) + '. Majority says ' + (majHot ? 'hotdog' : 'not hotdog') + ', but you be the judge.' + forfeitNote(round, t));
      pb.appendChild(bannerEl('split', 'Split decision!'));
    } else {
      setSummary('split', 'Split decision!', splitPhrase(t) + '.' + forfeitNote(round, t) + ' You’re the tie-breaker.');
      pb.appendChild(bannerEl('split', 'Split decision!'));
    }
    pb.hidden = false;

    if (t.ok.length >= 2) {
      var byTime = t.ok.slice().sort(function (a, b) { return round.results[a].total_ms - round.results[b].total_ms; });
      var fast = byTime[0], slow = byTime[byTime.length - 1];
      var tf = round.results[fast].total_ms, ts = round.results[slow].total_ms;
      var factor = tf > 0 ? ts / tf : Infinity;
      $('#speed-headline').textContent = MODELS[fast].name + ' was ' + (factor >= 10 ? Math.round(factor) : factor.toFixed(1)) + '× faster than ' + MODELS[slow].name;
      $('#speed-detail').textContent = byTime.map(function (k) { return MODELS[k].name + ' ' + fmtMs(round.results[k].total_ms); }).join(' · ') +
        ' total. (' + MODELS[slow].name + ' ' + MODELS[slow].slowQuip + ')';
      $('#summary-speed').hidden = false;
    }
    $('#summary-truth').hidden = false;
    if (round.pendingTruth) { var pt = round.pendingTruth; round.pendingTruth = null; applyTruth(round, pt); }
    updateTruthUI(round);
  }
  function showRetryFailed(t) {
    var b = $('#retry-failed');
    if (!t.failed.length) { b.hidden = true; return; }
    $('#retry-failed-text').textContent = t.failed.length === 1 ? 'Retry ' + MODELS[t.failed[0]].name : 'Retry ' + t.failed.length + ' failed';
    b.hidden = false;
  }

  function setSummary(kind, headline, detail) {
    var v = $('#summary-verdict');
    v.dataset.kind = kind;
    $('#summary-headline').textContent = headline;
    $('#summary-detail').textContent = detail;
  }

  // ---------- scoreboard ----------
  var SCORE_KEY = 'bun-tribunal:score';
  function emptyModelScore() { return { calls: 0, sumMs: 0, graded: 0, correct: 0, speedWins: 0 }; }
  function num(v) { return typeof v === 'number' && isFinite(v) && v >= 0 ? v : 0; }
  function emptyScore() {
    var s = { rounds: 0, contested: 0, unanimous: 0, models: {} };
    ORDER.forEach(function (k) { s.models[k] = emptyModelScore(); });
    return s;
  }
  function sanitizeScore(raw) {
    var s = emptyScore();
    if (!raw || typeof raw !== 'object') return s;
    s.rounds = num(raw.rounds); s.contested = num(raw.contested); s.unanimous = Math.min(num(raw.unanimous), s.contested);
    var models = raw.models && typeof raw.models === 'object' ? raw.models : {};
    ORDER.forEach(function (k) {
      var m = models[k];
      if (!m || typeof m !== 'object') return;
      Object.keys(s.models[k]).forEach(function (f) { s.models[k][f] = num(m[f]); });
      s.models[k].correct = Math.min(s.models[k].correct, s.models[k].graded);
    });
    return s;
  }
  var score = sanitizeScore(store(SCORE_KEY));

  function scoreRoundStats(round) {
    if (round.scored) return;
    round.scored = true;
    var snap = { ms: {}, contested: false, unanimous: false, fast: null };
    score.rounds++;
    var t = tally(round);
    t.ok.forEach(function (k) {
      var r = round.results[k];
      snap.ms[k] = r.total_ms || 0;
      score.models[k].calls++; score.models[k].sumMs += snap.ms[k];
    });
    if (t.ok.length >= 2) {
      score.contested++; snap.contested = true;
      if (!(t.hot.length && t.not.length)) { score.unanimous++; snap.unanimous = true; }
      var fast = t.ok.reduce(function (a, b) { return round.results[b].total_ms < round.results[a].total_ms ? b : a; });
      score.models[fast].speedWins++; snap.fast = fast;
    }
    round.scoreSnap = snap;
    saveScore();
  }
  function unscoreRoundStats(round) {
    var snap = round.scoreSnap;
    if (!round.scored || !snap) return;
    score.rounds = Math.max(0, score.rounds - 1);
    Object.keys(snap.ms).forEach(function (k) {
      var m = score.models[k];
      m.calls = Math.max(0, m.calls - 1); m.sumMs = Math.max(0, m.sumMs - snap.ms[k]);
    });
    if (snap.contested) score.contested = Math.max(0, score.contested - 1);
    if (snap.unanimous) score.unanimous = Math.max(0, score.unanimous - 1);
    if (snap.fast) score.models[snap.fast].speedWins = Math.max(0, score.models[snap.fast].speedWins - 1);
    round.scored = false; round.scoreSnap = null;
    saveScore();
  }
  function ungradeRound(round) {
    if (!round.truth) return;
    ORDER.forEach(function (k) {
      var r = round.results[k];
      if (!r) return;
      var m = score.models[k];
      m.graded = Math.max(0, m.graded - 1);
      if ((r.is_hotdog ? 'hotdog' : 'not_hotdog') === round.truth) m.correct = Math.max(0, m.correct - 1);
    });
    round.truth = null;
    saveScore();
  }

  function applyTruth(round, truth) {
    var prev = round.truth;
    if (prev === truth) return;
    ORDER.forEach(function (k) {
      var r = round.results[k];
      if (!r) return;
      var m = score.models[k];
      if (prev) { m.graded--; if ((r.is_hotdog ? 'hotdog' : 'not_hotdog') === prev) m.correct--; }
      m.graded++;
      if ((r.is_hotdog ? 'hotdog' : 'not_hotdog') === truth) m.correct++;
    });
    round.truth = truth;
    saveScore();
    updateTruthUI(round);
  }

  function updateTruthUI(round) {
    document.querySelectorAll('.truth-btn').forEach(function (btn) {
      btn.setAttribute('aria-pressed', String(btn.dataset.truth === round.truth));
    });
    var d = $('#truth-detail');
    if (!round.truth) { d.textContent = 'Tell us the truth to keep score.'; return; }
    var parts = ORDER.filter(function (k) { return round.results[k]; }).map(function (k) {
      var right = (round.results[k].is_hotdog ? 'hotdog' : 'not_hotdog') === round.truth;
      return MODELS[k].name + ' ' + (right ? 'got it right' : 'got it wrong');
    });
    d.textContent = 'Recorded. ' + parts.join(', ') + '.';
  }

  function saveScore() { store(SCORE_KEY, score); renderScore(); }
  function renderScore() {
    var body = $('#score-body');
    body.innerHTML = '';
    var accs = ORDER.map(function (k) { var m = score.models[k]; return m.graded ? m.correct / m.graded : null; });
    var best = Math.max.apply(null, accs.map(function (a) { return a == null ? -1 : a; }));
    ORDER.forEach(function (k, i) {
      var m = score.models[k];
      var acc = accs[i];
      var lead = acc != null && acc === best && accs.filter(function (a) { return a === best; }).length === 1;
      var tr = el('tr', { class: lead ? 'is-leader' : '' }, [
        el('th', { scope: 'row' }, [el('span', { class: 'swatch swatch--' + k, 'aria-hidden': 'true' }), MODELS[k].name, lead ? el('span', { class: 'crown', text: 'leads' }) : null]),
        el('td', null, [
          el('span', { class: 'acc' }, [
            el('span', { class: 'acc__track' }, [el('span', { class: 'acc__fill', style: 'width:' + (acc == null ? 0 : acc * 100) + '%' })]),
            el('span', { text: acc == null ? '—' : Math.round(acc * 100) + '%' }),
          ]),
        ]),
        el('td', { text: m.correct + ' / ' + m.graded }),
        el('td', { text: m.calls ? fmtMs(m.sumMs / m.calls) : '—' }),
        el('td', { text: String(m.speedWins) }),
      ]);
      body.appendChild(tr);
    });
    $('#score-foot').textContent = score.rounds
      ? score.rounds + ' round' + (score.rounds === 1 ? '' : 's') + ' played' +
        (score.contested ? ' · unanimous verdict in ' + score.unanimous + ' of ' + score.contested : '')
      : 'No rounds yet. Judge a photo, then record the truth to track accuracy.';
  }

  // ---------- input handling ----------
  var OK_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  function rasterize(src, size) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        var w = size || img.naturalWidth || 512, h = size || img.naturalHeight || 512;
        var cv = document.createElement('canvas');
        cv.width = w; cv.height = h;
        cv.getContext('2d').drawImage(img, 0, 0, w, h);
        cv.toBlob(function (b) { b ? resolve(b) : reject(new Error('Could not encode image')); }, 'image/png');
      };
      img.onerror = function () { reject(new Error('Could not decode image')); };
      img.src = src;
    });
  }

  function handleFile(file, label) {
    if (!file) return;
    if (!/^image\//.test(file.type)) { toast('That is not an image. Even a hotdog has standards.'); return; }
    var url = URL.createObjectURL(file);
    var ready = OK_TYPES.indexOf(file.type) >= 0 ? Promise.resolve(file) : rasterize(url);
    ready.then(function (blob) {
      showPreview(url, label || file.name || 'your photo');
      var name = blob === file
        ? (file.name || 'upload.' + (file.type.split('/')[1] || 'png'))
        : (file.name || 'upload').replace(/\.[^.]+$/, '') + '.png';
      runRound({ blob: blob, name: name, label: label || 'your photo' });
    }, function () { toast('Couldn’t read that image format. Try JPEG, PNG or WebP.'); });
  }

  var currentPreviewUrl = null;
  function showPreview(url, alt) {
    var img = $('#preview');
    if (currentPreviewUrl && currentPreviewUrl.indexOf('blob:') === 0 && currentPreviewUrl !== url) URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = url;
    img.src = url;
    $('#viewfinder').style.setProperty('--bg-img', 'url("' + url + '")');
    img.alt = 'Photo being judged: ' + alt;
    img.hidden = false;
    $('#empty-state').classList.add('is-hidden');
  }

  function openPicker(preferCamera) {
    var coarse = window.matchMedia('(pointer: coarse)').matches;
    (preferCamera && coarse ? $('#camera-input') : $('#file-input')).click();
  }

  function wireInputs() {
    $('#file-input').addEventListener('change', function (e) { handleFile(e.target.files[0]); e.target.value = ''; });
    $('#camera-input').addEventListener('change', function (e) { handleFile(e.target.files[0], 'a fresh snap'); e.target.value = ''; });
    $('#btn-upload').addEventListener('click', function () { openPicker(false); });
    $('#btn-shutter').addEventListener('click', function () {
      var s = $('#btn-shutter');
      s.classList.remove('is-firing'); void s.offsetWidth; s.classList.add('is-firing');
      openPicker(true);
    });
    $('#empty-state').addEventListener('click', function () { openPicker(false); });
    $('#btn-rerun').addEventListener('click', function () { if (lastInput) runRound(lastInput); });
    $('#retry-failed').addEventListener('click', function () { if (current) retry(tally(current).failed); });

    // paste
    document.addEventListener('paste', function (e) {
      var items = (e.clipboardData && e.clipboardData.items) || [];
      for (var i = 0; i < items.length; i++) {
        if (items[i].kind === 'file' && /^image\//.test(items[i].type)) {
          e.preventDefault();
          handleFile(items[i].getAsFile(), 'a pasted image');
          return;
        }
      }
    });

    // drag & drop anywhere
    var depth = 0;
    var overlay = $('#drop-overlay');
    function hasFiles(e) { return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], 'Files') >= 0; }
    window.addEventListener('dragenter', function (e) { if (!hasFiles(e)) return; e.preventDefault(); depth++; overlay.classList.add('is-on'); });
    window.addEventListener('dragover', function (e) { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener('dragleave', function () { depth = Math.max(0, depth - 1); if (!depth) overlay.classList.remove('is-on'); });
    window.addEventListener('drop', function (e) {
      if (!hasFiles(e)) return;
      e.preventDefault(); depth = 0; overlay.classList.remove('is-on');
      handleFile(e.dataTransfer.files[0]);
    });

    document.querySelectorAll('.truth-btn').forEach(function (btn) {
      btn.addEventListener('click', function () { if (current) applyTruth(current, btn.dataset.truth); });
    });
    $('#score-reset').addEventListener('click', function () {
      score = emptyScore();
      if (current) current.truth = null;
      saveScore();
      if (current) updateTruthUI(current);
      toast('Scoreboard reset. Fresh buns.');
    });
  }

  // ---------- samples (original, hand-drawn SVG) ----------
  var SAMPLES = [
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
  ];
  function svgUrl(svg) { return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg); }
  function buildSamples() {
    var ul = $('#samples');
    SAMPLES.forEach(function (s) {
      var url = svgUrl(s.svg);
      var btn = el('button', { type: 'button', class: 'sample', 'aria-label': 'Try sample: ' + s.label, 'data-sample': s.id }, [
        el('img', { src: url, alt: '', width: '72', height: '72' }),
        el('span', { text: s.title }),
      ]);
      btn.addEventListener('click', function () {
        rasterize(url, 512).then(function (blob) {
          showPreview(url, s.label);
          runRound({ blob: blob, name: 'sample-' + s.id + '.png', label: s.label });
        }, function () { toast('Couldn’t render that sample.'); });
      });
      ul.appendChild(el('li', null, [btn]));
    });
  }

  // ---------- confetti ----------
  function confetti() {
    if (reduceMotion.matches) return;
    var cv = $('#confetti');
    var ctx = cv.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = window.innerWidth, H = window.innerHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var colors = ['#ffd21f', '#e2342b', '#1fae54', '#ff8a1f', '#ffffff', '#c4442c'];
    var parts = [];
    for (var i = 0; i < 160; i++) {
      parts.push({
        x: W * (0.2 + Math.random() * 0.6), y: H * 0.25 + Math.random() * 40,
        vx: (Math.random() - 0.5) * 12, vy: -Math.random() * 12 - 4,
        w: 6 + Math.random() * 8, h: 4 + Math.random() * 5,
        r: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.4,
        c: colors[i % colors.length], sausage: Math.random() < 0.15,
      });
    }
    var start = performance.now();
    cv.classList.add('is-on');
    (function frame(now) {
      var t = now - start;
      ctx.clearRect(0, 0, W, H);
      parts.forEach(function (p) {
        p.vy += 0.35; p.vx *= 0.99; p.x += p.vx; p.y += p.vy; p.r += p.vr;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r);
        ctx.globalAlpha = Math.max(0, 1 - t / 2600);
        if (p.sausage) {
          ctx.fillStyle = '#e8ae63'; roundRect(ctx, -11, -5, 22, 10, 5);
          ctx.fillStyle = '#c4442c'; roundRect(ctx, -13, -3, 26, 6, 3);
        } else {
          ctx.fillStyle = p.c; ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        }
        ctx.restore();
      });
      if (t < 2600) requestAnimationFrame(frame);
      else { ctx.clearRect(0, 0, W, H); cv.classList.remove('is-on'); }
    })(start);
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); ctx.fill();
  }

  // ---------- pitch deck mode ----------
  var PITCH_KEY = 'bun-tribunal:pitch';
  function setPitch(on) {
    document.body.classList.toggle('pitch', on);
    $('#pitch-toggle').setAttribute('aria-pressed', String(on));
    $('#tagline').textContent = on
      ? 'It’s like a sommelier, but for sausages. Three of them.'
      : 'Three models enter. One sausage leaves.';
    store(PITCH_KEY, on);
  }

  // ---------- boot ----------
  ORDER.forEach(buildCard);
  ORDER.forEach(renderHealth);
  buildSamples();
  wireInputs();
  renderScore();
  $('#origin').textContent = window.location.origin;
  $('#footer-urls').textContent = ORDER.map(function (k) { return MODELS[k].name + ' → ' + MODELS[k].url; }).join(' · ') + ' · override with ?llm=…&clef=…&cnn=…';
  $('#pitch-toggle').addEventListener('click', function () { setPitch(!document.body.classList.contains('pitch')); });
  if (store(PITCH_KEY) === true || /[?&]pitch=1/.test(window.location.search)) setPitch(true);
  pollHealth();

  // test hook
  // ---------- phone status bar clock ----------
  // Real local time in the browser's own hour format, without AM/PM (like a phone status bar).
  var clockFmt = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' });
  function tickClock() {
    var t = clockFmt.formatToParts(new Date())
      .filter(function (p) { return p.type !== 'dayPeriod'; })
      .map(function (p) { return p.value; }).join('').trim();
    var c = $('#phone-clock');
    if (c && c.textContent !== t) c.textContent = t;
  }
  tickClock();
  setInterval(tickClock, 1000);

  window.__bunTribunal = { runSample: function (id) { var b = document.querySelector('[data-sample="' + id + '"]'); if (b) b.click(); } };
})();
