// Backend URLs and client tuning. Override the URLs per page load with query params, e.g.
//   http://localhost:8080/?clef=http://gpu-box:8001&cnn=http://localhost:9002&llm=http://localhost:9003
(() => {
  // Same host as the page, so it also works from a phone on your Wi-Fi (LAN=1 ./run.sh).
  let host = window.location.protocol === 'file:' ? 'localhost' : window.location.hostname || 'localhost';
  if (host.includes(':')) host = `[${host}]`; // IPv6 literal
  const defaults = {
    llm: `http://${host}:8003`,
    clef: `http://${host}:8001`,
    cnn: `http://${host}:8002`,
  };
  const params = new URLSearchParams(window.location.search);
  const pick = (key) => {
    const v = params.get(key);
    return (v && /^https?:\/\//i.test(v) ? v : defaults[key]).replace(/\/+$/, '');
  };

  window.HOTDOG_CONFIG = Object.freeze({
    llm: pick('llm'),
    clef: pick('clef'),
    cnn: pick('cnn'),
    healthIntervalMs: 15000, // poll interval when everything is ready
    healthRetryMs: 3000, // while something is loading/offline
    healthTimeoutMs: 4000,
    classifyTimeoutMs: 300000, // safety net only; the LLM server retries rate limits itself (60 s timeout each)
    uploadMaxSide: 1280, // photos larger than this are downscaled once in the browser before upload
    uploadJpegQuality: 0.9,
  });
})();
