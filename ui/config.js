// Backend URLs. Override per page load with query params, e.g.
//   http://localhost:8080/?clef=http://gpu-box:8001&cnn=http://localhost:9002&llm=http://localhost:9003
(function () {
  // Same host as the page, so it also works from a phone on your Wi-Fi (LAN=1 ./run.sh).
  var host = window.location.protocol === 'file:' ? 'localhost' : (window.location.hostname || 'localhost');
  if (host.indexOf(':') !== -1) host = '[' + host + ']'; // IPv6 literal
  var defaults = {
    clef: 'http://' + host + ':8001',
    cnn: 'http://' + host + ':8002',
    llm: 'http://' + host + ':8003',
  };
  var params = new URLSearchParams(window.location.search);
  function pick(key) {
    var v = params.get(key);
    return (v && /^https?:\/\//i.test(v) ? v : defaults[key]).replace(/\/+$/, '');
  }
  window.HOTDOG_CONFIG = {
    clef: pick('clef'),
    cnn: pick('cnn'),
    llm: pick('llm'),
    healthIntervalMs: 15000, // when everything is ready
    healthRetryMs: 3000, // while something is loading/offline
    healthTimeoutMs: 4000,
  };
})();
