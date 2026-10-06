#!/usr/bin/env python3
"""Mock Hotdog Arena backends for UI development (stdlib only).

Serves all three classifier endpoints from CONTRACT.md:
  CLEF-Flash mock on :8001 (~800 ms latency)
  CNN mock        on :8002 (~15 ms latency)
  LLM mock        on :8003 (~1500 ms "API round trip", adds reason / confidence_source / model_id)

Results are deterministic per image (seeded from the image bytes' hash).
All mocks share a base verdict, but the CNN flips it ~25% of the time and the
LLM ~35% of the time (independently), so you get to see 2-1 split decisions too.

Usage:  python mock-server.py [--clef-port 8001] [--cnn-port 8002] [--llm-port 8003] [--speed 1.0]
"""
import argparse
import hashlib
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {
    "clef-flash": {"device": "mock-gpu", "latency": 0.80, "jitter": 0.25},
    "cnn": {"device": "mock-cpu", "latency": 0.015, "jitter": 0.006},
    "llm": {"device": "remote-api", "latency": 1.50, "jitter": 0.40},
}
LLM_MODEL_ID = "mock-llm"
LLM_BASE_URL_HOST = "localhost"
LLM_REASONS = {
    "hotdog": [
        "A grilled sausage in a split bun with a mustard zigzag: textbook hotdog.",
        "Cylindrical meat, bread hug, condiments. All the hotdog boxes are ticked.",
        "I see a bun cradling a sausage, which is legally and spiritually a hotdog.",
        "Long, brown, in a bun and judging by the mustard, very proud of it. Hotdog.",
    ],
    "not_hotdog": [
        "No bun and no sausage in sight; this is ordinary, non-hotdog reality.",
        "Food-adjacent shapes, maybe, but nothing I would put mustard on.",
        "Elongated, yes. Hotdog, no. Some things are just long.",
        "I looked very hard for a sausage and found only disappointment.",
    ],
}
STARTUP_DELAY_S = 2.0  # /health reports ready:false this long (to show the "loading" state)
STARTED = time.time()
SPEED = 1.0


def extract_file(body: bytes, content_type: str) -> bytes | None:
    """Pull the bytes of the multipart field named "file"."""
    if "boundary=" not in content_type:
        return None
    boundary = content_type.split("boundary=", 1)[1].split(";")[0].strip().strip('"')
    for part in body.split(b"--" + boundary.encode()):
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        if b'name="file"' in head:
            return data[:-2] if data.endswith(b"\r\n") else data
    return None


def looks_like_image(data: bytes) -> bool:
    return (
        data.startswith(b"\x89PNG")
        or data.startswith(b"\xff\xd8")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
        or data.startswith(b"GIF8")
    )


def classify(model: str, data: bytes) -> dict:
    digest = hashlib.sha256(data).digest()
    base_is_hotdog = digest[0] % 2 == 0
    rng = random.Random(digest + model.encode())
    if model == "cnn":
        is_hotdog = base_is_hotdog if rng.random() > 0.25 else not base_is_hotdog
        conf = rng.uniform(0.56, 0.97)
    elif model == "llm":
        is_hotdog = base_is_hotdog if rng.random() > 0.35 else not base_is_hotdog
        conf = rng.choice([0.8, 0.85, 0.9, 0.95, 0.98])  # LLMs love round, confident numbers
    else:
        is_hotdog = base_is_hotdog
        conf = rng.uniform(0.84, 0.995)
    cfg = MODELS[model]
    latency = max(0.002, cfg["latency"] + rng.uniform(-cfg["jitter"], cfg["jitter"])) * SPEED
    t0 = time.perf_counter()
    time.sleep(latency)
    fwd_ms = (time.perf_counter() - t0) * 1000
    p_hot = conf if is_hotdog else 1 - conf
    label = "hotdog" if is_hotdog else "not_hotdog"
    if model == "llm":
        # no local forward pass: latency_ms is the whole API round trip
        return {
            "model": model,
            "label": label,
            "is_hotdog": is_hotdog,
            "confidence": conf,
            "probabilities": {"hotdog": round(p_hot, 4), "not_hotdog": round(1 - p_hot, 4)},
            "latency_ms": round(fwd_ms, 1),
            "total_ms": round(fwd_ms + rng.uniform(4, 15), 1),
            "reason": rng.choice(LLM_REASONS[label]),
            "confidence_source": "self-reported",
            "model_id": LLM_MODEL_ID,
        }
    return {
        "model": model,
        "label": "hotdog" if is_hotdog else "not_hotdog",
        "is_hotdog": is_hotdog,
        "confidence": round(conf, 4),
        "probabilities": {"hotdog": round(p_hot, 4), "not_hotdog": round(1 - p_hot, 4)},
        "latency_ms": round(fwd_ms, 1),
        "total_ms": round(fwd_ms + rng.uniform(2, 9), 1),
    }


def make_handler(model: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"HotdogMock/{model}"

        def log_message(self, fmt, *args):
            print(f"[{model}] " + fmt % args)

        def _send(self, status: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path.split("?")[0] != "/health":
                return self._send(404, {"detail": "not found"})
            ready = time.time() - STARTED > STARTUP_DELAY_S
            payload = {"status": "ok", "model": model, "device": MODELS[model]["device"],
                       "mock": True, "ready": ready}
            if model == "llm":
                payload.update(model_id=LLM_MODEL_ID, base_url_host=LLM_BASE_URL_HOST)
                if not ready:
                    payload["detail"] = "checking that the mock API answers"
            self._send(200, payload)

        def do_POST(self):
            if self.path.split("?")[0] != "/classify":
                return self._send(404, {"detail": "not found"})
            if time.time() - STARTED <= STARTUP_DELAY_S:
                return self._send(503, {"detail": "model not ready"})
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            data = extract_file(body, self.headers.get("Content-Type", ""))
            if not data:
                return self._send(400, {"detail": "missing multipart field 'file'"})
            if not looks_like_image(data):
                return self._send(400, {"detail": "could not decode image"})
            self._send(200, classify(model, data))

    return Handler


def main():
    global SPEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clef-port", type=int, default=8001)
    ap.add_argument("--cnn-port", type=int, default=8002)
    ap.add_argument("--llm-port", type=int, default=8003)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--speed", type=float, default=1.0, help="latency multiplier (e.g. 3 = slower)")
    args = ap.parse_args()
    SPEED = args.speed

    servers = [
        ThreadingHTTPServer((args.host, args.clef_port), make_handler("clef-flash")),
        ThreadingHTTPServer((args.host, args.cnn_port), make_handler("cnn")),
        ThreadingHTTPServer((args.host, args.llm_port), make_handler("llm")),
    ]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"Mock CLEF-Flash on http://{args.host}:{args.clef_port}  |  mock CNN on http://{args.host}:{args.cnn_port}"
          f"  |  mock LLM on http://{args.host}:{args.llm_port}")
    print("Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        for s in servers:
            s.shutdown()


if __name__ == "__main__":
    main()
