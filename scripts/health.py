"""Print the status of the three model servers (used by ./run.sh). Stdlib only.

    python scripts/health.py              # one-shot status
    python scripts/health.py --wait 20    # wait up to 20 s for every server to answer /health
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

SERVICES = [  # same order as the UI
    ("LLM", int(os.environ.get("PORT_LLM", 8003))),
    ("CLEF-Flash", int(os.environ.get("PORT_CLEF", 8001))),
    ("CNN", int(os.environ.get("PORT_CNN", 8002))),
]
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
C = {"green": "32", "yellow": "33", "red": "31", "dim": "2", "cyan": "36"}


def paint(text: str, color: str) -> str:
    return f"\033[{C[color]}m{text}\033[0m" if COLOR else text


def probe(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def probe_ui(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2):
            return True
    except Exception:
        return False


def describe(name: str, port: int, h: dict | None) -> str:
    """One status line: ready | loading… | not set up (with the fix from the server) | not running."""
    label = f"{name:<11}"
    if h is None:
        return f"  {paint('●', 'red')} {label} {paint('not running', 'red')}  (port {port})"
    detail = (h.get("detail") or h.get("error") or "").strip()
    extra = []
    if h.get("model_id"):
        extra.append(f"{h['model_id']} via {h.get('base_url_host', '?')}")
    if h.get("arch"):
        extra.append(h["arch"])
    if h.get("device") and h.get("device") not in ("api", "pending"):
        extra.append(h["device"])
    info = paint(f"  ({', '.join(extra)})", "dim") if extra else ""
    if h.get("ready"):
        return f"  {paint('●', 'green')} {label} {paint('ready', 'green')}{info}"
    if detail:
        return f"  {paint('●', 'yellow')} {label} {paint('not set up', 'yellow')}: {detail}"
    return f"  {paint('●', 'cyan')} {label} {paint('loading…', 'cyan')}{info}"


def main() -> int:
    wait = 0.0
    if "--wait" in sys.argv:
        wait = float(sys.argv[sys.argv.index("--wait") + 1])
    deadline = time.time() + wait
    try:
        while True:
            results = [(n, p, probe(p)) for n, p in SERVICES]
            if all(h is not None for *_, h in results) or time.time() >= deadline:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    for n, p, h in results:
        print(describe(n, p, h))
    ui = int(os.environ.get("PORT_UI", 8080))
    ui_up = probe_ui(ui)
    dot = paint("●", "green" if ui_up else "red")
    print(f"  {dot} {'Web app':<11} " + (paint("ready", "green") + paint(f"  (http://localhost:{ui})", "dim") if ui_up
                                          else paint("not running", "red") + f"  (port {ui})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
