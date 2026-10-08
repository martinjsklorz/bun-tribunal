"""Print one status line per model server and for the web app (used by ./run.sh). Stdlib only.

    python scripts/health.py              # one-shot status
    python scripts/health.py --wait 20    # first wait up to 20 s until every server and the web app answer

Ports come from PORT_LLM, PORT_CLEF, PORT_CNN and PORT_UI (defaults 8003, 8001, 8002, 8080), as set by run.sh.
Always exits 0 (130 on Ctrl+C): the lines themselves say what is wrong and how to fix it.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple


class Service(NamedTuple):
    name: str
    port: int


def env_port(var: str, default: int) -> int:
    return int(os.environ.get(var) or default)


SERVERS = [  # same order as the UI
    Service("LLM", env_port("PORT_LLM", 8003)),
    Service("CLEF-Flash", env_port("PORT_CLEF", 8001)),
    Service("CNN", env_port("PORT_CNN", 8002)),
]
WEB_APP = Service("Web app", env_port("PORT_UI", 8080))
LABEL_WIDTH = 11  # fits "CLEF-Flash"
# Talk to 127.0.0.1 directly, even when HTTP(S)_PROXY is set without a matching NO_PROXY.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
COLORS = {"green": "32", "yellow": "33", "red": "31", "dim": "2", "cyan": "36"}


def paint(text: str, color: str) -> str:
    return f"\033[{COLORS[color]}m{text}\033[0m" if USE_COLOR else text


def fetch(port: int, path: str, timeout: float = 2.0) -> bytes | None:
    """GET http://127.0.0.1:<port><path>; None if nothing answers (or answers with an error)."""
    try:
        with DIRECT.open(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
            return response.read()
    except (OSError, ValueError, http.client.HTTPException):  # URLError, HTTPError, timeouts, refusals: OSError
        return None


def probe_server(port: int) -> dict[str, Any] | None:
    """The server's /health JSON, or None if it isn't running."""
    body = fetch(port, "/health")
    if body is None:
        return None
    try:
        health = json.loads(body)
    except ValueError:
        return None
    return health if isinstance(health, dict) else None


def probe_web_app(port: int) -> bool:
    return fetch(port, "/") is not None


def probe_all(pool: ThreadPoolExecutor) -> tuple[list[dict[str, Any] | None], bool]:
    """Probe everything in parallel, so one slow server doesn't delay the others."""
    servers = [pool.submit(probe_server, s.port) for s in SERVERS]
    web_app = pool.submit(probe_web_app, WEB_APP.port)
    return [f.result() for f in servers], web_app.result()


def dot(color: str) -> str:
    return paint("●", color)


def describe_server(service: Service, health: dict[str, Any] | None) -> str:
    """One status line: ready | loading… | not set up (with the fix the server names) | not running."""
    label = f"{service.name:<{LABEL_WIDTH}}"
    if health is None:
        return f"  {dot('red')} {label} {paint('not running', 'red')}  (port {service.port})"
    extra = []
    if model_id := health.get("model_id"):
        host = health.get("base_url_host")
        extra.append(f"{model_id} via {host}" if host else str(model_id))
    if arch := health.get("arch"):
        extra.append(str(arch))
    if (device := health.get("device")) and device not in ("api", "pending"):
        extra.append(str(device))
    info = paint(f"  ({', '.join(extra)})", "dim") if extra else ""
    if health.get("ready"):
        return f"  {dot('green')} {label} {paint('ready', 'green')}{info}"
    if detail := str(health.get("detail") or health.get("error") or "").strip():
        return f"  {dot('yellow')} {label} {paint('not set up', 'yellow')}: {detail}"
    return f"  {dot('cyan')} {label} {paint('loading…', 'cyan')}{info}"


def describe_web_app(up: bool) -> str:
    label = f"{WEB_APP.name:<{LABEL_WIDTH}}"
    if up:
        return f"  {dot('green')} {label} {paint('ready', 'green')}" + paint(
            f"  (http://localhost:{WEB_APP.port})", "dim"
        )
    return f"  {dot('red')} {label} {paint('not running', 'red')}  (port {WEB_APP.port})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wait", type=float, default=0.0, metavar="SECONDS", help="wait until everything answers")
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait
    try:
        with ThreadPoolExecutor(max_workers=len(SERVERS) + 1) as pool:
            while True:
                servers, web_app_up = probe_all(pool)
                everything_up = web_app_up and all(h is not None for h in servers)
                if everything_up or time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
    except KeyboardInterrupt:
        return 130
    for service, health in zip(SERVERS, servers, strict=True):
        print(describe_server(service, health))
    print(describe_web_app(web_app_up))
    return 0


if __name__ == "__main__":
    sys.exit(main())
