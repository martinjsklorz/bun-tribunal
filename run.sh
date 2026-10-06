#!/usr/bin/env bash
# Bun Tribunal — set up .venv, install deps, start all four services.
#
#   ./run.sh              # start everything (each model reports "not set up" until it is)
#   ./run.sh setup        # only create .venv + install, don't start anything
#
# Env knobs:
#   PYTHON=python3.12     interpreter used to create the venv (needs >= 3.11)
#   VENV=/path/to/venv    venv location (default: ./.venv)
#   FORCE_INSTALL=1       reinstall even if requirements are unchanged
#   SKIP_INSTALL=1        never touch the venv's packages
#   DEVICE, DTYPE, CLEF_MODEL, CNN_ARTIFACTS are passed through to the servers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"
CMD="${1:-run}"

say() { printf '\033[1;33m🌭 %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- venv
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found. Install Python >= 3.11 or set PYTHON=..."
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || die "$("$PYTHON" -V) is too old — need Python >= 3.11 (e.g. PYTHON=python3.12 ./run.sh)"

if [ ! -x "$VENV/bin/python" ]; then
  say "Creating virtualenv in $VENV"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
fi
PY="$VENV/bin/python"

# ---------------------------------------------------------------- requirements
REQS=("$ROOT/clef-server/requirements.txt" "$ROOT/cnn/requirements.txt" "$ROOT/llm-server/requirements.txt" "$ROOT/ui/requirements.txt")
MODE="full"
for r in "${REQS[@]}"; do [ -f "$r" ] || die "missing $r"; done

# Re-install only when the requirement files changed.
STAMP="$VENV/.bun-tribunal-$MODE.sha256"
HASH="$("$PY" - "${REQS[@]}" <<'EOF'
import hashlib, sys
h = hashlib.sha256()
for p in sys.argv[1:]:
    h.update(open(p, "rb").read())
print(h.hexdigest())
EOF
)"

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  say "SKIP_INSTALL=1 — not touching packages"
elif [ "${FORCE_INSTALL:-0}" = "1" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$HASH" ]; then
  say "Installing $MODE requirements (clef-server, cnn, llm-server, ui) — first run with torch takes a few minutes"
  PIP_ARGS=()
  for r in "${REQS[@]}"; do PIP_ARGS+=(-r "$r"); done
  "$PY" -m pip install "${PIP_ARGS[@]}"
  echo "$HASH" > "$STAMP"
else
  say "Requirements up to date ($MODE)"
fi

[ "$CMD" = "setup" ] && { say "Setup done. Activate with: source ${VENV#$ROOT/}/bin/activate"; exit 0; }

# ---------------------------------------------------------------- pre-flight hints
ls "$ROOT"/cnn/artifacts/*.pt >/dev/null 2>&1 \
  || say "No CNN weights in cnn/artifacts/ yet — run cnn/train_hotdog_cnn.ipynb first (CNN will report not ready)."
say "CLEF-Flash loads ~19 GB of weights in the background; the UI shows 'loading' until it's ready."
[ -f "$ROOT/llm-server/.env" ] || [ -n "${LLM_API_KEY:-}${OPENAI_API_KEY:-}${LLM_BASE_URL:-}" ] \
  || say "LLM not configured — copy llm-server/.env.example to llm-server/.env (OpenAI key or local Ollama)."

# ---------------------------------------------------------------- start
PIDS=()
cleanup() {
  [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$ROOT/clef-server" && exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8001) & PIDS+=($!)
(cd "$ROOT/cnn/server"  && exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8002) & PIDS+=($!)
(cd "$ROOT/llm-server"  && exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8003) & PIDS+=($!)
(cd "$ROOT/ui"          && exec "$PY" -m http.server 8080 --bind 127.0.0.1) & PIDS+=($!)

say "CLEF-Flash : http://localhost:8001/health"
say "CNN        : http://localhost:8002/health"
say "LLM        : http://localhost:8003/health"
say "UI         : http://localhost:8080   (Ctrl+C stops everything)"
wait
