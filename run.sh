#!/usr/bin/env bash
# Bun Tribunal: one script for everything.
#
#   ./run.sh            set up what's missing (asks first), start everything, open the browser
#   ./run.sh llm        configure the LLM provider (OpenAI, OpenRouter, Ollama, LM Studio, ...)
#   ./run.sh train      train the CNN (~20 min on an Apple Silicon Mac)
#   ./run.sh download   download the CLEF-Flash weights (4-bit MLX, ~6 GB, on Apple Silicon; ~19 GB elsewhere)
#   ./run.sh status     show what's set up and what's running
#   ./run.sh setup      ask again about everything that is not set up yet
#   ./run.sh test       run all test suites
#   ./run.sh help       this help   (LAN=1 ./run.sh to try it on your phone)
#
# Works with macOS' built-in bash 3.2. Needs Python >= 3.11 (python.org, Homebrew or pyenv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV="${VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"
LOGS="$ROOT/logs"
PREFS="$ROOT/.run-prefs"          # remembers "don't ask again" answers
export PORT_LLM=8003 PORT_CLEF=8001 PORT_CNN=8002 PORT_UI="${PORT_UI:-8080}"   # PORT_UI=9090 ./run.sh if 8080 is taken
UI_URL="http://localhost:$PORT_UI"

# ------------------------------------------------------------------ output helpers
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'
else
  B=""; DIM=""; RED=""; GRN=""; YEL=""; CYN=""; RST=""
fi
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$RST"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s!%s %s\n' "$YEL" "$RST" "$*"; }
die()  { printf '\n%s✗ %s%s\n' "$RED" "$*" "$RST" >&2; exit 1; }

interactive() { [ -t 0 ] && [ -t 1 ] && [ "${NONINTERACTIVE:-0}" != "1" ]; }

# ask "Question" Y|N  -> returns 0 for yes. Non-interactive: takes the default.
ask() {
  local q="$1" def="${2:-N}" hint ans
  [ "$def" = "Y" ] && hint="[Y/n]" || hint="[y/N]"
  if ! interactive; then [ "$def" = "Y" ]; return; fi
  printf '  %s?%s %s %s ' "$CYN" "$RST" "$q" "$hint"
  read -r ans || ans=""
  ans="$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')"
  [ -z "$ans" ] && ans="$(printf '%s' "$def" | tr '[:upper:]' '[:lower:]')"
  [ "$ans" = "y" ] || [ "$ans" = "yes" ]
}
prompt() {  # prompt "Label" "default" -> echo answer
  local label="$1" def="${2:-}" ans
  if [ -n "$def" ]; then printf '  %s %s[%s]%s: ' "$label" "$DIM" "$def" "$RST" >&2; else printf '  %s: ' "$label" >&2; fi
  read -r ans || ans=""
  printf '%s' "${ans:-$def}"
}
prompt_secret() {
  local label="$1" ans
  printf '  %s: ' "$label" >&2
  read -rs ans || ans=""
  printf '\n' >&2
  printf '%s' "$ans"
}
pref_set()  { touch "$PREFS"; grep -qx "$1" "$PREFS" 2>/dev/null || echo "$1" >> "$PREFS"; }
pref_has()  { [ -f "$PREFS" ] && grep -qx "$1" "$PREFS"; }

# ------------------------------------------------------------------ python + venv
find_python() {
  local c
  for c in ${PYTHON:-} python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# ensure_env full|light  (light = just what the LLM wizard needs, no PyTorch)
ensure_env() {
  local mode="$1" reqs stamp hash base
  if [ ! -x "$PY" ]; then
    base="$(find_python)" || die "Python 3.11 or newer is required. Install it (e.g. 'brew install python@3.12') and run ./run.sh again."
    step "Creating a Python environment in ${VENV#"$ROOT"/} ($("$base" -V))"
    "$base" -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
  fi
  if [ "$mode" = "light" ]; then
    # A full install also covers the light set.
    [ -f "$VENV/.stamp-full" ] && [ "$(cat "$VENV/.stamp-full")" = "$(req_hash full)" ] && return 0
    reqs="llm-server/requirements.txt"
  else
    reqs="clef-server/requirements.txt cnn/requirements.txt llm-server/requirements.txt"
  fi
  stamp="$VENV/.stamp-$mode"; hash="$(req_hash "$mode")"
  if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$hash" ] || [ "${FORCE_INSTALL:-0}" = "1" ]; then
    if [ "$mode" = "full" ]; then
      step "Installing Python packages (PyTorch, MLX on Apple Silicon, transformers, FastAPI, …): a few minutes the first time"
    else
      step "Installing the LLM packages"
    fi
    local args="" quiet="--quiet"; for r in $reqs; do args="$args -r $r"; done
    [ "$mode" = "full" ] && quiet=""   # big download: show pip's progress
    # shellcheck disable=SC2086
    "$PY" -m pip install $quiet --disable-pip-version-check $args || die "Package installation failed (see above)."
    echo "$hash" > "$stamp"
    ok "packages installed"
  fi
}
req_hash() {
  local files
  if [ "$1" = "light" ]; then files="llm-server/requirements.txt"
  else files="clef-server/requirements.txt cnn/requirements.txt llm-server/requirements.txt"; fi
  # shellcheck disable=SC2086
  cat $files | cksum | cut -d' ' -f1
}

# ------------------------------------------------------------------ state checks
llm_configured()  { [ -f llm-server/.env ] || [ -n "${LLM_API_KEY:-}${OPENAI_API_KEY:-}${LLM_BASE_URL:-}" ]; }
llm_summary() {   # "model via base URL" for status output
  local m b; m="$(env_value LLM_MODEL)"; b="$(env_value LLM_BASE_URL)"
  m="${m:-${LLM_MODEL:-gpt-4o-mini}}"; b="${b:-${LLM_BASE_URL:-https://api.openai.com/v1}}"
  echo "$m via $b"
}
cnn_trained()     { [ -f cnn/artifacts/model_meta.json ] && ls cnn/artifacts/hotdog_cnn*.pt >/dev/null 2>&1; }
clef_downloaded() { [ -x "$PY" ] && (cd clef-server && "$PY" download_model.py --check >/dev/null 2>&1); }
# Which CLEF-Flash weights this machine uses: "mlx" (4-bit, Apple Silicon) or "torch" (full BF16). Asks Python,
# so a Rosetta shell on an M-series Mac still gets the right answer; falls back to uname before the venv exists.
clef_backend() {
  local b=""
  [ -x "$PY" ] && b="$(cd clef-server && "$PY" -c 'import clef_backend as c; print(c.choose_backend())' 2>/dev/null || true)"
  if [ -z "$b" ]; then
    case "${CLEF_BACKEND:-auto}" in mlx|torch) b="$CLEF_BACKEND" ;;
      *) if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then b=mlx; else b=torch; fi ;;
    esac
  fi
  echo "$b"
}
clef_size() { if [ "$(clef_backend)" = "mlx" ]; then echo "~6 GB"; else echo "~19 GB"; fi; }
env_value()       { [ -f llm-server/.env ] && sed -n "s/^$1=//p" llm-server/.env | tail -1 | sed 's/[[:space:]]#.*//' | tr -d '"'"'" || true; }
ram_gb() {
  if [ "$(uname)" = "Darwin" ]; then echo $(( $(sysctl -n hw.memsize) / 1073741824 ))
  elif [ -r /proc/meminfo ]; then awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo
  else echo 0; fi
}

# ------------------------------------------------------------------ ./run.sh llm
cmd_llm() {
  interactive || die "./run.sh llm needs an interactive terminal (or write llm-server/.env yourself, see .env.example)."
  ensure_env light
  step "Configure the LLM contender"
  say "  Any vision-capable chat model behind an OpenAI-compatible API works."
  say ""
  say "    1) OpenAI          gpt-4o-mini: fast, cheap, real probabilities (needs an API key)"
  say "    2) OpenRouter      hundreds of models, some free (needs an API key)"
  say "    3) Ollama          fully local and free (needs Ollama + a vision model)"
  say "    4) LM Studio       fully local and free"
  say "    5) Other           any OpenAI-compatible endpoint"
  say ""
  local choice base key model
  choice="$(prompt 'Choose 1-5' '1')"
  case "$choice" in
    1) base="https://api.openai.com/v1"; model="gpt-4o-mini"
       say "  Create a key at https://platform.openai.com/api-keys"
       key="$(prompt_secret 'OpenAI API key (sk-…)')" ;;
    2) base="https://openrouter.ai/api/v1"; model="google/gemini-2.5-flash"
       say "  Create a key at https://openrouter.ai/keys  (':free' models work but are heavily rate-limited)"
       key="$(prompt_secret 'OpenRouter API key (sk-or-…)')" ;;
    3) base="http://localhost:11434/v1"; model="qwen2.5vl:7b"; key="ollama"
       say "  Install from https://ollama.com, then: ollama pull qwen2.5vl:7b" ;;
    4) base="http://localhost:1234/v1"; model=""; key="lm-studio"
       say "  In LM Studio: load a vision model and start the local server." ;;
    5) base="$(prompt 'Base URL (ends in /v1)' '')"; model=""
       key="$(prompt_secret 'API key (leave empty if none)')" ;;
    *) die "Please choose 1-5." ;;
  esac
  model="$(prompt 'Model' "$model")"
  [ -n "$base" ] && [ -n "$model" ] || die "Base URL and model are required."

  local max_tokens=512
  case "$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')" in
    *reason*|*think*|*r1*|o1*|o3*|o4*|*/o1*|*/o3*|*/o4*|*gpt-5*) max_tokens=2000
      warn "looks like a reasoning model: allowing it 2000 tokens to think" ;;
  esac
  if [ -f llm-server/.env ]; then
    cp llm-server/.env llm-server/.env.bak && chmod 600 llm-server/.env.bak
    ok "previous settings saved to llm-server/.env.bak"
  fi
  touch llm-server/.env && chmod 600 llm-server/.env   # the API key is only readable by you
  cat > llm-server/.env <<EOF
# Written by ./run.sh llm. Edit freely; see .env.example for all options.
LLM_BASE_URL=$base
LLM_API_KEY=${key:-none}
LLM_MODEL=$model
LLM_MAX_TOKENS=$max_tokens
EOF
  ok "saved llm-server/.env (only readable by you)"

  step "Checking the connection"
  if (cd "$ROOT" && "$PY" scripts/llm_check.py); then
    if ask "Send one small test picture to the model? (costs a fraction of a cent, free locally)" Y; then
      "$PY" scripts/llm_check.py --classify || warn "the test call failed. Fix the settings and run ./run.sh llm again."
    fi
  else
    warn "not reachable yet. Fix the settings above and run ./run.sh llm again."
  fi
  say "  ${DIM}If Bun Tribunal is already running, restart it (Ctrl+C, then ./run.sh) to use the new settings.${RST}"
}

# ------------------------------------------------------------------ ./run.sh train
kaggle_ready() {
  [ -f "$HOME/.kaggle/kaggle.json" ] || [ -f "$HOME/.kaggle/access_token" ] \
    || [ -n "${KAGGLE_API_TOKEN:-}" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }
}
cmd_train() {
  ensure_env full
  step "Train the CNN (ConvNeXt-Tiny on Food-101)"
  if [ "${SMOKE:-0}" != "1" ] && [ "${DATA_SOURCE:-food101}" = "kaggle" ] && ! kaggle_ready; then
    say "  The hot dog dataset comes from Kaggle, which needs a free account and an API token:"
    say "    sign in at https://www.kaggle.com  →  Settings  →  API  →  create a new token"
    say "  Paste the token below. (Older kaggle.json? Paste its \"key\" value; you'll be asked for the username.)"
    interactive || die "Kaggle credentials missing (~/.kaggle/access_token or ~/.kaggle/kaggle.json)."
    local token user
    token="$(prompt_secret 'Kaggle API token')"
    [ -n "$token" ] || die "No token entered. Run ./run.sh train again when you have one."
    mkdir -p "$HOME/.kaggle" && chmod 700 "$HOME/.kaggle"
    if printf '%s' "$token" | grep -qE '^[0-9a-f]{32}$'; then   # legacy 32-hex key -> kaggle.json
      user="$(prompt 'Kaggle username' '')"
      [ -n "$user" ] || die "The username is needed with a legacy key."
      printf '{"username":"%s","key":"%s"}\n' "$user" "$token" > "$HOME/.kaggle/kaggle.json"
      chmod 600 "$HOME/.kaggle/kaggle.json"; ok "saved ~/.kaggle/kaggle.json"
    else
      printf '%s\n' "$token" > "$HOME/.kaggle/access_token"
      chmod 600 "$HOME/.kaggle/access_token"; ok "saved ~/.kaggle/access_token"
    fi
  fi
  say "  What happens: download datasets (one-time, ~5 GB of Food-101 into cnn/data/), fine-tune,"
  say "  evaluate, export to cnn/artifacts/. Roughly 15–25 min on an Apple Silicon Mac."
  if [ "$(uname)" = "Darwin" ]; then say "  Keep the lid open; the Mac is kept awake while training."; fi
  if cnn_trained && ! ask "A trained model already exists. Train a new one (it will replace it)?" N; then return 0; fi
  local runner=( "$PY" run_training.py )
  command -v caffeinate >/dev/null 2>&1 && runner=( caffeinate -i "${runner[@]}" )
  local rc=0
  (cd cnn && "${runner[@]}") || rc=$?
  [ "$rc" = "130" ] && die "Training interrupted. Run ./run.sh train to start again."
  [ "$rc" = "0" ] || die "Training failed. The report in cnn/artifacts/training_report.html shows where."
  ok "model saved to cnn/artifacts/. Start ./run.sh to use it (restart it if it is already running)"
}

# ------------------------------------------------------------------ ./run.sh download
cmd_download() {
  ensure_env full
  local what; what="$(cd clef-server && "$PY" download_model.py --describe)" || die "clef-server is not set up correctly."
  step "Download CLEF-Flash: $what"
  if clef_downloaded; then ok "already downloaded"; return 0; fi
  local ram need use; ram="$(ram_gb)"
  if [ "$(clef_backend)" = "mlx" ]; then need=16; use="7–9 GB"; else need=32; use="about 20 GB"; fi
  if [ "$ram" -gt 0 ] && [ "$ram" -lt "$need" ]; then
    warn "this machine has ${ram} GB of memory; CLEF-Flash needs ${use} while running (${need} GB+ recommended)."
    ask "Download anyway?" N || return 0
  fi
  (cd clef-server && "$PY" download_model.py) || die "Download failed. Run ./run.sh download again to resume."
  ok "downloaded. It loads in the background the next time you start ./run.sh"
}

# ------------------------------------------------------------------ first-run / setup wizard
offer_setup() {
  if ! interactive; then [ "${1:-}" = "verbose" ] && warn "setup questions need an interactive terminal"; return 0; fi
  local missing=0
  if ! llm_configured && ! pref_has skip-llm; then
    missing=1
    step "The LLM isn't configured yet"
    if ask "Set it up now? (takes a minute)" Y; then ( cmd_llm ) || warn "LLM setup didn't finish. Later: ./run.sh llm"
    else pref_set skip-llm; warn "skipped. Later: ./run.sh llm"; fi
  fi
  if ! cnn_trained && ! pref_has skip-train; then
    missing=1
    step "The CNN isn't trained yet"
    if ask "Train it now? (~20 min, ~5 GB one-time download)" N; then ( cmd_train ) || warn "training didn't finish. Later: ./run.sh train"
    else pref_set skip-train; warn "skipped. Later: ./run.sh train"; fi
  fi
  if ! clef_downloaded && ! pref_has skip-download; then
    missing=1
    step "CLEF-Flash weights aren't downloaded yet"
    if ask "Download them now? ($(clef_size))" N; then ( cmd_download ) || warn "download didn't finish. Later: ./run.sh download (it resumes)"
    else pref_set skip-download; warn "skipped. Later: ./run.sh download"; fi
  fi
  if [ "$missing" = "1" ]; then say ""; say "  ${DIM}Models that aren't set up show a 'not set up' card with the command to fix it.${RST}"
  elif [ "${1:-}" = "verbose" ]; then ok "everything is set up"; fi
  return 0
}

# ------------------------------------------------------------------ start
port_free() { "$PY" -c "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sys.exit(s.connect_ex(('127.0.0.1', $1)) == 0)"; }
open_browser() {
  [ "${NO_BROWSER:-0}" = "1" ] && return 0
  if command -v open >/dev/null 2>&1 && [ "$(uname)" = "Darwin" ]; then open "$UI_URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$UI_URL" >/dev/null 2>&1 || true; fi
}

PIDS=""; NAMES=""
start_bg() {  # start_bg name dir cmd...
  local name="$1" dir="$2"; shift 2
  (cd "$dir" && exec "$@") > "$LOGS/$name.log" 2>&1 &
  PIDS="$PIDS $!"; NAMES="$NAMES $name"
}
stop_all() {
  trap - EXIT INT TERM
  # shellcheck disable=SC2086
  [ -n "$PIDS" ] && kill $PIDS 2>/dev/null || true
  wait 2>/dev/null || true
  printf '\n%sStopped. Bye!%s\n' "$DIM" "$RST"
  exit 0
}

cmd_start() {
  if [ ! -f "$VENV/.stamp-full" ]; then
    step "Welcome to Bun Tribunal! First-time setup"
    say "  LLM vs CLEF-Flash vs CNN: three ways to tell a hot dog from everything else."
  fi
  ensure_env full
  offer_setup

  local p
  for p in $PORT_LLM $PORT_CLEF $PORT_CNN $PORT_UI; do
    port_free "$p" || die "Port $p is already in use. Is Bun Tribunal already running in another terminal? (find it with: lsof -i :$p)"
  done

  mkdir -p "$LOGS"
  trap stop_all EXIT INT TERM
  # Local only by default; LAN=1 also serves your Wi-Fi, e.g. to try it on a phone.
  local bind=127.0.0.1 lan_ip=""
  if [ "${LAN:-0}" = "1" ]; then
    bind=0.0.0.0
    lan_ip="$("$PY" -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.255.255.255', 1)); print(s.getsockname()[0])" 2>/dev/null || true)"
  fi
  step "Starting Bun Tribunal"
  start_bg llm  llm-server   "$PY" -m uvicorn app:app --host "$bind" --port "$PORT_LLM"
  start_bg clef clef-server  "$PY" -m uvicorn app:app --host "$bind" --port "$PORT_CLEF"
  start_bg cnn  cnn/server   "$PY" -m uvicorn app:app --host "$bind" --port "$PORT_CNN"
  start_bg ui   ui           "$PY" -m http.server "$PORT_UI" --bind "$bind"
  "$PY" scripts/health.py --wait 20
  say ""
  say "  ${B}Open ${CYN}$UI_URL${RST}   ${DIM}(logs in logs/ · Ctrl+C stops everything)${RST}"
  [ -n "$lan_ip" ] && say "  ${B}On your phone (same Wi-Fi): ${CYN}http://$lan_ip:$PORT_UI${RST}"
  open_browser

  # Keep running; report a service that dies (e.g. a crash) instead of failing silently.
  local dead="" i pid name
  while true; do
    sleep 2
    i=0
    for pid in $PIDS; do
      # NAMES is a space-separated list; $NAMES stays unquoted so echo drops the leading space.
      # shellcheck disable=SC2086
      i=$((i + 1)); name="$(echo $NAMES | cut -d' ' -f$i)"
      if ! kill -0 "$pid" 2>/dev/null && ! echo " $dead " | grep -q " $name "; then
        dead="$dead $name"
        printf '\n%s✗ %s stopped unexpectedly. Last lines of logs/%s.log:%s\n' "$RED" "$name" "$name" "$RST"
        tail -n 15 "$LOGS/$name.log" | sed 's/^/    /'
      fi
    done
  done
}

# ------------------------------------------------------------------ status / test / help
cmd_status() {
  step "Setup"
  if [ -f "$VENV/.stamp-full" ]; then ok "Python environment (.venv)"; else warn "Python environment not installed yet (./run.sh)"; fi
  if llm_configured; then ok "LLM configured: $(llm_summary)"; else warn "LLM not configured (./run.sh llm)"; fi
  if cnn_trained; then
    ok "CNN trained: $("$PY" -c "import json; m=json.load(open('cnn/artifacts/model_meta.json')); t=m.get('metrics',{}); print(m.get('arch','?'), '· test accuracy', format(t.get('test_acc',0),'.1%'), '·', m.get('trained_at','')[:10])" 2>/dev/null || echo yes)"
  else warn "CNN not trained (./run.sh train)"; fi
  if clef_downloaded; then ok "CLEF-Flash weights downloaded: $(cd clef-server && "$PY" download_model.py --describe 2>/dev/null)"; else warn "CLEF-Flash weights not downloaded (./run.sh download, $(clef_size))"; fi
  step "Servers"
  if [ -x "$PY" ]; then "$PY" scripts/health.py; else say "  not running"; fi
}

cmd_test() {
  ensure_env full
  local failed=0 dir
  for dir in clef-server cnn llm-server; do
    step "Tests: $dir"
    (cd "$dir" && "$PY" -m pytest -q -p no:cacheprovider) || failed=1
  done
  [ "$failed" = "0" ] && step "${GRN}All tests passed${RST}" || die "Some tests failed."
}

cmd_help() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  say ""
  say "Environment knobs: DEVICE=mps|cuda|cpu · ARCH=convnext_tiny|efficientnet_b0|mobilenet_v3_large (train)"
  say "  ADD_FOOD101=0 (train without Food-101) · LAN=1 (open to phones on your Wi-Fi) · NO_BROWSER=1"
  say "  NONINTERACTIVE=1 (never ask) · FORCE_INSTALL=1 (reinstall packages)"
}

case "${1:-start}" in
  start|"")      cmd_start ;;
  llm)           cmd_llm ;;
  train)         cmd_train ;;
  download)      cmd_download ;;
  setup)         ensure_env full; rm -f "$PREFS"; offer_setup verbose ;;
  status)        cmd_status ;;
  test|tests)    cmd_test ;;
  help|-h|--help) cmd_help ;;
  *) say "Unknown command: $1"; say ""; cmd_help; exit 1 ;;
esac
