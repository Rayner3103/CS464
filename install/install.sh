#!/usr/bin/env bash
# lscli installer (docs/lscli/spec.md section 8):
#
#   curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/learned-selector-cli/install/install.sh | bash
#
# Installs the learned-selector CLI and its daemon for THIS user: a git clone in $LSCLI_HOME/src, a venv with the pinned
# versions, the zero-shot models, ~/.local/bin/lscli, the config, and a systemd --user unit. Needs no root, except for
# the optional clock helper (asked; --clock-helper / --no-clock-helper). Re-running upgrades src/ and the venv and keeps
# the data. Without a terminal (piped into bash with no /dev/tty) every question takes its default (no).
set -euo pipefail

REPO_URL="https://github.com/Rayner3103/CS464.git"
REF="learned-selector-cli"
LSCLI_HOME="${LSCLI_HOME:-$HOME/.local/share/lscli}"
CONFIG_DIR="$HOME/.config/lscli"
BIN_DIR="$HOME/.local/bin"
UNIT_DIR="$HOME/.config/systemd/user"
PYTHON=""
REUSE_SITE=""
CLOCK_HELPER=ask
CLOCK_MHZ=""
LINGER=ask
START=1
YES=0
CHECK_ONLY=0
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
# Zero-shot models (the RTX 3080's rankers, cuBLASLt ranker and router), a release asset of $REF. The hash is pinned
# HERE, not fetched from the release, so a replaced asset is refused. Empty = none published yet: the daemon then
# trains this GPU's own models before it selects anything with a Triton ranker (install/pack_zero_shot.py makes one).
MODELS_URL=""
MODELS_SHA256=""

usage() {
  cat <<EOF
Usage: install.sh [options]
  --ref REF               git branch / tag to install (default $REF)
  --repo URL              repository (default $REPO_URL)
  --home DIR              state directory (default ~/.local/share/lscli; also \$LSCLI_HOME)
  --python BIN            interpreter for the daemon venv (default: python3.12 / python3.11 / python3 >= 3.11)
  --models-url URL        zero-shot model archive (with --models-sha256; overrides the pinned one)
  --models-sha256 HEX
  --clock-helper          install the SM clock helper (sudo) without asking; --clock-mhz MHZ sets its clock
  --no-clock-helper
  --linger | --no-linger  keep the daemon running when you are logged out (loginctl enable-linger)
  --no-start              install the unit but do not start it
  --yes                   answer every question with its default (non-interactive)
  --check-only            run the preflight checks and exit
  --reuse-site PYTHON     (development) give the venv PYTHON's site-packages instead of installing the requirements
  -h, --help
EOF
}

die() { echo "lscli install: $*" >&2; exit "${2:-1}"; }
say() { printf '==> %s\n' "$*"; }
ask() {  # ask "question" default(y|n) -> 0 for yes
  local q="$1" def="$2" ans=""
  if [[ $YES == 1 || ! -r /dev/tty ]]; then [[ $def == y ]]; return; fi
  read -r -p "$q [$( [[ $def == y ]] && echo Y/n || echo y/N )] " ans < /dev/tty || ans=""
  ans="${ans:-$def}"; [[ $ans == [yY]* ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="${2:?}"; shift ;;
    --repo) REPO_URL="${2:?}"; shift ;;
    --home) LSCLI_HOME="${2:?}"; shift ;;
    --python) PYTHON="${2:?}"; shift ;;
    --models-url) MODELS_URL="${2:?}"; shift ;;
    --models-sha256) MODELS_SHA256="${2:?}"; shift ;;
    --clock-helper) CLOCK_HELPER=yes ;;
    --no-clock-helper) CLOCK_HELPER=no ;;
    --clock-mhz) CLOCK_MHZ="${2:?}"; shift ;;
    --linger) LINGER=yes ;;
    --no-linger) LINGER=no ;;
    --no-start) START=0 ;;
    --yes) YES=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --reuse-site) REUSE_SITE="${2:?}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option $1" 2 ;;
  esac
  shift
done
LSCLI_HOME="$(mkdir -p "$LSCLI_HOME" && cd "$LSCLI_HOME" && pwd)"
chmod 700 "$LSCLI_HOME"

# ------------------------------------------------------------------ preflight (exit 4: unsupported system)
preflight() {
  say "preflight"
  [[ "$(uname -s)" == Linux ]] || die "Linux only" 4
  command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1 \
    || die "no systemd user session (systemctl --user)" 4
  command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi does not work: NVIDIA driver missing?" 4
  local n; n="$(nvidia-smi -L | wc -l)"
  [[ $n -ge 1 ]] || die "no NVIDIA GPU" 4
  [[ $n -eq 1 ]] || echo "warning: $n GPUs; lscli v1 uses device 0 only" >&2
  command -v git >/dev/null || die "git is needed" 4
  command -v gcc >/dev/null || die "gcc is needed (Triton builds its launcher with it)" 4
  if [[ -z "$PYTHON" ]]; then
    for c in python3.12 python3.11 python3; do
      if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
        PYTHON="$(command -v "$c")"; break
      fi
    done
  fi
  [[ -n "$PYTHON" ]] && "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || die "Python >= 3.11 is needed for the daemon venv (--python BIN)" 4
  "$PYTHON" -c 'import os, sysconfig, sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
    || die "the Python headers are missing for $PYTHON (e.g. apt install python3-dev); Triton needs them" 4
  "$PYTHON" -c 'import venv' 2>/dev/null || die "$PYTHON has no venv module (e.g. apt install python3-venv)" 4
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0); Python: $PYTHON"
}
preflight
[[ $CHECK_ONLY == 1 ]] && { echo "preflight OK"; exit 0; }

# ------------------------------------------------------------------ source
SRC="$LSCLI_HOME/src"
say "source: $REPO_URL @ $REF -> $SRC"
if [[ -d "$SRC/.git" ]]; then
  git -C "$SRC" fetch --quiet origin "$REF"
  git -C "$SRC" checkout --quiet --detach FETCH_HEAD
else
  rm -rf "$SRC"
  git clone --quiet --branch "$REF" "$REPO_URL" "$SRC" 2>/dev/null || {
    git clone --quiet "$REPO_URL" "$SRC" && git -C "$SRC" checkout --quiet --detach "$REF"; }
fi
echo "at $(git -C "$SRC" rev-parse --short HEAD)"

# ------------------------------------------------------------------ venv
VENV="$LSCLI_HOME/venv"
say "venv: $VENV"
if [[ -n "$REUSE_SITE" ]]; then
  [[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv --without-pip "$VENV"
  site="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  "$REUSE_SITE" -c 'import site; print("\n".join(site.getsitepackages()))' > "$site/lscli-reuse.pth"
else
  [[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  torch_pin="$(grep -E '^torch==' "$SRC/requirements.txt" | awk '{print $1}')"
  "$VENV/bin/python" -m pip install --quiet "$torch_pin" --index-url "$TORCH_INDEX"
  grep -vE '^torch==' "$SRC/requirements.txt" | "$VENV/bin/python" -m pip install --quiet -r /dev/stdin
fi
site="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$SRC" > "$site/lscli-src.pth"                 # the daemon venv runs lscli / tune / ... from src/
"$VENV/bin/python" -c 'import torch, triton, lightgbm, lscli' || die "the venv cannot import its packages"

# ------------------------------------------------------------------ models and kernels
if [[ -n "$MODELS_URL" ]]; then
  [[ -n "$MODELS_SHA256" ]] || die "--models-url needs --models-sha256 (or a pinned MODELS_SHA256)"
  say "zero-shot models"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  curl -fsSL "$MODELS_URL" -o "$tmp/models.tar.gz" || die "download failed: $MODELS_URL"
  echo "$MODELS_SHA256  $tmp/models.tar.gz" | sha256sum -c --quiet - || die "zero-shot models: sha256 mismatch"
  rm -rf "$LSCLI_HOME/models/zero_shot.new"
  mkdir -p "$LSCLI_HOME/models/zero_shot.new"
  tar -xzf "$tmp/models.tar.gz" -C "$LSCLI_HOME/models/zero_shot.new" --no-same-owner
  [[ -f "$LSCLI_HOME/models/zero_shot.new/MODELS.json" ]] || die "the model archive has no MODELS.json"
  rm -rf "$LSCLI_HOME/models/zero_shot"
  mv "$LSCLI_HOME/models/zero_shot.new" "$LSCLI_HOME/models/zero_shot"
else
  echo "note: no zero-shot models for this release; the daemon trains this GPU's models before it selects kernels"
fi
(cd "$SRC" && "$VENV/bin/python" -m tune bundle unpack >/dev/null 2>&1) \
  && echo "kernel bundle unpacked" || echo "note: kernel bundle not unpacked (other Triton / GPU, or none)"

# ------------------------------------------------------------------ config, command, unit
mkdir -p "$CONFIG_DIR" "$BIN_DIR" "$UNIT_DIR"
[[ -f "$CONFIG_DIR/config.toml" ]] || \
  LSCLI_CONFIG="$CONFIG_DIR/config.toml" "$VENV/bin/python" -c \
    'import os; from lscli import config; open(os.environ["LSCLI_CONFIG"], "w").write(config.dumps(config.DEFAULTS))'
cat > "$BIN_DIR/lscli" <<EOF
#!/bin/sh
export LSCLI_HOME="$LSCLI_HOME"
exec "$VENV/bin/python" -m lscli "\$@"
EOF
chmod 755 "$BIN_DIR/lscli"
sed -e "s|%h/.local/share/lscli|$LSCLI_HOME|g" \
    -e "s|^\[Service\]|[Service]\nEnvironment=LSCLI_HOME=$LSCLI_HOME|" \
    "$SRC/install/lscli.service" > "$UNIT_DIR/lscli.service"
systemctl --user daemon-reload

# ------------------------------------------------------------------ clock helper (the only root step)
if [[ $CLOCK_HELPER == ask ]]; then
  ask "Install the SM clock helper (needs sudo; measurements at a locked clock)?" n && CLOCK_HELPER=yes || CLOCK_HELPER=no
fi
if [[ $CLOCK_HELPER == yes ]]; then
  max="$(nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits | tr -d ' ')"
  def=$(( max * 85 / 100 ))
  if [[ -z "$CLOCK_MHZ" ]]; then
    if [[ $YES == 1 || ! -r /dev/tty ]]; then CLOCK_MHZ=$def
    else read -r -p "Lock MHz (max $max) [$def] " CLOCK_MHZ < /dev/tty || true; CLOCK_MHZ="${CLOCK_MHZ:-$def}"; fi
  fi
  [[ "$CLOCK_MHZ" =~ ^[0-9]{3,4}$ ]] && (( CLOCK_MHZ <= max )) || die "bad clock $CLOCK_MHZ (max $max)"
  say "clock helper (sudo): /usr/local/libexec/lscli-clock, lock at $CLOCK_MHZ MHz"
  tmp_sudo="$(mktemp)"
  printf '%s ALL=(root) NOPASSWD: /usr/local/libexec/lscli-clock lock, /usr/local/libexec/lscli-clock reset\n' \
    "$(id -un)" > "$tmp_sudo"
  sudo visudo -cf "$tmp_sudo" >/dev/null || die "the sudoers rule did not validate"
  sudo install -d -o root -g root -m 0755 /usr/local/libexec /etc/lscli
  sudo install -o root -g root -m 0755 "$SRC/install/lscli-clock" /usr/local/libexec/lscli-clock
  printf 'lock_mhz=%s\n' "$CLOCK_MHZ" | sudo tee /etc/lscli/clock.conf >/dev/null
  sudo chown root:root /etc/lscli/clock.conf && sudo chmod 0644 /etc/lscli/clock.conf
  sudo install -o root -g root -m 0440 "$tmp_sudo" /etc/sudoers.d/lscli
  rm -f "$tmp_sudo"
  LSCLI_CONFIG="$CONFIG_DIR/config.toml" LSCLI_HOME="$LSCLI_HOME" "$VENV/bin/python" -m lscli config clock.lock_mhz "$CLOCK_MHZ" >/dev/null
fi

# ------------------------------------------------------------------ linger + start
if [[ $LINGER == ask ]]; then
  ask "Keep the daemon running while you are logged out (loginctl enable-linger)?" n && LINGER=yes || LINGER=no
fi
[[ $LINGER == yes ]] && { loginctl enable-linger "$(id -un)" || echo "warning: enable-linger failed" >&2; }
if [[ $START == 1 ]]; then
  systemctl --user enable --now lscli.service
else
  systemctl --user enable lscli.service
fi

cat <<EOF

lscli is installed ($LSCLI_HOME).
  - next: register the Python environment your programs use, then run them with LSCLI_PATCH=1:
      lscli enable-env \$(command -v python3)
  - check: lscli status   |   lscli doctor
EOF
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) echo "  - add $BIN_DIR to your PATH" ;; esac
