#!/usr/bin/env bash
set -euo pipefail

# One-click startup script for WSL
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

VENV_DIR=".venv-linux"
REQ_FILE="requirements.txt"
ENV_FILE=".env"
ENV_EXAMPLE_FILE="env.example"
DEPS_STAMP="$VENV_DIR/.deps_installed"

echo "[1/5] Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found in WSL. Please install Python 3 first."
  exit 1
fi

echo "[2/5] Preparing virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/5] Checking .env..."
if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$ENV_EXAMPLE_FILE" ]; then
    cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
    echo "Created .env from env.example. Please edit API keys if needed."
  else
    echo "Warning: .env and env.example are both missing."
  fi
fi

echo "[4/5] Installing/updating dependencies..."
if [ ! -f "$DEPS_STAMP" ] || [ "$REQ_FILE" -nt "$DEPS_STAMP" ]; then
  python3 -m pip install --upgrade pip
  pip install -r "$REQ_FILE"
  touch "$DEPS_STAMP"
else
  echo "Dependencies are up to date."
fi

echo "[5/5] Starting API service..."
python -m app.run
