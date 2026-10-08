#!/usr/bin/env bash
# Start the Meridian web dashboard. First run creates .venv and installs
# requirements.txt; later runs only reinstall when requirements.txt changes.
#   ./meridian.sh                 → http://127.0.0.1:8788 (opens your browser)
#   MERIDIAN_WEB_PORT=8787 ./meridian.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "python3 not found — install Python 3.9+ first." >&2; exit 1; }

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtualenv in .venv ..."
    "$PY" -m venv .venv
fi
if ! cmp -s requirements.txt .venv/.requirements.installed; then
    echo "Installing dependencies ..."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
    cp requirements.txt .venv/.requirements.installed
fi

exec .venv/bin/python web_server.py "$@"
