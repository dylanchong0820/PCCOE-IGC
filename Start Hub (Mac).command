#!/bin/bash
# Double-click to start the S.E.D.O.S.S. farmer hub.
# First run: creates the project's Python environment and installs the packages (a few minutes).
# Later runs: starts the hub and opens it in the browser. Close this window (or Ctrl+C) to stop the hub.

cd "$(dirname "$0")" || exit 1

stop() {
  echo
  echo "$1"
  read -r -p "Press Enter to close this window."
  exit 1
}

echo "S.E.D.O.S.S. farmer hub"
echo

if [ ! -x .venv/bin/python ]; then
  PY=""
  for v in 3.14 3.13 3.12 3.11 3.10; do
    if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
  done
  if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1 &&
     python3 -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)'; then
    PY="python3"
  fi
  [ -n "$PY" ] || stop "Python 3.10 to 3.14 was not found. Install Python 3.14 from python.org/downloads, then double-click this file again."

  echo "First run: setting up with $PY. This takes a few minutes, please wait..."
  "$PY" -m venv .venv || stop "Could not create the Python environment."
fi

if ! .venv/bin/python -c "import numpy, pandas, xgboost, yaml, sklearn, matplotlib, requests" >/dev/null 2>&1; then
  echo "Installing the packages..."
  .venv/bin/python -m pip install --disable-pip-version-check -q -r ml/requirements.txt ||
    stop "Installing the packages failed. Check the internet connection and try again."
  if ! .venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
    stop "XGBoost needs the OpenMP library. Install Homebrew (brew.sh), run:  brew install libomp  then double-click this file again."
  fi
  echo "Setup finished."
  echo
fi

.venv/bin/python ml/run_bench.py --serve --open "$@"
