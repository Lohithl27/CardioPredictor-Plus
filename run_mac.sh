#!/bin/bash
# CardioPredictor+ : one-command setup and run for macOS (also works on Linux).
# Usage (in Terminal, from inside the CardioPredictor folder):
#     bash run_mac.sh
# Leave this Terminal window open. Closing it (or Ctrl+C) stops the server.

set -e
cd "$(dirname "$0")"

# --- Settings for the Mac ---------------------------------------------------
# No sensor is connected on the Mac, so the sensor is switched off and the
# heart rate is typed in by hand. The app will say "NOT CONNECTED" instead of
# showing a fake live reading. Change to SENSOR_MODE=simulate only for demos.
export SENSOR_MODE="${SENSOR_MODE:-off}"
export CAMERA_BACKEND="${CAMERA_BACKEND:-none}"
# Only this Mac can open the dashboard. Set HOST=0.0.0.0 to allow other devices.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# --- 1. Check Python (3.10 or newer) ----------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Run:  brew install python@3.12"
  echo "(or download it from https://www.python.org/downloads/)"
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Your python3 is too old: $(python3 --version)"
  echo "Install a newer one with:  brew install python@3.12"
  exit 1
fi

# --- 2. Python virtual environment (created once) ---------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# --- 3. Dependencies (fast on repeat runs) ----------------------------------
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

# --- 4. Train the model only if it is missing --------------------------------
if [ ! -f models/best_model.joblib ]; then
  python src/train.py
fi

export CARDIO_DEVICE_KEY="${CARDIO_DEVICE_KEY:-change-this-long-secret}"

echo ""
echo "======================================================"
echo " Dashboard (website): http://127.0.0.1:${PORT}/dashboard/"
echo " API docs:            http://127.0.0.1:${PORT}/docs"
echo " Sensor mode:         ${SENSOR_MODE} (heart rate is typed in by hand)"
echo " Stop the server:     press Ctrl+C in this window"
echo "======================================================"
echo ""

uvicorn api.main:app --host "${HOST}" --port "${PORT}"
