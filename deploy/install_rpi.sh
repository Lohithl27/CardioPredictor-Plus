#!/usr/bin/env bash
# One-time setup for Raspberry Pi OS. Run from the project folder:
#   bash deploy/install_rpi.sh
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"

echo "==> Installing system packages (camera, I2C, Python tools)"
sudo apt update
sudo apt install -y unzip python3-venv python3-pip python3-picamera2 i2c-tools libopenjp2-7

echo "==> Enabling I2C (for the MAX30102 sensor and optional OLED)"
sudo raspi-config nonint do_i2c 0
echo "    (I2C takes full effect after a reboot. Step 7 of the guide covers this.)"

echo "==> Creating virtual environment and installing Python packages"
cd "$PROJECT"
python3 -m venv --system-site-packages .venv   # keeps the apt picamera2 package visible
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-rpi.txt

echo "==> Training the model if it is missing"
if [ ! -f models/best_model.joblib ]; then
  .venv/bin/python src/train.py
fi

echo "==> Preparing data folder and settings"
mkdir -p data/patients
if [ ! -f /etc/cardiopredictor.env ]; then
  DASH_PW="$(openssl rand -base64 12 | tr -d '/+=' | cut -c1-16)"
  DEV_KEY="$(openssl rand -hex 16)"
  sudo tee /etc/cardiopredictor.env >/dev/null <<ENV
# Dashboard login password (any username is accepted).
CARDIO_DASHBOARD_PASSWORD=$DASH_PW
# Key for the optional ESP8266 device endpoints.
CARDIO_DEVICE_KEY=$DEV_KEY
ENV
  sudo chmod 600 /etc/cardiopredictor.env
  echo
  echo "    Dashboard password created: $DASH_PW"
  echo "    WRITE IT DOWN. It is also stored in /etc/cardiopredictor.env"
  echo
fi

echo "==> Installing the background service"
sed -e "s#__USER__#$USER_NAME#g" -e "s#__PROJECT__#$PROJECT#g" deploy/cardio.service \
  | sudo tee /etc/systemd/system/cardiopredictor.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now cardiopredictor.service

echo
echo "Done. Next: reboot with  sudo reboot  then open http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
echo "Check status with:  sudo systemctl status cardiopredictor"
