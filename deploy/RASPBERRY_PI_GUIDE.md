# CardioPredictor+ on Raspberry Pi: step by step

This guide runs the whole system on one Raspberry Pi 4 (4 GB). It keeps running
when you close your laptop, and any phone or PC on the same Wi-Fi can open the website.

## 1. What you need

| Part | Recommended | Notes |
|---|---|---|
| Computer | Raspberry Pi 4B, 4 GB | Runs everything. Use a 32 GB+ microSD, or better a USB SSD. |
| Pi 3 (4 GB listed) | Optional | Works as a second sensor station (see section 9). Heavy report and model work is slower on a Pi 3. |
| Sensor | MAX30102 module | Heart rate and SpO2. Wire to GPIO pins below. |
| Camera | See the table in section 5 | |
| Display (optional) | 0.96" SSD1306 OLED | Shows heart rate on the device. Not required. |

## 2. Wiring the MAX30102 (and optional OLED)

| Module pin | Pi pin | Notes |
|---|---|---|
| VIN | Pin 1 (3.3 V) | Use 3.3 V only. |
| GND | Pin 6 (GND) | |
| SDA | Pin 3 (GPIO 2) | I2C data |
| SCL | Pin 5 (GPIO 3) | I2C clock |
| INT | not connected | |

The OLED (if fitted) goes on the same SDA and SCL pins, with address 0x3C.
Check the wiring with `sudo i2cdetect -y 1`. You should see `57` (MAX30102), and `3c` if the OLED is fitted.

## 3. Install Raspberry Pi OS

1. Use Raspberry Pi Imager. Choose **Raspberry Pi OS (64-bit)**.
2. In the settings (gear icon), set the hostname (e.g. `cardio-pi`), enable SSH, set your Wi-Fi name, password and country.
3. Boot the Pi, then connect from your laptop: `ssh <user>@cardio-pi.local`.
4. Update: `sudo apt update && sudo apt full-upgrade -y && sudo reboot`.

## 4. Copy the project and install

On the Pi:

```bash
cd ~
# Copy the CardioPredictor folder here (USB stick, scp, or git clone)
cd CardioPredictor
bash deploy/install_rpi.sh
```

The script installs packages, trains the model if needed, sets up the background
service, and starts it. It takes 10–20 minutes on a Pi 4.

Then edit the passwords:

```bash
sudo nano /etc/cardiopredictor.env   # set CARDIO_DASHBOARD_PASSWORD and CARDIO_DEVICE_KEY
sudo systemctl restart cardiopredictor
```

## 5. Choosing a camera

| You have | Use | Setting |
|---|---|---|
| Camera Module 3 (12 MP) | Pi camera port (CSI) | `CAMERA_BACKEND=picamera2` |
| Camera Module 1 (5 MP) | Pi camera port (CSI) | `CAMERA_BACKEND=picamera2` |
| 2 MP module | Usually an OV2640 board, which Picamera2 does not support. Use a USB webcam instead. | `CAMERA_BACKEND=usb` |
| USB webcam | USB port | `CAMERA_BACKEND=usb`, `CAMERA_INDEX=0` |

To change the camera, edit `/etc/systemd/system/cardiopredictor.service` (the `Environment=CAMERA_BACKEND=` line), then run `sudo systemctl daemon-reload && sudo systemctl restart cardiopredictor`.

Test the camera: `curl -X POST http://localhost:8000/patients/<patient_id>/photo/capture`

## 6. Opening the website from other devices

1. Find the Pi's address: `hostname -I`. Example: `192.168.1.42`.
2. On any phone or PC on the **same Wi-Fi**, open: `http://192.168.1.42:8000/dashboard/`
   You can also try `http://cardio-pi.local:8000/dashboard/`.
3. Log in with the dashboard password from `/etc/cardiopredictor.env` (any username).

To stop the Pi getting a new address, set a fixed IP for it in your router's settings (DHCP reservation).

**Away from home:** do not open a port on your router. Install Tailscale on the Pi and on your phone (free personal plan), then use the Pi's Tailscale address. Share links will only work for people who can reach the Pi.

## 7. Everyday use

1. Open the dashboard and go to **1 · Patient register**.
2. Register a new patient. Tick **Take the registration photo** to take their photo at the same time. Their folder is created as `data/patients/<patient_id>/`.
3. Go to **2 · Finger measurement**. Place a finger on the sensor and press **Start measurement**. The reading is saved to the patient's folder.
4. Go to **3 · Risk assessment**. Heart rate is filled in from the measurement. Enter the other values and run the assessment.
5. Go to **4 · Reports** to download a PNG, Word or text report.
6. Go to **5 · Share** to create a read-only link for the patient. It expires after the number of days you choose.

## 8. Finding old data

Every patient has a folder:

```
data/patients/<patient_id>/
    patient.json       name, age, sex, notes, share link
    photos/            registration photo and any later photos
    measurements/      one JSON file per finger measurement
    assessments/       one JSON file per risk assessment (inputs and results)
    reports/           every PNG, TXT and DOCX report generated
```

Open the folder with `ls ~/CardioPredictor/data/patients/` or copy it to a USB stick.
Use `sqlite3 data/predictions.db` for the trend history.

## 9. Optional: second Pi as a sensor station

A Pi 3 can run only the sensor, camera and a tiny forwarder, while the Pi 4 does the work. This is an extension, not built yet. Ask before setting it up.

## 10. Checks and troubleshooting

| Symptom | Check |
|---|---|
| Dashboard does not load | `sudo systemctl status cardiopredictor` and `journalctl -u cardiopredictor -n 50` |
| Dashboard says SIMULATED | The sensor was not found. Run `sudo i2cdetect -y 1` and check the wiring. |
| "No finger" | Press the finger firmly over the sensor and keep still. The sensor should glow red. |
| Camera error | Check the cable and the `CAMERA_BACKEND` setting (section 5). |
| Reports fail | `journalctl -u cardiopredictor -n 50`. Often the packages are missing: re-run `.venv/bin/pip install -r requirements-rpi.txt`. |
| Server stops after reboot | `sudo systemctl enable cardiopredictor` |

## 11. Important limits

- The 12-year figure extends a 10-year model by assuming a steady yearly risk. It is an approximation, not a validated 12-year model.
- The MAX30102 is a hobby-grade sensor. SpO2 is approximate and not clinically calibrated.
- Blood pressure, cholesterol and glucose are typed in by hand.
- Patient photos and records are health data. Keep the Pi physically secure, back up `data/`, and get patient consent before storing photos.
