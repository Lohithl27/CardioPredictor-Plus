# Update: registration fix, phone/laptop photos, edit patient, HW-827 sensor

## 1. Find out WHY registration failed (the "Request failed (500)" error)
On the Raspberry Pi, open Terminal and run ONE of these:

If you start the server with the service:
    sudo journalctl -u cardiopredictor -n 40 --no-pager

If you start it by hand, look at the Terminal window where uvicorn is running.
The last lines say the real reason (for example "PermissionError").

The most common cause is that the `data` folder belongs to `root` (because the
installer or unzip ran with sudo) while the server runs as `pi`, so it cannot
create the patient folder. Fix it with:

    cd ~/CardioPredictor          # change if your folder is somewhere else
    sudo chown -R $USER:$USER data
    mkdir -p data/patients

After applying this update the dashboard shows the real reason on screen
instead of "Request failed (500)", and /health shows `data_folder_error`.

## 2. Apply the update (only the changed files)
Copy the `CardioPredictor_update_v6` folder to the Pi, then:

    cd CardioPredictor_update_v6
    bash apply_update.sh ~/CardioPredictor

It copies 8 files, fixes the data folder owner, and restarts the service
(or tells you to restart the server if you run it by hand).

## 3. What changed
- api/main.py            readable 500 errors, data-folder check at start-up, PUT /patients/{id} (edit), /health details
- api/static/index.html  photo from phone/laptop camera or any image file in the register form, "Edit details" button
- src/patients.py        accepts any image (JPEG, PNG, WebP...) and stores it as JPEG; edit; write check
- src/hardware/camera.py CAMERA_BACKEND now defaults to "auto": Pi camera, then USB, else a clear message
- src/hardware/hw827.py  NEW: HW-827 pulse sensor through an ADS1115 ADC
- src/vitals.py          heart rate from a single waveform (HW-827)
- src/sensor_service.py  SENSOR_MODE=hw827, no busy loop when the sensor FIFO is empty
- tests/                 5 new tests (58 pass)

## 4. Photos when the Pi has no camera
Open the dashboard from a phone or laptop on the same network (or Tailscale):
- Phone: "Use this device's camera" opens the phone camera. "Choose a photo file" opens the gallery.
- Laptop: "Choose a photo file" (any JPEG/PNG/WebP). Big photos are shrunk in the browser.
The Raspberry Pi camera is optional now.

## 5. HW-827 heart-rate sensor (needs an ADS1115 board)
The Raspberry Pi has no analog inputs, and the HW-827 gives an analog signal,
so it must go through an ADS1115 (I2C 16-bit ADC board, about Rs 150-300).
Power everything from 3.3 V, never 5 V.

    HW-827 +  (VCC)  -> Pi pin 1  (3.3 V)
    HW-827 -  (GND)  -> Pi pin 6  (GND)
    HW-827 S  (signal) -> ADS1115 A0
    ADS1115 VDD -> Pi pin 1 (3.3 V)     ADS1115 GND -> Pi pin 6 (GND)
    ADS1115 SDA -> Pi pin 3             ADS1115 SCL -> Pi pin 5
    ADS1115 ADDR -> GND  (address 0x48)

Check it is seen:   i2cdetect -y 1     (you should see 48)

Tell the server to use it. Add this line to /etc/cardiopredictor.env:

    SENSOR_MODE=hw827

    sudo nano /etc/cardiopredictor.env
    sudo systemctl restart cardiopredictor

The chip then reads "Sensor connected (HW-827 pulse sensor)". This sensor gives
heart rate only: SpO2 stays "-" (it has one light, not two). If the sensor is
not found the page says NOT CONNECTED with the reason, and you type the heart rate.
Leave SENSOR_MODE unset (auto) to try MAX30102 first, then HW-827, then simulation.

Finger detection uses a threshold of 40 ADC counts. If it says "No finger"
with a finger on, or shows a pulse with no finger, change it in
/etc/cardiopredictor.env, e.g. HW827_MIN_AC=25 or HW827_MIN_AC=80.
This was NOT tested on a real HW-827 yet (no hardware here): treat it as untested.
