# CardioPredictor+ — macOS guide

A research prototype that estimates 12-year cardiovascular risk from the
Framingham risk factors, with a clinic-style web dashboard. Patients are
registered, photographed, measured and assessed, and every result is saved in
that patient's own folder. Reports come out as PNG, TXT or DOCX, and each
patient can get a read-only share link.

> Research prototype. Not a medical device. Do not use it for real diagnosis.

**On this Mac there is no sensor connected**, so the heart rate is typed in by
hand. The dashboard says `Sensor: NOT CONNECTED — manual entry`. It does not
show a made-up live reading.

---

## 1. Requirements (one time)

You need **Python 3.10 or newer** (3.12 is recommended).

Open **Terminal** (press `⌘ Space`, type `Terminal`, press Return) and paste:

```
python3 --version
```

- If it prints `Python 3.10` or higher, you are ready. Go to step 2.
- If it prints an older version, or says `command not found`, install Python:

```
brew install python@3.12
```

If `brew` is not found, install Homebrew first (paste it as one line, then
follow what it prints):

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Or download Python from https://www.python.org/downloads/ and run the installer.

---

## 2. Put the project in Documents

Move `CardioPredictor_v5.zip` into your **Documents** folder, then paste:

```
cd ~/Documents
unzip -o CardioPredictor_v5.zip
cd CardioPredictor
```

You are now inside the project folder. Check it:

```
ls
```

You should see `api`, `src`, `models`, `run_mac.sh`, `README.md` and others.

---

## 3. Start the app (one command)

Paste this and press Return:

```
bash run_mac.sh
```

The first run takes a few minutes while packages install. When you see:

```
Dashboard (website): http://127.0.0.1:8000/dashboard/
```

open that address in **Safari** or **Chrome**.

**Leave the Terminal window open.** The app runs only while this window is open.
To stop the app, click the Terminal window and press `Ctrl + C`.

---

## 4. Try the whole workflow

Do these in the dashboard, in order:

1. **Register a patient**: type a Patient ID such as `P-1042` (letters,
   digits, `-` or `_`, 3 to 40 characters), a name, age and sex, then press
   **Register patient**. Then choose that patient in the *Select* box.
2. **Photo** (optional): press **Take photo with camera**. On the Mac it
   saves a placeholder image, because no camera is connected.
3. **Measure**: the *Start measurement* button is greyed out, because no sensor
   is connected. This is expected.
4. **Assess**: type the heart rate and the other values by hand, then press
   **Run risk assessment**. The result shows the 12-year risk and the factors that
   raised or lowered it for this patient.
5. **Report**: choose **PNG**, **TXT** or **DOCX** in the format box, then
   press **Generate & download**.
6. **Share**: press **Create share link**. The link opens a read-only page for
   that patient and expires after the number of days you choose.

Every result is saved to:

```
data/patients/<patient ID>/
    patient.json     name, age, sex
    photos/          photos
    measurements/    finger measurements (none, while no sensor is connected)
    assessments/     risk results, one file per assessment
    reports/         PNG, TXT and DOCX reports
```

Open it in Finder with:

```
open data/patients
```

---

## 5. Check that everything works (optional)

Run the automated tests. Paste these two lines one after the other:

```
source .venv/bin/activate
python -m pytest -q tests
```

The last line should say `passed`, with no `failed`.

---

## 6. Stop and start again later

- **Stop:** click the Terminal window and press `Ctrl + C`.
- **Start again:** open Terminal and paste:

```
cd ~/Documents/CardioPredictor
bash run_mac.sh
```

Your patients and results are kept, because they stay in the `data` folder.

---

## 7. Other settings (optional)

Paste these before `bash run_mac.sh` to change how the app runs:

| What you want | Paste this first |
|---|---|
| Demo with a simulated pulse (clearly labelled SIMULATED) | `export SENSOR_MODE=simulate` |
| Open the dashboard from other devices on the same Wi-Fi | `export HOST=0.0.0.0` |
| Use a different port | `export PORT=8080` |

Example:

```
export SENSOR_MODE=simulate
bash run_mac.sh
```

**Keep the Mac awake during a demo:** the server stops if the Mac sleeps. Keep
the laptop plugged in and awake while you demo.

---

## Troubleshooting

**`Address already in use` or `port 8000 is busy`**
Another copy of the app is running. Stop it, or free the port:

```
lsof -ti :8000 | xargs kill
bash run_mac.sh
```

**`Your python3 is too old`**
Run `brew install python@3.12`, then `bash run_mac.sh` again.

**`pip` fails with a network or SSL error**
Check your internet connection, then run `bash run_mac.sh` again. The steps are
safe to repeat.

**The page says "Model not loaded"**
The trained model is missing. Paste this and then run `bash run_mac.sh`:

```
source .venv/bin/activate
python src/train.py
```

**The dashboard does not load**
Make sure the Terminal window is still open and shows the server running. Then
open `http://127.0.0.1:8000/dashboard/` again.

**Reset all test data (deletes all patients and results — careful)**

```
rm -rf data/patients data/predictions.db
```

---

## Raspberry Pi

The Raspberry Pi version (with the MAX30102 sensor and camera, running 24/7)
is in **RASPBERRY_PI_DEPLOYMENT_GUIDE.docx** and **.pdf**. Follow that guide on
the Pi. This README covers the Mac only.

## Notes

- The 12-year figure is an approximation of the 10-year model. It assumes the
  yearly risk stays constant. It is not a separately trained 12-year model.
- The SpO2 value is stored with each reading, but it does not change the risk
  estimate, because the Framingham data has no SpO2 column.
- The code is in `src/` (training, validation, reports, storage),
  `api/` (the web server and dashboard in `api/static/`) and `tests/`.
