import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

# Keep tests away from real patient data: use a throwaway folder and a
# simulated sensor. These must be set before the app modules are imported.
os.environ["CARDIO_DATA_DIR"] = tempfile.mkdtemp(prefix="cardio-test-")
os.environ["SENSOR_MODE"] = "simulate"
os.environ.setdefault("CAMERA_BACKEND", "none")
