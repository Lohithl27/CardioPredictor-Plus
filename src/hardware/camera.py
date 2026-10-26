"""
camera.py
Still-image capture for the patient photo.

Backends (set CAMERA_BACKEND in the environment or the systemd unit):
  picamera2  Raspberry Pi CSI camera: Camera Module 1 (5MP), v2 (8MP), v3 (12MP).
  usb        Any USB webcam (uses OpenCV, device index CAMERA_INDEX, default 0).
  auto       (default) try the Pi camera, then a USB webcam. If neither is found
             it raises CameraError, and the dashboard then offers "Upload photo"
             or "Use this device's camera" (phone / laptop browser) instead.
  none       No camera: writes a labelled placeholder image (for testing).

A 2MP module is usually an OV2640-type board that Picamera2 does not support.
If yours is one of those, use a USB camera instead.
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime
from pathlib import Path


class CameraError(RuntimeError):
    pass


def _placeholder(path: Path, note: str) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 480), (232, 240, 238))
    d = ImageDraw.Draw(img)
    d.text((24, 210), "No camera connected", fill=(31, 111, 102))
    d.text((24, 240), note, fill=(92, 111, 109))
    d.text((24, 270), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fill=(92, 111, 109))
    img.save(path, "JPEG", quality=90)


def _capture_picamera2(dest: Path) -> str:
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise CameraError("picamera2 is not installed (sudo apt install python3-picamera2)") from exc
    cam = None
    try:
        cam = Picamera2()
        cam.configure(cam.create_still_configuration())
        cam.start()
        cam.capture_file(str(dest))
    except Exception as exc:          # no camera, camera busy, bad ribbon, wrong config
        raise CameraError(f"Pi camera failed: {exc}") from exc
    finally:
        if cam is not None:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass
    return "picamera2"


def _capture_usb(dest: Path) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise CameraError("OpenCV is not installed (pip install opencv-python-headless)") from exc
    if sys.platform.startswith("linux") and not glob.glob("/dev/video*"):
        raise CameraError("no /dev/video device found")
    index = int(os.environ.get("CAMERA_INDEX", "0"))
    try:
        cap = cv2.VideoCapture(index)
    except Exception as exc:
        raise CameraError(f"USB camera failed: {exc}") from exc
    try:
        for _ in range(10):           # let auto-exposure settle
            cap.read()
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise CameraError(f"USB camera at index {index} returned no image")
    cv2.imwrite(str(dest), frame)
    return "usb"


def capture_photo(dest: Path, backend: str | None = None) -> str:
    """Save one JPEG to `dest` and return the backend that produced it."""
    backend = (backend or os.environ.get("CAMERA_BACKEND", "auto")).lower()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if backend == "picamera2":
        return _capture_picamera2(dest)
    if backend == "usb":
        return _capture_usb(dest)
    if backend == "auto":
        problems = []
        for fn in (_capture_picamera2, _capture_usb):
            try:
                return fn(dest)
            except CameraError as exc:
                problems.append(str(exc))
        raise CameraError("No camera detected on the Raspberry Pi (" + "; ".join(problems) + ")")

    _placeholder(dest, "CAMERA_BACKEND=none (set auto, picamera2 or usb)")
    return "none"
