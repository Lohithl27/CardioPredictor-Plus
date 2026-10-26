"""
reports.py
Builds one report for a patient from their folder: the registration photo,
every finger measurement, and every risk assessment. Three formats:

  png   chart of 12-year risk over time + photo + readings table
  txt   plain-text report (opens anywhere)
  docx  Word document with the photo and a readings table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from patients import (  # noqa: E402
    load, list_records, first_photo_path, report_path,
)

DISCLAIMER = ("Research prototype. Risk values are model estimates, not a medical "
              "diagnosis. Heart rate and SpO2 come from a hobby-grade sensor. "
              "Consult a qualified clinician for medical advice.")


def _fmt_pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _fmt(v, unit="") -> str:
    return "-" if v is None else f"{v}{unit}"


def _rows(assessments: list[dict], measurements: list[dict]) -> list[dict]:
    """One row per assessment (with the measurement it used, if any)."""
    rows = []
    for a in assessments:
        inp = a.get("inputs", {})
        rows.append({
            "time": a.get("saved_at", a.get("timestamp", ""))[:19].replace("T", " "),
            "hr": inp.get("heartRate"), "hr_source": inp.get("hr_source", "manual"),
            "spo2": inp.get("spo2"), "sysBP": inp.get("sysBP"), "diaBP": inp.get("diaBP"),
            "chol": inp.get("totChol"), "glucose": inp.get("glucose"), "bmi": inp.get("BMI"),
            "risk10": a.get("risk_10_year_probability"),
            "risk12": a.get("risk_12_year_probability"),
            "category": a.get("risk_category"),
        })
    return rows


def _header(rec: dict) -> list[tuple[str, str]]:
    sex = rec.get("sex") or "-"
    return [("Patient ID", rec["patient_id"]), ("Name", rec["name"]),
            ("Age", _fmt(int(rec["age"]) if rec.get("age") is not None else None)), ("Sex", sex),
            ("Registered", (rec.get("created_at") or "")[:19].replace("T", " ")),
            ("Notes", rec.get("notes") or "-")]


def build_txt(patient_id: str, out: Path) -> Path:
    rec = load(patient_id)
    assessments = list_records(patient_id, "assessments")
    measurements = list_records(patient_id, "measurements")
    lines = ["CARDIOPREDICTOR+ PATIENT REPORT", "=" * 60]
    lines += [f"{k:<14}: {v}" for k, v in _header(rec)]
    photo = first_photo_path(patient_id)
    lines.append(f"{'Registration photo':<14}: {photo.name if photo else 'none'}")
    lines += ["", "FINGER MEASUREMENTS", "-" * 60]
    if not measurements:
        lines.append("None recorded.")
    for m in measurements:
        lines.append(f"{m.get('saved_at', '')[:19].replace('T', ' ')}  HR {_fmt(m.get('hr'), ' bpm'):<10}"
                     f" SpO2 {_fmt(m.get('spo2'), '%'):<8} quality: {m.get('quality', '-')}")
    lines += ["", "RISK ASSESSMENTS (12-year estimate; 10-year model output in brackets)", "-" * 60]
    if not assessments:
        lines.append("None recorded.")
    for r in _rows(assessments, measurements):
        lines.append(
            f"{r['time']}  12y {_fmt_pct(r['risk12'])} ({_fmt_pct(r['risk10'])})  {r['category'] or '-'}\n"
            f"    HR {_fmt(r['hr'], ' bpm')} [{r['hr_source']}]  SpO2 {_fmt(r['spo2'], '%')}  "
            f"BP {_fmt(r['sysBP'])}/{_fmt(r['diaBP'])}  Chol {_fmt(r['chol'])}  "
            f"Glucose {_fmt(r['glucose'])}  BMI {_fmt(r['bmi'])}")
    lines += ["", "-" * 60, DISCLAIMER]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_png(patient_id: str, out: Path) -> Path:
    rec = load(patient_id)
    assessments = list_records(patient_id, "assessments")
    measurements = list_records(patient_id, "measurements")
    rows = _rows(assessments, measurements)

    fig = plt.figure(figsize=(9, 12), dpi=110, facecolor="white")
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 2.2, 3.0, 0.6], hspace=0.55, wspace=0.3)

    # Header band
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis("off")
    ax_h.text(0, 0.9, "CardioPredictor+ Patient Report", fontsize=17, weight="bold", color="#14514a")
    for i, (k, v) in enumerate(_header(rec)[:5]):
        ax_h.text(0, 0.7 - i * 0.16, f"{k}: {v}", fontsize=10.5, color="#1b2b2a")

    # Registration photo
    ax_p = fig.add_subplot(gs[0, 2])
    ax_p.axis("off")
    photo = first_photo_path(patient_id)
    if photo:
        ax_p.imshow(plt.imread(photo))
        ax_p.set_title("First photo", fontsize=9, color="#5c6f6d")
    else:
        ax_p.text(0.5, 0.5, "No photo", ha="center", va="center", color="#5c6f6d")

    # Risk trend
    ax_t = fig.add_subplot(gs[1, :])
    if rows:
        x = list(range(1, len(rows) + 1))
        y12 = [(r["risk12"] or 0) * 100 for r in rows]
        y10 = [(r["risk10"] or 0) * 100 for r in rows]
        ax_t.plot(x, y12, marker="o", color="#1f6f66", lw=2, label="12-year estimate")
        ax_t.plot(x, y10, marker=".", color="#8a9998", lw=1.2, ls="--", label="10-year model output")
        ax_t.axhline(10, color="#b8862f", ls=":", lw=1)
        ax_t.axhline(20, color="#b3402f", ls=":", lw=1)
        ax_t.set_xticks(x)
        ax_t.set_xlabel("Assessment number (oldest to newest)")
        ax_t.set_ylabel("Risk (%)")
        ax_t.set_title("Risk over time", fontsize=11, loc="left")
        ax_t.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2, frameon=False)
        ax_t.grid(alpha=0.25)
    else:
        ax_t.axis("off")
        ax_t.text(0.5, 0.5, "No risk assessments yet", ha="center", color="#5c6f6d")

    # Table of every reading
    ax_tb = fig.add_subplot(gs[2, :])
    ax_tb.axis("off")
    ax_tb.set_title("All readings", fontsize=11, loc="left")
    if rows:
        cells = [[str(i + 1), r["time"][5:16], _fmt(r["hr"]), r["hr_source"], _fmt(r["spo2"]),
                  f"{_fmt(r['sysBP'])}/{_fmt(r['diaBP'])}", _fmt(r["chol"]),
                  _fmt_pct(r["risk12"]), r["category"] or "-"] for i, r in enumerate(rows)]
        table = ax_tb.table(cellText=cells, colLabels=["#", "Date/time (UTC)", "HR", "HR src",
                            "SpO2", "BP", "Chol", "12y risk", "Band"],
                            loc="upper center", cellLoc="center",
                            colWidths=[0.06, 0.2, 0.1, 0.13, 0.1, 0.16, 0.1, 0.12, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.25)
    else:
        ax_tb.text(0.5, 0.5, "No readings", ha="center", color="#5c6f6d")

    fig.text(0.5, 0.02, DISCLAIMER, ha="center", fontsize=7, color="#5c6f6d", wrap=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def build_docx(patient_id: str, out: Path) -> Path:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    rec = load(patient_id)
    assessments = list_records(patient_id, "assessments")
    measurements = list_records(patient_id, "measurements")
    doc = Document()
    title = doc.add_heading("CardioPredictor+ Patient Report", level=0)
    title.runs[0].font.color.rgb = RGBColor(0x14, 0x51, 0x4A)

    t = doc.add_table(rows=0, cols=2)
    t.style = "Light Grid Accent 1"
    for k, v in _header(rec):
        cells = t.add_row().cells
        cells[0].text, cells[1].text = k, str(v)

    photo = first_photo_path(patient_id)
    if photo:
        doc.add_heading("Registration photo", level=2)
        doc.add_picture(str(photo), width=Inches(2.6))

    doc.add_heading("Finger measurements", level=2)
    if not measurements:
        doc.add_paragraph("None recorded.")
    else:
        mt = doc.add_table(rows=1, cols=4)
        mt.style = "Light Grid Accent 1"
        for c, h in zip(mt.rows[0].cells, ["Time (UTC)", "Heart rate", "SpO2", "Signal quality"]):
            c.text = h
        for m in measurements:
            cells = mt.add_row().cells
            cells[0].text = m.get("saved_at", "")[:19].replace("T", " ")
            cells[1].text = _fmt(m.get("hr"), " bpm")
            cells[2].text = _fmt(m.get("spo2"), "%")
            cells[3].text = m.get("quality", "-")

    doc.add_heading("Risk assessments", level=2)
    doc.add_paragraph("12-year estimate shown first; the 10-year model output is in brackets.")
    rows = _rows(assessments, measurements)
    if not rows:
        doc.add_paragraph("None recorded.")
    else:
        rt = doc.add_table(rows=1, cols=8)
        rt.style = "Light Grid Accent 1"
        for c, h in zip(rt.rows[0].cells, ["Time (UTC)", "HR", "SpO2", "BP", "Chol",
                                            "Glucose", "12-year risk", "Band"]):
            c.text = h
        for r in rows:
            cells = rt.add_row().cells
            cells[0].text = r["time"]
            cells[1].text = f"{_fmt(r['hr'])} ({r['hr_source']})"
            cells[2].text = _fmt(r["spo2"])
            cells[3].text = f"{_fmt(r['sysBP'])}/{_fmt(r['diaBP'])}"
            cells[4].text = _fmt(r["chol"])
            cells[5].text = _fmt(r["glucose"])
            cells[6].text = f"{_fmt_pct(r['risk12'])} ({_fmt_pct(r['risk10'])})"
            cells[7].text = r["category"] or "-"

    doc.add_paragraph()
    note = doc.add_paragraph(DISCLAIMER)
    note.runs[0].font.size = Pt(8)
    doc.save(out)
    return out


BUILDERS = {"png": build_png, "txt": build_txt, "docx": build_docx}


def generate(patient_id: str, fmt: str) -> Path:
    if fmt not in BUILDERS:
        raise ValueError("format must be png, txt or docx")
    out = report_path(patient_id, fmt)
    out.parent.mkdir(parents=True, exist_ok=True)
    return BUILDERS[fmt](patient_id, out)
