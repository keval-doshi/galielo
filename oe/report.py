"""What a Galileo run found, for people: the key result, how it compares, and a PDF report.

    from oe.report import build_pdf
    build_pdf("runs/toronto5")     # -> runs/toronto5/galileo-report-toronto5.pdf

The PDF uses fonts that ship with matplotlib (STIX for text, DejaVu Sans for any symbol STIX
lacks), so it looks the same on every computer.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from .agent import format_value

# --------------------------------------------------------------------------- the key result

REFERENCES = (  # in m/s²
    (9.80665, "standard gravity on Earth"),
    (1.62, "gravity on the Moon"),
)


def load(run) -> tuple[dict, dict]:
    data = json.loads((Path(run) / "report.json").read_text(encoding="utf-8"))
    return data.get("report") or {}, data.get("meta") or {}


def number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _acceleration_unit(unit: str) -> float | None:
    """How many of this unit make 1 m/s², or None when it is not an acceleration unit."""
    u = str(unit or "").lower().replace(" ", "").replace("·", "").replace("*", "").replace("^", "")
    u = u.replace("⁻²", "-2").replace("²", "2")
    return {"m/s2": 1.0, "ms-2": 1.0, "m/s/s": 1.0, "cm/s2": 100.0, "cms-2": 100.0, "mm/s2": 1000.0}.get(u)


def _is_gravity(quantity: str) -> bool:
    q = str(quantity or "").lower()
    return "gravit" in q or "free-fall" in q or "free fall" in q or bool(re.search(r"(^|[\s,(])g($|[\s,)])", q))


def key_result(report: dict) -> dict | None:
    """The one number the experiment was about: g if it was measured, else the first result with an error bar."""
    results = [r for r in (report.get("results") or []) if isinstance(r, dict) and number(r.get("value")) is not None]
    if not results:
        return None
    for r in results:
        if _is_gravity(r.get("quantity")) and _acceleration_unit(r.get("unit")):
            return r
    with_error = [r for r in results if (number(r.get("uncertainty")) or 0) > 0]
    return (with_error or results)[0]


def other_results(report: dict, key: dict | None, limit: int = 3) -> list[dict]:
    rest = [r for r in (report.get("results") or []) if isinstance(r, dict) and r is not key and number(r.get("value")) is not None]
    return rest[:limit]


def comparison(result: dict | None) -> dict | None:
    """The reference value this result is naturally compared with, and how far from it it is."""
    if not result or not _is_gravity(result.get("quantity")):
        return None
    factor = _acceleration_unit(result.get("unit"))
    value = number(result.get("value"))
    if factor is None or value is None or value <= 0:
        return None
    ref_si, label = min(REFERENCES, key=lambda ref: abs(math.log(value / factor / ref[0])))
    reference = ref_si * factor
    sigma = number(result.get("uncertainty"))
    sigma = sigma if sigma and sigma > 0 else None
    difference = value - reference
    out = {"reference": reference, "label": label, "difference": difference, "percent": 100 * difference / reference,
           "sigma": sigma, "z": abs(difference) / sigma if sigma else None}
    side = "below" if difference < 0 else "above"
    sentence = f"{abs(out['percent']):.2g}% {side} {label} ({reference:g} {result.get('unit', '')})"
    if out["z"] is None:
        sentence += "."
    elif out["z"] <= 1:
        sentence += ": inside the error bar."
    elif out["z"] <= 2:
        sentence += f": within two error bars ({out['z']:.1f} σ)."
    else:
        sentence += f": {out['z']:.1f} σ away, more than the error bar explains."
    out["sentence"] = sentence
    out["agrees"] = out["z"] is not None and out["z"] <= 2
    return out


def _nice_step(raw: float) -> float:
    power = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * power >= raw:
            return multiple * power
    return 10 * power


def scale_geometry(value: float, sigma: float | None, reference: float) -> dict:
    """Axis range and ticks for drawing a measurement with its error bar next to a reference value."""
    sigma = sigma or 0.0
    span = max(4 * sigma, 2.5 * abs(value - reference), 0.004 * abs(reference)) or 1.0
    centre = (value + reference) / 2
    lo, hi = centre - span, centre + span
    step = _nice_step((hi - lo) / 5)
    ticks = [k * step for k in range(math.ceil(lo / step), math.floor(hi / step) + 1)]
    decimals = max(0, -math.floor(math.log10(step) + 1e-9)) + (1 if f"{step:.10g}".rstrip("0").endswith("25") else 0)
    return {"lo": lo, "hi": hi, "ticks": ticks, "labels": [f"{t:.{decimals}f}" for t in ticks]}


def value_parts(result: dict) -> tuple[str, str, str]:
    """('9.803', '0.089', 'm/s²') with two significant figures in the uncertainty."""
    text = format_value(result.get("value"), result.get("uncertainty"))
    value, _, error = text.partition(" ± ")
    return value, error, str(result.get("unit") or "")


def pdf_path(run) -> Path:
    run = Path(run)
    return run / f"galileo-report-{run.name}.pdf"


def ensure_pdf(run) -> Path | None:
    """The run's PDF, rebuilt when the report is newer. None when the run has no report yet."""
    run = Path(run)
    if not (run / "report.json").exists():
        return None
    out = pdf_path(run)
    if not out.exists() or out.stat().st_mtime < (run / "report.json").stat().st_mtime:
        build_pdf(run, out)
    return out


# --------------------------------------------------------------------------------- the PDF

INK = "#182A4A"
MUTED = "#5A6A85"
RULE = "#D5DEEA"
PEN = "#2447A8"
HIGHLIGHT = "#F7DF4B"
PANEL = "#F7F9FD"
GRID = "#E6ECF5"

_FONT_FILES = {
    "GSerif": "STIXGeneral.ttf",
    "GSerif-Bold": "STIXGeneralBol.ttf",
    "GSerif-Italic": "STIXGeneralItalic.ttf",
    "GSerif-BoldItalic": "STIXGeneralBolIta.ttf",
    "GSans": "DejaVuSans.ttf",
    "GSans-Bold": "DejaVuSans-Bold.ttf",
}
_CMAPS: dict[str, set] = {}


def _fonts() -> None:
    if _CMAPS:
        return
    import matplotlib
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in _FONT_FILES.items():
        font = TTFont(name, str(folder / file))
        pdfmetrics.registerFont(font)
        _CMAPS[name] = set(font.face.charToGlyph)
    for bold, italic, name in ((0, 0, "GSerif"), (1, 0, "GSerif-Bold"), (0, 1, "GSerif-Italic"), (1, 1, "GSerif-BoldItalic")):
        addMapping("GSerif", bold, italic, name)
    for bold, italic, name in ((0, 0, "GSans"), (1, 0, "GSans-Bold"), (0, 1, "GSans"), (1, 1, "GSans-Bold")):
        addMapping("GSans", bold, italic, name)


def _runs(text: str, font: str) -> list[tuple[str, str]]:
    """Split text into (piece, font) so that symbols missing from the font come from DejaVu Sans."""
    fallback = "GSans-Bold" if "Bold" in font else "GSans"
    pieces: list[tuple[str, str]] = []
    for ch in str(text):
        use = font if (ord(ch) in _CMAPS[font] or ch in "\n\t") else (fallback if ord(ch) in _CMAPS[fallback] else None)
        if use is None:
            ch, use = "?", font
        if pieces and pieces[-1][1] == use:
            pieces[-1] = (pieces[-1][0] + ch, use)
        else:
            pieces.append((ch, use))
    return pieces


def rich(text, font: str = "GSerif") -> str:
    """Paragraph markup for untrusted text: escaped, **bold** kept, `code` unquoted, symbols covered."""
    out = []
    for piece, use in _runs(str(text if text is not None else ""), font):
        piece = escape(piece).replace("\n", "<br/>")
        out.append(piece if use == font else f'<font name="{use}">{piece}</font>')
    markup = "".join(out)
    markup = re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"<b>\1</b>", markup)
    return markup.replace("`", "")


def _draw(canvas, x: float, y: float, text: str, font: str, size: float, anchor: str = "left") -> float:
    """drawString with symbol fallback. Returns the width drawn."""
    pieces = _runs(text, font)
    width = sum(canvas.stringWidth(piece, use, size) for piece, use in pieces)
    x = x - width / 2 if anchor == "middle" else x - width if anchor == "right" else x
    for piece, use in pieces:
        canvas.setFont(use, size)
        canvas.drawString(x, y, piece)
        x += canvas.stringWidth(piece, use, size)
    return width


def _styles():
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle

    base = ParagraphStyle("body", fontName="GSerif", fontSize=10.5, leading=14.5, textColor=INK)
    return {
        "body": base,
        "lead": ParagraphStyle("lead", base, fontSize=12.5, leading=17.5),
        "title": ParagraphStyle("title", base, fontName="GSerif-Bold", fontSize=21, leading=25, spaceAfter=2),
        "sub": ParagraphStyle("sub", base, fontSize=9.5, leading=12.5, textColor=MUTED),
        "h2": ParagraphStyle("h2", base, fontName="GSerif-Bold", fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=5, textColor=INK),
        "small": ParagraphStyle("small", base, fontSize=8.8, leading=11.5),
        "small_muted": ParagraphStyle("small_muted", base, fontSize=8.8, leading=11.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", base, fontSize=9.2, leading=12),
        "cell_head": ParagraphStyle("cell_head", base, fontName="GSerif-Bold", fontSize=9.2, leading=12, textColor=MUTED),
        "caption": ParagraphStyle("caption", base, fontSize=8.8, leading=11.5, textColor=MUTED, spaceBefore=3),
        "right": ParagraphStyle("right", base, fontSize=9.2, leading=12.5, textColor=MUTED, alignment=TA_RIGHT),
        "bullet": ParagraphStyle("bullet", base, leftIndent=11, bulletIndent=0, spaceAfter=3),
        "tool": ParagraphStyle("tool", base, fontSize=9, leading=12, leftIndent=11, spaceAfter=4),
        "step": ParagraphStyle("step", base, spaceBefore=4, spaceAfter=3),
        "mini": ParagraphStyle("mini", base, fontSize=13, leading=16.5),
        "wordmark": ParagraphStyle("wordmark", base, fontName="GSerif-Italic", fontSize=27, leading=30, textColor=PEN),
    }


def _result_panel(width: float, result: dict, comp: dict | None):
    """The measured value, highlighted, on a small ruler next to the reference value."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Flowable

    value_text, error_text, unit = value_parts(result)
    big = value_text + (f" ± {error_text}" if error_text else "")
    quantity = str(result.get("quantity") or "Result")
    value = number(result.get("value"))
    geometry = scale_geometry(value, comp["sigma"], comp["reference"]) if comp else None

    class Panel(Flowable):
        def wrap(self, *_):
            self.height = (61 if geometry else 33) * mm
            return width, self.height

        def draw(self):
            c, w, h, pad = self.canv, width, self.height, 7 * mm
            c.saveState()
            path = c.beginPath()
            path.roundRect(0, 0, w, h, 2.2 * mm)
            c.setFillColor(colors.HexColor(PANEL))
            c.setStrokeColor(colors.HexColor(RULE))
            c.setLineWidth(0.7)
            c.drawPath(path, stroke=1, fill=1)
            c.clipPath(path, stroke=0, fill=0)
            c.setStrokeColor(colors.HexColor(GRID))
            c.setLineWidth(0.35)
            step = 5 * mm
            for i in range(1, int(w / step) + 1):
                c.line(i * step, 0, i * step, h)
            for i in range(1, int(h / step) + 1):
                c.line(0, i * step, w, i * step)
            c.restoreState()

            c.setFillColor(colors.HexColor(MUTED))
            _draw(c, pad, h - pad - 3.2 * mm, quantity, "GSerif", 11)
            size = 31
            baseline = h - pad - 15.5 * mm
            text_width = sum(c.stringWidth(p, f, size) for p, f in _runs(big, "GSerif"))
            c.setFillColor(colors.HexColor(HIGHLIGHT))
            c.rect(pad - 1.6 * mm, baseline - 0.16 * size, text_width + 3.2 * mm, 0.62 * size, stroke=0, fill=1)
            c.setFillColor(colors.HexColor(INK))
            _draw(c, pad, baseline, big, "GSerif", size)
            if unit:
                _draw(c, pad + text_width + 2.6 * mm, baseline, unit, "GSerif", 18)
            if not geometry:
                return

            lo, hi = geometry["lo"], geometry["hi"]
            x0, x1 = pad, w - pad
            axis_y = 15 * mm

            def at(v):
                return x0 + (v - lo) / (hi - lo) * (x1 - x0)

            c.setStrokeColor(colors.HexColor(INK))
            c.setLineWidth(0.8)
            c.line(x0, axis_y, x1, axis_y)
            c.setFont("GSerif", 8)
            for tick, text in zip(geometry["ticks"], geometry["labels"]):
                c.line(at(tick), axis_y, at(tick), axis_y - 1.4 * mm)
                c.setFillColor(colors.HexColor(MUTED))
                _draw(c, at(tick), axis_y - 4.6 * mm, text, "GSerif", 8, "middle")
            ref_x = at(comp["reference"])
            c.setDash(2, 1.6)
            c.setLineWidth(0.9)
            c.line(ref_x, axis_y, ref_x, axis_y + 11 * mm)
            c.setDash()
            c.setFillColor(colors.HexColor(INK))
            ref_label = f"{comp['label']}, {comp['reference']:g}"
            anchor = "middle"
            label_width = c.stringWidth(ref_label, "GSerif-Italic", 8.6)
            if ref_x - label_width / 2 < x0:
                anchor = "left"
            elif ref_x + label_width / 2 > x1:
                anchor = "right"
            _draw(c, ref_x, axis_y + 12.2 * mm, ref_label, "GSerif-Italic", 8.6, anchor)
            mark_y = axis_y + 5 * mm
            sigma = comp["sigma"] or 0
            pen = colors.HexColor(PEN)
            if sigma:
                c.setStrokeColor(pen)
                c.setLineWidth(0.8)
                c.line(at(value - 2 * sigma), mark_y, at(value + 2 * sigma), mark_y)
                for end in (value - 2 * sigma, value + 2 * sigma):
                    c.line(at(end), mark_y - 1.1 * mm, at(end), mark_y + 1.1 * mm)
                c.setFillColor(colors.Color(pen.red, pen.green, pen.blue, alpha=0.28))
                c.rect(at(value - sigma), mark_y - 1.5 * mm, at(value + sigma) - at(value - sigma), 3 * mm, stroke=0, fill=1)
            c.setFillColor(pen)
            c.circle(at(value), mark_y, 1.25 * mm, stroke=0, fill=1)
            if sigma:
                c.setFillColor(colors.HexColor(MUTED))
                _draw(c, w - pad, h - pad - 3.2 * mm, "blue bar ±1σ, whiskers ±2σ", "GSerif", 8.2, "right")
            c.setFillColor(colors.HexColor(INK))
            _draw(c, x0, 3.6 * mm, comp["sentence"][0].upper() + comp["sentence"][1:], "GSerif", 9.6)

    return Panel()


def _figure(source, caption=None, title=None, max_height: float = 0, min_height: float = 0):
    """Optional heading, image and caption as one block. The image shrinks (down to min_height) to use
    the space left on the page instead of leaving a gap; below that, the block moves to the next page."""
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Flowable

    reader = ImageReader(io.BytesIO(source) if isinstance(source, bytes) else str(source))
    pixels_w, pixels_h = reader.getSize()

    class Figure(Flowable):
        def wrap(self, avail_w, avail_h):
            self.avail_w = avail_w
            self.title_h = (title.wrap(avail_w, avail_h)[1] + title.style.spaceBefore + title.style.spaceAfter) if title else 0
            self.caption_h = (caption.wrap(avail_w, avail_h)[1] + caption.style.spaceBefore) if caption else 0
            image_h = avail_w * pixels_h / pixels_w
            if max_height:
                image_h = min(image_h, max_height)
            room = avail_h - self.title_h - self.caption_h - 2
            if image_h > room >= min_height:
                image_h = room
            self.image_h = image_h
            self.image_w = image_h * pixels_w / pixels_h
            self.height = self.title_h + image_h + self.caption_h
            return avail_w, self.height

        def draw(self):
            y = self.height
            if title:
                y -= title.style.spaceBefore + title.height
                title.drawOn(self.canv, 0, y)
                y -= title.style.spaceAfter
            y -= self.image_h
            self.canv.drawImage(reader, (self.avail_w - self.image_w) / 2, y, self.image_w, self.image_h)
            if caption:
                y -= caption.style.spaceBefore + caption.height
                caption.drawOn(self.canv, 0, y)

    return Figure()


def _model_name(model: str | None) -> str:
    model = str(model or "")
    for prefix, name in (("muse-wonder", "Muse Wonder"), ("muse-spark", "Muse Spark")):
        if model.startswith(prefix):
            return f"{name} ({model})"
    return model or "the model"


def _track_label(name: str, names) -> str:
    """'wooden block' for 'wooden block#0' when it is the only track of that prompt."""
    prompt = name.rsplit("#", 1)[0]
    return prompt if sum(1 for n in names if n.rsplit("#", 1)[0] == prompt) == 1 else name


def _tracks(run: Path) -> dict[str, list[tuple[float, float, float]]]:
    rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    path = run / "tracks.csv"
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows[row["track"]].append((float(row["t"]), float(row["x_px"]), float(row["y_px"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _tracking_frame(run: Path, fps: float | None) -> tuple[bytes, float, str] | None:
    """A frame of the SAM overlay where the main object is far from rest, as JPEG bytes."""
    import cv2
    import numpy as np

    video = run / "overlay.mp4"
    tracks = _tracks(run)
    if not video.exists() or not tracks or not fps:
        return None
    name, points = max(tracks.items(), key=lambda item: len(item[1]))
    later = points[len(points) // 2:] or points
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    cx, cy = float(np.median(xs)), float(np.median(ys))
    t = max(later, key=lambda p: (p[1] - cx) ** 2 + (p[2] - cy) ** 2)[0]
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[1] > 1400:
        scale = 1400 / frame.shape[1]
        frame = cv2.resize(frame, (1400, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (data.tobytes(), t, name) if ok else None


def _filename(path: str) -> str:
    return re.split(r"[\\/]", str(path))[-1] if path else ""


def _when(run: Path, meta: dict) -> str:
    stamp = meta.get("finished_at")
    try:
        t = time.strptime(stamp, "%Y-%m-%d %H:%M") if stamp else time.localtime((run / "report.json").stat().st_mtime)
    except (ValueError, OSError):
        t = time.localtime()
    return f"{t.tm_mday} {time.strftime('%B %Y, %H:%M', t)}"


def _table(rows, widths, styles, header: bool = True):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
    ]
    if header:
        commands.append(("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.HexColor(INK)))
    table.setStyle(TableStyle(commands))
    return table


def build_pdf(run, out=None) -> Path:
    """Write the run's report as an A4 PDF and return its path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import (CondPageBreak, HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    _fonts()
    run = Path(run)
    out = Path(out) if out else pdf_path(run)
    report, meta = load(run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) if (run / "summary.json").exists() else {}
    s = _styles()
    width = A4[0] - 36 * mm
    experiment = str(report.get("experiment") or "Experiment")
    story: list = []

    def heading(text):
        return Paragraph(rich(text, "GSerif-Bold"), s["h2"])

    # header and title
    right = "<br/>".join(rich(x) for x in ("Experiment report", _when(run, meta), f"Run {run.name}"))
    head = Table([[Paragraph("Galileo", s["wordmark"]), Paragraph(right, s["right"])]], colWidths=[width * 0.5, width * 0.5])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    story += [head, HRFlowable(width="100%", thickness=0.9, color=colors.HexColor(PEN), spaceBefore=5, spaceAfter=11)]
    story.append(Paragraph(rich(experiment, "GSerif-Bold"), s["title"]))
    video_name = _filename(summary.get("video", ""))
    story.append(Paragraph(rich(
        "Measured from a video" + (f" ({video_name})" if video_name else "")
        + f": SAM 3.1 tracked the motion and {_model_name(meta.get('model'))} worked out the physics."), s["sub"]))
    story.append(Spacer(1, 9))

    # the result
    key = key_result(report)
    comp = comparison(key)
    if key:
        story += [_result_panel(width, key, comp), Spacer(1, 10)]
    if report.get("headline"):
        story.append(Paragraph(rich(report["headline"]), s["lead"]))
    others = other_results(report, key)
    if others:
        cells = []
        for r in others:
            value, error, unit = value_parts(r)
            text = value + (f" ± {error}" if error else "") + (f" {unit}" if unit else "")
            cells.append([Paragraph(rich(r.get("quantity")), s["small_muted"]), Paragraph(rich(text), s["mini"])])
        mini = Table([[c for c in cells]], colWidths=[width / len(cells)] * len(cells), hAlign="LEFT")
        mini.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                  ("LINEABOVE", (0, 0), (-1, 0), 0.4, colors.HexColor(RULE))]))
        story += [Spacer(1, 8), mini]
    if report.get("law"):
        law = Table([[Paragraph(f'<font name="GSerif-Bold" color="{MUTED}">The law</font><br/>{rich(report["law"])}', s["body"])]],
                    colWidths=[width], hAlign="LEFT")
        law.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                                 ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(PEN)),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                 ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [Spacer(1, 10), law]

    best = report.get("best_plot")
    if best and (run / _filename(best)).exists():
        story.append(_figure(run / _filename(best), Paragraph(
            "Tracked data (dots) with the fitted law (line); the residuals underneath show what the law does not explain.",
            s["caption"]), heading("The fit"), max_height=100 * mm, min_height=62 * mm))

    # the evidence
    fps = number(summary.get("fps"))
    frame = _tracking_frame(run, fps)
    tracks = summary.get("tracks") or {}
    facts = []
    prompts = summary.get("prompts") or []
    if prompts:
        facts.append(("Prompt", ", ".join(f"“{p}”" for p in prompts)))
    for name, track in list(tracks.items())[:4]:
        gap = track.get("largest_gap_frames")
        facts.append(("Tracked", f"{_track_label(name, tracks)}: {track.get('frames', '?')} of {summary.get('frames', '?')} frames "
                      f"({100 * float(track.get('coverage') or 0):.0f}%)" + ("" if gap is None else ", no gaps" if gap == 0 else f", largest gap {gap} frames")))
    if fps and summary.get("frames"):
        facts.append(("Clip", f"{summary['frames'] / fps:.1f} s at {fps:g} fps, {summary.get('width')}×{summary.get('height')} px"))
    calibrated = meta.get("scale_set_by_galileo")
    if calibrated:
        facts.append(("Length scale", f"{float(calibrated['px_per_cm']):.4g} px/cm, set by Galileo: {calibrated.get('reason') or 'no reason given'}"))
    elif number(summary.get("px_per_cm")):
        facts.append(("Length scale", f"{float(summary['px_per_cm']):.4g} px/cm from {summary.get('scale_source') or 'the reference'}"))
    else:
        facts.append(("Length scale", "None in the video, so positions are in pixels; lengths come from the notes."))
    fact_rows = [[Paragraph(rich(k), s["small_muted"]), Paragraph(rich(v), s["small"])] for k, v in facts]
    if frame:
        data, t, name = frame
        story += [_figure(data, Paragraph(rich(f"SAM 3.1's mask on “{_track_label(name, tracks)}” at t = {t:.2f} s, with its "
                                               "recent path."), s["caption"]),
                          heading("What SAM 3.1 tracked"), max_height=95 * mm, min_height=70 * mm), Spacer(1, 6)]
    else:
        story += [CondPageBreak(40 * mm), heading("What SAM 3.1 tracked")]
    story.append(_table(fact_rows, [width * 0.16, width * 0.84], s, header=False))
    if (run / "plot.png").exists():
        story += [Spacer(1, 8), _figure(run / "plot.png", Paragraph(
            "Position of every tracked object over time, and its path in the picture.", s["caption"]),
            max_height=66 * mm, min_height=48 * mm)]

    results = [r for r in (report.get("results") or []) if isinstance(r, dict)]
    if results:
        rows = [[Paragraph(x, s["cell_head"]) for x in ("Quantity", "Value", "Unit", "Note")]]
        for r in results:
            rows.append([Paragraph(rich(r.get("quantity")), s["cell"]),
                         Paragraph(rich(format_value(r.get("value"), r.get("uncertainty"))), s["cell"]),
                         Paragraph(rich(r.get("unit")), s["cell"]),
                         Paragraph(rich(r.get("note")), s["cell"])])
        story += [CondPageBreak(45 * mm), heading("All results"), _table(rows, [width * 0.22, width * 0.21, width * 0.08, width * 0.49], s)]
    if report.get("explanation"):
        story += [KeepTogether([heading("What this means"), Paragraph(rich(report["explanation"]), s["body"])])]

    # the reasoning
    comparisons = [c for c in (report.get("comparisons") or []) if isinstance(c, dict)]
    if comparisons:
        rows = [[Paragraph(x, s["cell_head"]) for x in ("Model", "BIC", "rms", "Verdict")]]
        for c in comparisons:
            bic, rms = number(c.get("bic")), number(c.get("rms"))
            rows.append([Paragraph(rich(c.get("model")), s["cell"]),
                         Paragraph("" if bic is None else f"{bic:.1f}", s["cell"]),
                         Paragraph("" if rms is None else f"{rms:.3g}", s["cell"]),
                         Paragraph(rich(c.get("verdict")), s["cell"])])
        story += [CondPageBreak(40 * mm), heading("Laws compared"),
                  _table(rows, [width * 0.33, width * 0.09, width * 0.09, width * 0.49], s),
                  Paragraph("BIC: lower is better; a difference above 10 is decisive.", s["caption"])]
    for title, items in (("Data quality", report.get("data_quality")), ("Next steps", report.get("next_steps"))):
        items = [i for i in (items or []) if str(i).strip()]
        if items:
            block = [heading(title)] + [Paragraph(rich(i), s["bullet"], bulletText="•") for i in items]
            story += [KeepTogether(block[:2])] + block[2:]

    transcript = meta.get("transcript") or []
    if transcript:
        story += [CondPageBreak(40 * mm), heading("How Galileo worked it out")]
        story.append(Paragraph(rich(
            f"{_model_name(meta.get('model'))} read the tracks and the notes, then called its own fitting tools. "
            "Every number in this report comes from those tools."), s["small_muted"]))
        story.append(Spacer(1, 4))
        for entry in transcript:
            if entry.get("kind") == "assistant":
                text = str(entry.get("text") or "")
                text = text if len(text) < 1400 else text[:1400] + "…"
                story.append(Paragraph(f'<font name="GSerif-Bold">Step {entry.get("step")}.</font> {rich(text)}', s["step"]))
            elif entry.get("kind") == "tool":
                name = str(entry.get("name") or "tool")
                args = entry.get("args") or {}
                shown = "" if name == "submit_report" else ", ".join(f"{k}={v}" for k, v in args.items())
                shown = shown if len(shown) < 300 else shown[:300] + "…"
                story.append(Paragraph(
                    f'<font name="GSans-Bold" size="8.2" color="{PEN}">{escape(name)}</font>'
                    f'<font name="GSans" size="8.2">({rich(shown, "GSans")})</font><br/>'
                    f'<font color="{MUTED}">{rich(entry.get("summary"))}</font>', s["tool"]))

    usage = meta.get("usage_tokens") or {}
    rows = []
    if video_name:
        rows.append(("Video", video_name))
    if meta.get("notes"):
        rows.append(("Notes given", meta["notes"]))
    rows.append(("Tracking", "SAM 3.1 on the Meta Model API: one call per prompt, masks for every frame"))
    analysis = _model_name(meta.get("model"))
    if meta.get("effort"):
        analysis += f" (effort {meta['effort']})"
    if meta.get("steps") is not None:
        analysis += f": {meta['steps']} model calls, {meta.get('tool_calls', 0)} tool calls, {float(meta.get('seconds') or 0):.0f} s"
    rows.append(("Analysis", analysis))
    if usage:
        tokens = f"{int(usage.get('input', 0)) + int(usage.get('cached', 0)):,} input ({int(usage.get('cached', 0)):,} cached), {int(usage.get('output', 0)):,} output"
        cost = number(meta.get("cost_usd_estimate"))
        rows.append(("Tokens", tokens + (f", about ${cost:.2f}" if cost is not None else "")))
    rows.append(("Files", f"runs/{run.name}: tracks.csv, overlay.mp4, report.json, agent_transcript.md and the fit plots"))
    story += [CondPageBreak(35 * mm), heading("Method and provenance"),
              _table([[Paragraph(rich(k), s["small_muted"]), Paragraph(rich(v), s["small"])] for k, v in rows],
                     [width * 0.18, width * 0.82], s, header=False)]

    footer_left = f"Galileo report: {experiment}"

    class NumberedCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                self.setStrokeColor(colors.HexColor(RULE))
                self.setLineWidth(0.5)
                self.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
                self.setFillColor(colors.HexColor(MUTED))
                _draw(self, 18 * mm, 8.5 * mm, footer_left[:90], "GSerif", 8)
                _draw(self, A4[0] - 18 * mm, 8.5 * mm, f"Page {self._pageNumber} of {total}", "GSerif", 8, "right")
                super().showPage()
            super().save()

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=19 * mm,
        title=f"Galileo report: {experiment}", author="Galileo (SAM 3.1 + Muse Wonder)",
        subject=str(report.get("headline") or ""), creator="Galileo",
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    return out