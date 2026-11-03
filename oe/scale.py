"""Pixels-to-centimetres from a reference object of known size (an A4 sheet by default)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .draw import label
from .masks import Detection


@dataclass
class ScaleResult:
    px_per_cm: float
    long_px: float
    short_px: float
    long_cm: float
    short_cm: float | None
    corners: np.ndarray  # 4x2 rotated-rectangle corners, for drawing

    @property
    def short_side_px_per_cm(self) -> float | None:
        return self.short_px / self.short_cm if self.short_cm else None

    @property
    def disagreement(self) -> float | None:
        """Relative difference between the scale from the long and the short side."""
        other = self.short_side_px_per_cm
        return None if other is None else abs(other / self.px_per_cm - 1.0)


def measure_reference(
    det: Detection, height: int, width: int, long_cm: float, short_cm: float | None = None
) -> ScaleResult:
    full = det.full_mask(height, width).astype(np.uint8)
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("The reference mask is empty.")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    # The contour runs through pixel centres, so each side is one pixel short.
    w, h = w + 1.0, h + 1.0
    long_px, short_px = max(w, h), min(w, h)
    return ScaleResult(
        px_per_cm=long_px / long_cm,
        long_px=long_px,
        short_px=short_px,
        long_cm=long_cm,
        short_cm=short_cm,
        corners=cv2.boxPoints(rect),
    )


def draw_reference_check(frame: np.ndarray, det: Detection, result: ScaleResult, path: Path) -> None:
    """Save a picture showing exactly what was measured, so a bad scale is easy to spot."""
    canvas = frame.copy()
    mask = det.full_mask(*frame.shape[:2])
    canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array([255, 160, 0])).astype(np.uint8)
    cv2.polylines(canvas, [np.int32(np.round(result.corners))], True, (0, 200, 255), 2, cv2.LINE_AA)
    lines = [
        f"long side {result.long_px:.1f} px = {result.long_cm:g} cm  ->  {result.px_per_cm:.3f} px/cm",
    ]
    if result.short_cm:
        lines.append(
            f"short side {result.short_px:.1f} px = {result.short_cm:g} cm  ->  "
            f"{result.short_side_px_per_cm:.3f} px/cm  (sides agree within {result.disagreement:.1%})"
        )
    for i, text in enumerate(lines):
        label(canvas, text, (14, 14 + 30 * i))
    cv2.imwrite(str(path), canvas)
