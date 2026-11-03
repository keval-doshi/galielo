"""Small drawing helpers shared by the overlay video and the reference check image."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.6, color=(255, 255, 255)) -> None:
    """White text on a translucent dark box, readable on any background. ``org`` is the top-left."""
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    pad = 5
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, canvas.shape[1]), min(y + h + baseline + pad, canvas.shape[0])
    if x1 > x0 and y1 > y0:
        region = canvas[y0:y1, x0:x1]
        region[:] = (region * 0.35).astype(np.uint8)
    cv2.putText(canvas, text, (x, y + h), FONT, scale, color, 1, cv2.LINE_AA)
