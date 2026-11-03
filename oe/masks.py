"""Masks from SAM 3.1 (or the offline colour fallback) as simple Detection objects."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """One object in one frame.

    ``left/top/right/bottom`` is a half-open box in frame pixels. ``mask`` is the
    box-local boolean raster, which SAM may send at a lower resolution than the box.
    """

    object_id: str
    frame: int
    left: int
    top: int
    right: int
    bottom: int
    mask: np.ndarray
    confidence: float | None = None

    def _scale(self) -> tuple[float, float]:
        h, w = self.mask.shape
        return (self.right - self.left) / w, (self.bottom - self.top) / h

    def centroid(self) -> tuple[float, float] | None:
        """Mask centre of mass in pixel-index coordinates (x right, y down)."""
        rows, cols = np.nonzero(self.mask)
        if rows.size == 0:
            return None
        sx, sy = self._scale()
        return (
            self.left + (cols.mean() + 0.5) * sx - 0.5,
            self.top + (rows.mean() + 0.5) * sy - 0.5,
        )

    def area(self) -> float:
        sx, sy = self._scale()
        return float(self.mask.sum()) * sx * sy

    def clipped_box_mask(self, height: int, width: int):
        """Box mask resized to the box and clipped to the frame: (y0, y1, x0, x1, mask) or None."""
        bw, bh = self.right - self.left, self.bottom - self.top
        if bw <= 0 or bh <= 0:
            return None
        local = self.mask
        if local.shape != (bh, bw):
            local = cv2.resize(local.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
        x0, y0 = max(self.left, 0), max(self.top, 0)
        x1, y1 = min(self.right, width), min(self.bottom, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return y0, y1, x0, x1, local[y0 - self.top : y1 - self.top, x0 - self.left : x1 - self.left]

    def full_mask(self, height: int, width: int) -> np.ndarray:
        out = np.zeros((height, width), dtype=bool)
        clipped = self.clipped_box_mask(height, width)
        if clipped is not None:
            y0, y1, x0, x1, local = clipped
            out[y0:y1, x0:x1] = local
        return out


def parse_sam_text(text: str, media: str = "video") -> tuple[list[Detection], list[str]]:
    """Parse SAM 3.1 output text into detections, using Meta's reference parser.

    Returns the detections (latest mask wins per frame and object) and any parser
    diagnostics as readable strings.
    """
    from meta_sam_parser import (
        CompletedOutcome,
        decode_mask_to_raster,
        image_segmentation_format,
        video_segmentation_format,
    )

    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text, emit=False)
    result = parser.finish(CompletedOutcome()).result
    notes = [f"{d.severity}: {d.code} (line {d.line}): {d.message}" for d in result.diagnostics]

    latest: dict[tuple[int, str], Detection] = {}
    for record in result.records:
        if record.kind != "mask":
            continue
        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
        raster = raster.reshape(record.mask.height, record.mask.width) > 0
        frame = record.frame.frame_index if record.frame is not None else 0
        b = record.bounds
        latest[(frame, record.object_id)] = Detection(
            object_id=record.object_id,
            frame=frame,
            left=int(round(b.left)),
            top=int(round(b.top)),
            right=int(round(b.right)),
            bottom=int(round(b.bottom)),
            mask=raster,
            confidence=record.confidence,
        )
    detections = sorted(latest.values(), key=lambda d: (d.frame, int(d.object_id) if d.object_id.isdigit() else 0))
    return detections, notes


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a colour like '#d32f2f', got {hex_color!r}")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def color_detections(
    frame_bgr: np.ndarray,
    hex_color: str,
    frame_index: int,
    tolerance: float = 35.0,
    min_area: int = 30,
    max_objects: int = 1,
) -> list[Detection]:
    """Offline fallback: find blobs close to a colour (no API needed).

    Good enough for a brightly coloured ball on a plain background, and for testing.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[hex_to_bgr(hex_color)]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    mask = (np.linalg.norm(lab - target, axis=2) < tolerance).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    order = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    found: list[Detection] = []
    for rank, index in enumerate(order[:max_objects]):
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            break
        x, y, w, h = (int(v) for v in stats[index, :4])
        local = labels[y : y + h, x : x + w] == index
        found.append(Detection(str(rank), frame_index, x, y, x + w, y + h, local, None))
    return found
