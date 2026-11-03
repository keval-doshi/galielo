"""Motion analysis on tracked positions: loading, smoothing, release detection, path fingerprints.

Positions have y pointing up and are in cm when the run has a scale, otherwise in pixels
(see load_run). Times are in seconds.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class Track:
    name: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frame: np.ndarray

    def window(self, start: float | None = None, end: float | None = None) -> "Track":
        keep = np.ones_like(self.t, dtype=bool)
        if start is not None:
            keep &= self.t >= start - 1e-9
        if end is not None:
            keep &= self.t <= end + 1e-9
        return Track(self.name, self.t[keep], self.x[keep], self.y[keep], self.frame[keep])

    def __len__(self) -> int:
        return len(self.t)


def load_run(run: Path, px_per_cm: float | None = None) -> tuple[dict[str, Track], dict]:
    """Read tracks.csv in real units when possible.

    Order of preference for the scale: ``px_per_cm`` given here (e.g. set by Galileo), then the
    run's own scale. Without any scale the positions stay in pixels (y up) and
    ``info["unit"]`` is "px".
    """
    run = Path(run)
    path = run / "tracks.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run track.py first")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rows = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["track"]].append(row)
    height = summary.get("height")
    if not height:
        height = 1 + max((float(r["y_px"]) for items in rows.values() for r in items), default=0.0)
    if px_per_cm is not None and px_per_cm <= 0:
        raise ValueError("px_per_cm must be positive")
    own_scale = summary.get("px_per_cm")
    has_cm = bool(rows) and all(items[0].get("x_cm") for items in rows.values())
    if px_per_cm:
        scale, unit, source = float(px_per_cm), "cm", "set during this analysis"
    elif has_cm:
        scale, unit, source = own_scale, "cm", summary.get("scale_source") or "from the run"
    else:
        scale, unit, source = None, "px", "none: positions are in pixels"
    tracks = {}
    for name, items in rows.items():
        items.sort(key=lambda r: int(r["frame"]))
        t = np.array([float(r["t"]) for r in items])
        frame = np.array([int(r["frame"]) for r in items])
        if unit == "cm" and not px_per_cm:
            x = np.array([float(r["x_cm"]) for r in items])
            y = np.array([float(r["y_cm"]) for r in items])
        else:
            x = np.array([float(r["x_px"]) for r in items])
            y = (height - 1) - np.array([float(r["y_px"]) for r in items])
            if scale:
                x, y = x / scale, y / scale
        tracks[name] = Track(name, t, x, y, frame)
    info = {"unit": unit, "px_per_cm": scale, "scale_source": source, "frame_height_px": height}
    return tracks, info


def load_tracks(run: Path, px_per_cm: float | None = None) -> dict[str, Track]:
    return load_run(run, px_per_cm)[0]


def unit_factor(unit: str) -> float:
    """Distance thresholds are written in cm; in pixel mode assume roughly 10 px per cm."""
    return 10.0 if unit == "px" else 1.0


def frame_step(t: np.ndarray) -> float:
    return float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30


def smooth(t: np.ndarray, values: np.ndarray, window_s: float = 0.1) -> np.ndarray:
    n = len(values)
    w = int(round(window_s / frame_step(t))) | 1
    w = min(max(w, 5), n - 1 if n % 2 == 0 else n)
    if n < 7 or w < 5:
        return values.astype(float).copy()
    return savgol_filter(values, w, 2)


def velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray, window_s: float = 0.1):
    if len(t) < 3:
        zeros = np.zeros_like(t, dtype=float)
        return zeros, zeros
    return np.gradient(smooth(t, x, window_s), t), np.gradient(smooth(t, y, window_s), t)


def find_release(t: np.ndarray, x: np.ndarray, y: np.ndarray, k: float = 1.0) -> float:
    """Time the object starts moving after being held still (t[0] if it never rests first).

    ``k`` scales the distance thresholds (given in cm) to the data's unit, e.g. 10 for pixels.
    """
    n = len(t)
    if n < 10:
        return float(t[0])
    dt = frame_step(t)
    rest_n = min(max(5, int(round(0.2 / dt))), n // 3)
    x0, y0 = np.median(x[:rest_n]), np.median(y[:rest_n])
    d = np.hypot(x - x0, y - y0)
    noise = max(1.4826 * float(np.median(np.abs(d[:rest_n] - np.median(d[:rest_n])))), 0.02 * k)
    threshold = max(8 * noise, 0.3 * k)
    frames_needed = max(3, int(round(0.05 / dt)))
    moving = d > threshold
    run = np.convolve(moving.astype(int), np.ones(frames_needed, dtype=int), mode="valid") == frames_needed
    hits = np.flatnonzero(run)
    if hits.size == 0 or hits[0] < rest_n:
        return float(t[0])
    j = int(hits[0])
    # walk back to the last frame that was still clearly at rest (3 sigma, so noise rarely fools it)
    while j > 0 and d[j] > max(3 * noise, 0.08 * k):
        j -= 1
    return float(t[j])


def continuous_end(t: np.ndarray, start: float, max_gap_s: float = 0.25) -> float:
    """Last time before the first gap longer than max_gap_s, after ``start``."""
    after = t[t >= start - 1e-9]
    if after.size == 0:
        return float(t[-1])
    gaps = np.flatnonzero(np.diff(after) > max_gap_s)
    return float(after[gaps[0]] if gaps.size else after[-1])


def first_impact(t: np.ndarray, y: np.ndarray, start: float, k: float = 1.0) -> float:
    """For a falling object: the time just before it reaches its lowest point (the first impact)."""
    keep = t >= start - 1e-9
    tt, yy = t[keep], smooth(t[keep], y[keep], 0.05)
    if tt.size < 5:
        return float(tt[-1]) if tt.size else start
    low = int(np.argmin(yy))
    # where it gets within 0.3 cm of the lowest point: the floor (or a stop) starts there
    near = np.flatnonzero(yy[: low + 1] <= yy[low] + 0.3 * k)
    idx = int(near[0]) if near.size else low
    return float(tt[max(idx - 1, 1)])


def leaves_surface(t: np.ndarray, y: np.ndarray, start: float, k: float = 1.0) -> float:
    """For a ball rolling off a table: when it starts to fall. ``start`` if it was in the air from the start."""
    keep = t >= start - 1e-9
    tt, yy = t[keep], y[keep]
    if tt.size < 8:
        return start
    ys = smooth(tt, yy, 0.05)
    head = max(3, int(round(0.1 / frame_step(tt))))
    y0 = float(np.median(ys[:head]))
    dropped = np.flatnonzero(ys < y0 - 0.3 * k)
    risen = np.flatnonzero(ys > y0 + 0.3 * k)
    if dropped.size == 0 or (risen.size and risen[0] < dropped[0]):
        return start  # thrown upward, or never falls: the flight began at release
    j = int(dropped[0])
    while j > 0 and ys[j] < y0 - 0.05 * k:
        j -= 1
    if j <= 2:
        return start
    return float(tt[j])


def straight_end(t: np.ndarray, x: np.ndarray, y: np.ndarray, start: float, max_turn_deg: float = 12.0, k: float = 1.0) -> float:
    """For motion along a line (ramp): the time just before the direction changes (leaves the ramp)."""
    keep = t >= start - 1e-9
    tt, xx, yy = t[keep], x[keep], y[keep]
    if tt.size < 8:
        return float(tt[-1]) if tt.size else start
    vx, vy = velocity(tt, xx, yy, 0.08)
    speed = np.hypot(vx, vy)
    moving = speed > max(0.25 * speed.max(), 2.0 * k)
    first = np.flatnonzero(moving)
    if first.size < 4:
        return float(tt[-1])
    ref = first[: max(4, first.size // 4)]
    ref_angle = math.atan2(vy[ref].mean(), vx[ref].mean())
    turn = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(vy, vx) - ref_angle)))))
    bad = np.flatnonzero(moving & (turn > max_turn_deg) & (np.arange(tt.size) > first[0]))
    return float(tt[max(int(bad[0]) - 2, 1)]) if bad.size else float(tt[-1])


def fit_circle_simple(x: np.ndarray, y: np.ndarray):
    a = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    r = math.sqrt(max(c + cx**2 + cy**2, 1e-12))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def principal_axis(x: np.ndarray, y: np.ndarray):
    """Direction of the main motion, the coordinate along it and how straight the path is."""
    points = np.c_[x, y]
    centre = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - centre, full_matrices=False)
    direction = vt[0]
    s = (points - centre) @ direction
    if s[-1] < s[0]:
        direction, s = -direction, -s
    straightness = float(singular[1] / singular[0]) if singular[0] > 0 else 1.0
    return direction, s - s[0], straightness


def dominant_period(t: np.ndarray, signal: np.ndarray):
    """(period in s, peak strength) of the strongest oscillation, or (None, 0) if not periodic."""
    if len(t) < 16:
        return None, 0.0
    dt = frame_step(t)
    grid = np.arange(t[0], t[-1], dt)
    if grid.size < 16:
        return None, 0.0
    values = np.interp(grid, t, signal)
    values = values - np.polyval(np.polyfit(grid, values, 1), grid)
    n_fft = 8 * grid.size  # zero-padding for a finer frequency grid
    power = np.abs(np.fft.rfft(values * np.hanning(grid.size), n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, dt)
    if power.size < 4:
        return None, 0.0
    i = int(np.argmax(power[1:])) + 1
    cycles = freqs[i] * (grid[-1] - grid[0])
    strength = float(power[i] / (np.median(power[1:]) + 1e-12))
    if cycles < 1.5:
        return None, strength
    return float(1 / freqs[i]), strength


def describe_track(track: Track, fps: float | None = None, unit: str = "cm") -> dict:
    """A compact fingerprint of one track for the agent (and for people). ``unit`` is "cm" or "px"."""
    u = unit
    t, x, y = track.t, track.x, track.y
    info: dict = {"track": track.name, "points": len(t)}
    if len(t) < 5:
        info["note"] = "too few points to analyse"
        return info
    release = find_release(t, x, y, unit_factor(u))
    end = continuous_end(t, release)
    w = track.window(release, end)
    before = t < release
    info.update(
        {
            "t_first": round(float(t[0]), 3),
            "t_last": round(float(t[-1]), 3),
            "release_t": round(release, 3),
            "moving_until_t": round(end, 3),
            f"rest_position_{u}": [round(float(np.median(x[before])), 2), round(float(np.median(y[before])), 2)]
            if before.sum() >= 3
            else None,
            f"x_range_{u}": [round(float(x.min()), 2), round(float(x.max()), 2)],
            f"y_range_{u}": [round(float(y.min()), 2), round(float(y.max()), 2)],
        }
    )
    if len(w) < 8:
        info["note"] = "barely moves"
        return info
    vx, vy = velocity(w.t, w.x, w.y)
    speed = np.hypot(vx, vy)
    direction, s, straightness = principal_axis(w.x, w.y)
    extent = float(s.max() - s.min())
    cx, cy, radius, circle_rms = fit_circle_simple(w.x, w.y)
    angles = np.unwrap(np.arctan2(w.x - cx, cy - w.y))
    period, strength = dominant_period(w.t, (np.c_[w.x, w.y] - np.c_[w.x, w.y].mean(axis=0)) @ direction)
    line_angle = math.degrees(math.atan2(direction[1], direction[0]))
    if line_angle > 90:  # a line has no arrow: report it in (-90, 90]
        line_angle -= 180
    elif line_angle <= -90:
        line_angle += 180
    is_arc = circle_rms < 0.05 * radius and radius < 6 * max(extent, 1e-9) and straightness >= 0.01
    info.update(
        {
            f"max_speed_{u}_s": round(float(speed.max()), 1),
            f"path_extent_{u}": round(extent, 2),
            "path_straightness": round(straightness, 4),
            "main_line_deg_from_horizontal": round(line_angle, 1),
            "circle_fit": {
                f"radius_{u}": round(radius, 2),
                f"rms_{u}": round(circle_rms, 3),
                f"centre_{u}": [round(cx, 2), round(cy, 2)],
                "centre_is": "above the path" if cy > float(np.mean(w.y)) else "below the path",
                "arc_span_deg": round(float(np.degrees(angles.max() - angles.min())), 1),
            }
            if is_arc
            else "the path is not an arc",
            "oscillation": {
                "period_s_rough": round(period, 3),
                "cycles": round((w.t[-1] - w.t[0]) / period, 1),
                "clear": strength > 50,
            }
            if period
            else None,
        }
    )
    hints = []
    periodic = bool(period) and strength > 50
    if periodic and is_arc and cy > float(np.mean(w.y)):
        hints.append(f"swings back and forth on an arc of radius {radius:.1f} {u} with the centre above: pendulum-like")
    elif periodic and straightness < 0.1:
        hints.append(f"oscillates back and forth along a line (period about {period:.2f} s): spring-like oscillator")
    elif straightness < 0.03:
        angle = abs(math.degrees(math.atan2(direction[1], direction[0])))
        angle = min(angle, 180 - angle)
        if angle > 75:
            hints.append("moves along a vertical line: free fall-like")
        elif angle < 3:
            hints.append("moves along a horizontal line")
        else:
            hints.append(f"moves along a straight line tilted {angle:.1f} degrees: incline-like")
    elif cy < float(np.mean(w.y)) and circle_rms < 0.1 * radius:
        hints.append("curves downward like a parabola: projectile-like")
    info["hints"] = hints
    return info