#!/usr/bin/env python3
"""Make a synthetic test video with a known answer, to check the pipeline before filming.

    python make_test_clip.py pendulum      # -> test_pendulum.mp4 (true g = 9.807 m/s², L = 55 cm)
    python make_test_clip.py ramp          # -> test_ramp.mp4     (ball rolling down a 20° ramp)
    python make_test_clip.py drop          # -> test_drop.mp4     (free fall at 120 fps)
    python make_test_clip.py projectile    # -> test_projectile.mp4 (ball rolls off a table)

Scale is 10 px/cm. A blue A4 sheet is visible for the first 0.5 s (the "calibration moment"),
then the object is released.
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from scipy.integrate import solve_ivp

from oe.config import G_STANDARD_CM
from oe.video import H264Writer

W, H, PPC = 1280, 720, 10.0  # frame size and pixels per cm
RELEASE = 0.5  # seconds of calibration before the release
BOB_HEX, SHEET_HEX = "#d32f2f", "#3d85c6"
SHIFT = 4  # sub-pixel drawing precision for OpenCV (coordinates * 16)


def to_px(x_cm: float, y_cm: float) -> tuple[float, float]:
    return x_cm * PPC, (H - 1) - y_cm * PPC


def bgr(hex_color: str) -> tuple[int, int, int]:
    v = hex_color.lstrip("#")
    return int(v[4:6], 16), int(v[2:4], 16), int(v[0:2], 16)


def background(rng: np.random.Generator) -> np.ndarray:
    ramp = np.linspace(0, 12, H, dtype=np.float32)[:, None, None]
    base = np.array([228, 232, 236], np.float32)[None, None, :] - ramp
    return np.clip(base + rng.normal(0, 2.5, (H, W, 3)), 0, 255).astype(np.uint8)


def draw_sheet(frame: np.ndarray, centre_cm: tuple[float, float]) -> None:
    cx, cy = centre_cm
    corners = [to_px(cx + dx * 14.85, cy + dy * 10.5) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    cv2.fillPoly(frame, [np.int32(np.round(np.array(corners) * 2**SHIFT))], bgr(SHEET_HEX), cv2.LINE_AA, SHIFT)


def draw_ball(frame: np.ndarray, x_cm: float, y_cm: float, radius_cm: float) -> None:
    px, py = to_px(x_cm, y_cm)
    centre = (int(round(px * 2**SHIFT)), int(round(py * 2**SHIFT)))
    cv2.circle(frame, centre, int(round(radius_cm * PPC * 2**SHIFT)), bgr(BOB_HEX), -1, cv2.LINE_AA, SHIFT)


def draw_line(frame, a_cm, b_cm, color, thickness):
    a = tuple(int(round(v * 2**SHIFT)) for v in to_px(*a_cm))
    b = tuple(int(round(v * 2**SHIFT)) for v in to_px(*b_cm))
    cv2.line(frame, a, b, color, thickness, cv2.LINE_AA, SHIFT)


def pendulum(fps: float, seconds: float, out: str, rng) -> None:
    pivot, length, theta0, gamma = (64.0, 68.0), 55.0, math.radians(15.0), 0.03
    times = np.arange(0, seconds, 1 / fps)
    moving = times[times >= RELEASE] - RELEASE
    sol = solve_ivp(
        lambda t, s: [s[1], -2 * gamma * s[1] - G_STANDARD_CM / length * math.sin(s[0])],
        (0, moving[-1] + 1 / fps), [theta0, 0.0], t_eval=moving, rtol=1e-10, atol=1e-12,
    )
    theta = np.concatenate([np.full((times < RELEASE).sum(), theta0), sol.y[0]])
    writer = H264Writer(out, W, H, fps)
    for i, t in enumerate(times):
        frame = background(rng)
        if t < RELEASE:
            draw_sheet(frame, (25.0, 30.0))
        bob = (pivot[0] + length * math.sin(theta[i]), pivot[1] - length * math.cos(theta[i]))
        draw_line(frame, pivot, bob, (60, 60, 60), 2)
        cv2.circle(frame, tuple(int(v) for v in to_px(*pivot)), 5, (40, 40, 40), -1)
        draw_ball(frame, *bob, 1.8)
        writer.write(frame)
    writer.close()
    period = 2 * math.pi * math.sqrt(length / G_STANDARD_CM)
    print(f"wrote {out}: true g = {G_STANDARD_CM / 100:.3f} m/s², L = {length} cm, small-angle T0 = {period:.4f} s, "
          f"amplitude 15°, damping {gamma}/s, released at {RELEASE} s")


def ramp(fps: float, out: str, rng) -> None:
    angle, mu, radius, length = math.radians(20.0), 0.015, 1.5, 115.0
    start = (8.0, 62.0)
    along = np.array([math.cos(angle), -math.sin(angle)])
    normal = np.array([math.sin(angle), math.cos(angle)])
    accel = 5 / 7 * G_STANDARD_CM * (math.sin(angle) - mu * math.cos(angle))
    travel = length - 6.0
    seconds = RELEASE + math.sqrt(2 * travel / accel)
    writer = H264Writer(out, W, H, fps)
    for t in np.arange(0, seconds, 1 / fps):
        frame = background(rng)
        if t < RELEASE:
            draw_sheet(frame, (22.0, 14.0))
        end = np.array(start) + along * length
        draw_line(frame, start, tuple(end), (40, 90, 140), 6)
        s = 3.0 + (0.5 * accel * (t - RELEASE) ** 2 if t >= RELEASE else 0.0)
        centre = np.array(start) + along * s + normal * (radius + 0.3)
        draw_ball(frame, *centre, radius)
        writer.write(frame)
    writer.close()
    print(f"wrote {out}: ramp 20°, rolling resistance {mu}, true acceleration {accel:.1f} cm/s², released at {RELEASE} s")


def drop(fps: float, out: str, rng) -> None:
    x, top, radius = 64.0, 66.0, 2.0
    fall = math.sqrt(2 * (top - radius - 2) / G_STANDARD_CM)
    writer = H264Writer(out, W, H, fps)
    for t in np.arange(0, RELEASE + fall, 1 / fps):
        frame = background(rng)
        if t < RELEASE:
            draw_sheet(frame, (25.0, 30.0))
        y = top - (0.5 * G_STANDARD_CM * (t - RELEASE) ** 2 if t >= RELEASE else 0.0)
        draw_ball(frame, x, y, radius)
        writer.write(frame)
    writer.close()
    print(f"wrote {out}: free fall, true g = {G_STANDARD_CM / 100:.3f} m/s², released at {RELEASE} s")


def projectile(fps: float, out: str, rng) -> None:
    table_y, edge_x, radius, speed = 40.0, 60.0, 1.5, 120.0
    x_start = 12.0
    roll = (edge_x - x_start) / speed
    flight = math.sqrt(2 * table_y / G_STANDARD_CM)
    seconds = RELEASE + roll + flight + 0.4
    writer = H264Writer(out, W, H, fps)
    for t in np.arange(0, seconds, 1 / fps):
        frame = background(rng)
        if t < RELEASE:
            draw_sheet(frame, (95.0, 45.0))
        draw_line(frame, (0.0, table_y - 0.3), (edge_x, table_y - 0.3), (40, 90, 140), 5)
        draw_line(frame, (0.0, 0.0), (128.0, 0.0), (60, 60, 60), 4)
        moving = max(t - RELEASE, 0.0)
        if moving <= roll:
            x, y = x_start + speed * moving, table_y + radius
        elif moving <= roll + flight:
            tf = moving - roll
            x, y = edge_x + speed * tf, table_y + radius - 0.5 * G_STANDARD_CM * tf**2
        else:
            x, y = edge_x + speed * flight + 40.0 * (moving - roll - flight), radius
        draw_ball(frame, x, max(y, radius), radius)
        writer.write(frame)
    writer.close()
    print(f"wrote {out}: rolls at {speed:.0f} cm/s, leaves a {table_y:.0f} cm table at {RELEASE + roll:.3f} s, "
          f"true g = {G_STANDARD_CM / 100:.3f} m/s²")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", choices=["pendulum", "ramp", "drop", "projectile"], nargs="?", default="pendulum")
    ap.add_argument("--fps", type=float, help="frame rate (default 60, or 120 for drop)")
    ap.add_argument("--seconds", type=float, default=10.0, help="pendulum clip length")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    out = f"test_{args.scene}.mp4"
    if args.scene == "pendulum":
        pendulum(args.fps or 60.0, args.seconds, out, rng)
    elif args.scene == "ramp":
        ramp(args.fps or 60.0, out, rng)
    elif args.scene == "projectile":
        projectile(args.fps or 60.0, out, rng)
    else:
        drop(args.fps or 120.0, out, rng)
    print(f"colours for the offline backend: object {BOB_HEX}, sheet {SHEET_HEX}")


if __name__ == "__main__":
    main()
