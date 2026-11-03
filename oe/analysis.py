"""High-level analysis of a tracked run, shared by fit.py, the Galileo agent and the app.

Runs with a scale are analysed in centimetres. Runs without one (no reference object in the
video) are analysed in pixels: angles and times are still exact, so a pendulum with a known
length still gives g, and Galileo can set the scale later from a known length (px_per_cm).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .equations import fit_expression
from .motion import (
    Track,
    continuous_end,
    describe_track,
    find_release,
    first_impact,
    frame_step,
    leaves_surface,
    load_run,
    principal_axis,
    smooth,
    straight_end,
    unit_factor,
    velocity,
)
from .physics import (
    fit_circle,
    fit_constant_acceleration,
    fit_constant_velocity,
    fit_domino_wave,
    fit_oscillator,
    fit_pendulum,
    fit_projectile,
)
from .plots import plot_fit

MODELS = (
    "pendulum", "oscillator", "freefall", "projectile", "incline-roll", "incline-slide",
    "constant-velocity", "domino-wave",
)
SERIES = ("x", "y", "s", "theta", "r", "speed")
SERIES_INFO = {
    "x": "x ({u})",
    "y": "y ({u})",
    "s": "distance along the main direction of motion ({u})",
    "theta": "angle about the fitted pivot (rad)",
    "r": "distance from the fitted pivot ({u})",
    "speed": "speed ({u}/s)",
}
NO_SCALE_NOTE = (
    "NO SCALE: lengths are in pixels (y up), so speeds and accelerations are in px/s and px/s². "
    "Times and angles are exact. To get real units, take a real length from the notes, measure the same "
    "thing in pixels, and call set_scale with px_per_cm = pixels / centimetres."
)
# Results that mean nothing without a real length scale.
_NEEDS_SCALE = {
    "g_m_s2", "g_err_m_s2", "g_vs_standard_pct", "g_stat_err_cm_s2", "scale_err_assumed",
    "friction_coeff", "friction_coeff_err", "friction_note", "efficiency", "frictionless_accel_cm_s2", "note",
    "arc_radius_err_cm", "length_err_cm",
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "track"


def clean(value, digits: int = 6):
    """JSON-friendly copy with private keys removed and floats rounded to `digits` significant figures."""
    if isinstance(value, dict):
        return {k: clean(v, digits) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [clean(v, digits) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.{digits}g}")
    return value


def _px_key(key: str) -> str:
    return key.replace("_cm_s2", "_px_s2").replace("_cm_s", "_px_s").replace("_cm", "_px")


def _rename_px(value):
    if isinstance(value, dict):
        return {(_px_key(k) if not k.startswith("_") else k): _rename_px(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename_px(v) for v in value]
    return value


def _to_pixels(result: dict, model: str, length_given: bool) -> dict:
    """Relabel a result computed in pixel units, dropping what needs a real scale."""
    keep_as_is = {"model", "track", "window", "plot", "title", "warnings", "length_source", "angle_source", "spring_note"}
    if model == "pendulum" and length_given:
        keep_as_is |= {"g_cm_s2", "g_err_cm_s2", "g_m_s2", "g_err_m_s2", "g_vs_standard_pct", "length_cm", "length_err_cm"}
    out = {}
    for key, value in result.items():
        if key.startswith("_") or key in keep_as_is:
            out[key] = value
        elif key in _NEEDS_SCALE:
            continue
        else:
            out[_px_key(key)] = _rename_px(value)
    out["length_unit"] = "px"
    out["units_note"] = NO_SCALE_NOTE
    if model == "pendulum" and length_given:
        out["units_note"] = (
            "No scale, but g is real: it uses the given length_cm and the period, which needs no scale. "
            "Arc radius and pivot are in pixels."
        )
    for panel in out.get("_curve", {}).get("panels", []):
        panel["label"] = panel["label"].replace("(cm)", "(px)")
    return out


def choose_track(tracks: dict[str, Track], name: str | None) -> Track:
    if not tracks:
        raise ValueError("the run has no tracks")
    if name:
        if name in tracks:
            return tracks[name]
        matches = [k for k in tracks if k.startswith(name)]
        if len(matches) == 1:
            return tracks[matches[0]]
        raise ValueError(f"no track named {name!r}; available: {', '.join(tracks)}")
    return max(tracks.values(), key=len)


def auto_window(track: Track, model: str, k: float = 1.0) -> tuple[float, float, str]:
    """Default fit window. ``k`` scales the cm thresholds to the data's unit (10 for pixels)."""
    t, x, y = track.t, track.x, track.y
    release = find_release(t, x, y, k)
    end = continuous_end(t, release)
    dt = frame_step(t)
    held = release > t[0] + 1e-9
    if model in ("pendulum", "oscillator"):
        start = release + 0.2 if held else float(t[0])
        how = "from 0.2 s after release" if held else "from the first point"
    else:  # every other model, and "generic" (equation fits)
        start = release + 2 * dt if held else float(t[0])
        how = "from 2 frames after release" if held else "from the first point"
    if model == "projectile":
        launch = leaves_surface(t, y, start, k)
        if launch > start:
            start = launch + dt
            how = "from when it leaves the surface"
        end = min(end, first_impact(t, y, start, k))
        how += ", until just before the lowest point"
    elif model == "freefall":
        end = min(end, first_impact(t, y, start, k))
        how += ", until just before the lowest point"
    elif model in ("incline-roll", "incline-slide", "constant-velocity"):
        end = min(end, straight_end(t, x, y, start, k=k))
        how += ", until the direction changes"
    else:
        how += ", until tracking stops"
    return float(start), float(end), how


def _windowed(track: Track, model: str, start, end, k: float = 1.0):
    if start is None or end is None:
        a, b, how = auto_window(track, model, k)
        start = a if start is None else start
        end = b if end is None else end
        how = how if (start == a and end == b) else "partly automatic"
    else:
        how = "given"
    w = track.window(start, end)
    if len(w) < 6:
        raise ValueError(f"only {len(w)} points between {start:.3f} s and {end:.3f} s; widen the window")
    return w, float(start), float(end), how


def _domino_events(tracks: dict[str, Track], prefix: str | None, k: float = 1.0):
    groups: dict[str, list[Track]] = {}
    for name, track in tracks.items():
        groups.setdefault(name.split("#")[0], []).append(track)
    if prefix:
        prefix = prefix.split("#")[0]
        if prefix not in groups:
            raise ValueError(f"no tracks start with {prefix!r}; groups: {', '.join(groups)}")
        group = groups[prefix]
    else:
        group = max(groups.values(), key=len)
    rest, starts, names = [], [], []
    for track in group:
        if len(track) < 8:
            continue
        release = find_release(track.t, track.x, track.y, k)
        if release <= track.t[0] + 1e-9:
            continue  # never seen at rest, or never moved
        before = track.t < release
        x0, y0 = float(np.median(track.x[before])), float(np.median(track.y[before]))
        if np.max(np.hypot(track.x - x0, track.y - y0)) < 1.0 * k:
            continue
        rest.append((x0, y0))
        starts.append(release)
        names.append(track.name)
    if len(rest) < 3:
        raise ValueError(f"only {len(rest)} pieces were seen standing and then falling; need 3+")
    points = np.array(rest)
    direction, _, _ = principal_axis(points[:, 0], points[:, 1])
    positions = (points - points[0]) @ direction
    return positions, np.array(starts), names


def fit_on_run(
    run: Path,
    model: str,
    track: str | None = None,
    start: float | None = None,
    end: float | None = None,
    length_cm: float | None = None,
    length_err_cm: float = 0.3,
    angle_deg: float | None = None,
    scale_err: float = 0.01,
    plot: bool = True,
    px_per_cm: float | None = None,
    mass_kg: float | None = None,
    mass_err_kg: float = 0.0,
) -> dict:
    """Fit one physical model to a run. Returns the full result (with private plot data)."""
    run = Path(run)
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    tracks, info = load_run(run, px_per_cm)
    u = info["unit"]
    k = unit_factor(u)
    if model == "domino-wave":
        positions, starts, names = _domino_events(tracks, track, k)
        result = fit_domino_wave(positions, starts, names)
        label = (track or names[0]).split("#")[0]
        title = (
            f"{label}: wave speed {result['wave_speed_cm_s']:.1f} ± {result['wave_speed_err_cm_s']:.1f} {u}/s "
            f"over {result['pieces']} pieces"
        )
        result["window"] = None
    else:
        chosen = choose_track(tracks, track)
        w, start, end, how = _windowed(chosen, model, start, end, k)
        name = chosen.name
        if model == "pendulum":
            result = fit_pendulum(w.t, w.x, w.y, length_cm, length_err_cm, scale_err)
            if u == "cm" or length_cm:
                title = (
                    f"{name}: g = {result['g_m_s2']:.3f} ± {result['g_err_m_s2']:.3f} m/s²  "
                    f"(L = {result['length_cm']:.2f} cm, T = {result['period_s']:.4f} s, amplitude {result['amplitude_deg']:.1f}°)"
                )
            else:
                title = f"{name}: T = {result['period_s']:.4f} s, amplitude {result['amplitude_deg']:.1f}° (no scale, no length given)"
        elif model == "oscillator":
            result = fit_oscillator(w.t, w.x, w.y, mass_kg, mass_err_kg)
            title = f"{name}: T = {result['period_s']:.4f} s, damping {result['damping_per_s']:.3f}/s"
            if mass_kg:
                title += f", k = {result['spring_constant_N_m']:.2f} ± {result['spring_constant_err_N_m']:.2f} N/m"
        elif model in ("freefall", "incline-roll", "incline-slide"):
            result = fit_constant_acceleration(w.t, w.x, w.y, model, angle_deg, scale_err)
            if model == "freefall":
                title = (
                    f"{name}: g = {result['g_m_s2']:.3f} ± {result['g_err_m_s2']:.3f} m/s² (free fall)"
                    if u == "cm"
                    else f"{name}: a = {result['g_cm_s2']:.1f} px/s² downward (free fall, no scale)"
                )
            else:
                title = f"{name}: a = {result['accel_cm_s2']:.1f} {u}/s² on {result['angle_deg']:.1f}°"
                if u == "cm":
                    title += f", friction coefficient {result['friction_coeff']:.3f}"
        elif model == "projectile":
            result = fit_projectile(w.t, w.x, w.y, scale_err)
            title = (
                f"{name}: g = {result['g_m_s2']:.3f} ± {result['g_err_m_s2']:.3f} m/s² (projectile)"
                if u == "cm"
                else f"{name}: a = {result['g_cm_s2']:.1f} px/s² downward (projectile, no scale)"
            )
        else:
            result = fit_constant_velocity(w.t, w.x, w.y)
            title = f"{name}: v = {result['speed_cm_s']:.1f} ± {result['speed_err_cm_s']:.1f} {u}/s"
        result["track"] = name
        result["window"] = {"start_s": start, "end_s": end, "how": how, "points": len(w)}
        label = name
    if u == "px":
        result = _to_pixels(result, model, bool(length_cm))
    else:
        result["length_unit"] = "cm"
        if px_per_cm:
            result["scale_note"] = f"scale {px_per_cm:.4g} px/cm set during this analysis"
    if plot:
        filename = f"fit_{model}_{slug(label)}.png"
        plot_fit(result["_curve"], title, run / filename)
        result["plot"] = filename
    result["title"] = title
    return result


def series_on_run(run: Path, track: str | None, series: str, start=None, end=None, px_per_cm=None):
    """Data for equation fitting: (t from window start, values, label, track, window info)."""
    if series not in SERIES:
        raise ValueError(f"series must be one of {', '.join(SERIES)}")
    tracks, info = load_run(Path(run), px_per_cm)
    chosen = choose_track(tracks, track)
    w, start, end, how = _windowed(chosen, "generic", start, end, unit_factor(info["unit"]))
    tt = w.t - w.t[0]
    if series == "x":
        data = w.x
    elif series == "y":
        data = w.y
    elif series == "s":
        _, data, _ = principal_axis(w.x, w.y)
    elif series in ("theta", "r"):
        (cx, cy, _), _, _ = fit_circle(w.x, w.y)
        data = np.arctan2(w.x - cx, cy - w.y) if series == "theta" else np.hypot(w.x - cx, w.y - cy)
    else:
        vx, vy = velocity(w.t, w.x, w.y)
        data = np.hypot(vx, vy)
    window = {"start_s": start, "end_s": end, "how": how, "points": len(w), "t_is": f"seconds since {start:.3f} s"}
    return tt, np.asarray(data, dtype=float), SERIES_INFO[series].format(u=info["unit"]), chosen.name, window


def fit_equation_on_run(
    run: Path, track, series, expression, initial=None, start=None, end=None, plot_name=None, px_per_cm=None
) -> dict:
    run = Path(run)
    tt, data, label, name, window = series_on_run(run, track, series, start, end, px_per_cm)
    result = fit_expression(tt, data, expression, initial)
    result.update({"track": name, "series": series, "series_label": label, "window": window})
    if plot_name:
        fine = np.linspace(tt[0], tt[-1], 1500)
        curve = {
            "panels": [
                {
                    "label": label, "t": tt, "data": data, "fit_t": fine,
                    "fit": result["_f"](fine, *result["_popt"]), "residual": result["_residual"],
                }
            ]
        }
        title = f"{name}: {series}(t) = {expression}   (rms {result['rms']:.3g}, BIC {result['bic']:.1f})"
        plot_fit(curve, title, run / plot_name)
        result["plot"] = plot_name
    return result


def samples_table(run: Path, track: str | None, start=None, end=None, max_rows: int = 40, px_per_cm=None) -> str:
    tracks, info = load_run(Path(run), px_per_cm)
    u = info["unit"]
    chosen = choose_track(tracks, track)
    w = chosen.window(start, end)
    if len(w) == 0:
        return "no points in that window"
    vx, vy = velocity(w.t, w.x, w.y) if len(w) >= 3 else (np.zeros(len(w)), np.zeros(len(w)))
    max_rows = int(min(max(max_rows, 5), 120))
    index = np.unique(np.linspace(0, len(w) - 1, min(max_rows, len(w))).round().astype(int))
    lines = [
        f"track {chosen.name}: {len(w)} points in the window, showing {len(index)} (lengths in {u}, y up)",
        f"t_s, x_{u}, y_{u}, speed_{u}_s",
    ]
    for i in index:
        lines.append(f"{w.t[i]:.3f}, {w.x[i]:.2f}, {w.y[i]:.2f}, {math.hypot(vx[i], vy[i]):.1f}")
    return "\n".join(lines)


def run_overview(run: Path, px_per_cm: float | None = None) -> dict:
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) if (run / "summary.json").exists() else {}
    tracks, info = load_run(run, px_per_cm)
    fps = summary.get("fps")
    described = [describe_track(track, fps, info["unit"]) for track in sorted(tracks.values(), key=len, reverse=True)[:12]]
    scale = {
        "length_unit": info["unit"],
        "px_per_cm": info["px_per_cm"],
        "source": info["scale_source"],
        "check": summary.get("scale_check"),
    }
    if info["unit"] == "px":
        scale["note"] = NO_SCALE_NOTE
    return {
        "video": {
            "width": summary.get("width"),
            "height": summary.get("height"),
            "fps": fps,
            "frames": summary.get("frames"),
            "duration_s": round(summary["frames"] / fps, 3) if summary.get("frames") and fps else None,
        },
        "scale": scale,
        "prompts": summary.get("prompts"),
        "coverage": summary.get("tracks"),
        "tracks": clean(described, 5),
        "more_tracks": max(0, len(tracks) - 12),
    }


# re-exported for fit.py
__all__ = [
    "MODELS", "SERIES", "auto_window", "clean", "fit_on_run", "fit_equation_on_run", "run_overview",
    "samples_table", "slug", "smooth",
]