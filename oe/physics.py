"""Physics fits on tracked positions.

Positions are in cm with y pointing up, times in seconds. Every fit returns a dict of numbers
plus a private ``_curve`` with plot panels (see oe.plots).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit, least_squares

from .config import G_STANDARD_CM
from .motion import principal_axis


def _panel(label, t, data, fit_t, fit, residual):
    return {"label": label, "t": t, "data": data, "fit_t": fit_t, "fit": fit, "residual": residual}


def fit_circle(x: np.ndarray, y: np.ndarray):
    """Least-squares circle through the points: ((cx, cy, r), (err_cx, err_cy, err_r), rms)."""
    a = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    r = math.sqrt(max(c + cx**2 + cy**2, 1e-12))
    solution = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, r])
    dof = max(len(x) - 3, 1)
    s2 = float((solution.fun**2).sum() / dof)
    cov = np.linalg.pinv(solution.jac.T @ solution.jac) * s2
    return tuple(solution.x), tuple(np.sqrt(np.diag(cov))), math.sqrt(s2)


def _dominant_frequency(t: np.ndarray, signal: np.ndarray) -> float:
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    resampled = np.interp(grid, t, signal - signal.mean())
    spectrum = np.abs(np.fft.rfft(resampled * np.hanning(len(grid))))
    freqs = np.fft.rfftfreq(len(grid), dt)
    return float(freqs[1:][np.argmax(spectrum[1:])])


def _damped_cosine(t, offset, amplitude, gamma, freq, phase):
    return offset + amplitude * np.exp(-gamma * t) * np.cos(2 * np.pi * freq * t + phase)


def fit_pendulum(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    length_cm: float | None = None,
    length_err_cm: float = 0.0,
    scale_err: float = 0.01,
) -> dict:
    """Fit a (lightly damped) pendulum and estimate g.

    The pivot and string length come from the circle the bob traces, so the pivot does not
    need to be visible. The period is corrected for amplitude and damping before g = 4 pi^2 L / T0^2.
    """
    (cx, cy, radius), (_, _, radius_err), rms = fit_circle(x, y)
    warnings = []
    if cy < float(np.median(y)):
        warnings.append("the fitted pivot is below the bob: is y pointing up, and is this a pendulum?")
    theta = np.arctan2(x - cx, cy - y)
    tt = t - t[0]
    f0 = _dominant_frequency(tt, theta)
    a0 = math.sqrt(2) * float(theta.std())
    best = None
    for phase0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        try:
            params, cov = curve_fit(_damped_cosine, tt, theta, p0=[theta.mean(), a0, 0.01, f0, phase0], maxfev=20000)
        except RuntimeError:
            continue
        sse = float(((_damped_cosine(tt, *params) - theta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, params, cov)
    if best is None:
        raise RuntimeError("the pendulum fit did not converge; is the window long enough (3+ swings)?")
    _, params, cov = best
    offset, amplitude, gamma, freq, phase = params
    amplitude = abs(amplitude)
    errs = np.sqrt(np.abs(np.diag(cov)))
    omega_d = 2 * math.pi * freq
    omega_0 = math.sqrt(omega_d**2 + gamma**2)  # undo the small damping shift
    period = 2 * math.pi / omega_0
    period_err = period * errs[3] / freq
    mean_amp = float(np.mean(amplitude * np.exp(-gamma * tt)))
    period_small = period / (1 + mean_amp**2 / 16 + 11 * mean_amp**4 / 3072)

    if length_cm:
        length, length_err, length_source = length_cm, length_err_cm, "measured by hand"
    else:
        length, length_source = radius, "radius of the traced arc"
        length_err = math.hypot(radius_err, scale_err * radius)
    g = 4 * math.pi**2 * length / period_small**2
    g_err = g * math.hypot(length_err / length, 2 * period_err / period_small)
    residual = theta - _damped_cosine(tt, *params)
    if (tt[-1] - tt[0]) * freq < 2.5:
        warnings.append("fewer than 2.5 swings in the window: the period is less certain than the error suggests")
    if math.degrees(mean_amp) > 25:
        warnings.append("large swing (>25 deg): the amplitude correction matters; smaller swings are more accurate")
    fine = np.linspace(tt[0], tt[-1], 2000)
    return {
        "model": "pendulum",
        "g_cm_s2": g,
        "g_err_cm_s2": g_err,
        "g_m_s2": g / 100,
        "g_err_m_s2": g_err / 100,
        "g_vs_standard_pct": 100 * (g / G_STANDARD_CM - 1),
        "length_cm": length,
        "length_err_cm": length_err,
        "length_source": length_source,
        "arc_radius_cm": radius,
        "arc_radius_err_cm": math.hypot(radius_err, scale_err * radius),
        "period_s": period,
        "period_err_s": period_err,
        "period_small_angle_s": period_small,
        "amplitude_deg": math.degrees(mean_amp),
        "damping_per_s": gamma,
        "quality_factor": omega_0 / (2 * gamma) if gamma > 0 else None,
        "pivot_cm": [cx, cy],
        "arc_rms_cm": rms,
        "angle_rms_deg": math.degrees(float(np.sqrt(np.mean(residual**2)))),
        "points": len(t),
        "warnings": warnings,
        "_curve": {
            "panels": [
                _panel("angle (deg)", tt, np.degrees(theta), fine, np.degrees(_damped_cosine(fine, *params)), np.degrees(residual))
            ]
        },
    }


def fit_constant_acceleration(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "incline-roll",
    angle_deg: float | None = None,
    scale_err: float = 0.01,
) -> dict:
    """Fit s(t) = s0 + v0 t + a t^2 / 2 along the motion.

    mode: "freefall" (vertical, gives g), "incline-roll" (ball rolling without slipping,
    gives a rolling-resistance coefficient) or "incline-slide" (gives kinetic friction).
    Distances carry the cm-scale uncertainty ``scale_err`` (relative), which is added to the errors.
    """
    tt = t - t[0]
    direction, s, straightness = principal_axis(x, y)
    warnings = []
    if straightness > 0.08:
        warnings.append(f"the path is not a straight line (spread {straightness:.0%}); is this the right model?")
    fine = np.linspace(tt[0], tt[-1], 500)

    if mode == "freefall":
        coeffs, cov = np.polyfit(tt, y, 2, cov=True)
        a, a_stat = 2 * coeffs[0], 2 * math.sqrt(cov[0, 0])
        a_err = math.hypot(a_stat, scale_err * a)
        result = {
            "model": mode,
            "g_cm_s2": -a,
            "g_err_cm_s2": a_err,
            "g_m_s2": -a / 100,
            "g_err_m_s2": a_err / 100,
            "g_stat_err_cm_s2": a_stat,
            "scale_err_assumed": scale_err,
            "g_vs_standard_pct": 100 * (-a / G_STANDARD_CM - 1),
            "start_speed_cm_s": float(coeffs[1]),
        }
        series, label = y, "height (cm)"
        tilt = abs(90 - abs(math.degrees(math.atan2(direction[1], direction[0]))))
        if tilt > 5:
            warnings.append(f"the drop is {tilt:.0f} degrees off vertical: is the camera level?")
    else:
        coeffs, cov = np.polyfit(tt, s, 2, cov=True)
        a, a_stat = 2 * coeffs[0], 2 * math.sqrt(cov[0, 0])
        a_err = math.hypot(a_stat, scale_err * a)
        measured_angle = math.degrees(math.atan2(abs(direction[1]), abs(direction[0])))
        angle = angle_deg if angle_deg is not None else measured_angle
        th = math.radians(angle)
        g = G_STANDARD_CM
        if mode == "incline-roll":
            mu = math.tan(th) - 7 * a / (5 * g * math.cos(th))
            mu_err = 7 * a_err / (5 * g * math.cos(th))
            ideal = 5 / 7 * g * math.sin(th)
        else:
            mu = math.tan(th) - a / (g * math.cos(th))
            mu_err = a_err / (g * math.cos(th))
            ideal = g * math.sin(th)
        result = {
            "model": mode,
            "accel_cm_s2": a,
            "accel_err_cm_s2": a_err,
            "accel_stat_err_cm_s2": a_stat,
            "scale_err_assumed": scale_err,
            "friction_coeff_err": mu_err,
            "friction_note": "the error does not include the ramp angle; 0.5 degrees of angle error moves "
            "the coefficient by about 0.009",
            "angle_deg": angle,
            "angle_source": "given" if angle_deg is not None else "from the path (assumes a level camera)",
            "path_angle_deg": measured_angle,
            "friction_coeff": mu,
            "frictionless_accel_cm_s2": ideal,
            "efficiency": a / ideal if ideal else None,
            "start_speed_cm_s": float(coeffs[1]),
            "note": "efficiency = a / (value for a solid ball rolling without slipping or resistance) for incline-roll, "
            "a / (g sin angle) for incline-slide. Rolling: solid ball 1.0, hollow ball 0.84; "
            "sliding without friction 1.4 on the rolling scale",
        }
        series, label = s, "distance along the path (cm)"
    fitted = np.polyval(coeffs, tt)
    result.update(
        {
            "path_spread": straightness,
            "rms_cm": float(np.sqrt(np.mean((series - fitted) ** 2))),
            "points": len(t),
            "warnings": warnings,
            "_curve": {"panels": [_panel(label, tt, series, fine, np.polyval(coeffs, fine), series - fitted)]},
        }
    )
    return result


def fit_projectile(t: np.ndarray, x: np.ndarray, y: np.ndarray, scale_err: float = 0.01) -> dict:
    """x = x0 + vx t, y = y0 + vy t - g t^2 / 2 (no drag)."""
    tt = t - t[0]
    cx, cov_x = np.polyfit(tt, x, 1, cov=True)
    cy, cov_y = np.polyfit(tt, y, 2, cov=True)
    g, g_stat = -2 * cy[0], 2 * math.sqrt(cov_y[0, 0])
    g_err = math.hypot(g_stat, scale_err * g)
    vx, vy = float(cx[0]), float(cy[1])
    fx, fy = np.polyval(cx, tt), np.polyval(cy, tt)
    # does x slow down? (air drag or a bad camera angle)
    qx = np.polyfit(tt, x, 2)
    warnings = []
    if abs(2 * qx[0]) > 0.1 * abs(g) and len(tt) > 8:
        warnings.append(f"horizontal motion is not uniform (a_x = {2 * qx[0]:.0f} cm/s^2): drag, spin or a tilted camera")
    fine = np.linspace(tt[0], tt[-1], 500)
    return {
        "model": "projectile",
        "g_cm_s2": g,
        "g_err_cm_s2": g_err,
        "g_m_s2": g / 100,
        "g_err_m_s2": g_err / 100,
        "g_vs_standard_pct": 100 * (g / G_STANDARD_CM - 1),
        "g_stat_err_cm_s2": g_stat,
        "scale_err_assumed": scale_err,
        "vx_cm_s": vx,
        "vx_err_cm_s": math.sqrt(cov_x[0, 0]),
        "vy0_cm_s": vy,
        "launch_speed_cm_s": math.hypot(vx, vy),
        "launch_angle_deg": math.degrees(math.atan2(vy, vx)),
        "rms_cm": float(np.sqrt(np.mean((x - fx) ** 2 + (y - fy) ** 2))),
        "points": len(t),
        "warnings": warnings,
        "_curve": {
            "panels": [
                _panel("x (cm)", tt, x, fine, np.polyval(cx, fine), x - fx),
                _panel("y (cm)", tt, y, fine, np.polyval(cy, fine), y - fy),
            ]
        },
    }


def fit_constant_velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    tt = t - t[0]
    direction, s, straightness = principal_axis(x, y)
    coeffs, cov = np.polyfit(tt, s, 1, cov=True)
    fitted = np.polyval(coeffs, tt)
    quad = np.polyfit(tt, s, 2)
    fine = np.linspace(tt[0], tt[-1], 500)
    return {
        "model": "constant-velocity",
        "speed_cm_s": float(coeffs[0]),
        "speed_err_cm_s": math.sqrt(cov[0, 0]),
        "direction_deg": math.degrees(math.atan2(direction[1], direction[0])),
        "leftover_accel_cm_s2": float(2 * quad[0]),
        "path_spread": straightness,
        "rms_cm": float(np.sqrt(np.mean((s - fitted) ** 2))),
        "points": len(t),
        "warnings": [],
        "_curve": {"panels": [_panel("distance along the path (cm)", tt, s, fine, np.polyval(coeffs, fine), s - fitted)]},
    }


def fit_domino_wave(positions_cm: np.ndarray, start_times: np.ndarray, names: list[str]) -> dict:
    """Speed of a toppling wave from where each piece stands and when it starts to fall.

    Regresses time on position (timing is the noisier quantity), so v = 1 / slope.
    """
    order = np.argsort(positions_cm)
    pos, times = positions_cm[order], start_times[order]
    names = [names[i] for i in order]
    if len(pos) < 3:
        raise ValueError("need at least 3 pieces that fell to measure a wave speed")
    coeffs, cov = np.polyfit(pos, times, 1, cov=True)
    slope, slope_err = float(coeffs[0]), math.sqrt(cov[0, 0])
    if abs(slope) < 1e-9:
        raise ValueError("all pieces started at the same time; this is not a toppling wave")
    speed = 1 / abs(slope)
    speed_err = speed * slope_err / abs(slope)
    predicted = np.polyval(coeffs, pos)
    t_line = np.linspace(times.min(), times.max(), 200)
    return {
        "model": "domino-wave",
        "wave_speed_cm_s": speed,
        "wave_speed_err_cm_s": speed_err,
        "direction": "+x" if slope > 0 else "-x",
        "pieces": len(pos),
        "median_spacing_cm": float(np.median(np.diff(pos))) if len(pos) > 1 else None,
        "timing_rms_s": float(np.sqrt(np.mean((times - predicted) ** 2))),
        "events": [{"track": n, "position_cm": round(float(p), 2), "start_t": round(float(s), 3)} for n, p, s in zip(names, pos, times)],
        "warnings": [],
        "_curve": {
            "panels": [
                _panel(
                    "position along the line (cm)",
                    times,
                    pos,
                    t_line,
                    (t_line - coeffs[1]) / coeffs[0],
                    pos - (times - coeffs[1]) / coeffs[0],
                )
            ],
            "xlabel": "time the piece starts to fall (s)",
            "markers": True,
        },
    }


def fit_oscillator(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, mass_kg: float | None = None, mass_err_kg: float = 0.0
) -> dict:
    """Damped back-and-forth motion along a line (a mass on a spring, a cart between springs...).

    Fits s(t) = s0 + A e^(-gamma t) cos(2 pi f t + phi) along the main direction of motion.
    With the moving mass, the spring constant is k = m * omega0^2.
    """
    tt = t - t[0]
    direction, s, straightness = principal_axis(x, y)
    s = s - s.mean()
    f0 = _dominant_frequency(tt, s)
    a0 = math.sqrt(2) * float(s.std())
    best = None
    for phase0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        try:
            params, cov = curve_fit(_damped_cosine, tt, s, p0=[0.0, a0, 0.05, f0, phase0], maxfev=20000)
        except RuntimeError:
            continue
        sse = float(((_damped_cosine(tt, *params) - s) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, params, cov)
    if best is None:
        raise RuntimeError("the oscillation fit did not converge; is the window long enough (3+ cycles)?")
    _, params, cov = best
    offset, amplitude, gamma, freq, phase = params
    freq, amplitude = abs(freq), abs(amplitude)
    errs = np.sqrt(np.abs(np.diag(cov)))
    omega_d = 2 * math.pi * freq
    omega_0 = math.sqrt(omega_d**2 + gamma**2)
    omega_0_err = 2 * math.pi * errs[3] * omega_d / omega_0
    residual = s - _damped_cosine(tt, *params)
    warnings = []
    if straightness > 0.1:
        warnings.append(f"the motion is not along a line (spread {straightness:.0%}); for swinging use the pendulum model")
    if (tt[-1] - tt[0]) * freq < 2.5:
        warnings.append("fewer than 2.5 cycles in the window: the period is less certain than the error suggests")
    fine = np.linspace(tt[0], tt[-1], 2000)
    result = {
        "model": "oscillator",
        "period_s": 1 / freq,
        "period_err_s": errs[3] / freq**2,
        "natural_period_s": 2 * math.pi / omega_0,
        "angular_frequency_rad_s": omega_0,
        "angular_frequency_err_rad_s": omega_0_err,
        "damping_per_s": gamma,
        "quality_factor": omega_0 / (2 * gamma) if gamma > 0 else None,
        "amplitude_cm": amplitude,
        "line_angle_deg_from_horizontal": math.degrees(math.atan2(direction[1], direction[0])),
        "rms_cm": float(np.sqrt(np.mean(residual**2))),
        "points": len(t),
        "warnings": warnings,
        "_curve": {
            "panels": [
                _panel("displacement along the line (cm)", tt, s, fine, _damped_cosine(fine, *params), residual)
            ]
        },
    }
    if mass_kg:
        k = mass_kg * omega_0**2
        k_err = k * math.hypot(mass_err_kg / mass_kg, 2 * omega_0_err / omega_0)
        result.update(
            {
                "mass_kg": mass_kg,
                "spring_constant_N_m": k,
                "spring_constant_err_N_m": k_err,
                "spring_note": "k = m * omega0^2; for a mass hanging on a spring, add about a third of the "
                "spring's own mass to m for a better value",
            }
        )
    return result