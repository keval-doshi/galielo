#!/usr/bin/env python3
"""Fit one physics model to a tracked run by hand (Galileo does this for you automatically).

    python fit.py runs/pendulum1 --model pendulum
    python fit.py runs/pendulum1 --model pendulum --length-cm 52.0     # length measured by hand
    python fit.py runs/spring1 --model oscillator --mass-kg 0.2         # gives the spring constant
    python fit.py runs/drop1 --model freefall
    python fit.py runs/ramp1 --model incline-roll --angle-deg 18.5
    python fit.py runs/dominoes1 --model domino-wave --track domino
    python fit.py runs/nosheet1 --model freefall --px-per-cm 9.6        # set the scale for a run without one

The time window starts just after release and ends when the motion phase ends, unless you
give --start/--end. Runs without a scale are fitted in pixels (a pendulum with --length-cm
still gives g). Writes fit_<model>_<track>.png and .json next to tracks.csv, and keeps the
calibrated numbers in physics_params.json for the simulator later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oe.analysis import MODELS, clean, fit_on_run

HIDDEN = {"model", "track", "window", "plot", "title", "warnings", "_curve", "events", "units_note", "note",
          "friction_note", "spring_note", "length_unit", "scale_note"}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder written by track.py")
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--track", help='which track, e.g. "red ball#0" (default: the longest); for domino-wave, the prompt')
    ap.add_argument("--start", type=float, help="only use data after this time (s); default: just after release")
    ap.add_argument("--end", type=float, help="only use data before this time (s)")
    ap.add_argument("--length-cm", type=float, help="pendulum: pivot-to-bob-centre length measured by hand")
    ap.add_argument("--length-err-cm", type=float, default=0.3, help="uncertainty of --length-cm (default 0.3)")
    ap.add_argument("--scale-err-pct", type=float, default=1.0, help="assumed cm-scale uncertainty in %% (default 1)")
    ap.add_argument("--angle-deg", type=float, help="incline: ramp angle measured by hand (otherwise taken from the path)")
    ap.add_argument("--mass-kg", type=float, help="oscillator: the moving mass, to get the spring constant")
    ap.add_argument("--mass-err-kg", type=float, default=0.0, help="uncertainty of --mass-kg")
    ap.add_argument("--px-per-cm", type=float, help="use this scale (pixels per cm in the tracked clip) instead of the run's own")
    args = ap.parse_args(argv)

    run = Path(args.run)
    try:
        result = fit_on_run(
            run, args.model, args.track, args.start, args.end,
            args.length_cm, args.length_err_cm, args.angle_deg, args.scale_err_pct / 100,
            px_per_cm=args.px_per_cm, mass_kg=args.mass_kg, mass_err_kg=args.mass_err_kg,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        sys.exit(f"error: {error}")

    print(result["title"])
    window = result.get("window")
    if window:
        print(f"  window: {window['start_s']:.3f}-{window['end_s']:.3f} s ({window['how']}), {window['points']} points")
    public = clean(result)
    for key, value in public.items():
        if key not in HIDDEN and value is not None:
            print(f"  {key:<28} {value}")
    for event in public.get("events") or []:
        print(f"  piece {event}")
    for warning in result.get("warnings", []):
        print(f"  warning: {warning}")
    if result.get("units_note"):
        print(f"  note: {result['units_note']}")

    params = {}
    unit_ok = public.get("length_unit") == "cm" or (args.model == "pendulum" and args.length_cm)
    if args.model in ("pendulum", "freefall", "projectile") and unit_ok and "g_cm_s2" in public:
        params["g_cm_s2"] = public["g_cm_s2"]
        if args.model == "pendulum":
            params["pendulum_damping_per_s"] = public.get("damping_per_s")
    elif args.model in ("incline-roll", "incline-slide") and public.get("friction_coeff") is not None:
        params["rolling_resistance" if args.model == "incline-roll" else "kinetic_friction"] = public["friction_coeff"]
    elif args.model == "domino-wave" and unit_ok:
        params["domino_wave_speed_cm_s"] = public["wave_speed_cm_s"]
    elif args.model == "oscillator" and public.get("spring_constant_N_m") is not None:
        params["spring_constant_N_m"] = public["spring_constant_N_m"]

    json_name = Path(result["plot"]).with_suffix(".json").name
    (run / json_name).write_text(json.dumps(public, indent=2), encoding="utf-8")
    if params:
        params_file = Path("physics_params.json")
        stored = json.loads(params_file.read_text(encoding="utf-8")) if params_file.exists() else {}
        stored.update(params)
        params_file.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    print(f"\nsaved {run / result['plot']} and {json_name}" + ("; calibrated values merged into physics_params.json" if params else ""))


if __name__ == "__main__":
    main()