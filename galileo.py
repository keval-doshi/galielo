#!/usr/bin/env python3
"""Galileo: let Muse Wonder work out the physics of a tracked run by itself.

    python galileo.py runs/pendulum1
    python galileo.py runs/pendulum1 --note "string 61.5 cm from the knot to the bob centre"
    python galileo.py runs/ramp1 --note "ramp 30 cm long, raised 8.2 cm at one end; glass marble"
    python galileo.py runs/pendulum1 --model muse-spark-1.3          # cheaper and faster

Run track.py first. Galileo writes report.md, report.json, agent_transcript.md, its fit plots and
a PDF report (galileo-report-<run>.pdf) into the run folder.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from oe import config
from oe.agent import Galileo, GalileoError
from oe.meta_api import MetaAPIError
from oe.report import build_pdf

EFFORTS = ["minimal", "low", "medium", "high", "xhigh", "max"]


def printer(kind: str, text: str) -> None:
    if kind == "assistant":
        print(textwrap.fill(text, 100, initial_indent="  ", subsequent_indent="  "))
    elif kind == "tool":
        print(f"  > {text}")
    elif kind == "result":
        print(f"    = {text}")
    elif kind == "report":
        print(f"\n{text}")
    else:
        print(text)
    sys.stdout.flush()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder written by track.py")
    ap.add_argument("--note", default="", help="what you measured by hand or know about the setup")
    ap.add_argument("--model", default=config.WONDER_MODEL, help=f"default {config.WONDER_MODEL}; muse-spark-1.3 is cheaper")
    ap.add_argument("--effort", choices=EFFORTS, help="reasoning effort (default: the model's own default)")
    ap.add_argument("--max-steps", type=int, default=14, help="maximum model calls (default 14)")
    args = ap.parse_args(argv)

    run = Path(args.run)
    if not (run / "tracks.csv").exists():
        print(f"error: {run / 'tracks.csv'} not found; run track.py first", file=sys.stderr)
        return 1
    agent = Galileo(run, model=args.model, effort=args.effort, notes=args.note, on_event=printer, max_steps=args.max_steps)
    try:
        agent.run_agent()
    except (GalileoError, MetaAPIError) as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1
    print(f"\nreport: {run / 'report.md'}   (how it got there: {run / 'agent_transcript.md'})")
    try:
        print(f"PDF:    {build_pdf(run)}")
    except Exception as error:  # noqa: BLE001 - the report itself is already saved
        print(f"(the PDF could not be made: {error})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())