"""Galileo: Muse Wonder as an experimental physicist with fitting tools.

The agent sees an overview of a tracked run plus up to three images, then calls tools
(get_samples, fit_model, fit_equation, set_scale) until it submits a report with submit_report.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import cv2

from . import config, meta_api
from .analysis import MODELS, SERIES, clean, fit_equation_on_run, fit_on_run, run_overview, samples_table, slug
from .equations import FUNCTIONS

# USD per million tokens: (input, cached input, output)
PRICES = {"muse-wonder": (4.0, 0.2, 20.0), "muse-spark": (1.25, 0.15, 4.25)}

SYSTEM_PROMPT = """You are Galileo, an experimental physicist inside a lab-notebook app.
A video was analysed: SAM 3.1 segmented and tracked the moving objects, and each track is a series of
centroid positions (x to the right, y up) against time in seconds. The overview says whether
lengths are in centimetres (a scale came from a reference object of known size, or was given) or in
pixels (no scale yet).

Your job: work out what the experiment is, find the physical law that describes the motion,
measure the key quantity with an honest uncertainty, and explain the result simply.

How to work:
1. Read the overview, the notes from the person and the images. State in one or two sentences what
   you think the experiment is.
2. Sort out the scale (see below), then fit the most plausible model with fit_model and check it
   (rms, warnings, window).
3. Test at least one alternative, with fit_model or with your own equation in fit_equation,
   and compare the candidates by BIC (lower is better; a difference above 10 is decisive).
4. If something looks off (scale check, gaps, odd residuals, wrong window), investigate with
   get_samples or refit with a different time window.
5. Finish by calling submit_report exactly once.

Scale:
- Times and angles never need a scale. Lengths, speeds and accelerations do.
- If lengths are in pixels and the notes give a real length (a pendulum length, a drop height, a
  ramp length, a distance between two marks, the size of an object), measure the same thing in
  pixels with the tools (path extent, drop distance from get_samples, arc radius...), compute
  px_per_cm = pixels / centimetres and call set_scale. Everything after that is in cm. Say in the
  report exactly how you calibrated, and treat the uncertainty of that length as a scale uncertainty.
- A pendulum with a known length needs no scale at all: pass length_cm to fit_model.
- If the notes give a better length than the reference object, you may still use it; explain why.
- With no scale and no known length, report what is scale-free (periods, angles, ratios, timings)
  and say what measurement would complete the result.

Physics notes:
- Pendulum: the period is very precise. If the notes give the length, pass it as length_cm and report
  the arc radius as a cross-check. For a rigid pendulum (a rod or a block rather than a small bob on a
  string), the tracked point's arc radius is not the effective length; use the given length and explain
  the difference instead of calling it an error.
- Spring or anything oscillating along a line: model "oscillator"; pass mass_kg (and mass_err_kg) if
  the notes give the moving mass to get the spring constant.
- Ramp: give angle_deg if the notes give the ramp height and length (angle = asin(height/length));
  otherwise the angle comes from the path and assumes a level camera. For a rolling ball the
  efficiency (measured acceleration / ideal solid-ball value) tells you about the ball and the
  rolling resistance: 1.0 solid ball with no losses, 0.84 hollow ball.
- Free fall and projectile: the result scales directly with the length scale; timing is precise.
- Dominoes: use fit_model with model "domino-wave" and the prompt name as track.
- Anything else: write the candidate laws yourself with fit_equation and compare them.
- g = 9.80665 m/s² on Earth (1.62 m/s² on the Moon) is a reference for comparison only. When the
  goal is to measure g, never assume it.

Rules:
- Use only numbers that come from the tools, the overview or the notes. Never invent values.
- Report uncertainties as ± one standard error and say what dominates them.
- Keep text between tool calls short. Use at most 10 tool calls before submit_report."""


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_WINDOW = {
    "start": {"type": "number", "description": "start of the time window in seconds (default: just after release)"},
    "end": {"type": "number", "description": "end of the time window in seconds (default: end of this motion phase)"},
}
_TRACK = {"type": "string", "description": 'track name, e.g. "red ball#0" (default: the longest track)'}

TOOLS = [
    _schema(
        "get_samples",
        "Look at raw tracked positions (and smoothed speed) of one track, evenly subsampled.",
        {"track": _TRACK, **_WINDOW, "max_rows": {"type": "integer", "description": "rows to show, 5-120 (default 40)"}},
        [],
    ),
    _schema(
        "fit_model",
        "Fit a standard physical model to one track (or to all pieces for domino-wave) and return the "
        "physical quantities with uncertainties, fit quality and warnings. Saves a plot.",
        {
            "model": {"type": "string", "enum": list(MODELS)},
            "track": _TRACK,
            **_WINDOW,
            "length_cm": {"type": "number", "description": "pendulum: known length, pivot to the bob centre (or the effective length of a rigid pendulum)"},
            "length_err_cm": {"type": "number", "description": "uncertainty of length_cm (default 0.3)"},
            "angle_deg": {"type": "number", "description": "ramp: angle measured by hand"},
            "scale_err": {"type": "number", "description": "relative uncertainty of the cm scale (default 0.01 = 1%)"},
            "mass_kg": {"type": "number", "description": "oscillator: the moving mass, to get the spring constant"},
            "mass_err_kg": {"type": "number", "description": "oscillator: uncertainty of mass_kg"},
        },
        ["model"],
    ),
    _schema(
        "set_scale",
        "Set the length scale for the rest of this analysis, when you can work it out from a real length "
        "(for example the drop height or the distance between two marks given in the notes, measured in "
        "pixels with the other tools). Afterwards every length is in cm. Returns the track overview in cm.",
        {
            "px_per_cm": {"type": "number", "description": "pixels per centimetre in the tracked video"},
            "reason": {"type": "string", "description": "which length you used and how you measured it in pixels"},
        },
        ["px_per_cm", "reason"],
    ),
    _schema(
        "fit_equation",
        "Fit your own candidate law to one data series of a track by least squares. The expression uses "
        "the variable t (seconds since the window start), parameter names of your choice, numbers, + - * / ** "
        f"and the functions {', '.join(sorted(FUNCTIONS))}, plus pi. Returns parameters with errors, rms, "
        "R², AIC, BIC and a residual check. Saves a plot. Default window: from just after release to the "
        "end of the track; pass start/end (for example the window fit_model reported) to fit one phase only.",
        {
            "track": _TRACK,
            "series": {
                "type": "string",
                "enum": list(SERIES),
                "description": "x or y, s (distance along the main direction), theta (angle about the fitted "
                "pivot, rad), r (distance from the pivot), speed; lengths in cm, or in px when there is no scale",
            },
            "expression": {"type": "string", "description": 'e.g. "A*exp(-k*t)*cos(w*t + p) + c"'},
            "initial": {
                "type": "object",
                "description": 'starting values for the parameters, e.g. {"A": 0.25, "w": 4.2}; good guesses matter '
                "for oscillations",
                "additionalProperties": {"type": "number"},
            },
            **_WINDOW,
        },
        ["series", "expression"],
    ),
    _schema(
        "submit_report",
        "Submit the final result. Call exactly once, at the end.",
        {
            "experiment": {"type": "string", "description": "short name, e.g. 'Simple pendulum'"},
            "headline": {"type": "string", "description": "one sentence with the main measured result and how it compares"},
            "law": {"type": "string", "description": "the law in plain text, e.g. 'θ(t) = θ0·e^(−γt)·cos(2πt/T + φ), T = 2π√(L/g)'"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quantity": {"type": "string"},
                        "value": {"type": "number"},
                        "uncertainty": {"type": "number"},
                        "unit": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["quantity", "value", "unit"],
                },
            },
            "comparisons": {
                "type": "array",
                "description": "candidate models you tried",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "bic": {"type": "number"},
                        "rms": {"type": "number"},
                        "verdict": {"type": "string"},
                    },
                    "required": ["model", "verdict"],
                },
            },
            "data_quality": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string", "description": "3-5 sentences for a general audience"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "best_plot": {"type": "string", "description": "file name of the plot that best shows the result"},
        },
        ["experiment", "headline", "law", "results", "explanation"],
    ),
]


class GalileoError(RuntimeError):
    pass


def format_value(value, uncertainty=None) -> str:
    """'9.824 ± 0.099' with two significant figures in the uncertainty."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if uncertainty is None or not math.isfinite(float(uncertainty)) or float(uncertainty) <= 0:
        return f"{value:.4g}"
    uncertainty = float(uncertainty)
    digits = max(0, 1 - math.floor(math.log10(uncertainty)))
    return f"{value:.{digits}f} ± {uncertainty:.{digits}f}"


def _price(model: str):
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return None


def _frame_from(video: Path, time_s: float, fps: float | None):
    cap = cv2.VideoCapture(str(video))
    try:
        if fps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_s * fps))))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class Galileo:
    def __init__(
        self,
        run: Path,
        *,
        model: str | None = None,
        effort: str | None = None,
        notes: str = "",
        client=None,
        on_event: Callable[[str, str], None] | None = None,
        max_steps: int = 14,
        max_tool_calls: int = 16,
    ):
        self.run = Path(run)
        self.model = model or config.WONDER_MODEL
        self.effort = effort or None
        self.notes = notes.strip()
        self._client = client
        self.on_event = on_event or (lambda kind, text: None)
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self.plots: list[str] = []
        self.equations = 0
        self.usage = {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
        self.transcript: list[dict] = []
        self.report: dict | None = None
        self.px_per_cm: float | None = None  # set by the agent with set_scale
        self.scale_reason: str | None = None

    # ---------------------------------------------------------------- tools
    def _call_tool(self, name: str, args: dict) -> dict:
        scale = self.px_per_cm
        if name == "get_samples":
            table = samples_table(
                self.run, args.get("track"), args.get("start"), args.get("end"), int(args.get("max_rows") or 40), scale
            )
            return {"table": table}
        if name == "fit_model":
            result = fit_on_run(
                self.run,
                args["model"],
                args.get("track"),
                args.get("start"),
                args.get("end"),
                args.get("length_cm"),
                float(args.get("length_err_cm") or 0.3),
                args.get("angle_deg"),
                float(args.get("scale_err") or 0.01),
                px_per_cm=scale,
                mass_kg=args.get("mass_kg"),
                mass_err_kg=float(args.get("mass_err_kg") or 0.0),
            )
            self.plots.append(result["plot"])
            return clean(result)
        if name == "fit_equation":
            self.equations += 1
            plot = f"eq_{self.equations}_{slug(str(args.get('track') or 'track'))}.png"
            result = fit_equation_on_run(
                self.run, args.get("track"), args["series"], args["expression"],
                args.get("initial") or {}, args.get("start"), args.get("end"), plot_name=plot, px_per_cm=scale,
            )
            self.plots.append(plot)
            return clean(result)
        if name == "set_scale":
            value = float(args["px_per_cm"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("px_per_cm must be a positive number")
            self.px_per_cm = value
            self.scale_reason = str(args.get("reason") or "")
            overview = run_overview(self.run, value)
            return {"ok": True, "px_per_cm": value, "length_unit": "cm", "tracks_now": overview["tracks"][:3]}
        if name == "submit_report":
            self.report = args
            return {"ok": True}
        raise ValueError(f"unknown tool {name!r}")

    @staticmethod
    def _summary(name: str, result: dict) -> str:
        if "error" in result:
            return f"error: {result['error']}"
        if name == "fit_model":
            return result.get("title", "done")
        if name == "fit_equation":
            return f"rms {result['rms']:.4g}, BIC {result['bic']:.1f}, R² {result['r_squared']:.5f}"
        if name == "get_samples":
            return result["table"].splitlines()[0]
        if name == "set_scale":
            return f"scale set to {result['px_per_cm']:.4g} px/cm"
        return "report received"

    # ------------------------------------------------------------- messages
    def _first_message(self) -> dict:
        overview = run_overview(self.run)
        lines = [
            f"Experiment folder: {self.run.name}",
            f"Notes from the person who filmed it: {self.notes or 'none'}",
            "",
            "Overview (JSON):",
            json.dumps(overview, ensure_ascii=False),
            "",
            "Images, when available: the calibration frame with the measured reference object, a frame "
            "from the tracking overlay during the motion, and the tracked positions over time and the paths.",
        ]
        if (overview.get("scale") or {}).get("length_unit") == "px":
            lines.insert(3, "This run has NO scale: lengths are in pixels. Use the notes to calibrate (see 'Scale').")
        content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
        images = []
        if (self.run / "ref_check.png").exists():
            images.append(cv2.imread(str(self.run / "ref_check.png")))
        tracks = overview.get("tracks") or []
        fps = (overview.get("video") or {}).get("fps")
        if tracks and tracks[0].get("release_t") is not None:
            mid = (tracks[0]["release_t"] + tracks[0].get("moving_until_t", tracks[0]["release_t"])) / 2
            video = self.run / "overlay.mp4" if (self.run / "overlay.mp4").exists() else self.run / "clip.mp4"
            frame = _frame_from(video, mid, fps) if video.exists() else None
            if frame is not None:
                images.append(frame)
        if (self.run / "plot.png").exists():
            images.append(cv2.imread(str(self.run / "plot.png")))
        for image in images:
            if image is not None:
                content.append({"type": "image_url", "image_url": {"url": meta_api.image_to_data_url(image, ".jpg", max_side=1280)}})
        return {"role": "user", "content": content}

    def _client_or_new(self):
        if self._client is None:
            self._client = meta_api.get_client(timeout=600)
        return self._client

    def _complete(self, messages, force: bool):
        kwargs = {"model": self.model, "messages": messages, "tools": TOOLS}
        kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit_report"}} if force else "auto"
        if self.effort:
            kwargs["extra_body"] = {"reasoning_effort": self.effort}
        try:
            return self._client_or_new().chat.completions.create(**kwargs)
        except Exception as error:  # noqa: BLE001 - shown to the user
            if force:  # some servers reject a forced tool choice; ask in words instead
                kwargs["tool_choice"] = "auto"
                kwargs["messages"] = messages + [{"role": "user", "content": "Call submit_report now with your final result."}]
                return self._client_or_new().chat.completions.create(**kwargs)
            raise GalileoError(f"{self.model} request failed: {error}") from error

    def _count(self, usage) -> None:
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        out = getattr(usage, "completion_tokens", 0) or 0
        out_details = getattr(usage, "completion_tokens_details", None)
        self.usage["input"] += prompt - cached
        self.usage["cached"] += cached
        self.usage["output"] += out
        self.usage["reasoning"] += (getattr(out_details, "reasoning_tokens", 0) or 0) if out_details else 0

    def cost_usd(self) -> float | None:
        price = _price(self.model)
        if not price:
            return None
        return (self.usage["input"] * price[0] + self.usage["cached"] * price[1] + self.usage["output"] * price[2]) / 1e6

    # ------------------------------------------------------------------ run
    def run_agent(self) -> dict:
        started = time.time()
        self.on_event("status", f"Galileo ({self.model}) is studying {self.run.name}...")
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, self._first_message()]
        nudges = 0
        steps = 0
        for step in range(self.max_steps):
            steps = step + 1
            force = self.tool_calls >= self.max_tool_calls or step == self.max_steps - 1 or nudges >= 2
            response = self._complete(messages, force)
            self._count(getattr(response, "usage", None))
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            text = (message.content or "").strip()
            entry = {"role": "assistant", "content": message.content or None}
            if calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                    for c in calls
                ]
            messages.append(entry)
            if text:
                self.on_event("assistant", text)
                self.transcript.append({"step": steps, "kind": "assistant", "text": text})
            if not calls:
                nudges += 1
                messages.append({"role": "user", "content": "Please continue with the tools, and finish by calling submit_report."})
                continue
            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as error:
                    args, result = {}, {"error": f"arguments were not valid JSON: {error}"}
                else:
                    shown = ", ".join(f"{k}={v}" for k, v in args.items() if k not in ("results", "comparisons", "explanation", "data_quality", "next_steps"))
                    self.on_event("tool", f"{name}({shown[:220]})")
                    self.tool_calls += 1
                    try:
                        result = self._call_tool(name, args)
                    except Exception as error:  # noqa: BLE001 - the agent should see and handle it
                        result = {"error": str(error)}
                summary = self._summary(name, result)
                self.on_event("result", summary)
                self.transcript.append({"step": steps, "kind": "tool", "name": name, "args": args, "summary": summary})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)[:12000]})
            if self.report is not None:
                break
        if self.report is None:
            raise GalileoError("Galileo stopped without a report; try again or switch the model")
        meta = {
            "model": self.model,
            "effort": self.effort,
            "steps": steps,
            "tool_calls": self.tool_calls,
            "seconds": round(time.time() - started, 1),
            "usage_tokens": self.usage,
            "cost_usd_estimate": None if self.cost_usd() is None else round(self.cost_usd(), 4),
            "notes": self.notes,
            "plots": self.plots,
            "scale_set_by_galileo": {"px_per_cm": self.px_per_cm, "reason": self.scale_reason} if self.px_per_cm else None,
            "finished_at": time.strftime("%Y-%m-%d %H:%M"),
            "transcript": self.transcript,
        }
        best = self.report.get("best_plot")
        if not best or not (self.run / Path(str(best)).name).exists():
            self.report["best_plot"] = self.plots[0] if self.plots else None
        else:
            self.report["best_plot"] = Path(str(best)).name
        (self.run / "report.json").write_text(json.dumps({"report": self.report, "meta": meta}, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.run / "report.md").write_text(render_report(self.report, meta), encoding="utf-8")
        (self.run / "agent_transcript.md").write_text(render_transcript(self.transcript, meta), encoding="utf-8")
        cost = meta["cost_usd_estimate"]
        self.on_event(
            "status",
            f"done in {meta['seconds']:.0f} s, {steps} model calls, {self.tool_calls} tool calls"
            + (f", about ${cost:.2f} of API usage" if cost is not None else ""),
        )
        self.on_event("report", self.report.get("headline", ""))
        return {"report": self.report, "meta": meta}


def _cell(text) -> str:
    """Safe text for a Markdown table cell."""
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def _code(text) -> str:
    """Inline code, so * and _ in equations are not read as Markdown."""
    text = str(text or "").replace("`", "'")
    return f"`{text}`" if text else ""


def render_report(report: dict, meta: dict) -> str:
    lines = [f"# {report.get('experiment', 'Experiment')}", "", f"**{report.get('headline', '')}**", ""]
    if report.get("law"):
        lines += [f"**Law:** {_code(report['law'])}", ""]
    results = report.get("results") or []
    if results:
        lines += ["| Quantity | Value | Unit | Note |", "|---|---|---|---|"]
        for r in results:
            lines.append(
                f"| {_cell(r.get('quantity'))} | {format_value(r.get('value'), r.get('uncertainty'))} | {_cell(r.get('unit'))} | {_cell(r.get('note'))} |"
            )
        lines.append("")
    if report.get("explanation"):
        lines += ["### What this means", "", report["explanation"], ""]
    comparisons = report.get("comparisons") or []
    if comparisons:
        lines += ["### Models compared", "", "| Model | BIC | rms | Verdict |", "|---|---|---|---|"]
        for c in comparisons:
            bic = "" if c.get("bic") is None else f"{float(c['bic']):.1f}"
            rms = "" if c.get("rms") is None else f"{float(c['rms']):.3g}"
            model = str(c.get("model", ""))
            model = _cell(_code(model)) if any(ch in model for ch in "*_^\\") else _cell(model)
            lines.append(f"| {model} | {bic} | {rms} | {_cell(c.get('verdict'))} |")
        lines.append("")
    if report.get("data_quality"):
        lines += ["### Data quality", ""] + [f"- {item}" for item in report["data_quality"]] + [""]
    if report.get("next_steps"):
        lines += ["### Next steps", ""] + [f"- {item}" for item in report["next_steps"]] + [""]
    if report.get("best_plot"):
        lines += [f"![fit]({report['best_plot']})", ""]
    calibrated = meta.get("scale_set_by_galileo")
    if calibrated:
        lines += [f"_Scale set by Galileo: {calibrated['px_per_cm']:.4g} px/cm ({calibrated.get('reason') or 'no reason given'})._", ""]
    cost = meta.get("cost_usd_estimate")
    lines.append(
        f"_Analysed by {meta['model']} in {meta['steps']} model calls and {meta['tool_calls']} tool calls"
        + (f", about ${cost:.2f} of API usage" if cost is not None else "")
        + f" ({meta['seconds']:.0f} s)._"
    )
    return "\n".join(lines) + "\n"


def render_transcript(entries: list[dict], meta: dict) -> str:
    lines = [f"# How Galileo worked it out ({meta['model']})", ""]
    if meta.get("notes"):
        lines += [f"Notes given: {meta['notes']}", ""]
    for e in entries:
        if e["kind"] == "assistant":
            lines += [f"**Step {e['step']}.** {e['text']}", ""]
        else:
            args = {k: v for k, v in e["args"].items() if k not in ("results", "comparisons", "explanation", "data_quality", "next_steps")}
            lines += [f"- {_code(e['name'])} {_code(json.dumps(args, ensure_ascii=False))}", f"  - {_code(e['summary']) if e['summary'].startswith('error') else e['summary']}", ""]
    return "\n".join(lines) + "\n"