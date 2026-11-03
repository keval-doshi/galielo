#!/usr/bin/env python3
"""Galileo app: upload a video, SAM 3.1 tracks it, Muse Wonder works out the physics.

    python app.py            # opens http://127.0.0.1:7860 in your browser
    python app.py --share    # also makes a temporary public link (anyone with it spends your API credits)
"""

from __future__ import annotations

import argparse
import html
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import gradio as gr

from oe import config
from oe.agent import Galileo
from oe.analysis import slug
from oe.report import comparison, ensure_pdf, key_result, number, other_results, scale_geometry, value_parts

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OTHER_OBJECT = "Other object of known size in the video"
NO_REFERENCE = "No reference: Galileo uses the lengths in my notes"
KNOWN_SCALE = "I know the scale (pixels per cm)"
SCALE_OPTIONS = {  # what SAM looks for, long side cm, short side cm (0 = unknown)
    "A4 sheet in the video (29.7 × 21 cm)": ("sheet of paper", 29.7, 21.0),
    "US Letter sheet in the video (27.94 × 21.59 cm)": ("sheet of paper", 27.94, 21.59),
    "Metre stick, fully in view (100 cm)": ("meter stick", 100.0, 0.0),
    "30 cm ruler, fully in view": ("ruler", 30.0, 0.0),
    OTHER_OBJECT: ("", None, 0.0),
    NO_REFERENCE: None,
    KNOWN_SCALE: None,
}
MODELS = [config.WONDER_MODEL, config.SPARK_MODEL]
EFFORTS = ["default", "minimal", "low", "medium", "high", "xhigh", "max"]
LOG_LINES = 300

# ------------------------------------------------------------------------------------ look

PAPER, GRID, INK, MUTED, RULE, PEN, PEN_DARK, PEN_SOFT, MARK, RED = (
    "#FBFCFE", "#E9EEF6", "#182A4A", "#5A6A85", "#D5DEEA", "#2447A8", "#1B3785", "#E8EDF9", "#F7DF4B", "#B42318",
)


def theme() -> gr.themes.Base:
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Instrument Sans", weights=(400, 500, 600)), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.LocalFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
    )
    values = dict(
        body_background_fill=PAPER, body_text_color=INK, body_text_color_subdued=MUTED,
        background_fill_primary="#FFFFFF", background_fill_secondary="#F5F7FB",
        border_color_primary=RULE, border_color_accent=PEN, border_color_accent_subdued=PEN_SOFT,
        color_accent=PEN, color_accent_soft=PEN_SOFT, link_text_color=PEN, link_text_color_hover=PEN_DARK,
        link_text_color_visited=PEN, link_text_color_active=PEN_DARK, code_background_fill="#F5F7FB",
        shadow_drop="none", shadow_drop_lg="none", shadow_spread="0px", block_shadow="none",
        block_background_fill="#FFFFFF", block_border_color=RULE, block_border_width="1px", block_radius="8px",
        block_label_background_fill="#FFFFFF", block_label_border_color=RULE, block_label_text_color=MUTED,
        block_title_text_color=INK, block_info_text_color=MUTED, panel_background_fill="#FFFFFF",
        panel_border_color=RULE, accordion_text_color=INK,
        input_background_fill="#FFFFFF", input_background_fill_focus="#FFFFFF", input_background_fill_hover="#FFFFFF",
        input_border_color=RULE, input_border_color_hover="#B9C5D8", input_border_color_focus=PEN,
        input_shadow="none", input_shadow_focus=f"0 0 0 3px {PEN_SOFT}", input_placeholder_color="#8794AA",
        button_primary_background_fill=PEN, button_primary_background_fill_hover=PEN_DARK,
        button_primary_border_color=PEN, button_primary_border_color_hover=PEN_DARK,
        button_primary_text_color="#FFFFFF", button_primary_text_color_hover="#FFFFFF",
        button_primary_shadow="none", button_primary_shadow_hover="none", button_primary_shadow_active="none",
        button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_hover="#F2F5FB",
        button_secondary_border_color=RULE, button_secondary_border_color_hover="#B9C5D8",
        button_secondary_text_color=INK, button_secondary_text_color_hover=INK,
        button_secondary_shadow="none", button_secondary_shadow_hover="none", button_secondary_shadow_active="none",
        button_transform_hover="none", button_transform_active="none", button_border_width="1px",
        loader_color=PEN, slider_color=PEN, checkbox_background_color_selected=PEN, checkbox_border_color_selected=PEN,
        error_text_color=RED, error_border_color=RED, error_background_fill="#FEF3F2",
        table_border_color=RULE, table_even_background_fill="#FFFFFF", table_odd_background_fill="#F8FAFD",
    )
    # The page is always light: the dark-mode values are the same as the light ones.
    known = set(inspect.signature(base.set).parameters)
    return base.set(**values, **{f"{key}_dark": value for key, value in values.items() if f"{key}_dark" in known})


HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap">'
    '<meta name="color-scheme" content="light">'
)

# Keeps the page light even when the computer is in dark mode.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

# Ticks the elapsed time of the running step every half second, without asking the server.
TIMER_JS = """
const tick = () => {
  element.querySelectorAll('[data-elapsed]').forEach((el) => {
    if (el.dataset.seenStamp !== el.dataset.stamp) {
      el.dataset.seenStamp = el.dataset.stamp;
      el.dataset.seen = String(Date.now());
    }
    const s = Number(el.dataset.elapsed) + (Date.now() - Number(el.dataset.seen)) / 1000;
    el.textContent = Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
  });
};
setInterval(tick, 500);
tick();
"""

SERIF = "'Instrument Serif', 'Iowan Old Style', Georgia, serif"
CSS = f"""
:root {{
  --gl-paper: {PAPER}; --gl-grid: {GRID}; --gl-ink: {INK}; --gl-muted: {MUTED}; --gl-rule: {RULE};
  --gl-pen: {PEN}; --gl-pen-soft: {PEN_SOFT}; --gl-mark: {MARK}; --gl-red: {RED};
}}
body {{
  background-color: var(--gl-paper) !important;
  background-image: linear-gradient(var(--gl-grid) 1px, transparent 1px),
                    linear-gradient(90deg, var(--gl-grid) 1px, transparent 1px) !important;
  background-size: 24px 24px !important;
}}
gradio-app, .gradio-container, .gradio-container .main, .gradio-container .wrap.app {{ background: transparent !important; }}
.gradio-container {{ width: 100% !important; max-width: 1360px !important; margin: 0 auto !important; padding-top: 10px !important; }}
footer {{ display: none !important; }}
.gl-html > .html-container {{ padding: 0 !important; }}

.gl-head {{ display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 22px; padding: 0 2px 10px;
  border-bottom: 1px solid var(--gl-pen); margin-bottom: 4px; }}
.gl-word {{ font-family: {SERIF}; font-style: italic; font-weight: 400; font-size: 2.75rem; line-height: 1;
  color: var(--gl-pen); margin: 0; letter-spacing: -0.01em; }}
.gl-tag {{ color: var(--gl-muted); margin: 0; font-size: 1.02rem; }}
.gl-tag b {{ color: var(--gl-ink); font-weight: 600; }}

#setup {{ gap: 10px; }}
.gl-step {{ display: flex; gap: 11px; align-items: baseline; margin: 8px 0 -2px; }}
.gl-n {{ flex: none; align-self: center; width: 24px; height: 24px; border-radius: 50%; border: 1.5px solid var(--gl-pen);
  color: var(--gl-pen); display: grid; place-items: center; font-size: 0.8rem; font-weight: 600; line-height: 1; background: #fff; }}
.gl-t {{ font-weight: 600; color: var(--gl-ink); margin: 0; font-size: 1rem; line-height: 1.3; }}
.gl-h {{ color: var(--gl-muted); font-weight: 400; font-size: 0.86rem; margin-left: 8px; }}
.gl-or {{ display: flex; align-items: center; gap: 12px; color: var(--gl-muted); font-size: 0.88rem; margin: 14px 0 0; }}
.gl-or::before, .gl-or::after {{ content: ""; flex: 1; border-top: 1px solid var(--gl-rule); }}
#run {{ font-size: 1.06rem; font-weight: 600; padding-top: 12px; padding-bottom: 12px; margin-top: 4px; }}
#setup .gl-hint {{ color: var(--gl-muted); font-size: 0.84rem; margin: -2px 0 0; }}

.gl-status {{ background: #fff; border: 1px solid var(--gl-rule); border-radius: 8px; padding: 14px 18px 12px; }}
.gl-steps {{ display: flex; align-items: center; list-style: none; margin: 0; padding: 0; }}
.gl-steps li {{ display: grid; grid-template-columns: auto auto; column-gap: 10px; align-items: center; margin: 0; padding: 0; }}
.gl-steps li::marker {{ content: none; }}
.gl-steps li.gl-link {{ display: block; flex: 1; height: 1.5px; min-width: 18px; margin: 0 14px; background: var(--gl-rule); }}
.gl-steps li.gl-link.done {{ background: var(--gl-pen); }}
.gl-dot {{ grid-row: span 2; width: 30px; height: 30px; border-radius: 50%; border: 1.5px solid var(--gl-rule);
  color: var(--gl-muted); display: grid; place-items: center; font-size: 0.9rem; font-weight: 600; background: #fff; }}
li.active .gl-dot {{ background: var(--gl-pen); border-color: var(--gl-pen); color: #fff; }}
li.done .gl-dot, li.skipped .gl-dot {{ border-color: var(--gl-pen); color: var(--gl-pen); }}
li.failed .gl-dot {{ border-color: var(--gl-red); color: var(--gl-red); }}
.gl-name {{ font-weight: 600; color: var(--gl-ink); line-height: 1.2; }}
li.pending .gl-name {{ color: var(--gl-muted); }}
.gl-sub {{ color: var(--gl-muted); font-size: 0.84rem; line-height: 1.2; font-variant-numeric: tabular-nums; white-space: nowrap; }}
li.failed .gl-sub {{ color: var(--gl-red); }}
.gl-now {{ margin: 11px 0 0; padding-top: 9px; border-top: 1px dashed var(--gl-rule); color: var(--gl-ink); font-size: 0.9rem;
  overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow-wrap: anywhere; }}
.gl-now.error {{ color: var(--gl-red); display: block; }}

.gl-stage {{ background: #fff; border: 1px solid var(--gl-rule); border-radius: 8px; padding: 22px 26px 20px; }}
.gl-stage h2 {{ font-family: {SERIF}; font-weight: 400; font-size: 2rem; line-height: 1.1; margin: 0 0 6px; color: var(--gl-ink); }}
.gl-stage p {{ margin: 0; color: var(--gl-muted); max-width: 64ch; }}
.gl-stage.gl-row {{ display: grid; grid-template-columns: 112px 1fr; gap: 22px; align-items: center; }}
.gl-stage.gl-error h2 {{ color: var(--gl-red); }}
.gl-stage.gl-caption {{ padding: 14px 20px; }}
.gl-stage.gl-caption p {{ color: var(--gl-ink); }}
.gl-pend {{ width: 112px; height: 122px; display: block; }}
.gl-pend .beam {{ stroke: var(--gl-ink); stroke-width: 2.4; stroke-linecap: round; }}
.gl-pend .arc {{ fill: none; stroke: var(--gl-pen); stroke-width: 1.2; stroke-dasharray: 2 4; opacity: .7; }}
.gl-pend .arm line {{ stroke: var(--gl-ink); stroke-width: 1.6; }}
.gl-pend .arm circle {{ fill: var(--gl-mark); stroke: var(--gl-ink); stroke-width: 1.6; }}
.gl-pend .pivot {{ fill: var(--gl-ink); }}
.gl-pend .arm {{ transform-box: view-box; transform-origin: 56px 12px; transform: rotate(17deg); }}
.gl-working .gl-pend .arm {{ animation: gl-swing 1.1s ease-in-out infinite alternate; }}
@keyframes gl-swing {{ from {{ transform: rotate(17deg); }} to {{ transform: rotate(-17deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .gl-working .gl-pend .arm {{ animation: none; }} }}

.gl-exp {{ font-family: {SERIF}; font-size: 1.55rem !important; line-height: 1.2; color: var(--gl-ink) !important; margin: 0 0 12px !important; }}
.gl-q {{ font-size: 0.95rem; }}
.gl-value {{ font-family: {SERIF}; font-size: clamp(2.9rem, 5.6vw, 4.5rem); line-height: 1.05; color: var(--gl-ink) !important;
  margin: 0 0 4px !important; max-width: none !important; font-variant-numeric: lining-nums; }}
.gl-value mark {{ color: inherit; padding: 0 0.1em; margin-left: -0.1em;
  background: linear-gradient(transparent 40%, var(--gl-mark) 40%, var(--gl-mark) 90%, transparent 90%); }}
.gl-value .gl-unit {{ font-size: 0.45em; margin-left: 0.25em; }}
.gl-ruler {{ width: 100%; height: auto; display: block; margin: 6px 0 4px; overflow: visible; }}
.gl-ruler .axis {{ stroke: var(--gl-ink); stroke-width: 1.3; }}
.gl-ruler .tick {{ stroke: var(--gl-ink); stroke-width: 1.1; }}
.gl-ruler .minor {{ stroke: var(--gl-muted); stroke-width: 0.8; }}
.gl-ruler .tl {{ fill: var(--gl-muted); font-size: 12px; text-anchor: middle; font-variant-numeric: tabular-nums; }}
.gl-ruler .ref {{ stroke: var(--gl-ink); stroke-width: 1.3; stroke-dasharray: 4 3; }}
.gl-ruler .rl {{ fill: var(--gl-ink); font-size: 12.5px; font-style: italic; }}
.gl-ruler .bar {{ fill: var(--gl-pen); fill-opacity: 0.24; }}
.gl-ruler .whisk {{ stroke: var(--gl-pen); stroke-width: 1.5; }}
.gl-ruler .pt {{ fill: var(--gl-pen); }}
.gl-ruler .key {{ fill: var(--gl-muted); font-size: 11.5px; text-anchor: end; }}
.gl-compare {{ color: var(--gl-ink) !important; font-size: 0.98rem; }}
.gl-more {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px 22px; margin: 18px 0 0;
  padding: 14px 0 0; border-top: 1px solid var(--gl-rule); }}
.gl-more dt {{ color: var(--gl-muted); font-size: 0.84rem; }}
.gl-more dd {{ margin: 2px 0 0; font-size: 1.15rem; font-weight: 500; line-height: 1.25; color: var(--gl-ink);
  font-variant-numeric: tabular-nums; }}
.gl-headline {{ margin-top: 16px !important; color: var(--gl-ink) !important; }}
.gl-meta {{ margin-top: 8px !important; font-size: 0.84rem; }}

#stage-video {{ border-radius: 8px; }}
#pdf {{ font-weight: 600; }}
#log textarea {{ font-family: var(--font-mono) !important; font-size: 12.5px !important; line-height: 1.55 !important;
  color: #24324F !important; background: #FCFDFE !important; }}
#tabs button[role="tab"] {{ font-weight: 500; color: var(--gl-muted); }}
#tabs .prose table, #tabs .prose th, #tabs .prose td {{ border-color: var(--gl-rule) !important; }}
#tabs .prose th {{ background: #F5F7FB; color: var(--gl-ink); }}
#tabs .prose code {{ background: #F5F7FB; border-color: var(--gl-rule); }}
#tabs .prose td, #tabs .prose th {{ word-break: normal !important; overflow-wrap: normal !important; vertical-align: top; }}
#tabs .prose th, #tabs .prose td:nth-child(2), #tabs .prose td:nth-child(3) {{ white-space: nowrap; }}
#tabs .prose td:first-child {{ min-width: 9em; }}
#bench, #setup {{ min-width: 0; }}
#tabs button[role="tab"][aria-selected="true"] {{ color: var(--gl-pen); font-weight: 600; }}
.gl-small {{ color: var(--gl-muted); font-size: 0.86rem; margin: 2px 0 0; }}
@media (max-width: 720px) {{
  .gl-word {{ font-size: 2.5rem; }}
  .gl-steps li:not(.gl-link) {{ grid-template-columns: auto; }}
  .gl-sub {{ display: none; }}
  .gl-stage.gl-row {{ grid-template-columns: 1fr; }}
}}
"""

HEADER = (
    '<header class="gl-head"><h1 class="gl-word">Galileo</h1>'
    "<p class=\"gl-tag\">Film something moving. <b>SAM 3.1</b> tracks it, <b>Muse Wonder</b> finds the law behind the "
    "motion and measures it, with an honest error bar.</p></header>"
)

PENDULUM = (
    '<svg class="gl-pend" viewBox="0 0 112 122" aria-hidden="true">'
    '<path class="arc" d="M 26.4 104.4 A 96 96 0 0 0 85.6 104.4"/>'
    '<line class="beam" x1="20" y1="12" x2="92" y2="12"/>'
    '<g class="arm"><line x1="56" y1="12" x2="56" y2="99"/><circle cx="56" cy="107" r="8.5"/></g>'
    '<circle class="pivot" cx="56" cy="12" r="3.2"/></svg>'
)


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def step_head(n: int, title: str, hint: str) -> str:
    return f'<div class="gl-step"><span class="gl-n">{n}</span><p class="gl-t">{esc(title)}<span class="gl-h">{esc(hint)}</span></p></div>'


def model_label(model: str | None) -> str:
    model = str(model or "")
    return "Muse Spark" if model.startswith("muse-spark") else "Muse Wonder" if model.startswith("muse-wonder") else (model or "Muse Wonder")


def _clock(seconds: float | None) -> str:
    seconds = max(0.0, float(seconds or 0))
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def status_html(track=("pending",), analyse=("pending",), report=("pending",), now: str = "", error: bool = False, model: str | None = None) -> str:
    """The three-step progress bar. Each state is (kind,) or (kind, seconds) with kind pending/active/done/skipped/failed."""
    steps = (("Track", "SAM 3.1", track), ("Analyse", model_label(model), analyse), ("Report", "PDF", report))
    items = []
    for index, (name, who, state) in enumerate(steps):
        kind = state[0]
        seconds = state[1] if len(state) > 1 else None
        if kind == "active":
            sub = f'{esc(who)}, <span data-elapsed="{float(seconds or 0):.2f}" data-stamp="{time.time():.3f}">{_clock(seconds)}</span>'
        elif kind == "done":
            sub = f"{esc(who)}, {_clock(seconds)}" if seconds is not None else f"{esc(who)}, done"
        elif kind == "skipped":
            sub = "reused, no new cost"
        elif kind == "failed":
            sub = "stopped"
        else:
            sub = esc(who)
        dot = {"done": "✓", "skipped": "✓", "failed": "!"}.get(kind, str(index + 1))
        if index:
            items.append(f'<li class="gl-link {"done" if steps[index - 1][2][0] in ("done", "skipped") else ""}" aria-hidden="true"></li>')
        items.append(f'<li class="{kind}"><span class="gl-dot">{dot}</span><span class="gl-name">{name}</span><span class="gl-sub">{sub}</span></li>')
    now_html = f'<p class="gl-now{" error" if error else ""}">{esc(now)}</p>' if now else ""
    return f'<div class="gl-status"><ol class="gl-steps">{"".join(items)}</ol>{now_html}</div>'


def idle_html() -> str:
    return (
        f'<div class="gl-stage gl-row">{PENDULUM}<div><h2>Your measurement lands here</h2>'
        "<p>Upload a clip, say what moves and what you know, then press <b>Track and analyse</b>. SAM 3.1 follows the "
        "object through every frame; Muse Wonder finds the law and the number, with its error bar.</p></div></div>"
    )


def working_html(title: str, text: str) -> str:
    return f'<div class="gl-stage gl-row gl-working">{PENDULUM}<div><h2>{esc(title)}</h2><p>{esc(text)}</p></div></div>'


def caption_html(text: str) -> str:
    return f'<div class="gl-stage gl-caption"><p>{esc(text)}</p></div>'


def problem_html(title: str, text: str) -> str:
    return f'<div class="gl-stage gl-error"><h2>{esc(title)}</h2><p>{esc(text)}</p></div>'


def ruler_svg(result: dict, comp: dict) -> str:
    """The measured value with its error bar on a ruler, next to the reference value."""
    value = number(result.get("value"))
    sigma = comp.get("sigma") or 0.0
    reference = comp["reference"]
    geometry = scale_geometry(value, sigma, reference)
    lo, hi = geometry["lo"], geometry["hi"]
    width, x0, x1, axis = 640, 10, 630, 58

    def at(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    parts = [f'<svg class="gl-ruler" viewBox="0 0 {width} 84" role="img" aria-label="{esc(comp["sentence"])}">']
    ticks = geometry["ticks"]
    if len(ticks) > 1:
        minor = (ticks[1] - ticks[0]) / 5
        k = int((lo - ticks[0]) // minor)
        while ticks[0] + k * minor <= hi:
            v = ticks[0] + k * minor
            if v >= lo:
                parts.append(f'<line class="minor" x1="{at(v):.1f}" y1="{axis}" x2="{at(v):.1f}" y2="{axis + 4}"/>')
            k += 1
    parts.append(f'<line class="axis" x1="{x0}" y1="{axis}" x2="{x1}" y2="{axis}"/>')
    for tick, label in zip(ticks, geometry["labels"]):
        parts.append(f'<line class="tick" x1="{at(tick):.1f}" y1="{axis}" x2="{at(tick):.1f}" y2="{axis + 8}"/>'
                     f'<text class="tl" x="{at(tick):.1f}" y="{axis + 22}">{esc(label)}</text>')
    ref_x = at(reference)
    label = f"{comp['label']}, {reference:g}"
    anchor = "middle" if 120 < ref_x < width - 120 else ("start" if ref_x <= 120 else "end")
    parts.append(f'<line class="ref" x1="{ref_x:.1f}" y1="{axis}" x2="{ref_x:.1f}" y2="17"/>'
                 f'<text class="rl" x="{ref_x:.1f}" y="11" text-anchor="{anchor}">{esc(label)}</text>')
    mark = axis - 19
    if sigma:
        a, b = at(value - 2 * sigma), at(value + 2 * sigma)
        parts.append(f'<line class="whisk" x1="{a:.1f}" y1="{mark}" x2="{b:.1f}" y2="{mark}"/>'
                     f'<line class="whisk" x1="{a:.1f}" y1="{mark - 5}" x2="{a:.1f}" y2="{mark + 5}"/>'
                     f'<line class="whisk" x1="{b:.1f}" y1="{mark - 5}" x2="{b:.1f}" y2="{mark + 5}"/>'
                     f'<rect class="bar" x="{at(value - sigma):.1f}" y="{mark - 6}" width="{at(value + sigma) - at(value - sigma):.1f}" height="12" rx="2"/>')
    parts.append(f'<circle class="pt" cx="{at(value):.1f}" cy="{mark}" r="5"/>')
    parts.append("</svg>")
    return "".join(parts)


def result_html(run: Path) -> str:
    """The big readout of a finished run."""
    data = json.loads((run / "report.json").read_text(encoding="utf-8"))
    report, meta = data.get("report") or {}, data.get("meta") or {}
    key = key_result(report)
    parts = ['<div class="gl-stage gl-result">', f'<p class="gl-exp">{esc(report.get("experiment") or "Experiment")}</p>']
    if key:
        value, error, unit = value_parts(key)
        comp = comparison(key)
        shown = esc(value) + (f" ± {esc(error)}" if error else "")
        parts.append(f'<p class="gl-q">{esc(key.get("quantity"))}</p>'
                     f'<p class="gl-value"><mark>{shown}</mark><span class="gl-unit">{esc(unit)}</span></p>')
        if comp:
            parts.append(ruler_svg(key, comp))
            sentence = comp["sentence"]
            parts.append(f'<p class="gl-compare">{esc(sentence[0].upper() + sentence[1:])}'
                         + (' <span class="gl-small">The blue bar is ±1σ, the whiskers ±2σ.</span>' if comp.get("sigma") else "") + "</p>")
        more = other_results(report, key)
        if more:
            cells = []
            for r in more:
                v, e, u = value_parts(r)
                cells.append(f"<div><dt>{esc(r.get('quantity'))}</dt><dd>{esc(v)}{(' ± ' + esc(e)) if e else ''} {esc(u)}</dd></div>")
            parts.append(f'<dl class="gl-more">{"".join(cells)}</dl>')
    if report.get("headline"):
        parts.append(f'<p class="gl-headline">{esc(report["headline"])}</p>')
    cost = number(meta.get("cost_usd_estimate"))
    if meta.get("model"):
        parts.append(
            f'<p class="gl-meta">{esc(model_label(meta.get("model")))} ({esc(meta.get("model"))}): {meta.get("steps", "?")} model calls, '
            f'{meta.get("tool_calls", "?")} tool calls, {float(meta.get("seconds") or 0):.0f} s'
            + (f", about ${cost:.2f} of API usage" if cost is not None else "") + "</p>"
        )
    parts.append("</div>")
    return "".join(parts)


# ------------------------------------------------------------------------------- run files


def list_runs() -> list[str]:
    if not RUNS.exists():
        return []
    runs = [p for p in RUNS.iterdir() if p.is_dir() and (p / "tracks.csv").exists()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]


def _path(run: Path, name: str | None):
    if not name:
        return None
    p = run / Path(name).name
    return str(p) if p.exists() else None


def _report_parts(run: Path):
    """(report markdown without the image line, best plot, transcript markdown)."""
    data = json.loads((run / "report.json").read_text(encoding="utf-8")) if (run / "report.json").exists() else None
    if not data:
        return "", None, ""
    text = (run / "report.md").read_text(encoding="utf-8") if (run / "report.md").exists() else ""
    text = "\n".join(line for line in text.splitlines() if not line.startswith("!["))
    transcript = (run / "agent_transcript.md").read_text(encoding="utf-8") if (run / "agent_transcript.md").exists() else ""
    return text, _path(run, data["report"].get("best_plot")), transcript


def _failure_reason(lines: list[str]) -> str:
    """The one line that says why track.py stopped."""
    for line in reversed(lines):
        text = line.strip()
        if text.lower().startswith("error:"):
            return text[6:].strip()
    for line in reversed(lines):  # an unexpected crash: the last line of the traceback says what it was
        if line.strip() and not line.startswith("Tracking with SAM"):
            return line.strip()
    return "track.py stopped without saying why"


def _tracking_parts(run: Path):
    return _path(run, "overlay.mp4"), _path(run, "plot.png"), _path(run, "ref_check.png")


def _tracked_sentence(run: Path) -> str:
    try:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "SAM 3.1 tracked the clip."
    tracks = summary.get("tracks") or {}
    if not tracks:
        return "SAM 3.1 tracked the clip."
    name, best = max(tracks.items(), key=lambda item: item[1].get("frames", 0))
    label = name.rsplit("#", 1)[0] if len(tracks) == 1 else name
    extra = f" and {len(tracks) - 1} more object{'s' if len(tracks) > 2 else ''}" if len(tracks) > 1 else ""
    return f"SAM 3.1 tracked “{label}”{extra} in {best.get('frames', '?')} of {summary.get('frames', '?')} frames."


def _pdf_for(run: Path):
    try:
        return ensure_pdf(run)
    except Exception as error:  # noqa: BLE001 - the report is still shown on the page
        gr.Warning(f"The PDF could not be made: {error}")
        return None


def _now_text(kind: str, text: str) -> str:
    text = " ".join(str(text).split())
    if kind == "assistant":
        return "Galileo: " + text
    if kind == "tool":
        return "Calling " + text
    if kind == "result":
        return "Result: " + text
    return text


# --------------------------------------------------------------------------------------- page


def build() -> gr.Blocks:
    with gr.Blocks(title="Galileo") as demo:
        gr.HTML(HEADER, elem_classes="gl-html")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=340, elem_id="setup"):
                gr.HTML(step_head(1, "Your video", "Camera still, motion side-on"), elem_classes="gl-html")
                video = gr.Video(sources=["upload"], show_label=False, height=180, elem_id="video-in")
                gr.HTML(step_head(2, "What moves", "a short phrase SAM 3.1 can find"), elem_classes="gl-html")
                prompt = gr.Textbox(
                    show_label=False, placeholder="e.g. wooden block, orange ball (empty: Muse Spark picks)", elem_id="prompt",
                )
                gr.HTML(step_head(3, "What you know", "lengths, heights or masses you measured"), elem_classes="gl-html")
                notes = gr.Textbox(
                    show_label=False, lines=2, max_lines=6, elem_id="notes",
                    placeholder="e.g. pendulum 33 cm from pivot to centre of mass; dropped from 1.6 m; ramp 30 cm long, "
                    "raised 8 cm; 200 g mass on a spring",
                )
                gr.HTML(step_head(4, "Where the scale comes from", "for lengths and speeds"), elem_classes="gl-html")
                scale_mode = gr.Dropdown(list(SCALE_OPTIONS), value=list(SCALE_OPTIONS)[0], show_label=False, elem_id="scale-mode")
                with gr.Row(visible=True, equal_height=True) as reference_box:
                    ref_prompt = gr.Textbox(value="sheet of paper", label="SAM looks for", scale=3, min_width=150)
                    ref_long = gr.Number(value=29.7, label="Long side (cm)", minimum=0, scale=2, min_width=100)
                    ref_short = gr.Number(value=21.0, label="Short side (cm)", minimum=0, scale=2, min_width=100)
                px_scale = gr.Number(value=None, label="Pixels per cm, measured on your original video", minimum=0, visible=False)
                with gr.Accordion("More settings: trim, reference time, model", open=False):
                    with gr.Row():
                        trim_start = gr.Number(label="Trim start (s), 0 = from the start", value=None, minimum=0)
                        trim_end = gr.Number(label="Trim end (s), 0 = to the end", value=None, minimum=0)
                    ref_time = gr.Number(value=0, label="Reference object fully visible at (s into the trimmed clip)", minimum=0)
                    with gr.Row():
                        model = gr.Dropdown(MODELS, value=MODELS[0], label="Galileo's model", allow_custom_value=True)
                        effort = gr.Dropdown(EFFORTS, value="default", label="Reasoning effort")
                track_button = gr.Button("Track and analyse", variant="primary", size="lg", elem_id="run")
                gr.HTML('<p class="gl-or">or open a run you already tracked</p>', elem_classes="gl-html")
                with gr.Row(equal_height=True):
                    previous = gr.Dropdown(list_runs(), value=None, show_label=False, scale=4, elem_id="previous")
                    refresh = gr.Button("Refresh list", scale=1, min_width=110)
                analyse_button = gr.Button("Analyse with Galileo", elem_id="again")
                gr.HTML('<p class="gl-hint">Opening a run is free. Analysing it again costs one Galileo run and no SAM call.</p>',
                        elem_classes="gl-html")
            with gr.Column(scale=7, min_width=440, elem_id="bench"):
                status = gr.HTML(status_html(), elem_id="status", elem_classes="gl-html", js_on_load=TIMER_JS)
                stage = gr.HTML(idle_html(), elem_id="stage", elem_classes="gl-html")
                stage_video = gr.Video(visible=False, show_label=False, interactive=False, autoplay=True, loop=True, elem_id="stage-video")
                pdf = gr.DownloadButton("Download the PDF report", visible=False, variant="primary", elem_id="pdf")
                log = gr.Textbox(label="Lab log", lines=9, max_lines=9, autoscroll=True, interactive=False, elem_id="log", visible=False)
                with gr.Tabs(elem_id="tabs", visible=False) as tabs:
                    with gr.Tab("Report", id="report", render_children=True):
                        report = gr.Markdown()
                        best_plot = gr.Image(label="Best fit", type="filepath", interactive=False)
                    with gr.Tab("Tracking", id="tracking", render_children=True):
                        overlay = gr.Video(label="SAM 3.1 tracking", interactive=False, loop=True)
                    with gr.Tab("Paths", id="paths", render_children=True):
                        paths = gr.Image(label="Tracked positions", type="filepath", interactive=False)
                    with gr.Tab("Scale check", id="scalecheck", render_children=True):
                        ref_image = gr.Image(label="What was measured for the cm scale", type="filepath", interactive=False)
                    with gr.Tab("How Galileo worked it out", id="how", render_children=True):
                        transcript = gr.Markdown()

        current_run = gr.State(None)
        clock = gr.State({})
        outputs = [status, stage, stage_video, pdf, log, report, best_plot, overlay, paths, ref_image, transcript,
                   previous, tabs, current_run, clock]
        # Gradio 6 does not show values that a *streaming* function sends to a hidden component, so the
        # streaming steps only update the progress bar, the stage text and the log, and plain functions
        # chained after them fill the videos and tabs. A step that fails sets current_run to None and the
        # later steps skip.

        def clear_results(model_name):
            return {
                status: status_html(track=("active", 0), now="Starting...", model=model_name),
                stage: working_html("Getting ready", "Checking the setup."), stage_video: gr.Video(value=None, visible=False),
                pdf: gr.DownloadButton(visible=False), log: gr.Textbox(value="", visible=True), report: "", best_plot: None,
                overlay: None, paths: None, ref_image: None, transcript: "", clock: {}, tabs: gr.Tabs(visible=True),
            }

        def track_step(video_path, prompt_text, mode, ref_prompt_text, long_cm, short_cm, ref_seconds, px_value, start, end, model_name):
            def stop(title, message):
                gr.Warning(message)
                return {status: status_html(now=message, error=True, model=model_name),
                        stage: problem_html(title, message), current_run: None}

            if not video_path:
                yield stop("Upload a video first", "Drop a clip into step 1, or open a run you already tracked.")
                return
            start = float(start) if start not in (None, "") and float(start) > 0 else None  # empty boxes arrive as 0
            end = float(end) if end not in (None, "") and float(end) > 0 else None
            if start is not None and end is not None and end <= start:
                yield stop("Check the trim times", "Trim end must be after trim start (or 0 to keep the rest of the video).")
                return
            name = f"{slug(Path(video_path).stem)[:30]}-{time.strftime('%m%d-%H%M%S')}"
            run = RUNS / name
            cmd = [sys.executable, "-u", str(ROOT / "track.py"), str(video_path), "--out", str(run)]
            if mode == NO_REFERENCE:
                cmd += ["--no-scale"]
            elif mode == KNOWN_SCALE:
                if not px_value or float(px_value) <= 0:
                    yield stop("Enter the scale", "Type how many pixels make one centimetre in your original video.")
                    return
                cmd += ["--px-per-cm", str(float(px_value))]
            else:
                if not (ref_prompt_text or "").strip():
                    yield stop("Describe the reference object", "Say what SAM should look for, e.g. 'black rectangle' or 'credit card'.")
                    return
                if not long_cm or float(long_cm) <= 0:
                    yield stop("Enter the reference size", "Type the long side of the reference object in cm.")
                    return
                cmd += [
                    "--ref-prompt", ref_prompt_text.strip(), "--ref-long-cm", str(float(long_cm)),
                    "--ref-short-cm", str(float(short_cm or 0)), "--ref-time", str(float(ref_seconds or 0)),
                ]
            target = (prompt_text or "").strip()
            cmd += ["--prompt", target] if target else ["--auto-prompts"]
            if start is not None:
                cmd += ["--start", str(start)]
            if end is not None:
                cmd += ["--end", str(end)]
            started = time.time()
            lines = [f"Tracking with SAM 3.1 into runs/{name} ..."]
            title = f"Tracking “{target}”" if target else "Finding what moves"
            yield {
                status: status_html(track=("active", 0), now="Preparing the clip", model=model_name),
                stage: working_html(title, "SAM 3.1 is finding it in every frame of the clip. This takes about a minute."),
                log: "\n".join(lines), current_run: None,
            }
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            process = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line.rstrip())
                update = {log: "\n".join(lines[-LOG_LINES:])}
                if line.strip() and (not line[:1].isspace() or "warning" in line.lower()):
                    update[status] = status_html(track=("active", time.time() - started), now=line.strip(), model=model_name)
                yield update
            if process.wait() != 0 or not (run / "tracks.csv").exists():
                reason = _failure_reason(lines)
                lines.append(f"TRACKING FAILED: {reason}")
                gr.Warning(f"Tracking failed: {reason[:400]}", duration=20)
                yield {
                    status: status_html(track=("failed",), now=reason, error=True, model=model_name),
                    stage: problem_html("Tracking failed", reason), log: "\n".join(lines[-LOG_LINES:]), current_run: None,
                }
                return
            seconds = time.time() - started
            yield {
                status: status_html(track=("done", seconds), analyse=("active", 0), now=_tracked_sentence(run), model=model_name),
                current_run: name, clock: {"track": seconds},
            }

        def show_tracking(run_name):
            if not run_name:
                return {current_run: None}  # nothing to show; the failure message stays
            run = RUNS / run_name
            video_out, paths_out, ref_out = _tracking_parts(run)
            return {
                stage: caption_html(_tracked_sentence(run) + " Muse Wonder is now choosing and testing physical laws."),
                stage_video: gr.Video(value=video_out, visible=bool(video_out)),
                overlay: video_out, paths: paths_out, ref_image: ref_out,
                previous: gr.Dropdown(choices=list_runs(), value=run_name), tabs: gr.Tabs(selected="tracking"),
            }

        def galileo_step(run_name, log_text, notes_text, model_name, effort_name, timing):
            if not run_name:
                yield {current_run: None}
                return
            run = RUNS / run_name
            timing = dict(timing or {})
            track_state = ("done", timing["track"]) if timing.get("track") is not None else ("skipped",)
            lines = (log_text or "").splitlines()
            events: queue.Queue = queue.Queue()
            failure: dict = {}

            def work():
                try:
                    Galileo(
                        run,
                        model=model_name or None,
                        effort=None if effort_name in (None, "", "default") else effort_name,
                        notes=notes_text or "",
                        on_event=lambda kind, text: events.put((kind, text)),
                    ).run_agent()
                except Exception as error:  # noqa: BLE001 - shown in the page
                    failure["error"] = str(error)
                finally:
                    events.put(("done", ""))

            started = time.time()
            threading.Thread(target=work, daemon=True).start()
            prefix = {"assistant": "Galileo: ", "tool": "  > ", "result": "    = "}
            yield {status: status_html(track=track_state, analyse=("active", 0), now="Galileo is reading the tracks and your notes", model=model_name)}
            while True:
                kind, text = events.get()
                if kind == "done":
                    break
                lines.append(prefix.get(kind, "") + text)
                yield {
                    log: "\n".join(lines[-LOG_LINES:]),
                    status: status_html(track=track_state, analyse=("active", time.time() - started), now=_now_text(kind, text), model=model_name),
                }
            if failure:
                lines.append(f"GALILEO FAILED: {failure['error']}")
                gr.Warning(f"Galileo hit a problem: {failure['error'][:400]}", duration=20)
                yield {
                    log: "\n".join(lines[-LOG_LINES:]),
                    status: status_html(track=track_state, analyse=("failed",), now=failure["error"][:600], error=True, model=model_name),
                    stage: problem_html("Galileo hit a problem", failure["error"][:600] + " The tracking is saved: pick this run "
                                        "under 'or open a run you already tracked' and try again, or switch the model."),
                    current_run: None,
                }
                return
            yield {clock: {**timing, "analyse": time.time() - started}}

        def show_report(run_name, timing, model_name):
            if not run_name:
                return {current_run: None}
            run = RUNS / run_name
            timing = dict(timing or {})
            text, plot, how = _report_parts(run)
            pdf_file = _pdf_for(run)
            return {
                status: status_html(
                    track=("done", timing["track"]) if timing.get("track") is not None else ("skipped",),
                    analyse=("done", timing.get("analyse")), report=("done", None) if pdf_file else ("failed",),
                    now="Done. The PDF report has the result, the fits, the tracking and Galileo's reasoning." if pdf_file
                    else "Done. The report is below (the PDF could not be made).", model=model_name,
                ),
                stage: result_html(run), stage_video: gr.Video(visible=False),
                pdf: gr.DownloadButton(value=str(pdf_file) if pdf_file else None, visible=bool(pdf_file)),
                report: text, best_plot: plot, transcript: how, tabs: gr.Tabs(selected="report"),
            }

        def start_previous(run_name, model_name):
            if not run_name or not (RUNS / run_name / "tracks.csv").exists():
                gr.Warning("Pick a run that has been tracked first.")
                return {current_run: None}
            run = RUNS / run_name
            video_out, paths_out, ref_out = _tracking_parts(run)
            return {
                current_run: run_name, clock: {},
                log: gr.Textbox(value=f"Re-analysing runs/{run_name} (SAM's tracking is reused, no new SAM call)", visible=True),
                status: status_html(track=("skipped",), analyse=("active", 0), now="Galileo is reading the tracks and your notes", model=model_name),
                stage: caption_html(_tracked_sentence(run) + " Muse Wonder is choosing and testing physical laws."),
                stage_video: gr.Video(value=video_out, visible=bool(video_out)), pdf: gr.DownloadButton(visible=False),
                report: "", best_plot: None, transcript: "", overlay: video_out, paths: paths_out, ref_image: ref_out,
                tabs: gr.Tabs(selected="tracking", visible=True),
            }

        def show_previous(run_name, model_name):
            if not run_name or not (RUNS / run_name).exists():
                return {current_run: None}
            run = RUNS / run_name
            video_out, paths_out, ref_out = _tracking_parts(run)
            common = {overlay: video_out, paths: paths_out, ref_image: ref_out,
                      log: gr.Textbox(value=f"Opened runs/{run_name}", visible=True), tabs: gr.Tabs(visible=True)}
            if not (run / "report.json").exists():
                return {
                    **common,
                    status: status_html(track=("done", None), now="Tracked, not analysed yet. Press 'Analyse with Galileo'.", model=model_name),
                    stage: caption_html(_tracked_sentence(run) + " Press 'Analyse with Galileo' to work out the physics."),
                    stage_video: gr.Video(value=video_out, visible=bool(video_out)), pdf: gr.DownloadButton(visible=False),
                    report: "", best_plot: None, transcript: "", tabs: gr.Tabs(selected="tracking", visible=True),
                }
            text, plot, how = _report_parts(run)
            meta = json.loads((run / "report.json").read_text(encoding="utf-8")).get("meta") or {}
            pdf_file = _pdf_for(run)
            return {
                **common,
                status: status_html(track=("done", None), analyse=("done", meta.get("seconds")), report=("done", None) if pdf_file else ("failed",),
                                    now=f"Opened runs/{run_name}.", model=meta.get("model") or model_name),
                stage: result_html(run), stage_video: gr.Video(visible=False),
                pdf: gr.DownloadButton(value=str(pdf_file) if pdf_file else None, visible=bool(pdf_file)),
                report: text, best_plot: plot, transcript: how, tabs: gr.Tabs(selected="report", visible=True),
            }

        (
            track_button.click(clear_results, inputs=model, outputs=outputs, show_progress="hidden")
            .then(
                track_step,
                inputs=[video, prompt, scale_mode, ref_prompt, ref_long, ref_short, ref_time, px_scale, trim_start, trim_end, model],
                outputs=outputs, show_progress="hidden",
            )
            .then(show_tracking, inputs=current_run, outputs=outputs, show_progress="hidden")
            .then(galileo_step, inputs=[current_run, log, notes, model, effort, clock], outputs=outputs, show_progress="hidden")
            .then(show_report, inputs=[current_run, clock, model], outputs=outputs, show_progress="hidden")
        )
        (
            analyse_button.click(start_previous, inputs=[previous, model], outputs=outputs, show_progress="hidden")
            .then(galileo_step, inputs=[current_run, log, notes, model, effort, clock], outputs=outputs, show_progress="hidden")
            .then(show_report, inputs=[current_run, clock, model], outputs=outputs, show_progress="hidden")
        )
        refresh.click(lambda: gr.Dropdown(choices=list_runs()), outputs=previous)
        previous.select(show_previous, inputs=[previous, model], outputs=outputs, show_progress="hidden")

        def on_scale_mode(mode):
            preset = SCALE_OPTIONS.get(mode)
            updates = {
                reference_box: gr.Row(visible=mode not in (NO_REFERENCE, KNOWN_SCALE)),
                px_scale: gr.Number(visible=mode == KNOWN_SCALE),
            }
            if preset is not None:
                updates[ref_prompt] = gr.Textbox(value=preset[0])
                updates[ref_long] = gr.Number(value=preset[1])
                updates[ref_short] = gr.Number(value=preset[2])
            return updates

        scale_mode.change(on_scale_mode, inputs=scale_mode, outputs=[reference_box, px_scale, ref_prompt, ref_long, ref_short])
    return demo


def favicon() -> str | None:
    """A 64 px pendulum icon for the browser tab (drawn once into the runs folder)."""
    path = RUNS / ".galileo-icon.png"
    if not path.exists():
        try:
            import cv2
            import numpy as np

            size = 256
            icon = np.zeros((size, size, 4), np.uint8)
            cv2.rectangle(icon, (0, 0), (size - 1, size - 1), (254, 252, 251, 255), -1)
            ink, pen, mark = (74, 42, 24, 255), (168, 71, 36, 255), (75, 223, 247, 255)
            cv2.line(icon, (40, 40), (216, 40), ink, 14, cv2.LINE_AA)
            cv2.line(icon, (128, 40), (176, 176), ink, 10, cv2.LINE_AA)
            cv2.circle(icon, (184, 196), 34, mark, -1, cv2.LINE_AA)
            cv2.circle(icon, (184, 196), 34, ink, 9, cv2.LINE_AA)
            cv2.circle(icon, (128, 40), 14, pen, -1, cv2.LINE_AA)
            cv2.imwrite(str(path), cv2.resize(icon, (64, 64), interpolation=cv2.INTER_AREA))
        except Exception:  # noqa: BLE001 - the default icon is fine too
            return None
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    demo = build()
    demo.queue(default_concurrency_limit=1).launch(
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        prevent_thread_lock=True,
        allowed_paths=[str(RUNS)],
        favicon_path=favicon(),
        theme=theme(),
        css=CSS,
        head=HEAD,
        js=FORCE_LIGHT_JS,
    )
    url = f"http://127.0.0.1:{args.port}/?__theme=light"
    print(f"Galileo is open at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    demo.block_thread()


if __name__ == "__main__":
    main()