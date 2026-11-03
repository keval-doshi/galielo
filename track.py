#!/usr/bin/env python3
"""Track objects in a phone video with SAM 3.1 and turn their motion into cm and seconds.

Typical use (pendulum, A4 sheet held in the swing plane during the first second):

    python track.py pendulum.mov --prompt "red ball" --end 15 --out runs/pendulum1

Outputs in the --out folder:
    clip.mp4        the trimmed, constant-frame-rate clip that was sent to SAM
    ref_check.png   what was measured for the cm scale (check this first!)
    tracks.csv      one row per object per frame: t, x/y in px and cm (y points up)
    plot.png        position over time and the path of each object
    overlay.mp4     the clip with masks, centroids and trails drawn on
    summary.json    scale, frame rate and per-track coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from oe import config  # noqa: F401  (loads .env)
from oe import meta_api
from oe.draw import label
from oe.masks import Detection, color_detections, parse_sam_text
from oe.scale import draw_reference_check, measure_reference
from oe.video import H264Writer, iter_frames, prepare_video, probe, read_frame

PALETTE = [(60, 60, 230), (230, 140, 30), (60, 190, 60), (200, 60, 200), (0, 200, 230), (160, 100, 40)]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "object"


def fail(message: str) -> None:
    print(f"\nerror: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="your clip (.mov/.mp4)")
    ap.add_argument("--prompt", action="append", default=[], help='what to track, e.g. "red ball". Repeat for more objects (one SAM call each).')
    ap.add_argument("--auto-prompts", action="store_true", help="let Muse Spark look at the first frame and suggest what to track")
    ap.add_argument("--out", help="output folder (default: runs/<video name>)")
    ap.add_argument("--start", type=float, help="trim: start time in seconds")
    ap.add_argument("--end", type=float, help="trim: end time in seconds")
    ap.add_argument("--fps", type=float, help="force a frame rate (default: keep the video's)")
    ap.add_argument("--max-size", type=int, default=1280, help="longest side after resizing (default 1280)")

    scale = ap.add_argument_group("scale (pixels -> cm)")
    scale.add_argument("--ref-prompt", default="sheet of paper", help='reference object prompt (default "sheet of paper")')
    scale.add_argument("--ref-long-cm", type=float, default=29.7, help="reference long side in cm (A4: 29.7, US Letter: 27.94)")
    scale.add_argument("--ref-short-cm", type=float, default=21.0, help="reference short side in cm, used as a cross-check (A4: 21.0). 0 to skip")
    scale.add_argument("--ref-time", type=float, default=0.0, help="when the reference is visible, in seconds into the trimmed clip (default 0)")
    scale.add_argument("--ref-media", help="separate image or video, same camera position, showing the reference")
    scale.add_argument("--px-per-cm", type=float, help="skip the reference and use this scale, measured on your ORIGINAL video (it is converted if the clip gets resized)")
    scale.add_argument("--no-scale", action="store_true", help="no reference: track in pixels; give known lengths to Galileo in the notes instead")

    backend = ap.add_argument_group("segmentation backend")
    backend.add_argument("--backend", choices=["sam", "color"], default="sam", help="sam = SAM 3.1 via the Meta Model API (default); color = offline colour matching")
    backend.add_argument("--color", action="append", default=[], help="color backend: hex colour for each --prompt, in order")
    backend.add_argument("--ref-color", help="color backend: hex colour of the reference sheet")
    backend.add_argument("--fresh", action="store_true", help="ignore cached SAM answers and call the API again")
    backend.add_argument("--no-overlay", action="store_true", help="skip overlay.mp4 (faster)")
    return ap.parse_args(argv)


class LazyClient:
    """Create the API client only if something actually needs the network."""

    def __init__(self) -> None:
        self._client = None

    def __call__(self):
        if self._client is None:
            try:
                self._client = meta_api.get_client()
            except meta_api.MetaAPIError as error:
                fail(str(error))
        return self._client


def prepare(args: argparse.Namespace, out: Path):
    clip = out / "clip.mp4"
    src = Path(args.video)
    if not src.exists():
        fail(f"video not found: {src}")
    if args.start is not None and args.start < 0:
        fail("--start cannot be negative")
    if args.end is not None and args.end <= (args.start or 0.0):
        fail("--end must be after --start (leave it out to keep the rest of the video)")
    stat = src.stat()
    key = {
        "source": str(src.resolve()), "size": stat.st_size, "mtime": stat.st_mtime,
        "start": args.start, "end": args.end, "fps": args.fps, "max_size": args.max_size,
    }
    key_file = out / "prep.json"
    if clip.exists() and key_file.exists():
        saved = json.loads(key_file.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            info = probe(clip)
            info.fps = saved["fps"]
            args.source_width = saved.get("source_width") or probe(src).width
            return clip, info
    print("preparing clip (trim, constant frame rate, resize)...")
    for stale in list(out.glob("sam_*.txt")) + list(out.glob("ref_*.txt")):
        stale.unlink()  # answers for the old clip no longer match its frames
    info = prepare_video(src, clip, max_size=args.max_size, fps=args.fps, start=args.start, end=args.end)
    args.source_width = probe(src).width
    key_file.write_text(json.dumps({"key": key, "fps": info.fps, "source_width": args.source_width}, indent=2), encoding="utf-8")
    return clip, info


def reference_frame(args, clip: Path, info, out: Path, time_s: float) -> np.ndarray:
    if args.ref_media:
        media = Path(args.ref_media)
        image = cv2.imread(str(media))
        if image is None:  # a video: prepare it the same way so the pixels match the clip
            ref_clip = out / "ref_clip.mp4"
            if not ref_clip.exists():
                prepare_video(media, ref_clip, max_size=args.max_size, fps=info.fps)
            ref_info = probe(ref_clip)
            image = read_frame(ref_clip, min(round(time_s * info.fps), max(ref_info.frames - 1, 0)))
        h, w = image.shape[:2]
        if (w, h) != (info.width, info.height):
            if abs(w / h - info.width / info.height) > 0.01:
                fail("--ref-media has a different shape from the clip; record it from the same position in video mode")
            image = cv2.resize(image, (info.width, info.height), interpolation=cv2.INTER_AREA)
        return image
    index = round(time_s * info.fps)
    return read_frame(clip, min(index, max(info.frames - 1, 0)))


def call_api(what: str, fn, *args, **kwargs):
    """Run one API call; on failure stop with a one-line reason instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except SystemExit:
        raise
    except meta_api.MetaAPIError as error:
        fail(f"{what} failed: {error}")
    except Exception as error:  # noqa: BLE001 - HTTP, network and timeout errors from the SDK
        fail(f"{what} failed: {type(error).__name__}: {error}")


def find_reference(args, clip: Path, info, out: Path, api):
    """Find the reference sheet with SAM 3.1: (detection, frame, time, prompt).

    Tries --ref-time first, then every 0.5 s up to 2 s later, and also "white paper" when the
    prompt is about paper, because people hold the sheet up at slightly different moments.
    """
    still = bool(args.ref_media) and cv2.haveImageReader(str(args.ref_media))
    last = max(info.duration - 1 / max(info.fps, 1), 0.0)
    times = [args.ref_time] if still else [t for t in (args.ref_time + 0.5 * k for k in range(5)) if t <= last + 1e-9]
    times = times or [min(args.ref_time, last)]
    prompts = [args.ref_prompt]
    lowered = args.ref_prompt.lower()
    if ("paper" in lowered or "sheet" in lowered) and lowered != "white paper":
        prompts.append("white paper")
    min_area = 0.001 * info.width * info.height  # ignore specks
    for prompt in prompts:
        for t in times:
            frame = reference_frame(args, clip, info, out, t)
            cache = out / f"ref_{slug(prompt)}_{int(round(t * 1000))}ms.txt"
            if cache.exists() and not args.fresh:
                text = cache.read_text(encoding="utf-8")
            else:
                print(f'measuring scale: SAM 3.1 looking for "{prompt}" at {t:.2f} s...')
                text = call_api("SAM 3.1 (finding the reference sheet)", meta_api.sam_image, api(), frame, prompt)
                cache.write_text(text, encoding="utf-8")
            detections, notes = parse_sam_text(text, media="image")
            for note in notes[:3]:
                print(f"  parser {note}")
            detections = [d for d in detections if d.area() >= min_area]
            if detections:
                return max(detections, key=lambda d: d.area()), frame, t, prompt
            print("  not found")
    looked_at = ", ".join(f"{t:g}" for t in times)
    names = " or ".join(f'"{p}"' for p in prompts)
    fail(
        f"could not find the reference sheet ({names} at {looked_at} s). Make sure the whole sheet is visible "
        "and flat at that moment, or set when it is visible (--ref-time, in the app 'The sheet is visible at'), "
        "describe it differently (--ref-prompt, in the app 'How SAM should find the sheet'; the synthetic test "
        'clips need "blue rectangle"), or give the scale with --px-per-cm.'
    )


def find_scale(args, clip, info, out, api) -> tuple[float | None, str]:
    if args.px_per_cm:
        if args.px_per_cm <= 0:
            fail("--px-per-cm must be positive")
        source_width = getattr(args, "source_width", None) or info.width
        scale = args.px_per_cm * info.width / source_width
        if abs(scale - args.px_per_cm) > 1e-9:
            print(f"scale: {args.px_per_cm:g} px/cm on the original {source_width}-px video = {scale:.4g} px/cm on the {info.width}-px working clip")
        else:
            print(f"scale: {scale:.4g} px/cm (given)")
        return scale, f"given: {args.px_per_cm:g} px/cm on the original video"
    if args.no_scale:
        return None, "none (pixels only)"
    if args.backend == "sam":
        det, frame, used_time, used_prompt = find_reference(args, clip, info, out, api)
    else:
        if not args.ref_color:
            fail("the color backend needs --ref-color for the reference, or pass --px-per-cm / --no-scale")
        frame = reference_frame(args, clip, info, out, args.ref_time)
        detections = color_detections(frame, args.ref_color, 0, max_objects=1)
        if not detections:
            fail(f"could not find the colour {args.ref_color} at {args.ref_time:.2f} s")
        det, used_time, used_prompt = detections[0], args.ref_time, f"colour {args.ref_color}"
    result = measure_reference(det, info.height, info.width, args.ref_long_cm, args.ref_short_cm or None)
    draw_reference_check(frame, det, result, out / "ref_check.png")
    print(f"scale: {result.px_per_cm:.3f} px/cm  (reference long side {result.long_px:.1f} px)")
    if result.disagreement is not None:
        if result.disagreement > 0.04:
            print(
                f"  warning: the long and short sides disagree by {result.disagreement:.1%}. The sheet is "
                "probably tilted, bent or partly hidden; distances may be off by about that much."
            )
        else:
            print(f"  cross-check: short side agrees within {result.disagreement:.1%}")
    args.scale_check = {
        "long_side_px": round(result.long_px, 1),
        "short_side_px": round(result.short_px, 1),
        "long_side_cm": result.long_cm,
        "short_side_cm": result.short_cm,
        "sides_disagree_by": None if result.disagreement is None else round(result.disagreement, 4),
        "reference_confidence": det.confidence,
        "reference_time_s": used_time,
        "reference_prompt": used_prompt,
    }
    return result.px_per_cm, f'"{used_prompt}" at {used_time:.2f} s, long side {args.ref_long_cm:g} cm'


def track_prompt(args, prompt: str, index: int, clip: Path, out: Path, api, state: dict) -> list[Detection]:
    if args.backend == "color":
        if index >= len(args.color):
            fail("the color backend needs one --color per --prompt")
        found: list[Detection] = []
        for frame_index, frame in enumerate(iter_frames(clip)):
            found += color_detections(frame, args.color[index], frame_index)
        return found

    cache = out / f"sam_{slug(prompt)}.txt"
    if cache.exists() and not args.fresh:
        text = cache.read_text(encoding="utf-8")
        print(f'"{prompt}": using cached SAM answer ({cache.name})')
    else:
        if "file_id" not in state:
            print(f"uploading clip ({clip.stat().st_size / 1e6:.1f} MB) to the Meta Model API...")
            state["file_id"] = call_api("uploading the clip", meta_api.upload_video, api(), clip)
        print(f'"{prompt}": SAM 3.1 is tracking it through the video...')
        started = time.time()
        text = call_api(f'SAM 3.1 (tracking "{prompt}")', meta_api.sam_video, api(), state["file_id"], prompt)
        cache.write_text(text, encoding="utf-8")
        print(f"  done in {time.time() - started:.0f}s")
    started = time.time()
    detections, notes = parse_sam_text(text, media="video")
    for note in notes[:5]:
        print(f"  parser {note}")
    print(f"  decoded {len(detections)} masks in {time.time() - started:.1f}s")
    return detections


def build_rows(tracks: dict[str, list[Detection]], info, scale: float | None) -> list[dict]:
    rows = []
    for name, detections in tracks.items():
        for det in detections:
            centre = det.centroid()
            if centre is None:
                continue
            x, y = centre
            row = {
                "track": name, "frame": det.frame, "t": det.frame / info.fps,
                "x_px": x, "y_px": y, "area_px": det.area(),
                "confidence": "" if det.confidence is None else det.confidence,
            }
            if scale:
                row["x_cm"] = x / scale
                row["y_cm"] = (info.height - 1 - y) / scale  # y points up, origin at bottom-left
            rows.append(row)
    rows.sort(key=lambda r: (r["track"], r["frame"]))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["track", "frame", "t", "x_px", "y_px", "x_cm", "y_cm", "area_px", "confidence"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def summarize(rows: list[dict], info) -> dict:
    by_track = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row["frame"])
    summary = {}
    for name, frames in by_track.items():
        frames = sorted(frames)
        gaps = np.diff(frames) - 1 if len(frames) > 1 else np.array([0])
        summary[name] = {
            "frames": len(frames),
            "coverage": len(frames) / max(info.frames, 1),
            "t_start": frames[0] / info.fps,
            "t_end": frames[-1] / info.fps,
            "largest_gap_frames": int(gaps.max()),
        }
    return summary


def plot(rows: list[dict], scale: float | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = "cm" if scale else "px"
    xk, yk = ("x_cm", "y_cm") if scale else ("x_px", "y_px")
    by_track = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)
    fig, (ax_t, ax_xy) = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, track) in enumerate(list(by_track.items())[:8]):
        t = [r["t"] for r in track]
        color = np.array(PALETTE[i % len(PALETTE)][::-1]) / 255
        ax_t.plot(t, [r[xk] for r in track], "-", color=color, label=f"{name}  x")
        ax_t.plot(t, [r[yk] for r in track], "--", color=color, label=f"{name}  y")
        ax_xy.plot([r[xk] for r in track], [r[yk] for r in track], ".-", ms=2, lw=0.8, color=color, label=name)
    ax_t.set(xlabel="time (s)", ylabel=f"position ({unit})", title="Position over time")
    ax_xy.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", title="Path (y up)" if scale else "Path (pixels, y down)")
    if not scale:
        ax_xy.invert_yaxis()
    ax_xy.set_aspect("equal", adjustable="datalim")
    for ax in (ax_t, ax_xy):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_overlay(clip: Path, info, tracks: dict[str, list[Detection]], scale: float | None, path: Path) -> None:
    by_frame: dict[int, list[tuple[int, str, Detection]]] = defaultdict(list)
    for i, (name, detections) in enumerate(tracks.items()):
        for det in detections:
            by_frame[det.frame].append((i, name, det))
    trails: dict[str, list[tuple[int, int]]] = defaultdict(list)
    writer = H264Writer(path, info.width, info.height, info.fps)
    try:
        for frame_index, frame in enumerate(iter_frames(clip)):
            canvas = frame.copy()
            for i, name, det in by_frame.get(frame_index, []):
                color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
                clipped = det.clipped_box_mask(info.height, info.width)
                if clipped is not None:
                    y0, y1, x0, x1, local = clipped
                    region = canvas[y0:y1, x0:x1]
                    region[local] = (0.55 * region[local] + 0.45 * color).astype(np.uint8)
                centre = det.centroid()
                if centre is not None:
                    point = (int(round(centre[0])), int(round(centre[1])))
                    trails[name].append(point)
                    cv2.circle(canvas, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, point, 4, tuple(int(c) for c in color), 1, cv2.LINE_AA)
                    label(canvas, name, (point[0] + 14, point[1] - 22), 0.45)
            for i, (name, points) in enumerate(trails.items()):
                recent = points[-90:]
                if len(recent) > 1:
                    cv2.polylines(canvas, [np.int32(recent)], False, PALETTE[i % len(PALETTE)], 2, cv2.LINE_AA)
            label(canvas, f"t = {frame_index / info.fps:6.3f} s", (14, 14))
            if scale:
                bar = int(round(10 * scale))
                y = info.height - 18
                cv2.line(canvas, (14, y), (14 + bar, y), (0, 0, 0), 5)
                cv2.line(canvas, (14, y), (14 + bar, y), (255, 255, 255), 3)
                label(canvas, "10 cm", (14, y - 30), 0.45)
            writer.write(canvas)
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    out = Path(args.out or Path("runs") / slug(Path(args.video).stem))
    out.mkdir(parents=True, exist_ok=True)
    api = LazyClient()

    clip, info = prepare(args, out)
    print(f"clip: {info.width}x{info.height} at {info.fps:g} fps, {info.frames} frames ({info.duration:.1f} s)")
    if args.backend == "sam" and info.frames > 15000:
        fail(f"the clip has {info.frames} frames and SAM 3.1 accepts at most 15,000; trim it with --start/--end (in the app: the Trim boxes)")
    if info.duration > 60:
        print("  note: long clip; trimming it to just the experiment makes tracking faster and cheaper")

    prompts = list(args.prompt)
    if args.auto_prompts:
        suggested = call_api("Muse Spark (choosing what to track)", meta_api.suggest_prompts, api(), read_frame(clip, 0))
        print(f"Muse Spark suggests tracking: {', '.join(suggested) or '(nothing)'}")
        prompts += [p for p in suggested if p not in prompts]
    if not prompts:
        fail('tell me what to track, e.g. --prompt "red ball" (or use --auto-prompts)')

    scale, scale_source = find_scale(args, clip, info, out, api)

    tracks: dict[str, list[Detection]] = {}
    state: dict = {}
    for index, prompt in enumerate(prompts):
        detections = track_prompt(args, prompt, index, clip, out, api, state)
        ids = sorted({d.object_id for d in detections}, key=lambda s: int(s) if s.isdigit() else 0)
        if not detections:
            print(f'  warning: nothing found for "{prompt}". Try a more specific phrase, more light, or a plainer background.')
        elif len(ids) > 1:
            print(f'  note: "{prompt}" matched {len(ids)} objects (ids {", ".join(ids)}); each becomes its own track')
        for object_id in ids:
            tracks[f"{prompt}#{object_id}"] = [d for d in detections if d.object_id == object_id]

    if not tracks:
        names = ", ".join(f'"{p}"' for p in prompts)
        fail(f"SAM 3.1 found nothing for {names}. Try a more specific phrase (e.g. \"orange ball\"), more light or a plainer background.")
    rows = build_rows(tracks, info, scale)
    write_csv(rows, out / "tracks.csv")
    summary = summarize(rows, info)
    print("\ntracks:")
    for name, s in summary.items():
        flag = "" if s["coverage"] >= 0.6 else "   <- lost for long stretches"
        print(
            f"  {name:<28} {s['frames']:>5} frames ({s['coverage']:.0%}), "
            f"{s['t_start']:.2f}-{s['t_end']:.2f} s, largest gap {s['largest_gap_frames']} frames{flag}"
        )
    if rows:
        plot(rows, scale, out / "plot.png")
    if not args.no_overlay and tracks:
        print("rendering overlay.mp4...")
        render_overlay(clip, info, tracks, scale, out / "overlay.mp4")

    (out / "summary.json").write_text(
        json.dumps(
            {
                "video": str(Path(args.video).resolve()),
                "fps": info.fps, "frames": info.frames, "width": info.width, "height": info.height,
                "px_per_cm": scale, "scale_source": scale_source,
                "scale_check": getattr(args, "scale_check", None),
                "backend": args.backend, "prompts": prompts, "tracks": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved to {out}/  (tracks.csv, plot.png, overlay.mp4, ref_check.png, summary.json)")


if __name__ == "__main__":
    main()