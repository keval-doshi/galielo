#!/usr/bin/env python3
"""Step 1: check that your Meta Model API key works and the models you need are there.

    python check_api.py
    python check_api.py --video test_pendulum.mp4 --prompt "red ball"   # also tests SAM 3.1 on video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from oe import config, meta_api
from oe.masks import parse_sam_text

OK, BAD = "  [ok]", "  [!!]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="optional short clip to test SAM 3.1 video tracking end to end")
    ap.add_argument("--prompt", default="red ball", help="what to look for in --video")
    args = ap.parse_args(argv)
    failures = 0

    print(f"API: {config.API_BASE_URL}")
    try:
        client = meta_api.get_client(timeout=300)
    except meta_api.MetaAPIError as error:
        print(f"{BAD} {error}")
        return 1
    print(f"{OK} MODEL_API_KEY found")

    try:
        available = {m.id for m in client.models.list()}
        for model in (config.SAM_MODEL, config.SPARK_MODEL, config.WONDER_MODEL):
            if model in available:
                print(f"{OK} model {model} is available")
            else:
                print(f"{BAD} model {model} is not in your model list")
                failures += 1
        others = sorted(available - {config.SAM_MODEL, config.SPARK_MODEL, config.WONDER_MODEL})
        if others:
            print(f"       also available: {', '.join(others)}")
    except Exception as error:  # noqa: BLE001 - show any API error plainly
        print(f"{BAD} could not list models: {error}")
        failures += 1

    try:
        started = time.time()
        answer = meta_api.ask_spark(client, "Reply with exactly one word: ready")
        print(f"{OK} {config.SPARK_MODEL} answered {answer.strip()[:40]!r} in {time.time() - started:.1f}s")
    except Exception as error:  # noqa: BLE001
        print(f"{BAD} {config.SPARK_MODEL} call failed: {error}")
        failures += 1

    # SAM 3.1 on a generated image: a red circle whose centre we know exactly.
    image = np.full((360, 640, 3), 235, np.uint8)
    cv2.rectangle(image, (60, 80), (220, 300), (200, 120, 40), -1)
    cv2.circle(image, (430, 180), 45, (40, 40, 210), -1, cv2.LINE_AA)
    try:
        started = time.time()
        text = meta_api.sam_image(client, image, "red circle")
        detections, notes = parse_sam_text(text, media="image")
        if detections:
            cx, cy = max(detections, key=lambda d: d.area()).centroid()
            error_px = float(np.hypot(cx - 430, cy - 180))
            if error_px <= 10:
                print(f"{OK} {config.SAM_MODEL} found the red circle {error_px:.1f} px from its true centre ({time.time() - started:.1f}s)")
            else:
                print(f"{BAD} {config.SAM_MODEL} answered, but its mask centre is {error_px:.0f} px from the circle")
                failures += 1
        else:
            print(f"{BAD} {config.SAM_MODEL} answered but found nothing. Raw: {text[:120]!r}")
            failures += 1
        for note in notes[:3]:
            print(f"       parser {note}")
    except Exception as error:  # noqa: BLE001
        print(f"{BAD} {config.SAM_MODEL} image call failed: {error}")
        failures += 1

    if args.video:
        try:
            started = time.time()
            file_id = meta_api.upload_video(client, Path(args.video))
            print(f"{OK} uploaded {args.video} as {file_id}")
            text = meta_api.sam_video(client, file_id, args.prompt)
            detections, _ = parse_sam_text(text, media="video")
            frames = len({d.frame for d in detections})
            print(f"{OK} {config.SAM_MODEL} tracked {args.prompt!r} in {frames} frames ({time.time() - started:.0f}s)")
        except Exception as error:  # noqa: BLE001
            print(f"{BAD} video test failed: {error}")
            failures += 1

    print("\nall good, you can start tracking." if not failures else f"\n{failures} problem(s) above.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
