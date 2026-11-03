"""Thin wrappers around the Meta Model API (OpenAI-compatible) for SAM 3.1 and Muse Spark.

SAM 3.1 request shapes follow Meta's own examples (github.com/meta-models/meta-sam):
- images: one unary Responses call with an ``input_image`` data URL;
- videos: upload the file to /v1/files (purpose ``user_data``), then a *streaming*
  Responses call with ``{"type": "input_video", "file_id": ...}``.
The model answers with special-token text, one line per frame, which
``oe.masks.parse_sam_text`` turns into masks.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import config


class MetaAPIError(RuntimeError):
    """Raised when the Meta Model API rejects or fails a request."""


def get_client(timeout: float = 900.0):
    """Return an OpenAI SDK client pointed at the Meta Model API."""
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - dependency guard
        raise MetaAPIError(
            "The 'openai' package is missing. Run: pip install -r requirements.txt"
        ) from error
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise MetaAPIError(
            "MODEL_API_KEY is not set. Copy .env.example to .env and paste the key "
            "from your Meta Model API dashboard (dev.meta.ai)."
        )
    return OpenAI(base_url=config.API_BASE_URL, api_key=key, timeout=timeout, max_retries=2)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def image_to_data_url(image: Any, fmt: str = ".png", max_side: int | None = None) -> str:
    """Encode a file path or a BGR numpy image as a data URL."""
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise MetaAPIError(f"Could not read image: {image}")
    if max_side:
        h, w = image.shape[:2]
        factor = max_side / max(h, w)
        if factor < 1:
            image = cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if fmt == ".jpg" else []
    ok, buffer = cv2.imencode(fmt, image, params)
    if not ok:
        raise MetaAPIError("Could not encode the image.")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}"


def _user_message(prompt: str, media_part: dict) -> list[dict]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}, media_part],
        }
    ]


def _sam_metadata(include_confidence: bool) -> dict[str, str]:
    # one_bit masks are smaller on the wire, which matters for long videos.
    metadata = {"mask_encoding": "one_bit"}
    if include_confidence:
        metadata["include_confidence"] = "true"
    return metadata


def collect_output_text(events: Iterable[Any]) -> str:
    """Collect SAM's text from a Responses event stream (SDK objects or dicts).

    The live SAM API streams one ``response.output_text.delta`` per frame line and
    finalizes the lane with ``response.content_part.done``.
    """
    deltas: list[str] = []
    final: str | None = None
    for event in events:
        kind = _get(event, "type", "")
        if kind == "response.output_text.delta":
            deltas.append(_get(event, "delta", "") or "")
        elif kind == "response.output_text.done":
            final = _get(event, "text", final)
        elif kind == "response.content_part.done":
            part = _get(event, "part")
            if _get(part, "type") == "output_text":
                final = _get(part, "text", final)
            elif _get(part, "type") == "refusal":
                raise MetaAPIError(f"SAM refused the request: {_get(part, 'refusal')}")
        elif kind == "response.refusal.done":
            raise MetaAPIError(f"SAM refused the request: {_get(event, 'refusal')}")
        elif kind == "response.failed":
            error = _get(_get(event, "response"), "error")
            raise MetaAPIError(f"SAM request failed: {_get(error, 'message', error)}")
        elif kind == "response.incomplete":
            details = _get(_get(event, "response"), "incomplete_details")
            print(f"warning: SAM response marked incomplete: {details}", file=sys.stderr)
        elif kind == "error":
            raise MetaAPIError(f"SAM stream error: {_get(event, 'message', event)}")
    return final if final is not None else "".join(deltas)


def upload_video(client: Any, path: Path) -> str:
    """Upload a video through the Files API and return its file id."""
    path = Path(path)
    with path.open("rb") as handle:
        uploaded = client.files.create(file=(path.name, handle, "video/mp4"), purpose="user_data")
    file_id = _get(uploaded, "id")
    if not file_id:
        raise MetaAPIError(f"The upload returned no file id: {uploaded!r}")
    return file_id


def sam_video(client: Any, file_id: str, prompt: str, include_confidence: bool = True) -> str:
    """Track every instance of ``prompt`` through an uploaded video."""
    stream = client.responses.create(
        model=config.SAM_MODEL,
        stream=True,
        input=_user_message(prompt, {"type": "input_video", "file_id": file_id}),
        metadata=_sam_metadata(include_confidence),
    )
    try:
        return collect_output_text(stream)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def sam_image(client: Any, image: Any, prompt: str, include_confidence: bool = True) -> str:
    """Segment every instance of ``prompt`` in one image (unary request)."""
    response = client.responses.create(
        model=config.SAM_MODEL,
        stream=False,
        input=_user_message(prompt, {"type": "input_image", "image_url": image_to_data_url(image)}),
        metadata=_sam_metadata(include_confidence),
    )
    if _get(response, "status") == "failed":
        error = _get(response, "error")
        raise MetaAPIError(f"SAM request failed: {_get(error, 'message', error)}")
    text = _get(response, "output_text", "") or ""
    if not text:
        for item in _get(response, "output", []) or []:
            for block in _get(item, "content", []) or []:
                if _get(block, "type") == "output_text":
                    text += _get(block, "text", "") or ""
    return text


def ask_spark(client: Any, prompt: str, image: Any | None = None, model: str | None = None) -> str:
    """One Muse Spark call with optional image, returning the text answer."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": image_to_data_url(image, ".jpg", max_side=1280)}}
        )
    response = client.chat.completions.create(
        model=model or config.SPARK_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content or ""


def suggest_prompts(client: Any, frame_bgr: np.ndarray, max_items: int = 4) -> list[str]:
    """Ask Muse Spark which objects in the frame are worth tracking, as SAM-friendly phrases."""
    instruction = (
        "This frame is from a physics experiment filmed from the side. List the distinct "
        "physical objects that are likely to move (balls, dominoes, pendulum bobs, toy cars, "
        "falling books). Do not list the background, hands, tables, rulers or sheets of paper. "
        f"Reply with ONLY a JSON array of at most {max_items} short noun phrases that a "
        'segmentation model can find, e.g. ["red ball", "domino"], most important first.'
    )
    answer = ask_spark(client, instruction, frame_bgr)
    match = re.search(r"\[.*?\]", answer, re.S)
    if not match:
        raise MetaAPIError(f"Muse Spark did not return a JSON list: {answer[:200]!r}")
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise MetaAPIError(f"Muse Spark returned malformed JSON: {match.group(0)[:200]!r}") from error
    phrases = [str(item).strip() for item in items if str(item).strip()]
    return phrases[:max_items]
