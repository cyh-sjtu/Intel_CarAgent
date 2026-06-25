"""Select one GroundingDINO candidate box with a VLM.

This module is intentionally a small testable bridge between open-vocabulary
detection and downstream box-prompted segmentation/localization.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from caragent_agent.config.config import config, ensure_api_key_env
from caragent_agent.perception.grounding.grounding_dino_openvino import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUT_DIR,
    GroundingDINOOpenVINO,
)
from caragent_agent.utils.llm_handler import UnifiedLLMClient


COLORS = [
    (239, 68, 68),
    (14, 165, 233),
    (34, 197, 94),
    (245, 158, 11),
    (168, 85, 247),
    (236, 72, 153),
    (20, 184, 166),
    (251, 113, 133),
    (132, 204, 22),
    (99, 102, 241),
    (249, 115, 22),
    (6, 182, 212),
]


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _encode_image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _clamp_box(box: list[float] | tuple[float, ...], width: int, height: int) -> list[int]:
    values = [float(v) for v in list(box)[:4]]
    x1, x2 = sorted((values[0], values[2]))
    y1, y2 = sorted((values[1], values[3]))
    return [
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    ]


def _text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox(xy, text, font=font)
    except Exception:
        w, h = draw.textsize(text, font=font)
        return xy[0], xy[1], xy[0] + w, xy[1] + h


def prepare_candidates(
    detections: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    max_candidates: int,
) -> list[dict[str, Any]]:
    width, height = image_size
    ranked = sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    candidates: list[dict[str, Any]] = []
    for idx, det in enumerate(ranked[:max_candidates], start=1):
        raw_box = det.get("box_int") or det.get("box") or [0, 0, 0, 0]
        box_int = _clamp_box(raw_box, width, height)
        x1, y1, x2, y2 = box_int
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        area_ratio = (box_w * box_h) / float(max(1, width * height))
        candidates.append(
            {
                "id": idx,
                "label": str(det.get("label") or "object"),
                "score": float(det.get("score", 0.0)),
                "box": [float(v) for v in det.get("box", box_int)],
                "box_int": box_int,
                "center_px": [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
                "area_ratio": round(area_ratio, 6),
            }
        )
    return candidates


def draw_candidate_overlay(image_path: Path, candidates: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(16)
    small_font = _load_font(14)
    if not candidates:
        draw.rectangle([10, 10, 230, 42], fill=(17, 24, 39), outline=(239, 68, 68), width=2)
        draw.text((18, 18), "No GroundingDINO candidates", fill=(255, 255, 255), font=small_font)
    for idx, candidate in enumerate(candidates):
        color = COLORS[idx % len(COLORS)]
        x1, y1, x2, y2 = candidate["box_int"]
        label = f"#{candidate['id']} {candidate['label']} {candidate['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        left, top, right, bottom = _text_box(draw, (x1, y1), label, font)
        text_w = right - left
        text_h = bottom - top
        bg_y1 = max(0, y1 - text_h - 8)
        draw.rectangle([x1, bg_y1, min(image.width - 1, x1 + text_w + 10), bg_y1 + text_h + 8], fill=color)
        draw.text((x1 + 5, bg_y1 + 4), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_selected_overlay(
    image_path: Path,
    candidates: list[dict[str, Any]],
    selection: dict[str, Any],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(18)
    selected_id = selection.get("selected_id")
    selected = next((item for item in candidates if item["id"] == selected_id), None)
    if selected is None:
        label = f"{selection.get('status', 'failed')}: no valid selected box"
        draw.rectangle([10, 10, min(image.width - 1, 520), 48], fill=(127, 29, 29))
        draw.text((18, 18), label, fill=(255, 255, 255), font=font)
    else:
        x1, y1, x2, y2 = selected["box_int"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 140), width=6)
        label = f"SELECTED #{selected_id} conf={float(selection.get('confidence', 0.0)):.2f}"
        left, top, right, bottom = _text_box(draw, (x1, y1), label, font)
        text_w = right - left
        text_h = bottom - top
        bg_y1 = max(0, y1 - text_h - 10)
        draw.rectangle([x1, bg_y1, min(image.width - 1, x1 + text_w + 12), bg_y1 + text_h + 10], fill=(0, 255, 140))
        draw.text((x1 + 6, bg_y1 + 5), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_crop_grid(image_path: Path, candidates: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    font = _load_font(20)
    cell_w, cell_h = 260, 220
    cols = min(4, max(1, len(candidates)))
    rows = max(1, (len(candidates) + cols - 1) // cols)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (15, 23, 42))
    draw_grid = ImageDraw.Draw(grid)
    for idx, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate["box_int"]
        pad_x = max(8, int((x2 - x1) * 0.08))
        pad_y = max(8, int((y2 - y1) * 0.08))
        crop_box = [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image.width, x2 + pad_x),
            min(image.height, y2 + pad_y),
        ]
        crop = image.crop(crop_box)
        crop.thumbnail((cell_w - 16, cell_h - 42), Image.Resampling.LANCZOS)
        row = idx // cols
        col = idx % cols
        origin_x = col * cell_w
        origin_y = row * cell_h
        color = COLORS[idx % len(COLORS)]
        grid.paste(crop, (origin_x + (cell_w - crop.width) // 2, origin_y + 36))
        label = f"#{candidate['id']} {candidate['label']} {candidate['score']:.2f}"
        draw_grid.rectangle([origin_x, origin_y, origin_x + cell_w - 1, origin_y + 32], fill=color)
        draw_grid.text((origin_x + 8, origin_y + 5), label, fill=(0, 0, 0), font=font)
        draw_grid.rectangle([origin_x, origin_y, origin_x + cell_w - 1, origin_y + cell_h - 1], outline=color, width=2)
    if not candidates:
        draw_grid.text((16, 16), "No candidates", fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def _json_from_text(text: str) -> dict[str, Any] | None:
    clean = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL | re.IGNORECASE)
    if fenced:
        clean = fenced.group(1).strip()
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    try:
        parsed = json.loads(clean)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_selection(parsed: dict[str, Any] | None, *, valid_ids: set[int], query: str) -> dict[str, Any] | None:
    if parsed is None:
        return None
    status = str(parsed.get("status") or "").strip().lower()
    if status == "no_valid_candidate":
        selected_id = parsed.get("selected_id")
        if selected_id not in (None, "", "null"):
            return None
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        parsed["status"] = "no_valid_candidate"
        parsed["selected_id"] = None
        parsed["confidence"] = max(0.0, min(1.0, confidence))
        parsed["reason"] = str(parsed.get("reason") or "").strip()
        parsed["query"] = query
        return parsed

    if status and status != "selected":
        return None

    try:
        selected_id = int(parsed.get("selected_id"))
    except Exception:
        return None
    if selected_id not in valid_ids:
        return None
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    parsed["status"] = "selected"
    parsed["selected_id"] = selected_id
    parsed["confidence"] = max(0.0, min(1.0, confidence))
    parsed["reason"] = str(parsed.get("reason") or "").strip()
    parsed["query"] = query
    return parsed


def build_vlm_messages(
    *,
    vlm_query: str,
    grounding_query: str,
    image_size: tuple[int, int],
    candidates: list[dict[str, Any]],
    candidate_overlay: Image.Image,
    crop_grid: Image.Image,
    retry_note: str = "",
) -> list[dict[str, Any]]:
    metadata = {
        "vlm_query": vlm_query,
        "grounding_query": grounding_query,
        "image_size": list(image_size),
        "candidates": candidates,
    }
    retry_text = f"\nPrevious response was invalid: {retry_note}\n" if retry_note else ""
    system_prompt = (
        "You are a robot perception box selector. Select one candidate bounding box only when "
        "a numbered candidate is a reasonable visual match for the requested target. If none of "
        "the candidates reasonably match, return no_valid_candidate. Do not invent boxes or coordinates."
    )
    user_prompt = f"""
Task: choose the single best candidate box for the query, or reject all candidates if none reasonably match.

Selection priority:
1. Semantic match to the query.
2. Visual confirmation in the annotated image and candidate crops.
3. Clear and complete object.
4. Distinctive partial target cue when the full object was not detected.
5. Higher GroundingDINO score.
6. More central / more salient object when all else is tied.

If the query includes attributes, spatial relations, or nearby context, the selected candidate should satisfy them visually.
If no candidate covers the full target but one candidate tightly covers a distinctive part of the target
(for example the fire extinguisher inside a fire-extinguisher cabinet), you may select that candidate
with low confidence and explain that it is a partial but useful target cue.
For this robot pipeline, a useful partial target cue is better than no selection because later SAM and
depth stages can still localize around it.
If multiple candidates look plausible, choose the most reasonable one.
If every candidate is unrelated to the requested target or contradicts the requested spatial/context cues,
do not choose the least-bad candidate. Return no_valid_candidate and explain briefly.

Return only strict JSON with one of these schemas:
{{"status":"selected","selected_id":1,"confidence":0.0,"reason":"short reason","query":"{vlm_query}"}}
{{"status":"no_valid_candidate","selected_id":null,"confidence":0.0,"reason":"short reason","query":"{vlm_query}"}}

Candidate metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}
{retry_text}
""".strip()
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _encode_image_to_data_url(candidate_overlay)}},
                {"type": "image_url", "image_url": {"url": _encode_image_to_data_url(crop_grid)}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


async def ask_vlm_to_select(
    *,
    vlm_query: str,
    grounding_query: str,
    image_size: tuple[int, int],
    candidates: list[dict[str, Any]],
    candidate_overlay_path: Path,
    crop_grid_path: Path,
    model: str,
) -> tuple[dict[str, Any], str]:
    ensure_api_key_env("qwen")
    client = UnifiedLLMClient()
    overlay = Image.open(candidate_overlay_path).convert("RGB")
    crops = Image.open(crop_grid_path).convert("RGB")
    messages = build_vlm_messages(
        vlm_query=vlm_query,
        grounding_query=grounding_query,
        image_size=image_size,
        candidates=candidates,
        candidate_overlay=overlay,
        crop_grid=crops,
    )
    response = await client.chat_completion(model, messages)
    parsed = _json_from_text(response)
    valid_ids = {item["id"] for item in candidates}
    normalized = _normalize_selection(parsed, valid_ids=valid_ids, query=vlm_query)
    if normalized is not None:
        return normalized, response

    retry_note = (
        "not strict JSON"
        if parsed is None
        else "selection JSON did not use an allowed status or a valid selected_id"
    )
    retry_messages = build_vlm_messages(
        vlm_query=vlm_query,
        grounding_query=grounding_query,
        image_size=image_size,
        candidates=candidates,
        candidate_overlay=overlay,
        crop_grid=crops,
        retry_note=retry_note,
    )
    retry_response = await client.chat_completion(model, retry_messages)
    retry_parsed = _json_from_text(retry_response)
    retry_normalized = _normalize_selection(retry_parsed, valid_ids=valid_ids, query=vlm_query)
    if retry_normalized is not None:
        return retry_normalized, retry_response
    failed = {
        "status": "vlm_parse_failed" if retry_parsed is None else "vlm_invalid_selection",
        "selected_id": None,
        "confidence": 0.0,
        "reason": retry_note,
        "query": vlm_query,
    }
    return failed, retry_response


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    grounding_png = output_dir / f"{stem}_grounding_openvino.png"
    grounding_json = output_dir / f"{stem}_grounding_openvino.json"
    candidates_png = output_dir / f"{stem}_vlm_candidates.png"
    crops_png = output_dir / f"{stem}_vlm_crops.png"
    selected_png = output_dir / f"{stem}_vlm_selected.png"
    selection_json = output_dir / f"{stem}_vlm_box_selection.json"

    vlm_query = str(args.query or "").strip()
    grounding_query = str(args.grounding_query or vlm_query).strip()
    if grounding_query and grounding_query[-1] not in ".。!?！？":
        grounding_query = f"{grounding_query} ."

    detector = GroundingDINOOpenVINO(model_dir=args.model_dir, model_id=args.model_id, device=args.grounding_device)
    grounding = detector.detect(
        image_path=image_path,
        text_prompt=grounding_query,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    image = Image.open(image_path).convert("RGB")
    candidates = prepare_candidates(
        grounding.get("detections", []),
        image_size=image.size,
        max_candidates=args.max_candidates,
    )
    grounding["overlay_path"] = str(grounding_png)
    grounding["vlm_candidates_path"] = str(candidates_png)
    grounding["candidate_crops_path"] = str(crops_png)
    grounding_json.write_text(json.dumps(grounding, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_candidate_overlay(image_path, prepare_candidates(grounding.get("detections", []), image_size=image.size, max_candidates=999), grounding_png)
    draw_candidate_overlay(image_path, candidates, candidates_png)
    make_crop_grid(image_path, candidates, crops_png)

    raw_vlm_response = ""
    if not candidates:
        selection = {
            "status": "no_detection",
            "selected_id": None,
            "confidence": 0.0,
            "reason": "GroundingDINO returned no candidate boxes.",
            "query": vlm_query,
        }
    else:
        selection, raw_vlm_response = asyncio.run(
            ask_vlm_to_select(
                vlm_query=vlm_query,
                grounding_query=grounding_query,
                image_size=image.size,
                candidates=candidates,
                candidate_overlay_path=candidates_png,
                crop_grid_path=crops_png,
                model=args.vlm_model,
            )
        )
    draw_selected_overlay(image_path, candidates, selection, selected_png)

    payload = {
        "image": str(image_path),
        "query": vlm_query,
        "vlm_query": vlm_query,
        "grounding_query": grounding_query,
        "selection": selection,
        "candidates": candidates,
        "vlm_model": args.vlm_model,
        "vlm_raw_response": raw_vlm_response,
        "paths": {
            "grounding_json": str(grounding_json),
            "grounding_overlay": str(grounding_png),
            "vlm_candidates": str(candidates_png),
            "vlm_crops": str(crops_png),
            "vlm_selected": str(selected_png),
            "selection_json": str(selection_json),
        },
    }
    selection_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GroundingDINO and ask a VLM to select one candidate box.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--query", required=True, help="Detailed target description for the VLM selector.")
    parser.add_argument("--grounding-query", default="", help="Short open-vocabulary detector prompt, e.g. chair.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--grounding-device", default="GPU")
    parser.add_argument("--box-threshold", default=0.25, type=float)
    parser.add_argument("--text-threshold", default=0.20, type=float)
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--vlm-model", default=str(config.get("vlm_model_analyse_images", "qwen3-vl-plus")))
    parser.add_argument("--model-dir", default=DEFAULT_MODELS_DIR, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_selection(args)
    selection = result["selection"]
    print(f"image: {result['image']}")
    print(f"vlm_query: {result['vlm_query']}")
    print(f"grounding_query: {result['grounding_query']}")
    print(f"status: {selection.get('status')}")
    print(f"selected_id: {selection.get('selected_id')}")
    print(f"confidence: {float(selection.get('confidence') or 0.0):.3f}")
    print(f"reason: {selection.get('reason')}")
    for key, value in result["paths"].items():
        print(f"{key}: {value}")
    return 0 if selection.get("status") in {"selected", "no_detection", "no_valid_candidate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
