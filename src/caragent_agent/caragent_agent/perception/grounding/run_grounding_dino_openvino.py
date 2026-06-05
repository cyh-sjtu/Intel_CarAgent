"""CLI for GroundingDINO OpenVINO inference on DK-2500."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .grounding_dino_openvino import (
    DEFAULT_IMAGE,
    DEFAULT_MODEL_ID,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUTPUT_DIR,
    GroundingDINOOpenVINO,
)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(image_path: Path, detections: list[dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    colors = [
        (255, 77, 77),
        (64, 196, 255),
        (88, 214, 141),
        (255, 193, 7),
        (171, 71, 188),
        (255, 112, 67),
    ]
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["box_int"]
        color = colors[idx % len(colors)]
        label = f"{det['label']} {det['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        text_h = bottom - top
        text_w = right - left
        bg_y1 = max(0, y1 - text_h - 4)
        draw.rectangle([x1, bg_y1, x1 + text_w + 6, bg_y1 + text_h + 4], fill=color)
        draw.text((x1 + 3, bg_y1 + 2), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GroundingDINO with OpenVINO.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, type=Path)
    parser.add_argument("--text", default="wooden round table . chair . door .")
    parser.add_argument("--model-dir", default=DEFAULT_MODELS_DIR, type=Path)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local directory containing the GroundingDINO processor.",
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--box-threshold", default=0.25, type=float)
    parser.add_argument("--text-threshold", default=0.20, type=float)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_grounding_openvino.png"
    json_path = output_dir / f"{stem}_grounding_openvino.json"

    detector = GroundingDINOOpenVINO(model_dir=args.model_dir, model_id=args.model_id, device=args.device)
    payload = detector.detect(
        image_path=image_path,
        text_prompt=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    payload["overlay_path"] = str(overlay_path)
    draw_detections(image_path, payload["detections"], overlay_path)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"image: {image_path}")
    print(f"text: {args.text}")
    print(f"detections: {len(payload['detections'])}")
    print(f"elapsed_ms: {payload['metadata']['elapsed_ms']:.2f}")
    for det in payload["detections"]:
        print(f"- {det['label']} score={det['score']:.3f} box={det['box_int']}")
    print(f"overlay: {overlay_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
