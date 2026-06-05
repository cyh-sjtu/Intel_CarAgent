"""Run local open-vocabulary detection with GroundingDINO.

This is intentionally independent from the ROS agent. It is a quick test bed for
object-level grounding before we wire the result into navigation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in values]


def _normalize_detection_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().tolist()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().tolist()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().tolist()

    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        raw_label = labels[idx] if idx < len(labels) else "object"
        label = str(raw_label)
        score = float(scores[idx]) if idx < len(scores) else 0.0
        x1, y1, x2, y2 = _to_float_list(box)
        detections.append(
            {
                "label": label,
                "score": score,
                "box": [x1, y1, x2, y2],
                "box_int": [round(x1), round(y1), round(x2), round(y2)],
            }
        )
    return detections


def run_grounding(
    image_path: Path,
    text: str,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    image = Image.open(image_path).convert("RGB")
    print(f"loading processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"loading model: {model_id}", flush=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}", flush=True)
    model = model.to(device)
    model.eval()

    print("preparing inputs", flush=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    print("running inference", flush=True)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    # Newer Transformers exposes GroundingDINO's text-aware postprocessor.
    if hasattr(processor, "post_process_grounded_object_detection"):
        postprocess = processor.post_process_grounded_object_detection
        params = inspect.signature(postprocess).parameters
        threshold_kwargs = {
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
        }
        if "box_threshold" in params:
            threshold_kwargs["box_threshold"] = box_threshold
        else:
            threshold_kwargs["threshold"] = box_threshold
        processed = postprocess(outputs, input_ids=inputs.get("input_ids"), **threshold_kwargs)
    else:
        processed = processor.post_process_object_detection(
            outputs,
            threshold=box_threshold,
            target_sizes=target_sizes,
        )

    detections = _normalize_detection_result(processed[0])
    metadata = {
        "image": str(image_path),
        "image_size": [image.width, image.height],
        "text": text,
        "model_id": model_id,
        "device": device,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "elapsed_ms": elapsed_ms,
    }
    return detections, metadata


def draw_detections(image_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Run local GroundingDINO detection on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument(
        "--text",
        required=True,
        help="Object prompt. GroundingDINO works best with period-separated phrases.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--box-threshold", default=0.25, type=float)
    parser.add_argument("--text-threshold", default=0.20, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_grounding.png"
    json_path = output_dir / f"{stem}_grounding.json"

    try:
        detections, metadata = run_grounding(
            image_path=image_path,
            text=args.text,
            model_id=args.model_id,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
    except OSError as exc:
        message = str(exc)
        if "1455" in message or "页面文件太小" in message:
            print(
                "\nModel loading failed because Windows page file / virtual memory is too small.\n"
                "Fix: increase the Windows paging file size, close memory-heavy apps, or run on a machine with more RAM.\n"
                "The Hugging Face cache is already on D: if you used run_with_d_cache.ps1.\n",
                flush=True,
            )
        raise
    draw_detections(image_path, detections, overlay_path)

    payload = {
        "metadata": metadata,
        "detections": detections,
        "overlay_path": str(overlay_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"image: {image_path}")
    print(f"text: {args.text}")
    print(f"detections: {len(detections)}")
    for det in detections:
        print(f"- {det['label']} score={det['score']:.3f} box={det['box_int']}")
    print(f"overlay: {overlay_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
