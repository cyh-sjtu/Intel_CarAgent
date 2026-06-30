"""Test EfficientSAM OpenVINO: consume GroundingDINO detection, segment with box prompt.

Mirrors run_efficientsam_from_box.py but uses OpenVINO IR instead of PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def choose_detection(detections: list[dict[str, Any]], label_query: str) -> dict[str, Any]:
    if not detections:
        raise ValueError("No detections found.")
    query_terms = [term.strip().lower() for term in label_query.split(",") if term.strip()]
    candidates = detections
    if query_terms:
        filtered = [d for d in detections if any(t in str(d.get("label", "")).lower() for t in query_terms)]
        if filtered:
            candidates = filtered
    return max(candidates, key=lambda d: float(d.get("score", 0.0)))


def save_overlay(
    image_path: Path,
    mask: np.ndarray,
    detection: dict[str, Any],
    iou: float,
    elapsed_ms: float,
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, Path, Path]:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    mask_path = output_dir / f"{output_stem}_mask_ov.png"
    overlay_path = output_dir / f"{output_stem}_mask_overlay_ov.png"
    json_path = output_dir / f"{output_stem}_segmentation_ov.json"

    # Green mask overlay
    overlay = image_np.copy()
    color = np.array([0, 255, 120], dtype=np.uint8)
    alpha = 0.42
    overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * (1.0 - alpha) + color * alpha).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # Draw detection box
    draw = ImageDraw.Draw(overlay_img)
    x1, y1, x2, y2 = [round(float(v)) for v in detection["box"]]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
    label = f"{detection.get('label', 'object')} det={float(detection.get('score', 0.0)):.2f} sam_iou={iou:.2f}"
    draw.rectangle([x1, max(0, y1 - 18), min(image.width - 1, x1 + len(label) * 7 + 4), y1], fill=(255, 80, 80))
    draw.text((x1 + 2, max(0, y1 - 16)), label, fill=(0, 0, 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    overlay_img.save(overlay_path)

    # Compute mask stats
    ys, xs = np.where(mask)
    mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else []
    centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else []
    area_px = int(mask.sum())

    payload = {
        "image": str(image_path),
        "source_detection": detection,
        "sam_iou": iou,
        "mask_area_px": area_px,
        "mask_bbox": mask_bbox,
        "mask_centroid_px": centroid,
        "elapsed_ms": elapsed_ms,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return mask_path, overlay_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EfficientSAM OpenVINO with GroundingDINO detection.")
    parser.add_argument("--grounding-json", required=True, type=Path, help="Output of run_grounding_dino.py or OpenVINO variant.")
    parser.add_argument("--label-query", default="", help="Preferred label substring (comma-separated).")
    parser.add_argument("--encoder-xml", required=True, type=Path, help="EfficientSAM encoder IR (.xml).")
    parser.add_argument("--decoder-xml", required=True, type=Path, help="EfficientSAM decoder IR (.xml).")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "AUTO"])
    parser.add_argument("--encoder-device", default="", help="Optional override for encoder device.")
    parser.add_argument("--decoder-device", default="", help="Optional override for decoder device.")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--output-stem", default="", help="Defaults to image stem.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # When run directly, ensure caragent_agent package is importable
    agent_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(agent_dir))

    from caragent_agent.perception.sam.efficient_sam_openvino import EfficientSAMOpenVINO

    grounding_path = args.grounding_json.resolve()
    data = json.loads(grounding_path.read_text(encoding="utf-8"))
    image_path = Path(data["metadata"]["image"]).resolve()
    detections = data.get("detections", [])

    if not detections:
        print("No detections in grounding JSON. Aborting.", file=sys.stderr)
        return 1

    detection = choose_detection(detections, args.label_query)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    # Load SAM
    sam = EfficientSAMOpenVINO(
        encoder_xml=args.encoder_xml,
        decoder_xml=args.decoder_xml,
        device=args.device,
        encoder_device=args.encoder_device or None,
        decoder_device=args.decoder_device or None,
    )

    # Encoder (once)
    t0 = time.perf_counter()
    embeddings = sam.get_embedding(image_np)

    # Decoder (per box)
    box = detection["box"]  # [x1, y1, x2, y2]
    mask, _, iou = sam.predict_mask(embeddings, tuple(box), orig_h, orig_w)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    output_stem = args.output_stem or image_path.stem
    mask_path, overlay_path, json_path = save_overlay(
        image_path, mask, detection, iou, elapsed_ms, args.output_dir.resolve(), output_stem,
    )

    print(f"image: {image_path}")
    print(f"detection: {detection.get('label')} score={float(detection.get('score', 0.0)):.3f}")
    print(f"box: {detection.get('box_int', box)}")
    print(f"sam_iou: {iou:.3f}")
    print(f"mask_area_px: {int(mask.sum())}")
    print(f"elapsed: {elapsed_ms:.1f} ms")
    print(f"mask: {mask_path}")
    print(f"overlay: {overlay_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
