"""Convert EfficientSAM ONNX exports to OpenVINO IR.

Models: encoder (image → embeddings) + decoder (embeddings + box → mask).

Run this ONCE to generate .xml/.bin IR files, then deploy them to the board.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def convert_onnx_to_ir(onnx_path: Path, output_path: Path, input_shape: dict | None = None) -> Path:
    """Convert a single ONNX model to OpenVINO IR (.xml + .bin)."""
    from openvino import Core

    print(f"Converting {onnx_path} ...", flush=True)
    core = Core()
    model = core.read_model(str(onnx_path))

    if input_shape:
        print(f"  Setting input shapes: {input_shape}", flush=True)
        model.reshape(input_shape)

    compiled = core.compile_model(model, "CPU")
    # simple sanity check
    for name, param in compiled.inputs[0].node.inputs() if hasattr(compiled.inputs[0], 'node') else []:
        pass

    output_stem = output_path / onnx_path.stem
    ov_xml = output_path / f"{onnx_path.stem}.xml"
    ov_bin = output_path / f"{onnx_path.stem}.bin"

    from openvino import save_model
    save_model(model, str(ov_xml))

    print(f"  Saved: {ov_xml}", flush=True)
    print(f"  Saved: {ov_bin}", flush=True)
    return ov_xml


def convert_encoder(onnx_path: Path, output_dir: Path, height: int, width: int) -> Path:
    """Convert encoder ONNX → IR with fixed input dimensions."""
    return convert_onnx_to_ir(
        onnx_path,
        output_dir,
        input_shape={"batched_images": [1, 3, height, width]},
    )


def convert_decoder(onnx_path: Path, output_dir: Path, num_points: int = 2) -> Path:
    """Convert decoder ONNX → IR. num_points=2 for box prompt (top-left + bottom-right)."""
    return convert_onnx_to_ir(
        onnx_path,
        output_dir,
        input_shape={
            "image_embeddings": [1, 256, 64, 64],
            "batched_point_coords": [1, 1, num_points, 2],
            "batched_point_labels": [1, 1, num_points],
            "orig_im_size": [2],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert EfficientSAM ONNX to OpenVINO IR.")
    parser.add_argument(
        "--encoder-onnx",
        required=True,
        type=Path,
        help="Path to efficient_sam_vitt_encoder.onnx",
    )
    parser.add_argument(
        "--decoder-onnx",
        required=True,
        type=Path,
        help="Path to efficient_sam_vitt_decoder.onnx",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for IR files.")
    parser.add_argument("--height", type=int, default=480, help="Input image height (fixed for IR).")
    parser.add_argument("--width", type=int, default=640, help="Input image width (fixed for IR).")
    parser.add_argument("--num-points", type=int, default=2, help="Max prompt points (2 for box).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.encoder_onnx.exists():
        print(f"ERROR: encoder ONNX not found: {args.encoder_onnx}", file=sys.stderr)
        return 1
    if not args.decoder_onnx.exists():
        print(f"ERROR: decoder ONNX not found: {args.decoder_onnx}", file=sys.stderr)
        return 1

    enc_xml = convert_encoder(args.encoder_onnx, args.output_dir, args.height, args.width)
    dec_xml = convert_decoder(args.decoder_onnx, args.output_dir, args.num_points)

    print(f"\nDone! IR files in: {args.output_dir}")
    print(f"  Encoder: {enc_xml}")
    print(f"  Decoder: {dec_xml}")
    print(f"\nDeploy these to the board, then use efficient_sam_openvino.py for inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
