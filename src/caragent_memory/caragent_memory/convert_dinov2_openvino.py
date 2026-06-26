#!/usr/bin/env python3
"""Export DINOv2 CLS image embedding to OpenVINO IR."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def export_dinov2(args: argparse.Namespace, output_dir: Path) -> Path:
    import numpy as np
    import torch
    from transformers import Dinov2Model

    try:
        import openvino as ov
    except Exception as exc:
        raise RuntimeError("OpenVINO Python package is required for conversion.") from exc

    class DINOv2CLSModule(torch.nn.Module):
        def __init__(self, model_ref: str) -> None:
            super().__init__()
            self.dino = Dinov2Model.from_pretrained(
                model_ref,
                local_files_only=not args.allow_download,
            )
            self.dino.eval()

        def forward(self, pixel_values):
            return self.dino(pixel_values=pixel_values).last_hidden_state[:, 0, :]

    output_dir.mkdir(parents=True, exist_ok=True)
    model = DINOv2CLSModule(args.model_id).eval()
    dummy = torch.zeros(1, 3, int(args.input_size), int(args.input_size), dtype=torch.float32)
    with torch.no_grad():
        torch_output = model(dummy).detach().cpu().numpy()
    if torch_output.reshape(-1).shape[0] != int(args.expected_dim):
        raise RuntimeError(
            f"PyTorch DINOv2 output has shape {torch_output.shape}, expected {args.expected_dim} values."
        )

    ov_model = ov.convert_model(model, example_input=dummy)
    xml_path = output_dir / f"{args.model_name}.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(args.fp16))

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), args.device)
    result = compiled([np.zeros_like(dummy.detach().cpu().numpy())])
    output = np.asarray(result[compiled.output(0)], dtype=np.float32).reshape(-1)
    if output.shape[0] != int(args.expected_dim):
        raise RuntimeError(f"Converted DINOv2 output has shape {output.shape}, expected ({args.expected_dim},).")
    return xml_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="~/caragent_ws/models/dinov2-small",
        help="Local Hugging Face DINOv2-small directory or model id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/caragent_ws/models/dinov2-small-openvino").expanduser(),
    )
    parser.add_argument("--model-name", default="openvino_model")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--expected-dim", type=int, default=384)
    parser.add_argument("--device", default="CPU", help="Device used for post-conversion validation.")
    parser.add_argument("--fp16", action="store_true", help="Save compressed FP16 weights.")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    model_ref = str(Path(args.model_id).expanduser()) if str(args.model_id).startswith("~") else str(args.model_id)
    args.model_id = model_ref
    xml_path = export_dinov2(args, output_dir)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "input_size": int(args.input_size),
        "expected_embedding_dim": int(args.expected_dim),
        "export_path": "Dinov2Model(...).last_hidden_state[:, 0, :]",
        "expected_xml": str(xml_path),
        "conversion_tool": "openvino.convert_model",
        "validation": {"openvino_output_dim": int(args.expected_dim), "status": "passed"},
    }
    (output_dir / "conversion_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
