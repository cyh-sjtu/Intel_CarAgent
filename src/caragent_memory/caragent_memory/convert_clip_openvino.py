#!/usr/bin/env python3
"""Export CLIP projected features to OpenVINO IR.

The selector expects the real CLIP image embedding, not ViT token hidden states.
For CLIP ViT-B/32 that means:

    vision_model CLS token (768) -> visual_projection (512) -> 512-D feature
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _write_metadata(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversion_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _metadata(args: argparse.Namespace, output_dir: Path) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "export": args.export,
        "input_size": int(args.input_size),
        "expected_embedding_dim": int(args.expected_dim),
        "dry_run": bool(args.dry_run),
        "export_path": (
            "CLIPModel.get_text_features(input_ids, attention_mask)"
            if args.export == "text"
            else "CLIPModel.get_image_features(pixel_values)"
        ),
        "notes": [
            "This export includes CLIP visual_projection.",
            "The expected output is a 512-D projected image embedding for ViT-B/32.",
            "Do not use raw CLIPVisionModel hidden-state exports shaped like [1, 50, 768].",
        ],
    }


def export_clip_image_features(args: argparse.Namespace, output_dir: Path) -> Path:
    import numpy as np
    import torch
    from transformers import CLIPModel

    try:
        import openvino as ov
    except Exception as exc:
        raise RuntimeError("OpenVINO Python package is required for conversion.") from exc

    class CLIPImageFeatureModule(torch.nn.Module):
        def __init__(self, model_id: str) -> None:
            super().__init__()
            self.clip = CLIPModel.from_pretrained(
                model_id,
                local_files_only=True,
                use_safetensors=False,
            )
            self.clip.eval()

        def forward(self, pixel_values):
            return self.clip.get_image_features(pixel_values=pixel_values)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = CLIPImageFeatureModule(args.model_id).eval()
    dummy = torch.zeros(
        1,
        3,
        int(args.input_size),
        int(args.input_size),
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch_output = model(dummy).detach().cpu().numpy()
    if torch_output.reshape(-1).shape[0] != int(args.expected_dim):
        raise RuntimeError(
            f"PyTorch CLIP image feature output has shape {torch_output.shape}, "
            f"expected {args.expected_dim} values."
        )

    ov_model = ov.convert_model(model, example_input=dummy)
    xml_path = output_dir / f"{args.model_name}.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(args.fp16))

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), args.device)
    result = compiled([np.zeros_like(dummy.detach().cpu().numpy())])
    output = np.asarray(result[compiled.output(0)], dtype=np.float32).reshape(-1)
    if output.shape[0] != int(args.expected_dim):
        raise RuntimeError(
            f"Converted OpenVINO model output has shape {output.shape}, "
            f"expected ({args.expected_dim},)."
        )
    return xml_path


def export_clip_text_features(args: argparse.Namespace, output_dir: Path) -> Path:
    import numpy as np
    import torch
    from transformers import CLIPModel

    try:
        import openvino as ov
    except Exception as exc:
        raise RuntimeError("OpenVINO Python package is required for conversion.") from exc

    class CLIPTextFeatureModule(torch.nn.Module):
        def __init__(self, model_id: str) -> None:
            super().__init__()
            self.clip = CLIPModel.from_pretrained(
                model_id,
                local_files_only=True,
                use_safetensors=False,
            )
            self.clip.eval()

        def forward(self, input_ids):
            text_outputs = self.clip.text_model(input_ids=input_ids)
            pooled_output = text_outputs[1]
            return self.clip.text_projection(pooled_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = CLIPTextFeatureModule(args.model_id).eval()
    input_ids = torch.zeros((1, 77), dtype=torch.long)
    attention_mask = torch.ones((1, 77), dtype=torch.long)

    with torch.no_grad():
        torch_output = model(input_ids).detach().cpu().numpy()
    if torch_output.reshape(-1).shape[0] != int(args.expected_dim):
        raise RuntimeError(
            f"PyTorch CLIP text feature output has shape {torch_output.shape}, "
            f"expected {args.expected_dim} values."
        )

    ov_model = ov.convert_model(model, example_input=input_ids)
    xml_path = output_dir / f"{args.model_name}.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(args.fp16))

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), args.device)
    del attention_mask
    result = compiled([np.zeros((1, 77), dtype=np.int64)])
    output = np.asarray(result[compiled.output(0)], dtype=np.float32).reshape(-1)
    if output.shape[0] != int(args.expected_dim):
        raise RuntimeError(
            f"Converted OpenVINO text model output has shape {output.shape}, "
            f"expected ({args.expected_dim},)."
        )
    return xml_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/caragent_ws/models/clip-vit-base-patch32").expanduser(),
        help="Output directory for OpenVINO IR and metadata.",
    )
    parser.add_argument("--model-name", default="image_encoder", help="Output IR basename.")
    parser.add_argument("--export", default="image", choices=["image", "text"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--expected-dim", type=int, default=512)
    parser.add_argument("--device", default="CPU", help="Device used for post-conversion validation.")
    parser.add_argument("--fp16", action="store_true", help="Save compressed FP16 weights.")
    parser.add_argument("--dry-run", action="store_true", help="Only write metadata and commands.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = _metadata(args, output_dir)

    if args.dry_run:
        metadata["recommended_commands"] = [
            "pip install torch transformers openvino",
            (
                "ros2 run caragent_memory convert_clip_openvino "
                "--model-id openai/clip-vit-base-patch32 "
                "--output-dir ~/caragent_ws/models/clip-vit-base-patch32"
            ),
            (
                "ros2 run caragent_memory convert_clip_openvino "
                "--export text --model-name text_encoder "
                "--model-id openai/clip-vit-base-patch32 "
                "--output-dir ~/caragent_ws/models/clip-vit-base-patch32"
            ),
            (
                "ros2 run caragent_memory select_keyframes "
                "--dataset ~/caragent_ws/keyframes/lab_001 "
                "--clip-model ~/caragent_ws/models/clip-vit-base-patch32/image_encoder.xml "
                "--device AUTO"
            ),
        ]
        metadata["expected_xml"] = str(output_dir / f"{args.model_name}.xml")
        _write_metadata(output_dir, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    xml_path = (
        export_clip_text_features(args, output_dir)
        if args.export == "text"
        else export_clip_image_features(args, output_dir)
    )
    metadata["expected_xml"] = str(xml_path)
    metadata["conversion_tool"] = "openvino.convert_model"
    metadata["validation"] = {
        "openvino_output_dim": int(args.expected_dim),
        "status": "passed",
    }
    _write_metadata(output_dir, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
