"""Convert Depth Anything V2 Hugging Face model to OpenVINO IR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_OUTPUT_DIR = Path("/home/car/caragent_ws/models/depth_anything_v2_openvino")


def patch_huggingface_endpoint(endpoint: str) -> None:
    if not endpoint:
        return
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_HOME = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = endpoint + "/{repo_id}/resolve/{revision}/{filename}"
    except Exception as exc:
        print(f"warning: could not patch huggingface_hub endpoint constants: {exc}", flush=True)
    print(f"using HF endpoint: {endpoint}", flush=True)


def convert(
    model_id: str,
    output_dir: Path,
    height: int,
    width: int,
    local_files_only: bool,
) -> None:
    import torch
    import openvino as ov
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("using local files only for Hugging Face assets", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading depth model: {model_id}", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    class DepthWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values):
            outputs = self.wrapped(pixel_values=pixel_values)
            return outputs.predicted_depth

    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)
    wrapped = DepthWrapper(model)
    print(f"converting with static input: 1x3x{height}x{width}", flush=True)
    ov_model = ov.convert_model(wrapped, example_input=(dummy,))
    model_xml = output_dir / "openvino_model.xml"
    ov.save_model(ov_model, model_xml)
    processor.save_pretrained(output_dir)
    (output_dir / "image_input_size.txt").write_text(f"{height} {width}\n", encoding="utf-8")
    (output_dir / "source_model.txt").write_text(model_id, encoding="utf-8")
    print(f"xml: {model_xml}", flush=True)
    print(f"bin: {model_xml.with_suffix('.bin')}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Depth Anything V2 to OpenVINO IR.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    patch_huggingface_endpoint(args.hf_endpoint)
    convert(
        model_id=args.model_id,
        output_dir=args.output_dir.expanduser().resolve(),
        height=args.height,
        width=args.width,
        local_files_only=args.local_files_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
