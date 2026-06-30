"""Convert GroundingDINO to OpenVINO IR for DK-2500.

Important: the Hugging Face Transformers implementation currently contains
operators that OpenVINO cannot convert directly (for example aten::cummax,
aten::cummin, aten::isin, aten::special_logit). This script follows the
OpenVINO Grounded-SAM notebook route instead: use the OpenVINO-compatible
GroundingDINO implementation from IDEA-Research/GroundingDINO.
"""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = "https://github.com/wenyi5608/GroundingDINO.git"
DEFAULT_REPO_DIR = Path("/home/car/caragent_ws/src/GroundingDINO_OpenVINO")
DEFAULT_OUTPUT_DIR = Path("/home/car/caragent_ws/models/grounding_dino_openvino")
DEFAULT_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_WEIGHTS = "groundingdino_swint_ogc.pth"
DEFAULT_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/"
    "groundingdino_swint_ogc.pth"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists():
        print(f"GroundingDINO repo exists: {repo_dir}")
        print("warning: ensure this is the OpenVINO-compatible fork: https://github.com/wenyi5608/GroundingDINO.git")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])


def ensure_weights(repo_dir: Path, weights_path: Path, weights_url: str) -> Path:
    if weights_path.exists():
        return weights_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(["wget", "-O", str(weights_path), weights_url], cwd=repo_dir)
    except Exception:
        run(["curl", "-L", "-o", str(weights_path), weights_url], cwd=repo_dir)
    return weights_path


def patch_huggingface_endpoint(endpoint: str) -> None:
    """Patch HF endpoint variables before Transformers loads BERT."""

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


def patch_transformers_compatibility() -> None:
    """Patch small Transformers API differences used by official GroundingDINO."""

    try:
        import torch
        from transformers import BertModel
    except Exception as exc:
        print(f"warning: could not import Transformers BERT for compatibility patch: {exc}", flush=True)
        return

    if not hasattr(BertModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask.to(dtype=getattr(self, "dtype", torch.float32))

        BertModel.get_head_mask = get_head_mask
        print("patched transformers.BertModel.get_head_mask compatibility shim", flush=True)

    original_get_extended_attention_mask = BertModel.get_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if isinstance(device, torch.device):
            device = None
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        params = inspect.signature(original_get_extended_attention_mask).parameters
        if "device" in params:
            return original_get_extended_attention_mask(
                self,
                attention_mask,
                input_shape,
                device=device,
                dtype=dtype,
            )
        return original_get_extended_attention_mask(
            self,
            attention_mask,
            input_shape,
            dtype=dtype,
        )

    BertModel.get_extended_attention_mask = get_extended_attention_mask
    print("patched transformers.BertModel.get_extended_attention_mask compatibility shim", flush=True)

    original_invert_attention_mask = BertModel.invert_attention_mask

    def invert_attention_mask(self, encoder_attention_mask, dtype=None):
        if dtype is None or isinstance(dtype, torch.device):
            dtype = getattr(self, "dtype", torch.float32)
        params = inspect.signature(original_invert_attention_mask).parameters
        if "dtype" in params:
            return original_invert_attention_mask(self, encoder_attention_mask, dtype=dtype)
        return original_invert_attention_mask(self, encoder_attention_mask)

    BertModel.invert_attention_mask = invert_attention_mask
    print("patched transformers.BertModel.invert_attention_mask compatibility shim", flush=True)

def convert(
    repo_dir: Path,
    config_path: Path,
    weights_path: Path,
    output_dir: Path,
    bert_path: str,
    local_files_only: bool,
) -> None:
    import torch
    import openvino as ov

    patch_transformers_compatibility()
    sys.path.insert(0, str(repo_dir))
    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map,
    )
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util import get_tokenlizer

    args = SLConfig.fromfile(str(config_path))
    args.device = "cpu"
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    if bert_path:
        args.text_encoder_type = bert_path
        print(f"using local/text encoder path: {bert_path}", flush=True)
    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("using local files only for Hugging Face assets", flush=True)
    model = build_model(args)
    forward_params = list(inspect.signature(model.forward).parameters)
    if len(forward_params) < 6:
        raise RuntimeError(
            "This GroundingDINO checkout does not expose the OpenVINO-compatible "
            "tensor-only forward signature. Use the fork from "
            "https://github.com/wenyi5608/GroundingDINO.git or pass "
            "--repo-dir pointing at that checkout."
        )
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    max_text_len = int(getattr(args, "max_text_len", 256))
    tokenizer = get_tokenlizer.get_tokenlizer(args.text_encoder_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    caption = "wooden round table . chair . door ."
    tokenized = tokenizer([caption], padding="longest", return_tensors="pt")
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized,
        special_tokens,
        tokenizer,
    )
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    dummy_image = torch.randn(1, 3, 1024, 1280)
    dummy_inputs = (
        dummy_image,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        position_ids,
        tokenized["token_type_ids"],
        text_self_attention_masks,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    print("converting GroundingDINO official implementation to OpenVINO IR", flush=True)
    traced_model = torch.jit.trace(
        model,
        example_inputs=dummy_inputs,
        strict=False,
        check_trace=False,
    )
    ov_model = ov.convert_model(traced_model, example_input=dummy_inputs)
    model_xml = output_dir / "openvino_model.xml"
    ov.save_model(ov_model, model_xml)

    # Keep config and weights path metadata next to IR for the runtime wrapper.
    (output_dir / "source_repo.txt").write_text(str(repo_dir), encoding="utf-8")
    (output_dir / "source_config.txt").write_text(str(config_path), encoding="utf-8")
    (output_dir / "image_input_size.txt").write_text("1024 1280\n", encoding="utf-8")
    print(f"xml: {model_xml}")
    print(f"bin: {model_xml.with_suffix('.bin')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert official GroundingDINO to OpenVINO IR.")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR, type=Path)
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--config", default="", help="Config path. Defaults to SwinT OGC config in repo.")
    parser.add_argument("--weights", default="", help="Weights path. Downloads SwinT OGC weights if missing.")
    parser.add_argument("--weights-url", default=DEFAULT_WEIGHTS_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument(
        "--hf-endpoint",
        default="",
        help="Optional Hugging Face endpoint mirror for BERT, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--bert-path",
        default="",
        help="Local bert-base-uncased directory, or another Transformers-compatible text encoder path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access network for Hugging Face text encoder assets.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    patch_huggingface_endpoint(args.hf_endpoint)
    repo_dir = args.repo_dir.expanduser().resolve()
    if not args.skip_clone:
        ensure_repo(repo_dir, args.repo_url)

    config_path = Path(args.config).expanduser().resolve() if args.config else repo_dir / DEFAULT_CONFIG
    weights_path = Path(args.weights).expanduser().resolve() if args.weights else repo_dir / DEFAULT_WEIGHTS
    weights_path = ensure_weights(repo_dir, weights_path, args.weights_url)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    # Some GroundingDINO utilities expect cwd to be the repo root.
    old_cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        convert(
            repo_dir,
            config_path,
            weights_path,
            args.output_dir.expanduser().resolve(),
            args.bert_path,
            args.local_files_only,
        )
    finally:
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
