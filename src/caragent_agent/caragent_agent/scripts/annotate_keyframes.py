"""Offline tool: enrich selected keyframes with VLM semantic descriptions.

Usage:
    python -m caragent_agent.scripts.annotate_keyframes \\
      --dataset-dir /path/to/session/selected \\
      --batch-size 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from typing import Any
from pathlib import Path

from caragent_agent.config.config import config, ensure_api_key_env, get_api_keys
from caragent_agent.impression_graph.scene_memory import SceneMemory
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import (
    vlm_single_image_request_message_kf,
)


def _try_load_openvino_clip_text_encoder(device: str = "GPU"):
    try:
        from caragent_memory.openvino_clip import OpenVINOClipTextEncoder

        workspace = Path(os.environ.get("CARAGENT_WORKSPACE", "/home/car/caragent_ws"))
        model_path = workspace / "models" / "clip-vit-base-patch32" / "text_encoder.xml"
        encoder = OpenVINOClipTextEncoder(model_path, device=device)
        return encoder, device
    except Exception as exc:
        print(f"OpenVINO CLIP text encoder unavailable on {device}: {exc}")
        return None, None


def _try_load_torch_clip():
    try:
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except Exception:
        return None, None, None


def _extract_semantic(response_data, req_id: int) -> str:
    if response_data is None:
        return ""
    if isinstance(response_data, dict):
        if not response_data:
            return ""
        if "error" in response_data:
            return ""
        value = response_data.get(req_id)
        if value is None:
            value = response_data.get(str(req_id))
        if value is None:
            value = next(iter(response_data.values()), "")
        if isinstance(value, dict) and "error" in value:
            return ""
        return "" if value is None else str(value).strip()
    return str(response_data).strip()


def _result_error(response_data: Any, req_id: int) -> str:
    if response_data is None:
        return "no response"
    if not isinstance(response_data, dict):
        return ""
    if "error" in response_data:
        return str(response_data.get("error") or "error")
    value = response_data.get(req_id)
    if value is None:
        value = response_data.get(str(req_id))
    if isinstance(value, dict) and "error" in value:
        return str(value.get("error") or "error")
    return ""


def _dashscope_key_pool() -> list[str]:
    return get_api_keys("qwen")


async def _run_annotation_batch(
    requests: list[dict[str, Any]],
    clients: list[UnifiedLLMClient],
) -> dict[int, Any]:
    async def _run_single(request: dict[str, Any]) -> tuple[int, Any]:
        req_id = int(request["request_id"])
        client_index = int(request.get("client_index", 0)) % len(clients)
        client = clients[client_index]
        try:
            response = await client.chat_completion(
                request["model"],
                request["messages"],
                **request.get("kwargs", {}),
            )
            return req_id, {req_id: response}
        except Exception as exc:
            return req_id, {req_id: {"error": str(exc)}}

    completed = await asyncio.gather(
        *(_run_single(request) for request in requests)
    )
    return {req_id: result for req_id, result in completed}


async def _retry_single_annotation(
    client: UnifiedLLMClient,
    node,
    *,
    model: str,
    reason: str,
) -> str:
    print(f"  kf_{node.kf_id}: {reason}, retrying once...")
    retry_request = {
        "request_id": node.kf_id,
        "model": model,
        "messages": vlm_single_image_request_message_kf(node),
    }
    retry_results = await client.batch_chat_completion([retry_request])
    retry_data = retry_results.get(node.kf_id)
    if isinstance(retry_data, dict) and "error" in retry_data:
        print(f"  kf_{node.kf_id}: retry failed - {retry_data['error']}")
        return ""
    return _extract_semantic(retry_data, node.kf_id)


def annotate(
    dataset_dir: Path,
    *,
    model: str = "qwen3-vl-plus",
    batch_size: int = 5,
    force: bool = False,
    compute_clip: bool = True,
    ids: list[int] | None = None,
) -> int:
    dataset_dir = dataset_dir.expanduser().resolve()
    print(f"Loading keyframes from {dataset_dir}")
    scene = SceneMemory(dataset_dir=dataset_dir, device=config.get("scene_memory", {}).get("device", "GPU"))

    nodes = list(scene.keyframe_nodes.values())
    if ids:
        id_set = set(ids)
        pending = [n for n in nodes if n.kf_id in id_set]
        if not pending:
            print(f"No keyframes matched the requested ids: {ids}")
            return 0
        print(f"Selective re-annotate: {len(pending)} of {len(nodes)} keyframes (ids={sorted(id_set)})")
    else:
        pending = [
            n for n in nodes if force or not n.semantic
        ]
        if not pending:
            print(f"All {len(nodes)} keyframes already have semantic descriptions.")
            return 0
        skipped = len(nodes) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-annotated keyframes (use --force to redo).")
    total = len(pending)
    batch_size = max(1, int(batch_size))
    print(f"Annotating {total} keyframes with model={model}, batch_size={batch_size}")

    ensure_api_key_env("qwen")

    clip_model, clip_preprocess, clip_device = None, None, None
    clip_text_encoder = None
    clip_lock = None
    if compute_clip:
        clip_model, clip_preprocess, clip_device = _try_load_torch_clip()
        if clip_model is not None:
            clip_lock = threading.Lock()
            print(f"CLIP model loaded on {clip_device} for text embedding.")
        else:
            clip_text_encoder, clip_device_ov = _try_load_openvino_clip_text_encoder(
                str(config.get("scene_memory", {}).get("device", "GPU"))
            )
            if clip_text_encoder is not None:
                clip_device = clip_device_ov
                print(f"OpenVINO CLIP text encoder loaded on {clip_device} for semantic text embedding (torch CLIP unavailable).")
            else:
                print("CLIP model unavailable, skipping semantic_clip_encoding.")

    key_pool = _dashscope_key_pool()
    if key_pool:
        print(f"Using DashScope API key pool: {len(key_pool)} key(s).")
        clients = [
            UnifiedLLMClient(
                api_key_overrides={"qwen": key},
                limiter_namespace=f"dashscope_key_{index}",
            )
            for index, key in enumerate(key_pool)
        ]
    else:
        clients = [UnifiedLLMClient()]
    annotated = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        requests = []
        client_index_by_req: dict[int, int] = {}
        for offset, node in enumerate(batch):
            messages = vlm_single_image_request_message_kf(node)
            client_index = (batch_start + offset) % len(clients)
            client_index_by_req[int(node.kf_id)] = client_index
            requests.append({
                "client_index": client_index,
                "request_id": node.kf_id,
                "model": model,
                "messages": messages,
            })

        try:
            results = asyncio.run(_run_annotation_batch(requests, clients))
        except Exception as exc:
            print(f"Batch [{batch_start}:{batch_start + len(batch)}] failed: {exc}")
            failed += len(batch)
            continue

        for node in batch:
            req_id = node.kf_id
            response_data = results.get(req_id)
            status = _result_error(response_data, req_id)
            if req_id not in results or status:
                status = status or "no response"
                print(f"  kf_{req_id}: FAILED — {status}")
                failed += 1
                continue

            semantic = _extract_semantic(response_data, req_id)
            if not semantic:
                try:
                    semantic = asyncio.run(
                        _retry_single_annotation(
                            clients[client_index_by_req.get(int(req_id), 0)],
                            node,
                            model=model,
                            reason="EMPTY response",
                        )
                    )
                except Exception as exc:
                    print(f"  kf_{req_id}: retry raised {exc}")
                    semantic = ""
            if not semantic:
                print(f"  kf_{req_id}: EMPTY response after retry, skipping.")
                failed += 1
                continue

            node.semantic = semantic

            if clip_text_encoder is not None:
                try:
                    node.semantic_clip_encoding = clip_text_encoder.encode_text(node.semantic)
                except Exception as exc:
                    print(f"  kf_{req_id}: OpenVINO CLIP encoding failed — {exc}")
            elif clip_model is not None:
                try:
                    import clip
                    import torch

                    with clip_lock:
                        tokens = clip.tokenize([node.semantic], truncate=True).to(clip_device)
                        with torch.no_grad():
                            text_features = clip_model.encode_text(tokens)
                            node.semantic_clip_encoding = text_features.squeeze(0).cpu().numpy()
                except Exception as exc:
                    print(f"  kf_{req_id}: CLIP encoding failed — {exc}")

            node.save_to_disk(dataset_dir / "constructed_memory" / "keyframe_nodes")
            annotated += 1
            print(f"  kf_{req_id}: OK ({annotated}/{total}) — {node.semantic[:80]}...")

    print(f"\nDone. Annotated {annotated}/{total} keyframes. failed={failed}")
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate selected CarAgent keyframes with VLM semantic descriptions."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the selected keyframe dataset (e.g. .../session_xxx/selected).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model name (default: from config vlm_model_get_semantic).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of VLM requests per batch (default: 5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if semantic already exists.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip computing semantic_clip_encoding.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated keyframe ids to annotate (overrides --force logic).",
    )
    args = parser.parse_args()

    model = args.model or config.get("vlm_model_get_semantic", "qwen3-vl-plus")
    ids = None
    if args.ids:
        ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    sys.exit(
        0
        if annotate(
            Path(args.dataset_dir),
            model=model,
            batch_size=args.batch_size,
            force=args.force,
            compute_clip=not args.skip_clip,
            ids=ids,
        )
        >= 0
        else 1
    )


if __name__ == "__main__":
    main()
