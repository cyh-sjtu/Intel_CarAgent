#!/usr/bin/env python3
"""Recompute semantic_clip_encoding for all keyframes using torch CLIP."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Recompute CLIP encodings")
    parser.add_argument("--dataset-dir", required=True, help="Path to scene memory dataset")
    args = parser.parse_args()

    from caragent_agent.impression_graph.scene_memory import SceneMemory

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    sm = SceneMemory(dataset_dir=dataset_dir, device="GPU")

    if sm.clip_model is None:
        print("torch CLIP not available on this SceneMemory instance.")
        return

    total = len(sm.keyframe_nodes)
    print(f"Recomputing semantic_clip_encoding for {total} keyframes using torch CLIP on {sm.device}...")

    import clip
    import torch

    updated = 0
    nodes_dir = dataset_dir / "constructed_memory" / "keyframe_nodes"

    for kf_id, node in sm.keyframe_nodes.items():
        if not node.semantic:
            continue

        with torch.no_grad():
            tokens = clip.tokenize([node.semantic], truncate=True).to(sm.device)
            text_features = sm.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            node.semantic_clip_encoding = (
                text_features.squeeze(0).cpu().numpy().astype(np.float32)
            )

        node.save_to_disk(nodes_dir)
        updated += 1
        if updated % 10 == 0:
            print(f"  {updated}/{total}...")

    print(f"Done. Updated {updated}/{total} keyframes.")


if __name__ == "__main__":
    main()
