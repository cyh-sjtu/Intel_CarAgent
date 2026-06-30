import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import numpy as np
from pathlib import Path

DATASET = Path(r"D:\CarAgent\ros2\caragent_ws\keyframes\session_20260524_005910")
CLIP_DIR = r"D:\CarAgent\model\clip"
DINO_DIR = r"D:\CarAgent\model\dinov2-small"

images = sorted((DATASET / "selected" / "left").glob("*.png"))
sample_n = min(30, len(images))
sample_images = [str(p) for p in images[:sample_n]]
sample_ids = [Path(p).stem for p in sample_images]

import torch
from PIL import Image

print(f"Comparing {sample_n} frames\n")

# ====== CLIP ======
print("=== PyTorch CLIP ===")
from transformers import CLIPModel, CLIPImageProcessor

full_model = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True)
full_model.eval()

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = full_model.vision_model
        self.proj = full_model.visual_projection
    def forward(self, x):
        return self.proj(self.vision(x).pooler_output)

clip_enc = ImageEncoder().eval()
clip_proc = CLIPImageProcessor.from_pretrained(CLIP_DIR, local_files_only=True)

def clip_encode(p):
    img = Image.open(p).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_enc(inputs["pixel_values"]).numpy().reshape(-1)
    return emb / np.linalg.norm(emb)

clip_embs = {}
for i, (fid, p) in enumerate(zip(sample_ids, sample_images)):
    clip_embs[fid] = clip_encode(p)
    if (i + 1) % 10 == 0:
        print(f"  CLIP: {i+1}/{sample_n}")

# ====== DINOv2 ======
print("=== DINOv2-small ===")
from transformers import Dinov2Model, AutoImageProcessor

dino_model = Dinov2Model.from_pretrained(DINO_DIR, local_files_only=True)
dino_model.eval()
dino_proc = AutoImageProcessor.from_pretrained(DINO_DIR, local_files_only=True)

def dino_encode(p):
    img = Image.open(p).convert("RGB")
    inputs = dino_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = dino_model(**inputs)
        emb = out.last_hidden_state[:, 0, :].numpy().reshape(-1)
    return emb / np.linalg.norm(emb)

dino_embs = {}
for i, (fid, p) in enumerate(zip(sample_ids, sample_images)):
    dino_embs[fid] = dino_encode(p)
    if (i + 1) % 10 == 0:
        print(f"  DINOv2: {i+1}/{sample_n}")

# ====== Stats ======
def stats(embs):
    ids_ = sorted(embs.keys())
    vecs = np.stack([embs[k] for k in ids_])
    sims = np.dot(vecs, vecs.T)[np.triu_indices(len(ids_), k=1)]
    return {k: float(getattr(np, k)(sims)) for k in ["min", "max", "mean", "median", "std"]}

clip_s = stats(clip_embs)
dino_s = stats(dino_embs)

print(f"\n=== Similarity distribution ({sample_n} frames) ===")
print(f"{'':12s}  {'CLIP ViT-B/32':>14s}  {'DINOv2-small':>14s}  {'Delta':>10s}")
for k in ["min", "max", "mean", "median", "std"]:
    delta = dino_s[k] - clip_s[k]
    print(f"{k:12s}  {clip_s[k]:14.4f}  {dino_s[k]:14.4f}  {delta:+10.4f}")

print()
print(f"Separation quality (min similarity, lower = better):")
print(f"  CLIP:   {clip_s['min']:.4f}")
print(f"  DINOv2: {dino_s['min']:.4f}")

# 5 most-different pairs
ids_ = sorted(clip_embs.keys())
cv = np.stack([clip_embs[k] for k in ids_])
dv = np.stack([dino_embs[k] for k in ids_])
triu = np.triu_indices(len(ids_), k=1)
cf = np.dot(cv, cv.T)[triu]
df = np.dot(dv, dv.T)[triu]

print("\n=== 5 most different pairs ===")
for i in range(5):
    ci = np.argsort(cf)[i]
    di = np.argsort(df)[i]
    print(f"CLIP  {ids_[triu[0][ci]]} vs {ids_[triu[1][ci]]}: {cf[ci]:.4f}     DINOv2 {ids_[triu[0][di]]} vs {ids_[triu[1][di]]}: {df[di]:.4f}")
