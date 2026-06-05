"""Filesystem helpers for CarAgent keyframe datasets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DATASET_DIRS = ("raw", "left", "right", "pose", "scan", "meta")


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    raw_path: Path
    left_path: Path
    right_path: Optional[Path]
    pose_path: Path
    meta_path: Path
    scan_path: Optional[Path]
    pose: dict
    meta: dict


def ensure_candidate_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in DATASET_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def resolve_record(dataset_root: Path, manifest_item: dict) -> FrameRecord:
    frame_id = str(manifest_item["frame_id"])
    pose_path = (dataset_root / manifest_item["pose_path"]).resolve()
    meta_path = (dataset_root / manifest_item["meta_path"]).resolve()
    raw_path = (dataset_root / manifest_item["raw_path"]).resolve()
    left_path = (dataset_root / manifest_item["left_path"]).resolve()
    right_value = manifest_item.get("right_path")
    scan_value = manifest_item.get("scan_path")
    right_path = (dataset_root / right_value).resolve() if right_value else None
    scan_path = (dataset_root / scan_value).resolve() if scan_value else None
    return FrameRecord(
        frame_id=frame_id,
        raw_path=raw_path,
        left_path=left_path,
        right_path=right_path,
        pose_path=pose_path,
        meta_path=meta_path,
        scan_path=scan_path,
        pose=read_json(pose_path),
        meta=read_json(meta_path),
    )


def copy_record_assets(record: FrameRecord, output_root: Path) -> dict:
    """Copy one frame record into a selected dataset and return manifest fields."""

    for dirname in DATASET_DIRS:
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    frame_id = record.frame_id
    copied = {
        "frame_id": frame_id,
        "raw_path": f"raw/{frame_id}.png",
        "left_path": f"left/{frame_id}.png",
        "right_path": f"right/{frame_id}.png" if record.right_path is not None else None,
        "pose_path": f"pose/{frame_id}_pose.json",
        "meta_path": f"meta/{frame_id}_meta.json",
        "scan_path": f"scan/{frame_id}_scan.npz" if record.scan_path is not None else None,
    }

    shutil.copy2(record.raw_path, output_root / copied["raw_path"])
    shutil.copy2(record.left_path, output_root / copied["left_path"])
    if record.right_path is not None:
        shutil.copy2(record.right_path, output_root / copied["right_path"])
    shutil.copy2(record.pose_path, output_root / copied["pose_path"])
    shutil.copy2(record.meta_path, output_root / copied["meta_path"])
    if record.scan_path is not None:
        shutil.copy2(record.scan_path, output_root / copied["scan_path"])

    return copied


def iter_frame_records(dataset_root: Path) -> Iterable[FrameRecord]:
    for item in load_manifest(dataset_root / "manifest.jsonl"):
        yield resolve_record(dataset_root, item)
