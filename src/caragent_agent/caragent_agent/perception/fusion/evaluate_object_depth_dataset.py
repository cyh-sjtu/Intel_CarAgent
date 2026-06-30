"""Evaluate object-depth models on a collected benchmark dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE = Path.home() / "caragent_ws"
DEFAULT_CALIB = DEFAULT_WORKSPACE / "calibration" / "stereo_current" / "stereo_calibration.npz"
DEFAULT_EXTR = DEFAULT_WORKSPACE / "calibration" / "lidar_camera" / "lidar_camera_extrinsics_calibrated.json"
DEFAULT_GROUNDING_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "grounding_dino_openvino"
DEFAULT_GROUNDING_MODEL_ID = DEFAULT_WORKSPACE / "models" / "grounding-dino-tiny"
DEFAULT_REL_DEPTH_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "depth_anything_v2_openvino"
DEFAULT_ABS_DEPTH_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "depth_anything_v2_metric_indoor_small_openvino"
DEFAULT_SAM_ENCODER_XML = DEFAULT_WORKSPACE / "models" / "efficient_sam_openvino" / "efficient_sam_vitt_encoder.xml"
DEFAULT_SAM_DECODER_XML = DEFAULT_WORKSPACE / "models" / "efficient_sam_openvino" / "efficient_sam_vitt_decoder.xml"
DEFAULT_LEARNED_STEREO_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "hitnet_openvino"

MODE_CHOICES = ("stereo", "stereo_learned", "mono_relative_lidar", "mono_absolute")


def read_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def first_float(data: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            return value_f
    return None


def format_prompt(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    terms = [term.strip() for term in normalized.replace(";", ".").split(".") if term.strip()]
    if not terms:
        terms = [normalized]
    return " . ".join(terms) + " ."


def config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def file_fingerprint(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha1": digest.hexdigest()[:16],
    }


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> None:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(cmd)
        + f"\n\nRETURN_CODE: {proc.returncode}\nELAPSED_SEC: {elapsed:.3f}\n\nSTDOUT:\n"
        + proc.stdout
        + "\n\nSTDERR:\n"
        + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed, see {log_path}")


def read_detection_count(grounding_json: Path) -> int | None:
    if not grounding_json.exists():
        return None
    try:
        payload = json.loads(grounding_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    detections = payload.get("detections")
    return len(detections) if isinstance(detections, list) else None


class DatasetEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_dir = args.dataset_dir.resolve()
        self.output_dir = self.dataset_dir / "evaluations" / args.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_csv = self.output_dir / "summary.csv"
        self.summary_jsonl = self.output_dir / "summary.jsonl"
        self.metrics_csv = self.output_dir / "metrics_by_mode.csv"
        self.metrics_json = self.output_dir / "metrics_by_mode.json"
        self.command_logs = self.output_dir / "command_logs"
        self.command_logs.mkdir(exist_ok=True)
        self.cache_index_path = self.output_dir / "cache_index.json"
        self.cache_index = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_index_path.exists():
            return {}
        try:
            return json.loads(self.cache_index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_index_path.write_text(
            json.dumps(self.cache_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def evaluate(self) -> None:
        samples = read_manifest(self.dataset_dir)
        rows = []
        existing_jsonl = []
        if self.summary_jsonl.exists():
            existing_jsonl = [line for line in self.summary_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        for sample in samples:
            for mode in self.args.modes:
                row = self.evaluate_one(sample, mode)
                rows.append(row)
                if row.get("new_result"):
                    with self.summary_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({k: v for k, v in row.items() if k != "new_result"}, ensure_ascii=False) + "\n")
        all_rows = [json.loads(line) for line in existing_jsonl]
        all_rows.extend({k: v for k, v in row.items() if k != "new_result"} for row in rows)
        all_rows = self._dedupe_rows(all_rows)
        self.write_csv(all_rows)
        self.write_metrics(all_rows)
        print(f"summary_csv: {self.summary_csv}")
        print(f"summary_jsonl: {self.summary_jsonl}")
        print(f"metrics_csv: {self.metrics_csv}")

    @staticmethod
    def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
        order: list[tuple[str, str, str]] = []
        for row in rows:
            key = (
                str(row.get("sample_id", "")),
                str(row.get("mode", "")),
                str(row.get("config_hash", "")),
            )
            if key not in deduped:
                order.append(key)
            deduped[key] = row
        return [deduped[key] for key in order]

    def evaluate_one(self, sample: dict[str, Any], mode: str) -> dict[str, Any]:
        sample_id = str(sample["sample_id"])
        target = str(sample.get("target") or self.args.target or "chair")
        label = str(sample.get("label_query") or self.args.label_query or target)
        truth = sample.get("truth_distance_m")
        truth_f = None if truth in {"", None} else float(truth)
        sample_dir = self.output_dir / sample_id / mode
        sample_dir.mkdir(parents=True, exist_ok=True)
        key_payload = {
            "sample_id": sample_id,
            "mode": mode,
            "target": target,
            "label": label,
            "truth": truth_f,
            "models": {
                "grounding": str(self.args.grounding_model_dir),
                "grounding_id": str(self.args.grounding_model_id),
                "relative_depth": str(self.args.depth_model_dir),
                "absolute_depth": str(self.args.absolute_depth_model_dir),
                "learned_stereo": str(self.args.learned_stereo_model_dir),
                "learned_stereo_file": str(self.args.learned_stereo_model_file),
                "learned_stereo_type": str(self.args.learned_stereo_model_type),
                "sam_encoder": str(self.args.sam_encoder_xml),
                "sam_decoder": str(self.args.sam_decoder_xml),
            },
            "calibration": {
                "calib_file": file_fingerprint(self.args.calib_file),
                "extrinsics_json": file_fingerprint(self.args.extrinsics_json),
            },
            "stereo_params": {
                "num_disparities": self.args.stereo_num_disparities,
                "block_size": self.args.stereo_block_size,
                "min_depth": self.args.stereo_min_depth,
                "max_depth": self.args.stereo_max_depth,
            },
        }
        key = config_hash(key_payload)
        cache_key = f"{sample_id}:{mode}:{key}"
        cached = self.cache_index.get(cache_key)
        if cached and Path(cached.get("result_json", "")).exists() and not self.args.force:
            row = json.loads(Path(cached["result_json"]).read_text(encoding="utf-8"))
            row["cached"] = True
            row["new_result"] = False
            print(f"skip cached {sample_id} {mode}")
            return row

        row = self._run_one(sample, sample_dir, mode, target, label, truth_f, key_payload)
        row["new_result"] = True
        result_json = sample_dir / "result.json"
        result_json.write_text(
            json.dumps({k: v for k, v in row.items() if k != "new_result"}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.cache_index[cache_key] = {"result_json": str(result_json), "updated_at": datetime.now().isoformat(timespec="seconds")}
        self._save_cache()
        return row

    def _run_one(
        self,
        sample: dict[str, Any],
        sample_dir: Path,
        mode: str,
        target: str,
        label: str,
        truth: float | None,
        key_payload: dict[str, Any],
    ) -> dict[str, Any]:
        left = Path(sample["left_image"])
        right_text = str(sample.get("right_image") or "")
        scan_text = str(sample.get("scan") or "")
        right = Path(right_text) if right_text else None
        scan = Path(scan_text) if scan_text else None
        sample_id = str(sample["sample_id"])

        grounding_json = sample_dir / f"{left.stem}_grounding_openvino.json"
        seg_json = sample_dir / f"{sample_id}_segmentation_ov.json"
        if not seg_json.exists() or self.args.force:
            grounding_log = self.command_logs / f"{sample_id}_{mode}_grounding.log"
            try:
                run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "caragent_agent.perception.grounding.run_grounding_dino_openvino",
                        "--image",
                        str(left),
                        "--text",
                        format_prompt(target),
                        "--model-dir",
                        str(self.args.grounding_model_dir),
                        "--model-id",
                        str(self.args.grounding_model_id),
                        "--device",
                        self.args.grounding_device,
                        "--output-dir",
                        str(sample_dir),
                    ],
                    self.args.workspace,
                    grounding_log,
                )
            except RuntimeError:
                return self._base_row(
                    sample,
                    mode,
                    target,
                    label,
                    truth,
                    status="grounding_failed",
                    key_payload=key_payload,
                    source_json=str(grounding_log),
                )
            detection_count = read_detection_count(grounding_json)
            if detection_count == 0:
                return self._base_row(
                    sample,
                    mode,
                    target,
                    label,
                    truth,
                    status="no_detection",
                    key_payload=key_payload,
                    source_json=str(grounding_json),
                )
            sam_log = self.command_logs / f"{sample_id}_{mode}_sam.log"
            try:
                run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "caragent_agent.perception.sam.run_efficientsam_openvino",
                        "--grounding-json",
                        str(grounding_json),
                        "--label-query",
                        label,
                        "--encoder-xml",
                        str(self.args.sam_encoder_xml),
                        "--decoder-xml",
                        str(self.args.sam_decoder_xml),
                        "--device",
                        self.args.sam_device,
                        "--decoder-device",
                        self.args.sam_decoder_device,
                        "--output-dir",
                        str(sample_dir),
                        "--output-stem",
                        sample_id,
                    ],
                    self.args.workspace,
                    sam_log,
                )
            except RuntimeError:
                return self._base_row(
                    sample,
                    mode,
                    target,
                    label,
                    truth,
                    status="segmentation_failed",
                    key_payload=key_payload,
                    source_json=str(sam_log),
                )

        if mode == "stereo":
            if right is None or not right.exists():
                return self._base_row(sample, mode, target, label, truth, status="missing_right_image", key_payload=key_payload)
            stereo_log = self.command_logs / f"{sample_id}_{mode}.log"
            try:
                run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "caragent_agent.perception.fusion.run_stereo_object_depth",
                        "--left-image",
                        str(left),
                        "--right-image",
                        str(right),
                        "--segmentation-json",
                        str(seg_json),
                        "--calib-file",
                        str(self.args.calib_file),
                        "--output-dir",
                        str(sample_dir),
                        "--num-disparities",
                        str(self.args.stereo_num_disparities),
                        "--block-size",
                        str(self.args.stereo_block_size),
                        "--min-depth",
                        str(self.args.stereo_min_depth),
                        "--max-depth",
                        str(self.args.stereo_max_depth),
                    ],
                    self.args.workspace,
                    stereo_log,
                )
                stereo_json = sample_dir / f"{left.stem}_stereo_object_3d.json"
                payload = json.loads(stereo_json.read_text(encoding="utf-8"))
                payload["json_path"] = str(stereo_json)
            except Exception:
                return self._base_row(
                    sample,
                    mode,
                    target,
                    label,
                    truth,
                    status="stereo_failed",
                    key_payload=key_payload,
                    source_json=str(stereo_log),
                )
            stats = ((payload.get("object_camera_project") or {}).get("stats") or {}).get("x_forward_m") or {}
            mask = payload.get("mask") or {}
            return self._row_from_stats(sample, mode, target, label, truth, stats, "ok", payload, key_payload, mask)

        if mode == "stereo_learned":
            if right is None or not right.exists():
                return self._base_row(sample, mode, target, label, truth, status="missing_right_image", key_payload=key_payload)
            cmd = [
                sys.executable,
                "-m",
                "caragent_agent.perception.fusion.run_learned_stereo_object_depth",
                "--left-image",
                str(left),
                "--right-image",
                str(right),
                "--segmentation-json",
                str(seg_json),
                "--calib-file",
                str(self.args.calib_file),
                "--output-dir",
                str(sample_dir),
                "--model-dir",
                str(self.args.learned_stereo_model_dir),
                "--device",
                self.args.learned_stereo_device,
                "--model-type",
                self.args.learned_stereo_model_type,
                "--min-depth",
                str(self.args.stereo_min_depth),
                "--max-depth",
                str(self.args.stereo_max_depth),
            ]
            if str(self.args.learned_stereo_model_file):
                cmd.extend(["--model-file", str(self.args.learned_stereo_model_file)])
            learned_log = self.command_logs / f"{sample_id}_{mode}.log"
            try:
                run_cmd(cmd, self.args.workspace, learned_log)
                learned_json = sample_dir / f"{left.stem}_learned_stereo_object_3d.json"
                payload = json.loads(learned_json.read_text(encoding="utf-8"))
                payload["json_path"] = str(learned_json)
            except Exception:
                return self._base_row(
                    sample,
                    mode,
                    target,
                    label,
                    truth,
                    status="learned_stereo_failed",
                    key_payload=key_payload,
                    source_json=str(learned_log),
                )
            stats = ((payload.get("object_camera_project") or {}).get("stats") or {}).get("x_forward_m") or {}
            mask = payload.get("mask") or {}
            return self._row_from_stats(sample, mode, target, label, truth, stats, "ok", payload, key_payload, mask)

        depth_model = self.args.depth_model_dir if mode == "mono_relative_lidar" else self.args.absolute_depth_model_dir
        depth_out = sample_dir / "depth"
        depth_log = self.command_logs / f"{sample_id}_{mode}_depth.log"
        try:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "caragent_agent.perception.depth.run_depth_anything_openvino",
                    "--image",
                    str(left),
                    "--model-dir",
                    str(depth_model),
                    "--device",
                    self.args.depth_device if mode == "mono_relative_lidar" else self.args.absolute_depth_device,
                    "--output-dir",
                    str(depth_out),
                ],
                self.args.workspace,
                depth_log,
            )
        except RuntimeError:
            return self._base_row(
                sample,
                mode,
                target,
                label,
                truth,
                status="depth_failed",
                key_payload=key_payload,
                source_json=str(depth_log),
            )
        depth_npy = depth_out / f"{left.stem}_depth.npy"
        if mode == "mono_absolute":
            try:
                from caragent_agent.perception.fusion.project_scan_fit_monodepth import load_optional_mask, robust_depth_stats

                import numpy as np
                from PIL import Image

                image = Image.open(left).convert("RGB")
                depth = np.load(depth_npy).astype(np.float32)
                mask = load_optional_mask(seg_json, (image.width, image.height))
                stats = robust_depth_stats(depth, mask) if mask is not None else {}
            except Exception:
                return self._base_row(
                    sample,
                    mode,
                    target,
                    label,
                    truth,
                    status="absolute_depth_stats_failed",
                    key_payload=key_payload,
                    source_json=str(depth_npy),
                )
            payload = {
                "mode": mode,
                "depth_npy": str(depth_npy),
                "object_mask_metric_depth_m": stats,
                "mask": {"valid_ratio": None},
            }
            return self._row_from_stats(sample, mode, target, label, truth, stats, "ok", payload, key_payload, {})

        if scan is None or not scan.exists():
            return self._base_row(sample, mode, target, label, truth, status="missing_scan", key_payload=key_payload)
        fit_out = sample_dir / "fit"
        fit_log = self.command_logs / f"{sample_id}_{mode}_fit.log"
        try:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "caragent_agent.perception.fusion.project_scan_fit_monodepth",
                    "--image",
                    str(left),
                    "--scan",
                    str(scan),
                    "--mono-depth-npy",
                    str(depth_npy),
                    "--calib-file",
                    str(self.args.calib_file),
                    "--extrinsics-json",
                    str(self.args.extrinsics_json),
                    "--segmentation-json",
                    str(seg_json),
                    "--output-dir",
                    str(fit_out),
                ],
                self.args.workspace,
                fit_log,
            )
            fit_json = fit_out / f"{left.stem}_scan_monodepth_fit.json"
            payload = json.loads(fit_json.read_text(encoding="utf-8"))
            payload["json_path"] = str(fit_json)
        except Exception:
            return self._base_row(
                sample,
                mode,
                target,
                label,
                truth,
                status="fit_failed",
                key_payload=key_payload,
                source_json=str(fit_log),
            )
        stats = payload.get("object_mask_metric_depth_m") or {}
        return self._row_from_stats(sample, mode, target, label, truth, stats, "ok", payload, key_payload, {})

    def _base_row(
        self,
        sample: dict[str, Any],
        mode: str,
        target: str,
        label: str,
        truth: float | None,
        status: str,
        key_payload: dict[str, Any],
        source_json: str = "",
    ) -> dict[str, Any]:
        return {
            "sample_id": sample["sample_id"],
            "mode": mode,
            "target": target,
            "label_query": label,
            "truth_distance_m": truth,
            "recommended_depth_m": None,
            "error_m": None,
            "abs_error_m": None,
            "status": status,
            "config_hash": config_hash(key_payload),
            "cached": False,
            "source_json": source_json,
        }

    def _row_from_stats(
        self,
        sample: dict[str, Any],
        mode: str,
        target: str,
        label: str,
        truth: float | None,
        stats: dict[str, Any],
        status: str,
        payload: dict[str, Any],
        key_payload: dict[str, Any],
        mask: dict[str, Any],
    ) -> dict[str, Any]:
        rec = first_float(stats, ["p10", "p05", "median"])
        err = None if rec is None or truth is None else float(rec) - float(truth)
        row = self._base_row(sample, mode, target, label, truth, status, key_payload)
        row.update(
            {
                "recommended_depth_m": rec,
                "error_m": err,
                "abs_error_m": abs(err) if err is not None else None,
                "p05_m": first_float(stats, ["p05"]),
                "p10_m": first_float(stats, ["p10"]),
                "median_m": first_float(stats, ["median"]),
                "p90_m": first_float(stats, ["p90", "p95"]),
                "valid_ratio": mask.get("valid_ratio"),
                "payload_json": str((self.output_dir / str(sample["sample_id"]) / mode / "result.json")),
                "source_json": payload.get("json_path") or "",
                "cached": False,
            }
        )
        return row

    def write_csv(self, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "sample_id",
            "mode",
            "target",
            "label_query",
            "truth_distance_m",
            "recommended_depth_m",
            "error_m",
            "abs_error_m",
            "p05_m",
            "p10_m",
            "median_m",
            "p90_m",
            "valid_ratio",
            "status",
            "config_hash",
            "cached",
            "payload_json",
            "source_json",
        ]
        with self.summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def write_metrics(self, rows: list[dict[str, Any]]) -> None:
        metrics = []
        modes = sorted({str(row.get("mode") or "") for row in rows if row.get("mode")})
        for mode in modes:
            mode_rows = [row for row in rows if row.get("mode") == mode]
            ok_rows = [
                row
                for row in mode_rows
                if row.get("status") == "ok" and row.get("abs_error_m") not in {"", None}
            ]
            abs_errors = [float(row["abs_error_m"]) for row in ok_rows]
            signed_errors = [float(row["error_m"]) for row in ok_rows if row.get("error_m") not in {"", None}]
            if abs_errors:
                abs_sorted = sorted(abs_errors)
                mid = len(abs_sorted) // 2
                median_abs = abs_sorted[mid] if len(abs_sorted) % 2 else (abs_sorted[mid - 1] + abs_sorted[mid]) / 2.0
                p90_abs = abs_sorted[min(len(abs_sorted) - 1, int(math.ceil(0.9 * len(abs_sorted)) - 1))]
                mean_abs = sum(abs_errors) / len(abs_errors)
                rmse = math.sqrt(sum(err * err for err in signed_errors) / len(signed_errors)) if signed_errors else None
                mean_error = sum(signed_errors) / len(signed_errors) if signed_errors else None
            else:
                median_abs = p90_abs = mean_abs = rmse = mean_error = None
            metrics.append(
                {
                    "mode": mode,
                    "n_total": len(mode_rows),
                    "n_ok": len(ok_rows),
                    "mean_abs_error_m": mean_abs,
                    "median_abs_error_m": median_abs,
                    "p90_abs_error_m": p90_abs,
                    "rmse_m": rmse,
                    "mean_error_m": mean_error,
                }
            )
        with self.metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "mode",
                "n_total",
                "n_ok",
                "mean_abs_error_m",
                "median_abs_error_m",
                "p90_abs_error_m",
                "rmse_m",
                "mean_error_m",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)
        self.metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an object-depth benchmark dataset.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--modes", default="stereo,mono_relative_lidar,mono_absolute")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target", default="")
    parser.add_argument("--label-query", default="")
    parser.add_argument("--calib-file", default=DEFAULT_CALIB, type=Path)
    parser.add_argument("--extrinsics-json", default=DEFAULT_EXTR, type=Path)
    parser.add_argument("--grounding-model-dir", default=DEFAULT_GROUNDING_MODEL_DIR, type=Path)
    parser.add_argument("--grounding-model-id", default=DEFAULT_GROUNDING_MODEL_ID)
    parser.add_argument("--grounding-device", default="GPU")
    parser.add_argument("--depth-model-dir", default=DEFAULT_REL_DEPTH_MODEL_DIR, type=Path)
    parser.add_argument("--depth-device", default="GPU")
    parser.add_argument("--absolute-depth-model-dir", default=DEFAULT_ABS_DEPTH_MODEL_DIR, type=Path)
    parser.add_argument("--absolute-depth-device", default="GPU")
    parser.add_argument("--learned-stereo-model-dir", default=DEFAULT_LEARNED_STEREO_MODEL_DIR, type=Path)
    parser.add_argument("--learned-stereo-model-file", default="", type=Path)
    parser.add_argument("--learned-stereo-device", default="GPU")
    parser.add_argument("--learned-stereo-model-type", default="eth3d", choices=["eth3d", "middlebury", "flyingthings"])
    parser.add_argument("--sam-device", default="GPU")
    parser.add_argument("--sam-decoder-device", default="CPU")
    parser.add_argument("--sam-encoder-xml", default=DEFAULT_SAM_ENCODER_XML, type=Path)
    parser.add_argument("--sam-decoder-xml", default=DEFAULT_SAM_DECODER_XML, type=Path)
    parser.add_argument("--stereo-num-disparities", default=96, type=int)
    parser.add_argument("--stereo-block-size", default=5, type=int)
    parser.add_argument("--stereo-min-depth", default=0.15, type=float)
    parser.add_argument("--stereo-max-depth", default=8.0, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.calib_file = args.calib_file.expanduser().resolve()
    args.extrinsics_json = args.extrinsics_json.expanduser().resolve()
    args.grounding_model_dir = args.grounding_model_dir.expanduser().resolve()
    args.depth_model_dir = args.depth_model_dir.expanduser().resolve()
    args.absolute_depth_model_dir = args.absolute_depth_model_dir.expanduser().resolve()
    args.learned_stereo_model_dir = args.learned_stereo_model_dir.expanduser().resolve()
    args.learned_stereo_model_file = Path(str(args.learned_stereo_model_file)).expanduser()
    if str(args.learned_stereo_model_file):
        args.learned_stereo_model_file = args.learned_stereo_model_file.resolve()
    args.sam_encoder_xml = args.sam_encoder_xml.expanduser().resolve()
    args.sam_decoder_xml = args.sam_decoder_xml.expanduser().resolve()
    args.modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    unknown = sorted(set(args.modes) - set(MODE_CHOICES))
    if unknown:
        raise ValueError(f"unsupported modes: {unknown}; choices={MODE_CHOICES}")
    DatasetEvaluator(args).evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
