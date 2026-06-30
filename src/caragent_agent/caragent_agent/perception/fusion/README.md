# Perception Fusion

LiDAR-camera depth fusion: project 2D LaserScan into camera frame, fit a scale
model between monocular relative depth and LiDAR metric range, produce per-frame
absolute depth maps, and estimate object-level metric depth.

For the detailed object-localization geometry, calibration design, and stereo
fusion math, see
[`OBJECT_LOCALIZATION_TECHNICAL_NOTE.md`](OBJECT_LOCALIZATION_TECHNICAL_NOTE.md).

## Package structure

All inference and fusion scripts now live under `caragent_agent.perception`:

```
caragent_agent/perception/
  grounding/   — GroundingDINO open-vocab detection (OpenVINO + PyTorch)
  depth/       — Depth Anything V2 monocular depth (OpenVINO + PyTorch)
  sam/         — EfficientSAM box-prompted segmentation (OpenVINO)
  fusion/      — LiDAR-camera fusion (this package)
    project_scan_fit_monodepth.py    — frame-level scale fitting
    run_stereo_object_depth.py       — stereo SGBM object depth
    live_scan_monodepth_validation.py — live ROS2 validation UI
    fuse_scan_depth.py              — legacy angular-bearing heuristic
```

Calibration tools live in `caragent_vision`:

```
caragent_vision/
  collect_lidar_camera_correspondences.py — interactive click-to-correspond
  calibrate_lidar_camera_extrinsics.py    — Gauss-Newton optimization
  calibrate_stereo_camera.py              — stereo calibration
```

## Keyframe dataset

Use the rebuilt map/keyframe session:

```text
~/caragent_ws/keyframes/session_20260526_155459/selected
```

Recommended test frames for LiDAR + monocular-depth fusion:

- `000013`: elevator, corridor, plant, person, doors
- `000015`: elevator, corridor, doors, wall panels
- `000001`: chairs, pillar, box, open office furniture
- `000148`: pillars, open space, chairs/tables

Calibration files:

```text
~/caragent_ws/calibration/stereo_old/stereo_calibration.npz
~/caragent_ws/calibration/lidar_camera/lidar_camera_extrinsics_calibrated.json
```

## Full Pipeline (Dev Board)

检测 → 分割 → 单目深度 → LiDAR 投影 + 尺度拟合 → 绝对深度图

### Environment

```bash
source /opt/ros/humble/setup.bash
cd ~/caragent_ws
source install/setup.bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/caragent_ws/hf_cache
export HUGGINGFACE_HUB_CACHE=$HOME/caragent_ws/hf_cache/hub
export TRANSFORMERS_CACHE=$HOME/caragent_ws/hf_cache/transformers

OUTDIR=$HOME/caragent_ws/perception_outputs/test
mkdir -p $OUTDIR
```

### Pick a frame

```bash
SESSION=$(ls -dt $HOME/caragent_ws/keyframes/session_* | head -1)/selected
echo "SESSION=$SESSION"
ls $SESSION/left/ | head -10
```

### Step 1: Detection (GroundingDINO OpenVINO)

```bash
FRAME=000013

python3 -m caragent_agent.perception.grounding.run_grounding_dino_openvino \
  --image $SESSION/left/${FRAME}.png \
  --text "elevator . door . plant . person . chair . table ." \
  --model-dir $HOME/caragent_ws/models/grounding_dino_openvino \
  --model-id $HOME/caragent_ws/models/grounding-dino-tiny \
  --device GPU \
  --output-dir $OUTDIR
```

### Step 2: Segmentation (EfficientSAM OpenVINO)

```bash
LABEL="door"

python3 -m caragent_agent.perception.sam.run_efficientsam_openvino \
  --grounding-json $OUTDIR/${FRAME}_grounding_openvino.json \
  --label-query "$LABEL" \
  --encoder-xml $HOME/caragent_ws/models/efficient_sam_openvino/efficient_sam_vitt_encoder.xml \
  --decoder-xml $HOME/caragent_ws/models/efficient_sam_openvino/efficient_sam_vitt_decoder.xml \
  --device GPU \
  --decoder-device CPU \
  --output-dir $OUTDIR \
  --output-stem $FRAME
```

### Step 3: Monocular Depth (Depth Anything V2 OpenVINO)

```bash
python3 -m caragent_agent.perception.depth.run_depth_anything_openvino \
  --image $SESSION/left/${FRAME}.png \
  --model-dir $HOME/caragent_ws/models/depth_anything_v2_openvino \
  --device GPU \
  --output-dir $OUTDIR
```

### Step 4: LiDAR Projection + Scale Fitting

```bash
CALIB=$HOME/caragent_ws/calibration/stereo_old/stereo_calibration.npz
EXTRINSICS=$HOME/caragent_ws/calibration/lidar_camera/lidar_camera_extrinsics_calibrated.json

python3 -m caragent_agent.perception.fusion.project_scan_fit_monodepth \
  --image $SESSION/left/${FRAME}.png \
  --scan $SESSION/scan/${FRAME}_scan.npz \
  --mono-depth-npy $OUTDIR/${FRAME}_depth.npy \
  --calib-file $CALIB \
  --extrinsics-json $EXTRINSICS \
  --output-dir $OUTDIR/scan_monodepth_fit \
  --segmentation-json $OUTDIR/${FRAME}_segmentation_ov.json
```

`--segmentation-json` is optional; without it the script produces a full-frame metric depth map. With it, it also outputs object-level depth statistics within the mask.

Default fit strategy: `log,quadratic` with P90-first selection (10% tolerance) then MAE tiebreaker.

### Step 4 outputs

| File | Content |
|------|---------|
| `*_mono_metric_depth.npy` | Full-frame absolute depth map (meters) |
| `*_mono_metric_depth_color.png` | Color visualization |
| `*_scan_projected_to_image.png` | LiDAR projection overlay (inlier/outlier/edge-rejected) |
| `*_scan_monodepth_fit_plot.png` | Mono depth vs LiDAR distance scatter + fitted curves |
| `*_scan_monodepth_fit.json` | Fit parameters, errors, per-sample diagnostics |

## Live Object Localization Validation

Live ROS2 validation UI for tape-measure comparisons. Press `r` to capture a
sample and run the currently selected object-localization mode.

The UI supports three switchable modes:

| Mode | CLI value | Sensors required | Depth backend |
|------|-----------|------------------|---------------|
| Stereo | `stereo` | left image, right image | SGBM disparity inside the SAM mask |
| Mono relative + LiDAR | `mono_relative_lidar` | left image, LaserScan | Depth Anything V2 relative depth fitted to projected LiDAR anchors |
| Mono absolute | `mono_absolute` | left image | Depth Anything V2 Metric Indoor Small, directly in meters |

Requires camera nodes for all modes. The LiDAR node is only required by
`mono_relative_lidar`; the right camera is only required by `stereo`.

```bash
source /opt/ros/humble/setup.bash
cd ~/caragent_ws
source install/setup.bash

python3 -m caragent_agent.perception.fusion.live_scan_monodepth_validation \
  --target "door" \
  --label-query "door" \
  --truth-distance-m 2.00 \
  --localization-mode mono_relative_lidar \
  --grounding-device GPU \
  --depth-device GPU \
  --absolute-depth-device GPU \
  --sam-device GPU \
  --sam-decoder-device CPU
```

Start directly in absolute metric mode:

```bash
python3 -m caragent_agent.perception.fusion.live_scan_monodepth_validation \
  --target "chair" \
  --label-query "chair" \
  --truth-distance-m 2.00 \
  --localization-mode mono_absolute \
  --absolute-depth-model-dir ~/caragent_ws/models/depth_anything_v2_metric_indoor_small_openvino \
  --grounding-device GPU \
  --absolute-depth-device GPU \
  --sam-device GPU \
  --sam-decoder-device CPU
```

### Controls

| Key | Action |
|-----|--------|
| `m` | Cycle localization mode: stereo -> mono relative + LiDAR -> mono absolute |
| `r` | Capture current frame and run the selected mode |
| `t` | Change target prompt (GroundingDINO) |
| `l` | Change SAM label query |
| `d` | Change measured reference distance (meters) |
| `q` / `Esc` | Quit |

### Results

Appended to:

```text
~/caragent_ws/perception_outputs/scan_monodepth_validation/validation_results.csv
~/caragent_ws/perception_outputs/scan_monodepth_validation/validation_results.jsonl
```

Key columns:

| Column | Meaning |
|--------|---------|
| `localization_mode` | Which backend produced `recommended_depth_m` |
| `recommended_depth_m`, `recommended_error_m` | Selected mode's recommended object distance and error vs truth |
| `mono_p10_m`, `mono_error_m` | Mono relative + LiDAR result |
| `absolute_p10_m`, `absolute_error_m` | Metric Depth Anything result |
| `stereo_p10_m`, `stereo_error_m`, `stereo_status` | Stereo SGBM result |
| `selected_fit`, `fit_mae_m`, `fit_p90_m` | LiDAR scale-fit diagnostics for `mono_relative_lidar` |

Use `--enable-stereo-preview` if you want stereo to run as a side preview while
testing another selected mode. By default only the selected mode runs, which
keeps repeated tests faster.

To use a different output directory:

```bash
  --output-dir ~/caragent_ws/perception_outputs/chair_test
```

## Object Depth Benchmark Dataset

The dashboard provides a dedicated `Object Dataset` card for collecting a small
benchmark dataset and comparing object-depth backends quantitatively. It is
designed for quick controlled tests such as chair/door samples at known
distances.

Default dataset root:

```text
~/caragent_ws/perception_datasets/object_depth
```

Each dataset is stored as:

```text
<dataset_name>/
  dataset.json
  live_config.json
  manifest.jsonl
  samples/
    <sample_id>_left.png
    <sample_id>_right.png
    <sample_id>_scan.npz
    <sample_id>.json
  previews/
    <timestamp>_grounding.json
    <timestamp>_segmentation_ov.json
    <timestamp>_mask_overlay_ov.png
  evaluations/
```

### Collection Workflow

Open Dashboard -> `Object Dataset`, then:

1. Select an existing dataset to extend, or enter a new dataset name.
2. Set the GroundingDINO detection prompt, SAM label query, measured distance,
   and optional note.
3. Click `Start collector`.
4. In the OpenCV collector window:
   - `v`: validate the current frame with GroundingDINO + EfficientSAM and show
     the detection/segmentation overlay.
   - `c`: capture the current left/right image and LaserScan into the dataset.
   - `q`: quit the collector.
5. Change target/truth values in the dashboard and click `Update target/truth`
   before capturing the next distance or object class.

The validation overlay is intentionally part of the collection loop. If
GroundingDINO or SAM misses the target, do not capture that frame as a trusted
distance sample.

Manual collector command:

```bash
python3 -m caragent_agent.perception.fusion.collect_object_depth_dataset \
  --dataset-root ~/caragent_ws/perception_datasets/object_depth \
  --dataset-name chair_door_distance_v1 \
  --target "chair . door ." \
  --label-query "chair" \
  --truth-distance-m 2.00 \
  --grounding-device GPU \
  --sam-device GPU \
  --sam-decoder-device CPU
```

### Evaluation Workflow

The dashboard `Run evaluation` button runs the selected modes and writes a
cached comparison table. Existing sample/mode/model-config results are skipped
unless `Force rerun` is enabled.

Default modes:

```text
stereo,stereo_learned,mono_relative_lidar,mono_absolute
```

`stereo_learned` is a learned stereo-disparity backend intended for lightweight
models such as HITNet. Put an OpenVINO IR or ONNX model under:

```text
~/caragent_ws/models/hitnet_openvino
```

The evaluator looks for `openvino_model.xml`, `model.xml`, `model_float32.xml`,
`model_float16.xml`, `model_float32.onnx`, or `model.onnx`. It converts predicted
disparity to metric forward depth with the existing stereo calibration
(`depth = fx * baseline / disparity`) and reports the same object-mask statistics
as the SGBM stereo backend.

Evaluation outputs:

```text
<dataset_name>/evaluations/<run_name>/
  summary.csv
  summary.jsonl
  cache_index.json
  command_logs/
  <sample_id>/<mode>/result.json
```

`summary.csv` is the main comparison file. Key columns:

| Column | Meaning |
|--------|---------|
| `sample_id` | Captured sample id |
| `mode` | Evaluated backend |
| `truth_distance_m` | Manually measured reference distance |
| `recommended_depth_m` | Model-estimated object distance |
| `error_m`, `abs_error_m` | Signed and absolute error vs reference |
| `p05_m`, `p10_m`, `median_m`, `p90_m` | Robust object-mask depth statistics |
| `config_hash` | Cache key for model/config compatibility |
| `cached` | Whether this row was reused from cache |

Manual evaluation command:

```bash
python3 -m caragent_agent.perception.fusion.evaluate_object_depth_dataset \
  --dataset-dir ~/caragent_ws/perception_datasets/object_depth/chair_door_distance_v1 \
  --run-name baseline \
  --modes stereo,stereo_learned,mono_relative_lidar,mono_absolute \
  --grounding-device GPU \
  --depth-device GPU \
  --absolute-depth-device GPU \
  --learned-stereo-device GPU \
  --sam-device GPU \
  --sam-decoder-device CPU
```

## Depth Edge Filtering

The scan-monodepth fit script filters out LiDAR correspondences that land on
strong mono-depth edges (Sobel gradient > 90th percentile, dilated 5px). These
correspondences are inherently unreliable — the LiDAR scanline may hit a
different surface than what the pixel sees (e.g. wall behind an object edge).

Filtered points are drawn purple in the projection overlay and recorded in
`depth_edge_filter` within the fit JSON. If too few points survive filtering,
the filter is skipped for that frame.
