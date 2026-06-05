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

## Live Metric Depth Validation

Live ROS2 validation UI for tape-measure comparisons. Press `r` to capture a sample and run the full pipeline.

Requires camera + LiDAR nodes already running (e.g. via `caragent_full.launch.py`).

```bash
source /opt/ros/humble/setup.bash
cd ~/caragent_ws
source install/setup.bash

python3 -m caragent_agent.perception.fusion.live_scan_monodepth_validation \
  --target "door" \
  --label-query "door" \
  --truth-distance-m 2.00 \
  --grounding-device GPU \
  --depth-device GPU \
  --sam-device GPU \
  --sam-decoder-device CPU
```

### Controls

| Key | Action |
|-----|--------|
| `r` | Capture current frame + scan and run full pipeline |
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

Key columns: `mono_p10_m`, `mono_error_m`, `stereo_p10_m`, `stereo_error_m`, `stereo_status`, `selected_fit`, `fit_mae_m`, `fit_p90_m`.

Use `--no-enable-stereo` to skip stereo SGBM and test only scan + monocular depth.

To use a different output directory:

```bash
  --output-dir ~/caragent_ws/perception_outputs/chair_test
```

## Depth Edge Filtering

The scan-monodepth fit script filters out LiDAR correspondences that land on
strong mono-depth edges (Sobel gradient > 90th percentile, dilated 5px). These
correspondences are inherently unreliable — the LiDAR scanline may hit a
different surface than what the pixel sees (e.g. wall behind an object edge).

Filtered points are drawn purple in the projection overlay and recorded in
`depth_edge_filter` within the fit JSON. If too few points survive filtering,
the filter is skipped for that frame.
