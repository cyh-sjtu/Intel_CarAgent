# Object Localization From RGB, Mask, Relative Depth, Stereo, and LaserScan

This note documents the CarAgent object-localization pipeline. The goal is to
estimate a target object's metric 3D position from:

- an RGB image (left stereo),
- an open-vocabulary detection box (GroundingDINO),
- an EfficientSAM mask,
- Depth Anything V2 relative depth,
- a 2D LaserScan snapshot,
- calibrated stereo SGBM depth (cross-validation),
- and calibrated sensor extrinsics.

The pipeline has matured from an angular-bearing heuristic into a full
calibrated projection + multi-curve scale fitting + depth-edge filtering stack.

## Files

| Script | Role |
|--------|------|
| `project_scan_fit_monodepth.py` | Core: project LiDAR into camera, fit mono→metric scale, produce metric depth map |
| `run_stereo_object_depth.py` | Stereo SGBM object depth (cross-validation) |
| `live_scan_monodepth_validation.py` | Live ROS2 UI for tape-measure benchmarking |
| `fuse_scan_depth.py` | Legacy angular-bearing heuristic (useful for quick single-beam diagnostics) |

Typical output files:

```text
outputs/000123_scan_projected_to_image.png   — LiDAR projection overlay
outputs/000123_scan_monodepth_fit_plot.png   — scatter + fitted curves
outputs/000123_mono_metric_depth.npy         — full-frame absolute depth (meters)
outputs/000123_mono_metric_depth_color.png   — colorized depth
outputs/000123_scan_monodepth_fit.json       — fit params, errors, diagnostics
outputs/000123_stereo_object_3d.json         — stereo SGBM object depth
outputs/000123_stereo_object_overlay.png     — stereo rectified mask overlay
```

## Coordinate Frames

The URDF defines `base_link` as:

```text
+X = robot forward
+Y = robot left
+Z = up
```

Key sensor poses (from `lidar_camera_extrinsics_calibrated.json`):

```text
base_link → laser:
  translation ≈ (0.12, 0.00, 0.30) m
  yaw ≈ π rad  (LiDAR angle zero points opposite robot forward)

base_link → camera_left:
  translation ≈ (0.30, 0.03, 0.185) m
  rpy ≈ (0, 0, 0)
```

Camera optical frame (OpenCV convention):

```text
+X optical = image right
+Y optical = image down
+Z optical = forward
```

Conversion from project-style camera (+X forward, +Y left, +Z up) to optical:

```text
x_opt = -y_project
y_opt = -z_project
z_opt =  x_project
```

## Calibrated Projection Pipeline

The core script `project_scan_fit_monodepth.py` replaces the old FOV-approximated
angular heuristic with a full calibrated projection. For every valid LaserScan
point:

1. **LiDAR → base_link**: rotate by `rodrigues(laser_rpy)`, translate by `(laser_x, laser_y, laser_z)`.

2. **base_link → camera (project frame)**: subtract camera translation, rotate by `rodrigues(camera_rpy)`.

3. **Camera project → optical**: apply the axis convention swap above.

4. **Optical → image plane**: `cv2.projectPoints` with the calibrated camera matrix `K` and distortion coefficients `D`.

```text
p_laser = [r·cos(θ), r·sin(θ), 0]ᵀ
p_base = T_base_laser · p_laser
p_cam  = R_camera_base · (p_base - t_camera)
(u, v) = project(K, D, optical(p_cam))
```

Each projected scan point produces a `(u, v, Z_optical)` triplet: pixel
coordinates plus the ground-truth metric distance along the camera optical axis.
The corresponding monocular depth value is sampled at `(u, v)` from the Depth
Anything V2 output.

Points are filtered to those that:
- fall inside the image bounds,
- have `Z_optical` within `[min_camera_z, max_camera_z]` (default 0.20–6.0 m),
- have finite, positive mono depth.

## Depth Edge Filtering

LiDAR-camera correspondences are unreliable when the scanline hits a depth
discontinuity — the pixel may see one surface while the LiDAR spot lands on
another (e.g. object edge against a distant wall). These contaminated
correspondences produce outlier residuals that degrade the scale fit.

The edge filter works on the monocular depth map:

1. **Normalize**: clip depth to [P2, P98], remap to [0, 1].
2. **Smooth**: Gaussian blur (5×5) to suppress texture noise.
3. **Gradient**: Sobel operator → gradient magnitude `√(Gx² + Gy²)`.
4. **Threshold**: mark pixels with gradient > P90 of the gradient distribution.
5. **Dilate**: morphological dilation (default 5px) to expand the rejection zone around edges.

Projected LiDAR points that land on edge pixels are excluded from the scale fit.
They are drawn purple in the projection overlay for visual inspection.

If too few points survive filtering (default: < 40), the filter is skipped for
that frame — a single flat wall produces no depth edges and doesn't need
filtering.

The rationale: on object boundaries, mono depth transitions sharply from near
(object) to far (background). A LiDAR beam grazing that boundary may report a
metric range from either surface, but the pixel's mono depth is an
interpolation of both. Excluding these boundary pixels eliminates the worst
outliers before fitting even begins.

## Multi-Curve Scale Fitting

The relationship between monocular relative depth `d` and metric distance `Z` is
not known a priori — it depends on the scene, the depth model, and the
normalization. Five parametric forms are fit simultaneously:

| Mode | Formula | Params |
|------|---------|--------|
| `linear` | Z = p₀·d + p₁ | 2 |
| `inverse` | Z = p₀/d + p₁ | 2 |
| `log` | Z = p₀·log₁ₚ(d) + p₁ | 2 |
| `sqrt` | Z = p₀·√d + p₁ | 2 |
| `quadratic` | Z = p₀·d² + p₁·d + p₂ | 3 |

### Robust fitting (RANSAC-style)

For each mode, 250 random trials are run:
- Sample 24 points (or all if fewer).
- Fit parameters via least squares on the subset.
- Score by inlier count (residual < max(0.20 m, 2.5 × median residual)).
- Keep the trial with the most inliers; break ties by lowest MAE on inliers.

After selecting the best trial, a final fit is performed on all inliers.

### Curve selection

With five curves fit, the selection rule is "P90-first, MAE tiebreaker":

1. Find the best (lowest) P90 absolute error across all fits.
2. Collect all fits within 10% of that best P90.
3. From that set, pick the one with the lowest MAE.
4. If still tied, pick the one with more inliers.

P90 (90th percentile of absolute error) is used as the primary metric because it
guards against tail risk: a fit with good median error but a long tail of bad
predictions is dangerous for navigation. The 10% tolerance band prevents
overfitting to a single metric when multiple curves perform similarly.

Default fit modes: `log,quadratic`. These two cover the most common mono-depth
relationships (log-like for Depth Anything's inverse-depth tendency, quadratic
for scenes where the relationship has curvature).

### Full-frame metric depth

Once the best fit is selected, the same formula is applied to every pixel of the
monocular depth map, producing a full-frame absolute depth image:

```text
metric_depth[h, w] = f_best(mono_depth[h, w])
```

Pixels where the formula produces non-positive or non-finite values are marked
NaN.

## Stereo SGBM Cross-Validation

`run_stereo_object_depth.py` provides an independent metric depth estimate using
calibrated stereo block matching (SGBM). It serves as cross-validation for the
mono+scan results.

Pipeline:

1. **Rectify** left and right images using `cv2.stereoRectify` + `cv2.initUndistortRectifyMap`.
2. **Rectify the SAM mask** to match the rectified left image.
3. **Compute disparity** via `cv2.StereoSGBM_create` (96 disparities, block size 5, 3-way mode).
4. **Convert disparity to depth**: `Z = fx · baseline / disparity` (using `P1[0,0]` from stereo rectification).
5. **Filter by mask**: keep only pixels inside the rectified SAM mask where disparity > 0.5.
6. **Transform**: optical → project frame → base_link (using URDF camera offset).
7. **Report statistics**: P5/P10/median/P90/P95 of forward distance (`x_forward_m`), plus height extent (`z_up_m` P5–P95).

The stereo baseline is ~0.06 m (left/right camera separation in URDF). At this
baseline, stereo is reliable for near-range objects (~0.3–3 m) with sufficient
texture. LaserScan is more robust for metric range but limited to the 2D scan
plane (~0.30 m height). The two modalities complement each other:
- LaserScan gives a hard metric anchor at the scan plane.
- Stereo gives dense depth anywhere in the mask (including above/below the scan
  plane).
- Mono+scan fit gives dense metric depth for the whole frame after scale
  calibration.

## Live Validation Workflow

`live_scan_monodepth_validation.py` is the primary tool for accuracy
benchmarking. It subscribes to ROS2 topics (`/stereo/left/image_raw`,
`/stereo/right/image_raw`, `/scan`) and runs one selected localization mode on
demand:

```bash
python3 -m caragent_agent.perception.fusion.live_scan_monodepth_validation \
  --target "door" \
  --truth-distance-m 2.00 \
  --localization-mode mono_relative_lidar \
  --grounding-device GPU \
  --depth-device GPU \
  --absolute-depth-device GPU \
  --sam-device GPU \
  --sam-decoder-device CPU
```

Three modes are available:

| Mode | CLI value | Main estimate |
|------|-----------|---------------|
| Stereo | `stereo` | SGBM depth inside the rectified SAM mask |
| Mono relative + LiDAR | `mono_relative_lidar` | Relative Depth Anything V2 fitted to projected LiDAR metric anchors |
| Mono absolute | `mono_absolute` | Depth Anything V2 Metric Indoor Small output directly in meters |

The UI starts in the CLI-selected mode and cycles modes with `m`. Press `r` to
run the current mode. This keeps repeated tape-measure testing fast: object
detection and SAM still run for every sample, but the expensive depth backend is
only the one being evaluated. `--enable-stereo-preview` can be used to run
stereo as an extra side output while another mode is selected.

On pressing `r`:
1. Capture the currently required inputs for the selected mode.
2. Run GroundingDINO → EfficientSAM → Depth Anything V2 → scale fit → stereo SGBM.
3. Log one row to `validation_results.csv` and `validation_results.jsonl`.

Key CSV columns:
- `mono_p10_m`, `mono_error_m` — mono+scan P10 depth and error vs truth.
- `stereo_p10_m`, `stereo_error_m` — stereo SGBM P10 depth and error vs truth.
- `selected_fit`, `fit_mae_m`, `fit_p90_m` — which curve was selected and its quality.
- `edge_rejected_samples` — how many LiDAR correspondences were filtered at depth edges.

The UI shows four panels: live camera, top-down LiDAR view, last segmentation
overlay, and metric depth + stereo visualization.

Current implementation note:

- `localization_mode` records which backend produced the main recommendation.
- `recommended_depth_m`, `recommended_error_m` are always the selected mode's result.
- `mono_p10_m`, `mono_error_m` are filled by `mono_relative_lidar`.
- `absolute_p10_m`, `absolute_error_m` are filled by `mono_absolute`.
- `stereo_p10_m`, `stereo_error_m`, `stereo_status` are filled by `stereo`.
- `selected_fit`, `fit_mae_m`, `fit_p90_m`, `edge_rejected_samples` are LiDAR
  scale-fit diagnostics for `mono_relative_lidar`.

## Object-Level Depth Recommendation

For navigation, the recommended object distance is the **P10 of metric depth
within the SAM mask**. Rationale:

- P10 (10th percentile) is conservative: it captures the nearest substantial
  portion of the object while rejecting isolated outlier pixels (specular
  highlights, depth holes).
- P5 can be too aggressive (single noisy pixel); median can be too optimistic
  (biased by background bleed at mask boundaries).
- The fit JSON reports P5/P10/P25/median/P75/P90/P95 so the consumer can choose
  a different policy.

## Output JSON Schema

`project_scan_fit_monodepth.py` produces:

```json
{
  "samples": {
    "projected_inside_image": 180,
    "used_for_fit": 142,
    "edge_candidates": 38,
    "edge_rejected": 38,
    "edge_filter_applied": true
  },
  "depth_edge_filter": {
    "available": true,
    "percentile": 90.0,
    "threshold": 0.042,
    "dilate_px": 5,
    "edge_pixel_count": 18423,
    "enabled": true,
    "applied": true
  },
  "selection_rule": {
    "name": "p90_guarded_mae",
    "p90_tolerance": 0.10,
    "description": "Choose fits within tolerance of best P90, then lowest MAE."
  },
  "fits": [
    {
      "mode": "log",
      "formula": "z = p0 * log1p(d) + p1",
      "params": [-1.234, 5.678],
      "inlier_count": 130,
      "mae_m": 0.145,
      "p90_abs_error_m": 0.312
    }
  ],
  "selected_fit": { ... },
  "object_mask_metric_depth_m": {
    "p10": 1.85,
    "median": 2.12,
    "p90": 2.45
  },
  "outputs": {
    "metric_depth_npy": "...",
    "projection": "...",
    "fit_plot": "..."
  }
}
```

`run_stereo_object_depth.py` produces:

```json
{
  "object_camera_project": {
    "median_xyz_m": [1.92, -0.15, 0.42],
    "stats": {
      "x_forward_m": { "p10": 1.78, "median": 1.92, "p90": 2.08 }
    }
  },
  "object_base": {
    "median_xyz_m": [2.22, -0.12, 0.605],
    "height_m_p05_p95": 0.38
  },
  "mask": {
    "valid_stereo_pixels": 452,
    "valid_ratio": 0.73
  }
}
```

## fuse_scan_depth.py (Legacy Angular Heuristic)

`fuse_scan_depth.py` is the earlier approach: pick the bottom pixel of the SAM
mask, convert to a horizontal bearing (using calibrated intrinsics when
available, or FOV approximation as fallback), sample LaserScan ranges in an
angular window, cluster by contiguous beam index + range jumps, and select the
nearest cluster as the target surface.

This script now supports calibrated intrinsics via `--calib-file`, making its
bearing computation equivalent to the projection pipeline's angular component.
It remains useful for:
- Quick single-beam diagnostics without running the full scale fit.
- Cases where only a bearing + range estimate is needed (no dense depth map).
- Top-down visualization (`_fused_topdown.png`) showing scan points, selected
  beams, and estimated target position.

The main limitation vs `project_scan_fit_monodepth.py` is that it uses a single
query pixel rather than projecting all scan points and fitting a global scale
model. It cannot produce a full-frame metric depth map.

## Summary

The current pipeline is:

```text
GroundingDINO (open-vocab detection)
  → EfficientSAM (box-prompted mask)
    → Depth Anything V2 (monocular relative depth)
      → project_scan_fit_monodepth (LiDAR projection + edge filter + multi-curve fit)
        → full-frame metric depth map + object depth statistics
      → run_stereo_object_depth (SGBM stereo depth inside mask, cross-validation)

Live validation: live_scan_monodepth_validation.py wraps the above in a ROS2 UI.
```

The key improvements over the original angular heuristic:
- **Calibrated projection** replaces FOV approximation with full `K, D` intrinsics
  and calibrated extrinsics.
- **Global scale fitting** (multi-curve + RANSAC + P90-first selection) replaces
  single-beam range sampling.
- **Depth edge filtering** eliminates unreliable LiDAR-pixel correspondences at
  depth discontinuities.
- **Stereo SGBM** provides an independent metric depth signal for cross-validation.
- **Full-frame metric depth** output enables downstream consumers (VLM verification,
  approach-point planning) to query depth at any pixel, not just one query point.
