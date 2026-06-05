# caragent_memory

CarAgent keyframe recording and selection tools.

This package records stereo keyframe candidates after a LiDAR map is built. The
final navigation target is still executed by LiDAR localization and Nav2; the
camera frames are used as semantic memory anchors.

## Online Collection

Start localization, camera, and the recorder with collection speed limits:

```bash
ros2 launch caragent_memory caragent_keyframe_collect.launch.py \
  session_name:=lab_001 \
  camera_device:=/dev/video0 \
  laser_port:=/dev/ttyUSB1 \
  stm32_port:=/dev/ttyUSB0
```

Then drive the car manually by publishing `/cmd_vel`. The launch limits STM32
commands to `0.16 m/s` and `0.45 rad/s` for stable image capture.

Manual capture:

```bash
ros2 service call /keyframe_recorder/capture_once std_srvs/srv/Trigger {}
```

Candidate dataset:

```text
~/caragent_ws/keyframes/<session_name>/
  raw/
  left/
  right/
  pose/
  scan/
  meta/
  manifest.jsonl
  session.json
```

## CLIP + DINOv2 Selection

Convert or document the model export:

```bash
ros2 run caragent_memory convert_clip_openvino --dry-run
```

The converter exports `CLIPModel.get_image_features(pixel_values)`, which keeps
the CLIP `visual_projection` head. For ViT-B/32 the selector expects a 512-D
image embedding. Do not use a raw vision-model hidden-state export such as
`[1, 50, 768]`; flattening that tensor would break cosine similarity.

After an OpenVINO image encoder XML is available:

```bash
ros2 run caragent_memory select_keyframes \
  --dataset ~/caragent_ws/keyframes/lab_001 \
  --clip-model ~/caragent_ws/models/clip-vit-base-patch32/image_encoder.xml \
  --dinov2-model ~/caragent_ws/models/dinov2-small \
  --device AUTO \
  --dinov2-device auto
```

The selector stores two embeddings for every selected keyframe:

- `clip_encoding` is a 512-D CLIP image embedding for image-text retrieval.
- `dinov2_encoding` is a 384-D DINOv2-small image embedding for frame-to-frame
  visual similarity, deduplication, and place recognition.

DINOv2 is the default frame deduplication backend. To reproduce the previous
CLIP-based deduplication behavior, pass `--dedupe-backend clip`.

Selection output is written to `<dataset>/selected`, including
`selected_manifest.jsonl`, `rejected_manifest.jsonl`, `review.html`,
`embeddings/`, and `constructed_memory/keyframe_nodes/`.

## v1 Rules

- Online candidates: first frame, then `>=1.5s` and either `>=0.65m` movement or
  `>=30deg` yaw change.
- Image quality: Laplacian variance, brightness mean, and brightness standard
  deviation.
- Offline selection: OpenVINO CLIP image embeddings for semantic retrieval,
  DINOv2 image embeddings for visual deduplication, plus pose/yaw coverage
  rules.
- No semantic weighting, OCR, object detection, or stereo-depth graph edges in v1.
