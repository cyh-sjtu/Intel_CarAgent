# caragent_vision

`caragent_vision` 是汇博双目摄像头 ROS2 驱动与标定工具包。它为关键帧语义记忆、当前画面确认和物体精细定位提供图像输入，是语义导航系统的视觉观测入口。

## 模块定位

| 功能 | 当前实现 |
| --- | --- |
| 双目采集 | 从 side-by-side UVC 帧中拆分左右目图像 |
| ROS 发布 | 发布原始拼接图、左目图、右目图 |
| 高分辨率支持 | 默认支持 `3840x1200` raw、左右目各 `1920x1200` |
| 双目标定 | 棋盘格采集与内外参标定工具 |
| LiDAR-相机标定 | 采集对应点并优化外参，用于后续深度/投影工具 |

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `launch/huibo_stereo_camera.launch.py` | 相机启动入口，透传分辨率、设备、标定和发布参数 |
| `caragent_vision/stereo_camera_node.py` | PyAV/OpenCV 采集、左右目拆分、ROS Image 发布 |
| `capture_stereo_calibration.py` | 双目标定图片采集 |
| `calibrate_stereo_camera.py` | 双目标定计算与 overlay 输出 |
| `collect_lidar_camera_correspondences.py` / `live_lidar_camera_correspondences.py` | LiDAR-相机对应点采集 |
| `calibrate_lidar_camera_extrinsics.py` | LiDAR-相机外参优化 |

## 发布话题

| 话题 | 内容 |
| --- | --- |
| `/stereo/image_raw` | 原始 side-by-side 拼接图 |
| `/stereo/left/image_raw` | 左目 raw 图像 |
| `/stereo/right/image_raw` | 右目 raw 图像 |
| `/stereo/left/image_rect` | 可选左目矫正图 |
| `/stereo/right/image_rect` | 可选右目矫正图 |
| `/stereo/disparity` | 可选视差图 |

## 数据流关系

```text
Huibo stereo camera
  └─ stereo_camera_node
      ├─ /stereo/image_raw       → keyframe_recorder_node
      ├─ /stereo/left/image_raw  → Agent 当前画面 / 视觉工具
      └─ /stereo/right/image_raw → 双目深度 / 物体定位工具
```

## 已实现能力

- PyAV 作为默认采集后端，OpenCV 作为可选后端。
- 支持 camera raw frame 和左右目分辨率分别配置。
- 默认 raw preview 不强制使用旧标定文件，避免高分辨率下错误 rectification。
- 提供双目标定和 LiDAR-相机外参标定工具链。

## 边界说明

- 默认发布 raw split 图像；只有在标定文件与当前分辨率匹配时才启用 rectification/disparity。
- 本包不直接执行 CLIP/DINOv2、GroundingDINO 或 SAM 推理，这些由 `caragent_memory` 和 `caragent_agent` 承担。
