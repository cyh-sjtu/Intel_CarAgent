# caragent_vision

汇博双目摄像头 ROS2 驱动包 + 标定工具集。

## 发布话题

| 话题 | 内容 |
|------|------|
| `/stereo/image_raw` | 原始拼接图（1280×480） |
| `/stereo/left/image_raw` | 左目图像（640×480） |
| `/stereo/right/image_raw` | 右目图像（640×480） |

## 启动

```bash
ros2 launch caragent_vision huibo_stereo_camera.launch.py \
  device:=/dev/video0 \
  width:=1280 \
  height:=480 \
  fps:=30.0 \
  show_image:=true
```

## 标定工具

| 脚本 | 用途 |
|------|------|
| `capture_stereo_calibration.py` | 拍摄棋盘格标定图像对 |
| `calibrate_stereo_camera.py` | 双目内外参标定（输出 stereo_calibration.npz） |
| `collect_lidar_camera_correspondences.py` | 交互式点击标定 LiDAR-相机对应点 |
| `calibrate_lidar_camera_extrinsics.py` | 高斯牛顿优化 LiDAR-相机外参 |
| `live_lidar_camera_correspondences.py` | 实时 LiDAR 点云投影可视化 |

标定数据存放于 `~/caragent_ws/calibration/`，包含双目标定和 LiDAR-相机外参。

## 依赖

- `caragent_bringup` — 硬件 bringup（串口、雷达）
- 上层语义功能（关键帧采集、CLIP 编码）由 `caragent_memory` 和 `caragent_agent` 负责
