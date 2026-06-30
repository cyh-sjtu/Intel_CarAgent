# caragent_description

CarAgent 机器人 URDF 模型包。定义 chassis 和全部传感器（激光雷达、双目摄像头、IMU）的物理尺寸与静态坐标变换。

## 目录

| 路径 | 说明 |
|------|------|
| [urdf/caragent.urdf](urdf/caragent.urdf) | 机器人 URDF 模型文件 |
| [launch/description.launch.py](launch/description.launch.py) | robot_state_publisher 启动，发布静态 TF |

## TF 树

```text
base_link
├── laser        (0.12, 0, 0.30)  rpy=(0, 0, π)   ← 激光雷达朝后
├── camera_left  (0.30, 0.03, 0.185)
├── camera_right (0.30, -0.03, 0.185)
└── imu_link     (0.20, -0.033, 0.16)
```

所有传感器位姿以此 URDF 为唯一数据源，不再在每个 launch 文件中手写 `static_transform_publisher`。
