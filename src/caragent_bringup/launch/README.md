# Launch 文件

| 文件 | 用途 |
|------|------|
| `caragent_full.launch.py` | 统一入口，按 `mode:=slam\|localization\|navigation` 切换 |
| `rplidar_c1_slam.launch.py` | LiDAR + STM32 + slam_toolbox 建图 |
| `rplidar_c1_localization.launch.py` | LiDAR + STM32 + slam_toolbox 定位模式 |
