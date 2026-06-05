# caragent_stm32_driver

STM32 串口桥接节点：接收 STM32 上发的 ODOM 里程计数据，发布 `/odom` 话题和 `odom → base_link` TF；同时将 `/cmd_vel` 速度指令下发至 STM32 控制电机。

## 数据方向

```text
STM32 ─(ODOM)──→ 解析 → /odom + TF
STM32 ←(CMD)─── 串口 ← /cmd_vel
```

## 节点

| 节点 | 可执行文件 |
|------|-----------|
| `stm32_driver_node` | `stm32_driver_node` |

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `stm32_port` | `/dev/ttyUSB1` | STM32 串口设备 |
| `baud_rate` | `115200` | 波特率 |
| `enable_cmd_vel` | `false` | 启用电机控制 |
| `base_link_yaw_offset_deg` | `180.0` | STM32 坐标系到 ROS 坐标系的偏航修正 |
| `zero_odom_on_start` | `true` | 启动时归零里程计原点 |

完整参数列表见 [stm32_driver_node.py](caragent_stm32_driver/stm32_driver_node.py) 的 `declare_parameter` 部分。
