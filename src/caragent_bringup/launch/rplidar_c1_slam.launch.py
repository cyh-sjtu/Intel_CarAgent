from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    laser_port = LaunchConfiguration("laser_port")
    frame_id = LaunchConfiguration("frame_id")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    use_slam = LaunchConfiguration("use_slam")
    lidar_package = LaunchConfiguration("lidar_package")
    lidar_launch_file = LaunchConfiguration("lidar_launch_file")
    lidar_inverted = LaunchConfiguration("lidar_inverted")
    lidar_angle_compensate = LaunchConfiguration("lidar_angle_compensate")
    lidar_scan_mode = LaunchConfiguration("lidar_scan_mode")
    slam_params_file = LaunchConfiguration("slam_params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    use_static_odom = LaunchConfiguration("use_static_odom")
    use_stm32_driver_node = LaunchConfiguration("use_stm32_driver_node")
    stm32_port = LaunchConfiguration("stm32_port")
    base_baud_rate = LaunchConfiguration("base_baud_rate")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
    odom_yaw_offset_deg = LaunchConfiguration("odom_yaw_offset_deg")
    map_file_name = LaunchConfiguration("map_file_name")
    map_start_at_dock = LaunchConfiguration("map_start_at_dock")
    odom_yaw_sign = LaunchConfiguration("odom_yaw_sign")
    base_link_yaw_offset_deg = LaunchConfiguration("base_link_yaw_offset_deg")
    linear_velocity_sign = LaunchConfiguration("linear_velocity_sign")
    angular_velocity_sign = LaunchConfiguration("angular_velocity_sign")
    tf_publish_rate_hz = LaunchConfiguration("tf_publish_rate_hz")
    tf_future_offset_sec = LaunchConfiguration("tf_future_offset_sec")
    zero_odom_on_start = LaunchConfiguration("zero_odom_on_start")
    enable_cmd_vel = LaunchConfiguration("enable_cmd_vel")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    cmd_send_rate_hz = LaunchConfiguration("cmd_send_rate_hz")
    cmd_timeout_sec = LaunchConfiguration("cmd_timeout_sec")
    max_linear_mps = LaunchConfiguration("max_linear_mps")
    max_angular_radps = LaunchConfiguration("max_angular_radps")
    log_cmd_serial = LaunchConfiguration("log_cmd_serial")
    log_pc_debug = LaunchConfiguration("log_pc_debug")

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare(lidar_package),
                    "launch",
                    lidar_launch_file,
                ]
            )
        ),
        launch_arguments={
            "serial_port": laser_port,
            "frame_id": frame_id,
            "inverted": lidar_inverted,
            "angle_compensate": lidar_angle_compensate,
            "scan_mode": lidar_scan_mode,
        }.items(),
    )

    stm32_driver_node = Node(
        package="caragent_stm32_driver",
        executable="stm32_driver_node",
        name="stm32_driver_node",
        output="screen",
        parameters=[
            {
                "stm32_port": stm32_port,
                "baud_rate": ParameterValue(base_baud_rate, value_type=int),
                "odom_frame": odom_frame,
                "base_frame": base_frame,
                "odom_yaw_offset_deg": ParameterValue(odom_yaw_offset_deg, value_type=float),
                "odom_yaw_sign": ParameterValue(odom_yaw_sign, value_type=float),
                "base_link_yaw_offset_deg": ParameterValue(base_link_yaw_offset_deg, value_type=float),
                "linear_velocity_sign": ParameterValue(linear_velocity_sign, value_type=float),
                "angular_velocity_sign": ParameterValue(angular_velocity_sign, value_type=float),
                "tf_publish_rate_hz": ParameterValue(tf_publish_rate_hz, value_type=float),
                "tf_future_offset_sec": ParameterValue(tf_future_offset_sec, value_type=float),
                "zero_odom_on_start": ParameterValue(zero_odom_on_start, value_type=bool),
                "enable_cmd_vel": ParameterValue(enable_cmd_vel, value_type=bool),
                "cmd_vel_topic": cmd_vel_topic,
                "cmd_send_rate_hz": ParameterValue(cmd_send_rate_hz, value_type=float),
                "cmd_timeout_sec": ParameterValue(cmd_timeout_sec, value_type=float),
                "max_linear_mps": ParameterValue(max_linear_mps, value_type=float),
                "max_angular_radps": ParameterValue(max_angular_radps, value_type=float),
                "log_cmd_serial": ParameterValue(log_cmd_serial, value_type=bool),
                "log_pc_debug": ParameterValue(log_pc_debug, value_type=bool),
            }
        ],
        condition=IfCondition(use_stm32_driver_node),
    )

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("caragent_description"),
                    "launch",
                    "description.launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    odom_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_to_base_tf",
        arguments=[
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            odom_frame,
            base_frame,
        ],
        condition=IfCondition(use_static_odom),
    )

    slam_toolbox = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params_file,
            {
                "use_sim_time": use_sim_time,
                "odom_frame": odom_frame,
                "base_frame": base_frame,
                "map_frame": "map",
                "scan_topic": "/scan",
                "transform_timeout": 0.3,
                "map_file_name": map_file_name,
                "map_start_at_dock": map_start_at_dock,
            },
        ],
        condition=IfCondition(use_slam),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "laser_port",
                default_value="/dev/ttyUSB0",
                description="RPLIDAR serial device, for example /dev/ttyUSB0.",
            ),
            DeclareLaunchArgument(
                "frame_id",
                default_value="laser",
                description="Laser scan frame id.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock if true.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz2 if true.",
            ),
            DeclareLaunchArgument(
                "use_slam",
                default_value="true",
                description="Start slam_toolbox if true.",
            ),
            DeclareLaunchArgument(
                "map_file_name",
                default_value="",
                description="Path to serialized slam_toolbox map for incremental mapping. Empty = start fresh.",
            ),
            DeclareLaunchArgument(
                "map_start_at_dock",
                default_value="false",
                description="Start pose at the dock position saved in the serialized map.",
            ),
            DeclareLaunchArgument(
                "use_static_odom",
                default_value="false",
                description="Publish a temporary static odom->base_link transform instead of relying on STM32 odom.",
            ),
            DeclareLaunchArgument(
                "use_stm32_driver_node",
                default_value="true",
                description="Start STM32 serial odometry bridge if true.",
            ),
            DeclareLaunchArgument(
                "stm32_port",
                default_value="/dev/ttyUSB1",
                description="STM32 serial device for ODOM telemetry.",
            ),
            DeclareLaunchArgument(
                "base_baud_rate",
                default_value="115200",
                description="STM32 serial baud rate.",
            ),
            DeclareLaunchArgument(
                "odom_frame",
                default_value="odom",
                description="Odometry frame id.",
            ),
            DeclareLaunchArgument(
                "base_frame",
                default_value="base_link",
                description="Robot base frame id.",
            ),
            DeclareLaunchArgument(
                "odom_yaw_offset_deg",
                default_value="0.0",
                description="Yaw offset applied when converting STM32 odometry into ROS odom/base_link coordinates.",
            ),
            DeclareLaunchArgument(
                "odom_yaw_sign",
                default_value="1.0",
                description="Yaw sign applied when converting STM32 odometry into ROS coordinates.",
            ),
            DeclareLaunchArgument(
                "base_link_yaw_offset_deg",
                default_value="180.0",
                description="Fixed yaw from STM32 odometry axes to ROS base_link axes. Current car uses 180deg so base_link +X points to the front.",
            ),
            DeclareLaunchArgument(
                "linear_velocity_sign",
                default_value="1.0",
                description="Linear velocity sign applied to STM32 odometry.",
            ),
            DeclareLaunchArgument(
                "angular_velocity_sign",
                default_value="1.0",
                description="Angular velocity sign applied to STM32 odometry.",
            ),
            DeclareLaunchArgument(
                "tf_publish_rate_hz",
                default_value="50.0",
                description="TF publish rate for odom->base_link.",
            ),
            DeclareLaunchArgument(
                "tf_future_offset_sec",
                default_value="0.02",
                description="Small timestamp offset applied to odom->base_link TF.",
            ),
            DeclareLaunchArgument(
                "zero_odom_on_start",
                default_value="true",
                description="Use the first valid STM32 odometry sample as ROS odom origin.",
            ),
            DeclareLaunchArgument(
                "enable_cmd_vel",
                default_value="false",
                description="Enable /cmd_vel serial output to STM32. Keep false for pure SLAM/debug.",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic",
                default_value="/cmd_vel",
                description="Twist topic used when enable_cmd_vel is true.",
            ),
            DeclareLaunchArgument(
                "cmd_send_rate_hz",
                default_value="20.0",
                description="Rate for sending CMD velocity commands to STM32.",
            ),
            DeclareLaunchArgument(
                "cmd_timeout_sec",
                default_value="0.3",
                description="Stop sending motion command if cmd_vel is stale for this many seconds.",
            ),
            DeclareLaunchArgument(
                "max_linear_mps",
                default_value="0.12",
                description="Maximum linear speed sent from ROS to STM32.",
            ),
            DeclareLaunchArgument(
                "max_angular_radps",
                default_value="0.8",
                description="Maximum angular speed sent from ROS to STM32.",
            ),
            DeclareLaunchArgument(
                "log_cmd_serial",
                default_value="false",
                description="Log CMD serial writes from stm32_driver_node.",
            ),
            DeclareLaunchArgument(
                "log_pc_debug",
                default_value="false",
                description="Log PCDBG/RCDBG serial debug lines returned by STM32.",
            ),
            DeclareLaunchArgument(
                "lidar_package",
                default_value="sllidar_ros2",
                description="Installed SLLIDAR/RPLIDAR ROS2 driver package name.",
            ),
            DeclareLaunchArgument(
                "lidar_launch_file",
                default_value="sllidar_c1_launch.py",
                description="Launch file from the SLLIDAR/RPLIDAR driver package.",
            ),
            DeclareLaunchArgument(
                "lidar_inverted",
                default_value="false",
                description="Whether to invert SLLIDAR scan data order. Toggle this if left/right scan data is mirrored.",
            ),
            DeclareLaunchArgument(
                "lidar_angle_compensate",
                default_value="true",
                description="Enable SLLIDAR angle compensation.",
            ),
            DeclareLaunchArgument(
                "lidar_scan_mode",
                default_value="Standard",
                description="SLLIDAR scan mode.",
            ),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("caragent_bringup"),
                        "config",
                        "slam_toolbox_mapper_params.yaml",
                    ]
                ),
                description="slam_toolbox parameter file.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("caragent_bringup"),
                        "rviz",
                        "caragent_slam.rviz",
                    ]
                ),
                description="RViz config file.",
            ),
            lidar_launch,
            stm32_driver_node,
            odom_to_base_tf,
            description_launch,
            slam_toolbox,
            rviz,
        ]
    )
