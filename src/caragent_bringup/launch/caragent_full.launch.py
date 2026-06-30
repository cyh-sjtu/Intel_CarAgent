from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode = LaunchConfiguration("mode")

    # ---- sensors ----
    laser_port = LaunchConfiguration("laser_port")
    stm32_port = LaunchConfiguration("stm32_port")
    lidar_inverted = LaunchConfiguration("lidar_inverted")
    lidar_angle_compensate = LaunchConfiguration("lidar_angle_compensate")
    lidar_scan_mode = LaunchConfiguration("lidar_scan_mode")

    # ---- common ----
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    enable_camera = LaunchConfiguration("enable_camera")

    # ---- STM32 driver ----
    base_link_yaw_offset_deg = LaunchConfiguration("base_link_yaw_offset_deg")
    enable_cmd_vel = LaunchConfiguration("enable_cmd_vel")
    log_pc_debug = LaunchConfiguration("log_pc_debug")

    # ---- SLAM / localization ----
    map_file_name = LaunchConfiguration("map_file_name")
    use_slam = LaunchConfiguration("use_slam")
    map_start_at_dock = LaunchConfiguration("map_start_at_dock")

    # ---- navigation ----
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    use_map_server = LaunchConfiguration("use_map_server")
    map_yaml_file = LaunchConfiguration("map_yaml_file")
    enable_left_only_goal_proxy = LaunchConfiguration("enable_left_only_goal_proxy")
    max_linear_mps = LaunchConfiguration("max_linear_mps")
    max_angular_radps = LaunchConfiguration("max_angular_radps")
    pre_align_strategy = LaunchConfiguration("pre_align_strategy")
    path_heading_lookahead_m = LaunchConfiguration("path_heading_lookahead_m")

    # ---- camera ----
    camera_device = LaunchConfiguration("camera_device")
    camera_calib_file = LaunchConfiguration("camera_calib_file")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_left_width = LaunchConfiguration("camera_left_width")
    camera_right_width = LaunchConfiguration("camera_right_width")
    camera_fps = LaunchConfiguration("camera_fps")

    # ============================================================
    # Camera (all modes, when enabled)
    # ============================================================
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("caragent_vision"),
                    "launch",
                    "huibo_stereo_camera.launch.py",
                ]
            )
        ),
        launch_arguments={
            "device": camera_device,
            "calib_file": camera_calib_file,
            "width": camera_width,
            "height": camera_height,
            "left_width": camera_left_width,
            "right_width": camera_right_width,
            "fps": camera_fps,
        }.items(),
        condition=IfCondition(enable_camera),
    )

    # ============================================================
    # Mode: SLAM
    # ============================================================
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("caragent_bringup"),
                    "launch",
                    "rplidar_c1_slam.launch.py",
                ]
            )
        ),
        launch_arguments={
            "laser_port": laser_port,
            "stm32_port": stm32_port,
            "use_sim_time": use_sim_time,
            "use_rviz": use_rviz,
            "base_link_yaw_offset_deg": base_link_yaw_offset_deg,
            "lidar_inverted": lidar_inverted,
            "lidar_angle_compensate": lidar_angle_compensate,
            "lidar_scan_mode": lidar_scan_mode,
            "use_slam": use_slam,
            "enable_cmd_vel": enable_cmd_vel,
            "log_pc_debug": log_pc_debug,
            "map_file_name": map_file_name,
        }.items(),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'slam'"])),
    )

    # ============================================================
    # Mode: localization
    # ============================================================
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("caragent_bringup"),
                    "launch",
                    "rplidar_c1_localization.launch.py",
                ]
            )
        ),
        launch_arguments={
            "laser_port": laser_port,
            "stm32_port": stm32_port,
            "use_sim_time": use_sim_time,
            "use_rviz": use_rviz,
            "base_link_yaw_offset_deg": base_link_yaw_offset_deg,
            "lidar_inverted": lidar_inverted,
            "lidar_angle_compensate": lidar_angle_compensate,
            "lidar_scan_mode": lidar_scan_mode,
            "map_file_name": map_file_name,
            "map_start_at_dock": map_start_at_dock,
            "enable_cmd_vel": enable_cmd_vel,
            "log_pc_debug": log_pc_debug,
        }.items(),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'localization'"])),
    )

    # ============================================================
    # Mode: navigation
    # ============================================================
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("caragent_navigation"),
                    "launch",
                    "caragent_nav2.launch.py",
                ]
            )
        ),
        launch_arguments={
            "laser_port": laser_port,
            "stm32_port": stm32_port,
            "map_file_name": map_file_name,
            "use_sim_time": use_sim_time,
            "use_nav_rviz": use_rviz,
            "nav2_params_file": nav2_params_file,
            "use_map_server": use_map_server,
            "map_yaml_file": map_yaml_file,
            "base_link_yaw_offset_deg": base_link_yaw_offset_deg,
            "lidar_inverted": lidar_inverted,
            "enable_cmd_vel": enable_cmd_vel,
            "enable_left_only_goal_proxy": enable_left_only_goal_proxy,
            "max_linear_mps": max_linear_mps,
            "max_angular_radps": max_angular_radps,
            "pre_align_strategy": pre_align_strategy,
            "path_heading_lookahead_m": path_heading_lookahead_m,
            "log_pc_debug": log_pc_debug,
        }.items(),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'navigation'"])),
    )

    default_map_file = PathJoinSubstitution(
        [
            EnvironmentVariable("HOME"),
            "caragent_ws",
            "maps",
            "map_202605191353",
        ]
    )
    default_map_yaml_file = PathJoinSubstitution(
        [
            EnvironmentVariable("HOME"),
            "caragent_ws",
            "maps",
            "map_202605191353.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode",
                default_value="slam",
                description="Operation mode: slam, localization, or navigation.",
            ),
            DeclareLaunchArgument(
                "laser_port",
                default_value="/dev/ttyUSB0",
                description="RPLIDAR serial device.",
            ),
            DeclareLaunchArgument(
                "stm32_port",
                default_value="/dev/ttyUSB1",
                description="STM32 serial device for ODOM telemetry.",
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
                "map_file_name",
                default_value="",
                description="slam_toolbox map base path without .posegraph/.data suffix. Empty for new map.",
            ),
            DeclareLaunchArgument(
                "map_start_at_dock",
                default_value="false",
                description="Start localization near the first posegraph node. Keep false when lidar_initialpose_node handles global localization.",
            ),
            DeclareLaunchArgument(
                "map_yaml_file",
                default_value=default_map_yaml_file,
                description="Occupancy grid YAML map for nav2_map_server.",
            ),
            DeclareLaunchArgument(
                "use_map_server",
                default_value="false",
                description="Start nav2_map_server. Keep false when localization already publishes /map.",
            ),
            DeclareLaunchArgument(
                "use_slam",
                default_value="true",
                description="Start slam_toolbox async SLAM (slam mode only).",
            ),
            DeclareLaunchArgument(
                "enable_cmd_vel",
                default_value="false",
                description="Forward /cmd_vel to STM32. Keep false for slam/localization, true for navigation.",
            ),
            DeclareLaunchArgument(
                "log_pc_debug",
                default_value="false",
                description="Log PCDBG/RCDBG serial debug lines returned by STM32.",
            ),
            DeclareLaunchArgument(
                "enable_left_only_goal_proxy",
                default_value="false",
                description="Start optional left-only goal proxy in navigation mode for A/B testing.",
            ),
            DeclareLaunchArgument(
                "max_linear_mps",
                default_value="0.40",
                description="STM32 cmd_vel linear clamp for navigation mode.",
            ),
            DeclareLaunchArgument(
                "max_angular_radps",
                default_value="3.50",
                description="STM32 cmd_vel angular clamp for navigation mode.",
            ),
            DeclareLaunchArgument(
                "pre_align_strategy",
                default_value="direct_bearing",
                description="Left-only pre-align strategy: direct_bearing or path_heading.",
            ),
            DeclareLaunchArgument(
                "path_heading_lookahead_m",
                default_value="0.70",
                description="Lookahead distance on the computed global path for path_heading pre-align.",
            ),
            DeclareLaunchArgument(
                "base_link_yaw_offset_deg",
                default_value="180.0",
                description="Fixed STM32 odometry axis to ROS base_link yaw correction.",
            ),
            DeclareLaunchArgument(
                "lidar_inverted",
                default_value="false",
                description="SLLIDAR scan data inversion.",
            ),
            DeclareLaunchArgument(
                "lidar_angle_compensate",
                default_value="true",
                description="SLLIDAR angle compensation.",
            ),
            DeclareLaunchArgument(
                "lidar_scan_mode",
                default_value="Standard",
                description="SLLIDAR scan mode.",
            ),
            DeclareLaunchArgument(
                "nav2_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("caragent_navigation"),
                        "config",
                        "nav2_params.yaml",
                    ]
                ),
                description="Nav2 parameter file (navigation mode only).",
            ),
            DeclareLaunchArgument(
                "enable_camera",
                default_value="false",
                description="Launch Huibo stereo camera driver.",
            ),
            DeclareLaunchArgument(
                "camera_device",
                default_value="/dev/video0",
                description="Huibo stereo camera device.",
            ),
            DeclareLaunchArgument(
                "camera_calib_file",
                default_value="",
                description="Path to stereo_calibration.npz (optional).",
            ),
            DeclareLaunchArgument(
                "camera_width",
                default_value="3840",
                description="Side-by-side stereo camera frame width.",
            ),
            DeclareLaunchArgument(
                "camera_height",
                default_value="1200",
                description="Side-by-side stereo camera frame height.",
            ),
            DeclareLaunchArgument(
                "camera_left_width",
                default_value="1920",
                description="Left image width in the side-by-side frame.",
            ),
            DeclareLaunchArgument(
                "camera_right_width",
                default_value="1920",
                description="Right image width in the side-by-side frame.",
            ),
            DeclareLaunchArgument(
                "camera_fps",
                default_value="30.0",
                description="Requested stereo camera FPS.",
            ),
            slam_launch,
            localization_launch,
            navigation_launch,
            camera_launch,
        ]
    )
