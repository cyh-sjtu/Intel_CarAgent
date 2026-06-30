from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    laser_port = LaunchConfiguration("laser_port")
    stm32_port = LaunchConfiguration("stm32_port")
    map_file_name = LaunchConfiguration("map_file_name")
    map_start_at_dock = LaunchConfiguration("map_start_at_dock")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_nav_rviz = LaunchConfiguration("use_nav_rviz")
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    use_map_server = LaunchConfiguration("use_map_server")
    map_yaml_file = LaunchConfiguration("map_yaml_file")
    base_link_yaw_offset_deg = LaunchConfiguration("base_link_yaw_offset_deg")
    lidar_inverted = LaunchConfiguration("lidar_inverted")
    enable_cmd_vel = LaunchConfiguration("enable_cmd_vel")
    enable_left_only_goal_proxy = LaunchConfiguration("enable_left_only_goal_proxy")
    max_linear_mps = LaunchConfiguration("max_linear_mps")
    max_angular_radps = LaunchConfiguration("max_angular_radps")
    pre_align_strategy = LaunchConfiguration("pre_align_strategy")
    path_heading_lookahead_m = LaunchConfiguration("path_heading_lookahead_m")
    log_pc_debug = LaunchConfiguration("log_pc_debug")

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
            "map_file_name": map_file_name,
            "map_start_at_dock": map_start_at_dock,
            "use_sim_time": use_sim_time,
            "use_rviz": "false",
            "enable_cmd_vel": enable_cmd_vel,
            "cmd_vel_topic": "/cmd_vel",
            "cmd_send_rate_hz": "30.0",
            "cmd_timeout_sec": "0.25",
            "max_linear_mps": max_linear_mps,
            "max_angular_radps": max_angular_radps,
            "log_cmd_serial": "true",
            "log_pc_debug": log_pc_debug,
            "base_link_yaw_offset_deg": base_link_yaw_offset_deg,
            "lidar_inverted": lidar_inverted,
        }.items(),
    )

    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("nav2_bringup"),
                    "launch",
                    "navigation_launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": nav2_params_file,
            "autostart": "true",
        }.items(),
    )

    map_server = LifecycleNode(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        namespace="",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "yaml_filename": map_yaml_file,
            }
        ],
        condition=IfCondition(use_map_server),
    )

    map_server_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["map_server"],
            }
        ],
        condition=IfCondition(use_map_server),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(
            PythonExpression(
                ["'", use_nav_rviz, "' == 'true' and '", enable_left_only_goal_proxy, "' != 'true'"]
            )
        ),
    )

    rviz_left_only = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [
                    FindPackageShare("caragent_navigation"),
                    "rviz",
                    "caragent_nav_left_only.rviz",
                ]
            ),
        ],
        condition=IfCondition(
            PythonExpression(
                ["'", use_nav_rviz, "' == 'true' and '", enable_left_only_goal_proxy, "' == 'true'"]
            )
        ),
    )

    left_only_goal_proxy = Node(
        package="caragent_navigation",
        executable="left_only_goal_proxy",
        name="left_only_goal_proxy",
        output="screen",
        parameters=[
            {
                "input_goal_topic": "/caragent/left_only_goal",
                "cmd_vel_topic": "/cmd_vel",
                "action_name": "navigate_to_pose",
                "spin_action_name": "spin",
                "global_frame": "map",
                "base_frame": "base_link",
                "odom_topic": "/odom",
                "scan_topic": "/scan",
                "pre_align_enabled": True,
                "pre_align_strategy": pre_align_strategy,
                "path_heading_action_name": "compute_path_to_pose",
                "path_heading_lookahead_m": path_heading_lookahead_m,
                "path_heading_min_goal_distance_m": 0.35,
                "path_heading_timeout_sec": 3.0,
                "path_heading_fallback_to_direct": True,
                "final_align_enabled": True,
                "arrival_tolerance_m": 0.25,
                "yaw_tolerance_deg": 4.0,
                "settle_time_sec": 0.7,
                "fast_omega": 3.40,
                "mid_omega": 2.50,
                "slow_omega": 1.50,
                "fast_threshold_deg": 20.0,
                "mid_threshold_deg": 10.0,
                "rotation_timeout_sec": 15.0,
                "rotation_loop_rate_hz": 20.0,
                "right_turn_shortcut_deg": 90.0,
                "safety_check_enabled": True,
                "safety_radius_m": 0.38,
                "safety_front_radius_m": 0.45,
                "safety_side_radius_m": 0.34,
                "safety_rear_radius_m": 0.30,
                "safety_scan_max_age_sec": 0.5,
                "nav2_final_align_fallback": True,
                "nav2_final_align_timeout_sec": 20.0,
            }
        ],
        condition=IfCondition(enable_left_only_goal_proxy),
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
                "map_file_name",
                default_value=default_map_file,
                description="slam_toolbox serialized map base path without suffix.",
            ),
            DeclareLaunchArgument(
                "map_start_at_dock",
                default_value="false",
                description="Start localization near the first posegraph node. Keep false when lidar_initialpose_node handles global localization.",
            ),
            DeclareLaunchArgument(
                "map_yaml_file",
                default_value=default_map_yaml_file,
                description="Occupancy grid YAML map for Nav2 map_server.",
            ),
            DeclareLaunchArgument(
                "use_map_server",
                default_value="false",
                description="Optionally start nav2_map_server to publish /map. Keep false when slam_toolbox localization already publishes /map.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock if true.",
            ),
            DeclareLaunchArgument(
                "use_nav_rviz",
                default_value="true",
                description="Start navigation RViz if true.",
            ),
            DeclareLaunchArgument(
                "enable_cmd_vel",
                default_value="false",
                description="Forward Nav2 /cmd_vel to STM32. Keep false for planning-only tests.",
            ),
            DeclareLaunchArgument(
                "enable_left_only_goal_proxy",
                default_value="false",
                description="Start optional left-only pre/final alignment goal proxy for A/B testing.",
            ),
            DeclareLaunchArgument(
                "max_linear_mps",
                default_value="0.40",
                description="STM32 cmd_vel linear clamp for navigation mode.",
            ),
            DeclareLaunchArgument(
                "max_angular_radps",
                default_value="3.50",
                description="STM32 cmd_vel angular clamp for autonomous navigation.",
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
                "log_pc_debug",
                default_value="false",
                description="Log PCDBG/RCDBG serial debug lines returned by STM32.",
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
                description="Nav2 parameter file.",
            ),
            DeclareLaunchArgument(
                "base_link_yaw_offset_deg",
                default_value="180.0",
                description="Fixed STM32 odometry axis to ROS base_link yaw correction.",
            ),
            DeclareLaunchArgument(
                "lidar_inverted",
                default_value="false",
                description="Fixed SLLIDAR scan inversion setting.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("caragent_navigation"),
                        "rviz",
                        "caragent_nav.rviz",
                    ]
                ),
                description="RViz config file.",
            ),
            localization_launch,
            map_server,
            map_server_lifecycle,
            nav2_navigation,
            left_only_goal_proxy,
            rviz,
            rviz_left_only,
        ]
    )
