from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    laser_port = LaunchConfiguration("laser_port")
    stm32_port = LaunchConfiguration("stm32_port")
    map_file_name = LaunchConfiguration("map_file_name")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_nav_rviz = LaunchConfiguration("use_nav_rviz")
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    use_map_server = LaunchConfiguration("use_map_server")
    map_yaml_file = LaunchConfiguration("map_yaml_file")
    base_link_yaw_offset_deg = LaunchConfiguration("base_link_yaw_offset_deg")
    lidar_inverted = LaunchConfiguration("lidar_inverted")
    enable_cmd_vel = LaunchConfiguration("enable_cmd_vel")

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
            "use_sim_time": use_sim_time,
            "use_rviz": "false",
            "enable_cmd_vel": enable_cmd_vel,
            "cmd_vel_topic": "/cmd_vel",
            "cmd_send_rate_hz": "30.0",
            "cmd_timeout_sec": "0.25",
            "max_linear_mps": "0.28",
            "max_angular_radps": "0.85",
            "log_cmd_serial": "true",
            "log_pc_debug": "true",
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
        condition=IfCondition(use_nav_rviz),
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
            rviz,
        ]
    )
