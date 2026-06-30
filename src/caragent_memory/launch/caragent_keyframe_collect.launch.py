from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    session_name = LaunchConfiguration("session_name")
    output_root = LaunchConfiguration("output_root")
    laser_port = LaunchConfiguration("laser_port")
    stm32_port = LaunchConfiguration("stm32_port")
    camera_device = LaunchConfiguration("camera_device")
    camera_backend = LaunchConfiguration("camera_backend")
    camera_show_image = LaunchConfiguration("camera_show_image")
    camera_calib_file = LaunchConfiguration("camera_calib_file")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_left_width = LaunchConfiguration("camera_left_width")
    camera_right_width = LaunchConfiguration("camera_right_width")
    camera_fps = LaunchConfiguration("camera_fps")
    map_file_name = LaunchConfiguration("map_file_name")
    map_start_at_dock = LaunchConfiguration("map_start_at_dock")
    use_rviz = LaunchConfiguration("use_rviz")
    lidar_inverted = LaunchConfiguration("lidar_inverted")
    lidar_angle_compensate = LaunchConfiguration("lidar_angle_compensate")
    lidar_scan_mode = LaunchConfiguration("lidar_scan_mode")

    min_time_sec = LaunchConfiguration("min_time_sec")
    min_distance_m = LaunchConfiguration("min_distance_m")
    min_yaw_deg = LaunchConfiguration("min_yaw_deg")
    manual_only = LaunchConfiguration("manual_only")
    init_pose_delay_sec = LaunchConfiguration("init_pose_delay_sec")
    max_tf_age_sec = LaunchConfiguration("max_tf_age_sec")
    enforce_tf_age = LaunchConfiguration("enforce_tf_age")
    collect_max_linear_mps = LaunchConfiguration("collect_max_linear_mps")
    collect_max_angular_radps = LaunchConfiguration("collect_max_angular_radps")

    default_map_file = PathJoinSubstitution(
        [
            EnvironmentVariable("HOME"),
            "caragent_ws",
            "maps",
            "map_202605191353",
        ]
    )

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
            "use_rviz": use_rviz,
            "enable_cmd_vel": "true",
            "cmd_vel_topic": "/cmd_vel",
            "cmd_send_rate_hz": "20.0",
            "cmd_timeout_sec": "0.3",
            "max_linear_mps": collect_max_linear_mps,
            "max_angular_radps": collect_max_angular_radps,
            "lidar_inverted": lidar_inverted,
            "lidar_angle_compensate": lidar_angle_compensate,
            "lidar_scan_mode": lidar_scan_mode,
        }.items(),
    )

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
            "backend": camera_backend,
            "device": camera_device,
            "calib_file": camera_calib_file,
            "show_image": camera_show_image,
            "width": camera_width,
            "height": camera_height,
            "left_width": camera_left_width,
            "right_width": camera_right_width,
            "fps": camera_fps,
            "publish_raw": "true",
            "publish_left": "true",
            "publish_right": "true",
        }.items(),
    )

    recorder = Node(
        package="caragent_memory",
        executable="keyframe_recorder_node",
        name="keyframe_recorder_node",
        output="screen",
        parameters=[
            {
                "image_topic": "/stereo/image_raw",
                "scan_topic": "/scan",
                "odom_topic": "/odom",
                "map_frame": "map",
                "base_frame": "base_link",
                "map_file_name": map_file_name,
                "session_name": session_name,
                "output_root": output_root,
                "left_width": ParameterValue(camera_left_width, value_type=int),
                "right_width": ParameterValue(camera_right_width, value_type=int),
                "min_time_sec": ParameterValue(min_time_sec, value_type=float),
                "min_distance_m": ParameterValue(min_distance_m, value_type=float),
                "min_yaw_deg": ParameterValue(min_yaw_deg, value_type=float),
                "manual_only": ParameterValue(manual_only, value_type=bool),
                "init_pose_delay_sec": ParameterValue(init_pose_delay_sec, value_type=float),
                "max_tf_age_sec": ParameterValue(max_tf_age_sec, value_type=float),
                "enforce_tf_age": ParameterValue(enforce_tf_age, value_type=bool),
                "use_latest_tf_on_failure": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("session_name", default_value=""),
            DeclareLaunchArgument(
                "output_root",
                default_value=PathJoinSubstitution([EnvironmentVariable("HOME"), "caragent_ws", "keyframes"]),
            ),
            DeclareLaunchArgument("laser_port", default_value="/dev/ttyUSB1"),
            DeclareLaunchArgument("stm32_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
            DeclareLaunchArgument("camera_backend", default_value="pyav"),
            DeclareLaunchArgument("camera_show_image", default_value="false"),
            DeclareLaunchArgument("camera_calib_file", default_value=""),
            DeclareLaunchArgument("camera_width", default_value="3840"),
            DeclareLaunchArgument("camera_height", default_value="1200"),
            DeclareLaunchArgument("camera_left_width", default_value="1920"),
            DeclareLaunchArgument("camera_right_width", default_value="1920"),
            DeclareLaunchArgument("camera_fps", default_value="30.0"),
            DeclareLaunchArgument("map_file_name", default_value=default_map_file),
            DeclareLaunchArgument("map_start_at_dock", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("lidar_inverted", default_value="false"),
            DeclareLaunchArgument("lidar_angle_compensate", default_value="true"),
            DeclareLaunchArgument("lidar_scan_mode", default_value="Standard"),
            DeclareLaunchArgument("min_time_sec", default_value="1.5"),
            DeclareLaunchArgument("min_distance_m", default_value="0.65"),
            DeclareLaunchArgument("min_yaw_deg", default_value="30.0"),
            DeclareLaunchArgument("manual_only", default_value="false"),
            DeclareLaunchArgument("init_pose_delay_sec", default_value="3.0"),
            DeclareLaunchArgument("max_tf_age_sec", default_value="0.5"),
            DeclareLaunchArgument("enforce_tf_age", default_value="false"),
            DeclareLaunchArgument("collect_max_linear_mps", default_value="0.16"),
            DeclareLaunchArgument("collect_max_angular_radps", default_value="0.45"),
            localization_launch,
            camera_launch,
            recorder,
        ]
    )
