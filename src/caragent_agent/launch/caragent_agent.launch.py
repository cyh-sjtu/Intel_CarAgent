from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    dataset_dir = LaunchConfiguration("dataset_dir")

    default_config = PathJoinSubstitution(
        [FindPackageShare("caragent_agent"), "config", "config.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="CarAgent agent configuration file.",
            ),
            DeclareLaunchArgument(
                "dataset_dir",
                default_value="",
                description="Override scene_memory.dataset_dir (keyframe selected path).",
            ),
            Node(
                package="caragent_agent",
                executable="agent_ros_node",
                name="caragent_agent",
                output="screen",
                parameters=[{"config_file": config_file, "dataset_dir": dataset_dir}],
            ),
        ]
    )
