from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    backend = LaunchConfiguration("backend")
    device = LaunchConfiguration("device")
    video_format = LaunchConfiguration("video_format")
    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    fps = LaunchConfiguration("fps")
    left_width = LaunchConfiguration("left_width")
    right_width = LaunchConfiguration("right_width")
    rtbufsize = LaunchConfiguration("rtbufsize")
    calib_file = LaunchConfiguration("calib_file")
    show_image = LaunchConfiguration("show_image")
    display_max_width = LaunchConfiguration("display_max_width")
    display_max_height = LaunchConfiguration("display_max_height")
    display_scale = LaunchConfiguration("display_scale")
    publish_raw = LaunchConfiguration("publish_raw")
    publish_left = LaunchConfiguration("publish_left")
    publish_right = LaunchConfiguration("publish_right")
    publish_rect = LaunchConfiguration("publish_rect")
    publish_disparity = LaunchConfiguration("publish_disparity")

    camera_node = Node(
        package="caragent_vision",
        executable="stereo_camera_node",
        name="caragent_stereo_camera_node",
        output="screen",
        parameters=[
            {
                "backend": backend,
                "device": device,
                "video_format": video_format,
                "width": ParameterValue(width, value_type=int),
                "height": ParameterValue(height, value_type=int),
                "fps": ParameterValue(fps, value_type=float),
                "left_width": ParameterValue(left_width, value_type=int),
                "right_width": ParameterValue(right_width, value_type=int),
                "rtbufsize": rtbufsize,
                "calib_file": calib_file,
                "show_image": ParameterValue(show_image, value_type=bool),
                "display_max_width": ParameterValue(display_max_width, value_type=int),
                "display_max_height": ParameterValue(display_max_height, value_type=int),
                "display_scale": ParameterValue(display_scale, value_type=float),
                "publish_raw": ParameterValue(publish_raw, value_type=bool),
                "publish_left": ParameterValue(publish_left, value_type=bool),
                "publish_right": ParameterValue(publish_right, value_type=bool),
                "publish_rect": ParameterValue(publish_rect, value_type=bool),
                "publish_disparity": ParameterValue(publish_disparity, value_type=bool),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "backend",
                default_value="pyav",
                description="Camera backend: pyav (recommended) or opencv.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="/dev/video0",
                description="Huibo stereo UVC device. /dev/video<N>, 'video=USB2.0 Camera RGB', or numeric index.",
            ),
            DeclareLaunchArgument(
                "video_format",
                default_value="auto",
                description="PyAV format: auto (detect from OS), dshow (Windows), v4l2 (Linux).",
            ),
            DeclareLaunchArgument(
                "width",
                default_value="3840",
                description="Side-by-side stereo frame width.",
            ),
            DeclareLaunchArgument(
                "height",
                default_value="1200",
                description="Side-by-side stereo frame height.",
            ),
            DeclareLaunchArgument(
                "fps",
                default_value="30.0",
                description="Requested camera FPS.",
            ),
            DeclareLaunchArgument(
                "left_width",
                default_value="1920",
                description="Left image width in the side-by-side frame.",
            ),
            DeclareLaunchArgument(
                "right_width",
                default_value="1920",
                description="Right image width in the side-by-side frame.",
            ),
            DeclareLaunchArgument(
                "rtbufsize",
                default_value="64M",
                description="PyAV V4L2/DShow buffer size.",
            ),
            DeclareLaunchArgument(
                "calib_file",
                default_value="",
                description="Path to stereo_calibration.npz (optional).",
            ),
            DeclareLaunchArgument(
                "show_image",
                default_value="true",
                description="Show OpenCV preview window.",
            ),
            DeclareLaunchArgument(
                "display_max_width",
                default_value="1600",
                description="Maximum preview window/image width. 0 disables width limit.",
            ),
            DeclareLaunchArgument(
                "display_max_height",
                default_value="900",
                description="Maximum preview window/image height. 0 disables height limit.",
            ),
            DeclareLaunchArgument(
                "display_scale",
                default_value="0.0",
                description="Fixed preview scale. 0.0 means auto-fit to display_max_width/height.",
            ),
            DeclareLaunchArgument(
                "publish_raw",
                default_value="true",
                description="Publish raw side-by-side frame on /stereo/image_raw.",
            ),
            DeclareLaunchArgument(
                "publish_left",
                default_value="true",
                description="Publish left image on /stereo/left/image_raw.",
            ),
            DeclareLaunchArgument(
                "publish_right",
                default_value="true",
                description="Publish right image on /stereo/right/image_raw.",
            ),
            DeclareLaunchArgument(
                "publish_rect",
                default_value="false",
                description="Publish rectified left/right images (requires calib_file).",
            ),
            DeclareLaunchArgument(
                "publish_disparity",
                default_value="false",
                description="Publish disparity map on /stereo/disparity (requires calib_file + publish_rect).",
            ),
            camera_node,
        ]
    )
