import os
from glob import glob

from setuptools import find_packages, setup

package_name = "caragent_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="car",
    maintainer_email="car@todo.todo",
    description="CarAgent stereo camera bringup and image publishing.",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "stereo_camera_node = caragent_vision.stereo_camera_node:main",
            "capture_stereo_calibration = caragent_vision.capture_stereo_calibration:main",
            "calibrate_stereo_camera = caragent_vision.calibrate_stereo_camera:main",
            "live_lidar_camera_correspondences = caragent_vision.live_lidar_camera_correspondences:main",
            "calibrate_lidar_camera_extrinsics = caragent_vision.calibrate_lidar_camera_extrinsics:main",
        ],
    },
)
