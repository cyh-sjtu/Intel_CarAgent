import os
from glob import glob

from setuptools import find_packages, setup

package_name = "caragent_memory"

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
    description="CarAgent keyframe recording and OpenVINO CLIP selection tools.",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "keyframe_recorder_node = caragent_memory.keyframe_recorder_node:main",
            "select_keyframes = caragent_memory.select_keyframes:main",
            "build_scene_memory = caragent_memory.build_scene_memory:main",
            "convert_clip_openvino = caragent_memory.convert_clip_openvino:main",
            "convert_dinov2_openvino = caragent_memory.convert_dinov2_openvino:main",
        ],
    },
)
