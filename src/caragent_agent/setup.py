from glob import glob
import os

from setuptools import find_packages, setup


package_name = "caragent_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "prompts"), glob("caragent_agent/prompts/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "networkx",
        "numpy",
        "Pillow",
        "PyYAML",
    ],
    zip_safe=True,
    maintainer="car",
    maintainer_email="car@todo.todo",
    description="CarAgent LLM agent with scene memory and Nav2 integration",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "agent_ros_node = caragent_agent.agent_ros_node:main",
            "agent_web_demo = caragent_agent.scripts.demo_ui.async_agent_web_demo:main",
            "annotate_keyframes = caragent_agent.scripts.annotate_keyframes:main",
        ],
    },
)
