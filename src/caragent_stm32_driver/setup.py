import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'caragent_stm32_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='car',
    maintainer_email='car@todo.todo',
    description='STM32 serial driver: odometry bridge and cmd_vel relay',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stm32_driver_node = caragent_stm32_driver.stm32_driver_node:main',
        ],
    },
)
