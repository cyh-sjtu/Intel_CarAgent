import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'caragent_ui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'static'), glob('caragent_ui/static/*')),
    ],
    package_data={package_name: ['static/*']},
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='car',
    maintainer_email='car@todo.todo',
    description='CarAgent system dashboard web UI',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'dashboard_node = caragent_ui.dashboard_node:main',
        ],
    },
)
