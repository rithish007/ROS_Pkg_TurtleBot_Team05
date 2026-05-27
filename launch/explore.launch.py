#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # SLAM
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('tuos_tb3_tools'),
                'launch',
                'slam.launch.py'
            )
        ),
        # Default environment is 'real' — change to environment:=sim for Gazebo
        # launch_arguments={'environment': 'sim'}.items(),
    )

    # Zone Exploration Node
    nav_node = Node(
        package='ele434_team05_2026',
        executable='zone_exploration.py',
        name='explore_node',
        output='screen',
        sigterm_timeout='10',
    )

    nav_node_delayed = TimerAction(
        period=5.0,
        actions=[nav_node]
    )

    return LaunchDescription([
        slam_launch,
        nav_node_delayed,
    ])
