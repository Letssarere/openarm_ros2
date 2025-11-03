#!/usr/bin/env python3

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> List[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument("parent", default_value="world",
                              description="Parent frame for static transform"),
        DeclareLaunchArgument("child", default_value="openarm_left_link0",
                              description="Child frame for static transform"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.0"),
        DeclareLaunchArgument("roll", default_value="0.0"),
        DeclareLaunchArgument("pitch", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
    ]


def generate_launch_description() -> LaunchDescription:
    declared = declare_arguments()

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_base_static_tf",
        output="screen",
        arguments=[
            LaunchConfiguration("x"),
            LaunchConfiguration("y"),
            LaunchConfiguration("z"),
            LaunchConfiguration("roll"),
            LaunchConfiguration("pitch"),
            LaunchConfiguration("yaw"),
            LaunchConfiguration("parent"),
            LaunchConfiguration("child"),
        ],
    )

    return LaunchDescription(declared + [static_tf])

