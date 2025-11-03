#!/usr/bin/env python3

from typing import List

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def declare_arguments() -> List[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument("arm", default_value="left", choices=["left", "right"],
                              description="Which arm to control"),
        DeclareLaunchArgument("execute", default_value="true",
                              description="If false, plan only without executing"),
        DeclareLaunchArgument("max_velocity_scaling", default_value="0.3"),
        DeclareLaunchArgument("max_acceleration_scaling", default_value="0.3"),
        DeclareLaunchArgument("planning_time", default_value="5.0"),
        DeclareLaunchArgument("goal_position_tolerance", default_value="0.005"),
        DeclareLaunchArgument("goal_orientation_tolerance", default_value="0.1"),
        DeclareLaunchArgument("use_approximate_ik", default_value="true"),
        DeclareLaunchArgument("server_timeout", default_value="-1.0",
                              description="Timeout (s) waiting for MoveIt action servers (-1 for infinite)"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]


def launch_setup(context: LaunchContext, *_) -> List[Node]:
    arm_value = LaunchConfiguration("arm").perform(context)
    execute_value = LaunchConfiguration("execute").perform(context).lower() in ("true", "1")
    max_vel = float(LaunchConfiguration("max_velocity_scaling").perform(context))
    max_acc = float(LaunchConfiguration("max_acceleration_scaling").perform(context))
    planning_time = float(LaunchConfiguration("planning_time").perform(context))
    goal_position_tol = float(LaunchConfiguration("goal_position_tolerance").perform(context))
    goal_orientation_tol = float(LaunchConfiguration("goal_orientation_tolerance").perform(context))
    use_approximate_ik = LaunchConfiguration("use_approximate_ik").perform(context).lower() in ("true", "1")
    server_timeout = float(LaunchConfiguration("server_timeout").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("true", "1")

    group = "right_arm" if arm_value == "right" else "left_arm"
    ee_link = "openarm_right_hand_tcp" if arm_value == "right" else "openarm_left_hand_tcp"
    base_frame = "openarm_right_link0" if arm_value == "right" else "openarm_left_link0"

    moveit_config = MoveItConfigsBuilder(
        "openarm", package_name="openarm_bimanual_moveit_config"
    ).to_moveit_configs()
    moveit_params = moveit_config.to_dict()

    node_params = {
        "arm": arm_value,
        "planning_group": group,
        "ee_link": ee_link,
        "planning_frame": base_frame,
        "max_velocity_scaling": max_vel,
        "max_acceleration_scaling": max_acc,
        "planning_time": planning_time,
        "goal_position_tolerance": goal_position_tol,
        "goal_orientation_tolerance": goal_orientation_tol,
        "use_approximate_ik": use_approximate_ik,
        "execute": execute_value,
        "server_timeout": server_timeout,
        "use_sim_time": use_sim_time,
    }

    moveit_params.update(node_params)

    node = Node(
        package="openarm_cartesian_control",
        executable="pose_goal_bridge",
        name=f"{arm_value}_pose_bridge",
        namespace=f"{arm_value}_pose_bridge",
        output="screen",
        remappings=[
            ("joint_states", "/joint_states"),
        ],
        parameters=[moveit_params],
    )

    return [node]


def generate_launch_description() -> LaunchDescription:
    declared = declare_arguments()
    return LaunchDescription(declared + [OpaqueFunction(function=launch_setup)])
