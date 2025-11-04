from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare('openarm_rl_inference')
    param_file = PathJoinSubstitution([pkg_share, 'config', 'rl_inference.yaml'])

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/best_agent.pt',
        description='Path to the RL policy checkpoint file.',
    )
    scaler_path_arg = DeclareLaunchArgument(
        'scaler_fallback_path',
        default_value='',
        description='Optional path to obs_scaler.npz when checkpoint lacks stats.',
    )
    rate_arg = DeclareLaunchArgument(
        'rate_hz',
        default_value='30.0',
        description='Inference loop frequency in Hz.',
    )
    controller_name_arg = DeclareLaunchArgument(
        'controller_name',
        default_value='left_joint_trajectory_controller',
        description='Controller name used within the controller namespace.',
    )
    controller_topic_arg = DeclareLaunchArgument(
        'controller_topic',
        default_value='/left_joint_trajectory_controller/joint_trajectory',
        description='JointTrajectory topic for the left arm controller.',
    )
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/left_pose_command',
        description='PoseStamped topic supplying end-effector targets.',
    )

    node = Node(
        package='openarm_rl_inference',
        executable='openarm_rl_inference',
        output='screen',
        parameters=[
            param_file,
            {
                'model': {
                    'path': LaunchConfiguration('model_path'),
                },
                'inference': {
                    'rate_hz': LaunchConfiguration('rate_hz'),
                },
                'controller': {
                    'name': LaunchConfiguration('controller_name'),
                    'topic': LaunchConfiguration('controller_topic'),
                },
                'pose': {
                    'topic': LaunchConfiguration('pose_topic'),
                },
                'scaler': {
                    'fallback_path': LaunchConfiguration('scaler_fallback_path'),
                },
            },
        ],
    )

    return LaunchDescription([
        model_path_arg,
        scaler_path_arg,
        rate_arg,
        controller_name_arg,
        controller_topic_arg,
        pose_topic_arg,
        node,
    ])
