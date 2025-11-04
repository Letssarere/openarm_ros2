import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .ckpt_loader import load_ckpt
from .kinematics import quat_normalize
from .safety import clamp_step, clip_joint_limits, ema
from .scaling import load_scaler_from_npz, standardize


class OpenArmRLInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__('openarm_rl_inference')
        self._clock: Clock = self.get_clock()
        self._declare_parameters()
        self._load_policy()
        self._init_state()
        self._create_interfaces()
        self._start_time = self._clock.now()
        self._wait_timeout = Duration(seconds=float(self.get_parameter('boot.wait_for_topics_sec').value))
        rate_hz = float(self.get_parameter('inference.rate_hz').value)
        if rate_hz <= 0.0:
            raise ValueError('inference.rate_hz must be > 0')
        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)

    def _declare_parameters(self) -> None:
        self.declare_parameters('', [
            ('model.path', 'models/best_agent.pt'),
            ('model.device', 'cpu'),
            ('model.deterministic', True),
            ('inference.rate_hz', 30.0),
            ('inference.action_scale', 0.5),
            ('inference.ema_alpha', 0.2),
            ('inference.step_delta_limit', 0.05),
            ('inference.offset_mode', 'current'),
            ('controller.name', 'left_joint_trajectory_controller'),
            ('controller.topic', '/left_joint_trajectory_controller/joint_trajectory'),
            (
                'joints.names',
                [
                    'openarm_left_joint1',
                    'openarm_left_joint2',
                    'openarm_left_joint3',
                    'openarm_left_joint4',
                    'openarm_left_joint5',
                    'openarm_left_joint6',
                    'openarm_left_joint7',
                ],
            ),
            (
                'joints.limits.lower',
                [-3.490659, -3.316125, -1.570796, 0.0, -1.570796, -0.785398, -1.570796],
            ),
            (
                'joints.limits.upper',
                [1.396263, 0.174533, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796],
            ),
            ('pose.frame', 'openarm_left_link0'),
            ('pose.topic', '/left_pose_command'),
            ('boot.warmup_zero_actions', 3),
            ('boot.wait_for_topics_sec', 5.0),
            ('staleness.pose_timeout_sec', 0.5),
            ('staleness.joint_timeout_sec', 0.5),
            ('scaler.fallback_path', ''),
        ])

    def _load_policy(self) -> None:
        model_path = Path(self.get_parameter('model.path').value).expanduser()
        device_str = self.get_parameter('model.device').value
        self._device = torch.device(device_str)
        info = load_ckpt(model_path, device=device_str)
        self._policy = info['policy']
        if hasattr(self._policy, 'to'):
            try:
                self._policy.to(self._device)
            except Exception:
                pass
        self._policy.eval()
        self._obs_mean = np.asarray(info.get('mean'), dtype=np.float32)
        self._obs_std = np.asarray(info.get('std'), dtype=np.float32)
        if self._obs_mean.shape[0] != 28 or self._obs_std.shape[0] != 28:
            raise ValueError('Expected 28-D observation statistics')
        fallback_path = self.get_parameter('scaler.fallback_path').value
        if fallback_path:
            mean, std = load_scaler_from_npz(fallback_path)
            if mean.shape[0] != 28 or std.shape[0] != 28:
                raise ValueError('Fallback scaler must be 28-D')
            self._obs_mean = mean
            self._obs_std = std
        det_param = self.get_parameter('model.deterministic').value
        self._deterministic = bool(det_param)

    def _init_state(self) -> None:
        joint_names = list(self.get_parameter('joints.names').value)
        if len(joint_names) != 7:
            raise ValueError('joints.names must list 7 joints for the left arm')
        self._joint_names = joint_names
        self._joint_limits_lower = np.asarray(self.get_parameter('joints.limits.lower').value, dtype=np.float32)
        self._joint_limits_upper = np.asarray(self.get_parameter('joints.limits.upper').value, dtype=np.float32)
        if self._joint_limits_lower.shape[0] != 7 or self._joint_limits_upper.shape[0] != 7:
            raise ValueError('Joint limit arrays must have length 7')
        self._joint_index_order: Optional[np.ndarray] = None
        self._home_position: Optional[np.ndarray] = None
        self._q = np.zeros(7, dtype=np.float32)
        self._dq = np.zeros(7, dtype=np.float32)
        self._pose_xyzquat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._last_action = np.zeros(7, dtype=np.float32)
        self._last_q_target: Optional[np.ndarray] = None
        self._joint_state_time: Optional[Time] = None
        self._pose_time: Optional[Time] = None
        self._joint_ready = False
        self._pose_ready = False
        self._joint_warned = False
        self._pose_warned = False
        self._warmup_counter = int(self.get_parameter('boot.warmup_zero_actions').value)
        self._action_scale = float(self.get_parameter('inference.action_scale').value)
        self._ema_alpha = float(self.get_parameter('inference.ema_alpha').value)
        self._step_delta_limit = float(self.get_parameter('inference.step_delta_limit').value)
        self._offset_mode = str(self.get_parameter('inference.offset_mode').value)
        self._pose_timeout = Duration(seconds=float(self.get_parameter('staleness.pose_timeout_sec').value))
        self._joint_timeout = Duration(seconds=float(self.get_parameter('staleness.joint_timeout_sec').value))

    def _create_interfaces(self) -> None:
        joint_topic = '/joint_states'
        pose_topic_param = self.get_parameter('pose.topic').value
        pose_topic = str(pose_topic_param).strip() if pose_topic_param else '/left_pose_command'
        if pose_topic and not pose_topic.startswith('/'):
            pose_topic = f'/{pose_topic}'

        controller_name = str(self.get_parameter('controller.name').value)
        topic_param = self.get_parameter('controller.topic').value
        traj_topic = str(topic_param).strip() if topic_param else ''
        if not traj_topic:
            base_name = controller_name or 'left_joint_trajectory_controller'
            traj_topic = f'/{base_name}/joint_trajectory'
        elif not traj_topic.startswith('/'):
            traj_topic = f'/{traj_topic}'

        self._controller_name = controller_name
        self._controller_topic = traj_topic

        self.create_subscription(JointState, joint_topic, self._on_joint_state, 20)
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 20)
        self._traj_publisher = self.create_publisher(JointTrajectory, traj_topic, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name:
            return
        if self._joint_index_order is None:
            try:
                indices = [msg.name.index(name) for name in self._joint_names]
            except ValueError as exc:
                self.get_logger().error('JointState missing required joint: %s', exc)
                return
            self._joint_index_order = np.array(indices, dtype=np.int32)
            self.get_logger().info('JointState indices mapped for left arm')
        idx = self._joint_index_order
        positions = np.array([msg.position[i] for i in idx], dtype=np.float32)
        self._q = positions
        if msg.velocity and len(msg.velocity) >= len(msg.position):
            velocities = np.array([msg.velocity[i] for i in idx], dtype=np.float32)
            self._dq = velocities
        else:
            self._dq = np.zeros_like(self._dq)
        if self._home_position is None:
            self._home_position = positions.copy()
        self._joint_ready = True
        self._joint_state_time = self._clock.now()
        self._joint_warned = False

    def _on_pose(self, msg: PoseStamped) -> None:
        orientation = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        quat = quat_normalize(orientation)
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self._pose_xyzquat = np.hstack((np.asarray(position, dtype=np.float32), quat))
        self._pose_time = self._clock.now()
        self._pose_ready = True
        self._pose_warned = False

    def _on_timer(self) -> None:
        now = self._clock.now()
        if not self._joint_ready or not self._pose_ready:
            if now - self._start_time > self._wait_timeout:
                self.get_logger().warn('Waiting for initial JointState/Pose messages...')
                self._start_time = now
            return
        if self._joint_state_time is not None and now - self._joint_state_time > self._joint_timeout:
            if not self._joint_warned:
                self.get_logger().warn('JointState topic stale; holding command publish')
                self._joint_warned = True
            return
        pose_stale = False
        if self._pose_time is None or now - self._pose_time > self._pose_timeout:
            pose_stale = True
            if not self._pose_warned:
                self.get_logger().warn('Pose command stale; holding last target')
                self._pose_warned = True
        if pose_stale and self._last_q_target is not None:
            self._publish_trajectory(self._last_q_target)
            return
        if self._warmup_counter > 0:
            self._warmup_counter -= 1
            self._last_action = np.zeros_like(self._last_action)
            offset = self._select_offset()
            q_target = offset.copy()
            if self._last_q_target is not None and self._step_delta_limit > 0.0:
                q_target = clamp_step(self._last_q_target, q_target, self._step_delta_limit)
            q_target = clip_joint_limits(q_target, self._joint_limits_lower, self._joint_limits_upper)
            self._last_q_target = q_target.astype(np.float32)
            self._publish_trajectory(self._last_q_target)
            return
        obs = np.concatenate((self._q, self._dq, self._pose_xyzquat, self._last_action)).astype(np.float32)
        obs_std = standardize(obs, self._obs_mean, self._obs_std)
        obs_tensor = torch.from_numpy(obs_std).to(self._device).unsqueeze(0)
        with torch.no_grad():
            policy_out = self._policy(obs_tensor)
        action = self._extract_action(policy_out)
        action = np.clip(action, -1.0, 1.0)
        if self._ema_alpha > 0.0:
            action = ema(self._last_action, action, self._ema_alpha)
        offset = self._select_offset()
        q_target = offset + self._action_scale * action
        if self._last_q_target is not None and self._step_delta_limit > 0.0:
            q_target = clamp_step(self._last_q_target, q_target, self._step_delta_limit)
        q_target = clip_joint_limits(q_target, self._joint_limits_lower, self._joint_limits_upper)
        if self._action_scale > 0.0:
            effective_action = np.clip((q_target - offset) / self._action_scale, -1.0, 1.0)
        else:
            effective_action = np.zeros_like(action)
        self._last_action = effective_action.astype(np.float32)
        self._last_q_target = q_target.astype(np.float32)
        self._publish_trajectory(self._last_q_target)

    def _extract_action(self, policy_out) -> np.ndarray:
        if isinstance(policy_out, (list, tuple)) and policy_out:
            action_tensor = policy_out[0]
            std_tensor = policy_out[1] if len(policy_out) > 1 else None
        else:
            action_tensor = policy_out
            std_tensor = None
        if not torch.is_tensor(action_tensor):
            raise TypeError('Policy output must be a torch.Tensor')
        if self._deterministic or std_tensor is None:
            tensor = action_tensor
        else:
            if not torch.is_tensor(std_tensor):
                raise TypeError('Expected tensor std from stochastic policy')
            std = torch.exp(std_tensor)
            dist = torch.distributions.Normal(action_tensor, std)
            tensor = dist.sample()
        action = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if action.shape[0] != 7:
            raise ValueError(f'Policy action should have 7 dims, got {action.shape}')
        return action

    def _select_offset(self) -> np.ndarray:
        mode = self._offset_mode.lower()
        if mode == 'current':
            return self._q.copy()
        if mode == 'home' and self._home_position is not None:
            return self._home_position.copy()
        return self._q.copy()

    def _publish_trajectory(self, q_target: np.ndarray) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = q_target.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(math.floor(0.18 * 1e9))
        msg.points = [point]
        self._traj_publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = OpenArmRLInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = ['main', 'OpenArmRLInferenceNode']
