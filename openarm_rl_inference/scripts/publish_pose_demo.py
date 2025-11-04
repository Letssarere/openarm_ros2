#!/usr/bin/env python3
import math
import random
from typing import Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from openarm_rl_inference.kinematics import rpy_to_quat

XYZ_RANGE = {
    'x': (0.20, 0.45),
    'y': (-0.05, 0.45),
    'z': (0.40, 0.70),
}
ROLL_RANGE = (-0.7, 0.7)
PITCH_FIXED = 4.712
YAW_FIXED = 1.571


class PoseDemoPublisher(Node):
    def __init__(self) -> None:
        super().__init__('openarm_pose_demo')
        self.declare_parameter('frame', 'base_link')
        self.declare_parameter('topic', '/left_pose_command')
        self.declare_parameter('rate_hz', 6.0)
        topic = self.get_parameter('topic').value
        self._frame = str(self.get_parameter('frame').value)
        rate = float(self.get_parameter('rate_hz').value)
        self._publisher = self.create_publisher(PoseStamped, topic, 10)
        self._phase = random.random() * 2.0 * math.pi
        self._timer = self.create_timer(1.0 / max(rate, 1.0), self._on_timer)

    def _sample_xyz(self, phase: float) -> Tuple[float, float, float]:
        x_min, x_max = XYZ_RANGE['x']
        y_min, y_max = XYZ_RANGE['y']
        z_min, z_max = XYZ_RANGE['z']
        x = (x_min + x_max) * 0.5 + (x_max - x_min) * 0.4 * math.sin(phase)
        y = (y_min + y_max) * 0.5 + (y_max - y_min) * 0.4 * math.cos(phase * 0.7)
        z = (z_min + z_max) * 0.5 + (z_max - z_min) * 0.35 * math.sin(phase * 0.5)
        return x, y, z

    def _on_timer(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        x, y, z = self._sample_xyz(self._phase)
        msg.pose.position.x = float(np.clip(x, *XYZ_RANGE['x']))
        msg.pose.position.y = float(np.clip(y, *XYZ_RANGE['y']))
        msg.pose.position.z = float(np.clip(z, *XYZ_RANGE['z']))
        roll = np.clip(0.5 * math.sin(self._phase), *ROLL_RANGE)
        quat = rpy_to_quat(roll, PITCH_FIXED, YAW_FIXED)
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._publisher.publish(msg)
        self._phase += 0.12
        if self._phase > 2.0 * math.pi:
            self._phase -= 2.0 * math.pi


def main() -> None:
    rclpy.init()
    node = PoseDemoPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
