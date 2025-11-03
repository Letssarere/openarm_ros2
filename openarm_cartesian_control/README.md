# openarm_cartesian_control

Tools for commanding OpenArm v10 TCP targets from Pose goals and querying current TCP poses.

## Nodes

### PoseGoal Bridge
- **Executable**: `pose_goal_bridge`
- **Namespace**: `/left_pose_bridge` or `/right_pose_bridge`
- **Input**: `~/goal_pose` (`geometry_msgs/PoseStamped`)
- **Output**: `~/result` (`std_msgs/String`) — textual status for the last request
- Plans and executes via `moveit::planning_interface::MoveGroupInterface` using the arm's MoveIt group and TCP link.

Start MoveIt fake hardware demo and launch the bridge for one arm:

```bash
ros2 launch openarm_bimanual_moveit_config demo.launch.py arm_type:=v10 use_fake_hardware:=true
ros2 launch openarm_cartesian_control cartesian_pose_bridge.launch.py arm:=left
# widen tolerances if needed, e.g. goal_orientation_tolerance:=0.3 use_approximate_ik:=true
ros2 topic pub /left_pose_bridge/goal_pose geometry_msgs/PoseStamped \
"{header:{frame_id:openarm_left_link0}, pose:{position:{x:0.45,y:0.10,z:0.20}, orientation:{w:1.0}}}"
```

Important parameters (see launch file defaults):
- `planning_group`: `left_arm` or `right_arm`
- `ee_link`: `openarm_left_hand_tcp` or `openarm_right_hand_tcp`
- `planning_frame`: TCP command frame (`openarm_*_link0` or `world` via TF)
- `max_velocity_scaling`, `max_acceleration_scaling`, `planning_time`, `server_timeout`
- `goal_position_tolerance`, `goal_orientation_tolerance` (defaults 5 mm / 0.1 rad)
- `use_approximate_ik` to fall back on MoveIt approximate IK solver when the direct pose target fails

### TF Probe Service
- **Executable**: `tf_probe_node`
- **Service**: `~/get_pose` (`openarm_cartesian_control/srv/GetPose`)
- Returns the pose of `tip_frame` expressed in `base_frame` using TF.

Example:
```bash
ros2 run openarm_cartesian_control tf_probe_node
ros2 service call /tf_probe/get_pose openarm_cartesian_control/srv/GetPose \
"{base_frame: openarm_left_link0, tip_frame: openarm_left_hand_tcp}"
```

## Static TF utility

When running in a world frame without a broadcaster, publish a fixed world→base transform:

```bash
ros2 launch openarm_cartesian_control tf_world_fix.launch.py child:=openarm_left_link0
```

## Build

The package uses `ament_cmake` and depends on MoveIt 2.

```bash
colcon build --packages-select openarm_cartesian_control
source install/setup.bash
```

## Development checklist
- Joint trajectory controllers active (`ros2 control list_controllers`)
- `tf_probe_node` returns valid base→TCP pose
- Pose goals plan and execute from `world` or arm base frames
