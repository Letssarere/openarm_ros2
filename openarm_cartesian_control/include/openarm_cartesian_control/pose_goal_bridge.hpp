#ifndef OPENARM_CARTESIAN_CONTROL__POSE_GOAL_BRIDGE_HPP_
#define OPENARM_CARTESIAN_CONTROL__POSE_GOAL_BRIDGE_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <std_msgs/msg/string.hpp>

namespace openarm_cartesian_control
{

class PoseGoalBridge : public rclcpp::Node
{
public:
  explicit PoseGoalBridge(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~PoseGoalBridge() override;

private:
  void initialize_move_group();
  void goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void publish_result(bool success, const std::string & reason, double planning_time = 0.0,
    double trajectory_duration = 0.0, std::size_t waypoint_count = 0);

  std::shared_ptr<rclcpp::Node> move_group_node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> move_group_executor_;
  std::thread move_group_spin_thread_;

  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::mutex planning_mutex_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_pub_;

  std::string arm_;
  std::string planning_group_;
  std::string ee_link_;
  std::string default_frame_;
  double max_velocity_scaling_;
  double max_acceleration_scaling_;
  double planning_time_limit_;
  double goal_position_tolerance_;
  double goal_orientation_tolerance_;
  bool use_approximate_ik_;
  bool execute_;
  double server_timeout_seconds_;
  bool ready_;
};

}  // namespace openarm_cartesian_control

#endif  // OPENARM_CARTESIAN_CONTROL__POSE_GOAL_BRIDGE_HPP_
