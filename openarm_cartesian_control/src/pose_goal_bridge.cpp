#include "openarm_cartesian_control/pose_goal_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/exceptions.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_ros/buffer.h>
#include <moveit/utils/moveit_error_code.h>

namespace openarm_cartesian_control
{

PoseGoalBridge::PoseGoalBridge(const rclcpp::NodeOptions & options)
: rclcpp::Node("pose_goal_bridge", options),
  move_group_executor_(std::make_shared<rclcpp::executors::SingleThreadedExecutor>()),
  ready_(false)
{
  arm_ = this->declare_parameter<std::string>("arm", "left");
  std::string default_group = arm_ == "right" ? "right_arm" : "left_arm";
  std::string default_ee = arm_ == "right" ? "openarm_right_hand_tcp" : "openarm_left_hand_tcp";
  std::string default_frame = arm_ == "right" ? "openarm_right_link0" : "openarm_left_link0";

  planning_group_ = this->declare_parameter<std::string>("planning_group", default_group);
  ee_link_ = this->declare_parameter<std::string>("ee_link", default_ee);
  default_frame_ = this->declare_parameter<std::string>("planning_frame", default_frame);

  max_velocity_scaling_ = this->declare_parameter<double>("max_velocity_scaling", 0.3);
  max_acceleration_scaling_ = this->declare_parameter<double>("max_acceleration_scaling", 0.3);
  planning_time_limit_ = this->declare_parameter<double>("planning_time", 5.0);
  goal_position_tolerance_ = this->declare_parameter<double>("goal_position_tolerance", 0.005);
  goal_orientation_tolerance_ = this->declare_parameter<double>("goal_orientation_tolerance", 0.1);
  use_approximate_ik_ = this->declare_parameter<bool>("use_approximate_ik", true);
  execute_ = this->declare_parameter<bool>("execute", true);
  server_timeout_seconds_ = this->declare_parameter<double>("server_timeout", -1.0);

  auto qos = rclcpp::SystemDefaultsQoS();
  goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "goal_pose", qos,
    std::bind(&PoseGoalBridge::goal_pose_callback, this, std::placeholders::_1));

  result_pub_ = this->create_publisher<std_msgs::msg::String>("result", qos);

  initialize_move_group();
}

PoseGoalBridge::~PoseGoalBridge()
{
  if (move_group_executor_) {
    move_group_executor_->cancel();
  }
  if (move_group_spin_thread_.joinable()) {
    move_group_spin_thread_.join();
  }
  if (move_group_executor_ && move_group_node_) {
    move_group_executor_->remove_node(move_group_node_);
  }
}

void PoseGoalBridge::initialize_move_group()
{
  rclcpp::NodeOptions mg_options;
  mg_options.use_intra_process_comms(false);
  mg_options.automatically_declare_parameters_from_overrides(true);
  mg_options.arguments({"--ros-args", "-r", "joint_states:=/joint_states"});
  const std::string move_group_node_name = std::string(this->get_name()) + "_move_group";
  move_group_node_ = rclcpp::Node::make_shared(move_group_node_name, mg_options);

  const auto overrides = this->get_node_parameters_interface()->get_parameter_overrides();
  for (const auto & entry : overrides) {
    const auto & parameter_name = entry.first;
    const auto & parameter_value = entry.second;
    try {
      move_group_node_->declare_parameter(parameter_name, parameter_value);
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
      // Parameter already declared, nothing else to do.
    }
  }

  move_group_executor_->add_node(move_group_node_);
  move_group_spin_thread_ = std::thread([this]() {
    try {
      move_group_executor_->spin();
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(this->get_logger(), "Move group executor exception: %s", ex.what());
    }
  });

  rclcpp::Duration timeout = server_timeout_seconds_ < 0.0
    ? rclcpp::Duration::from_seconds(-1.0)
    : rclcpp::Duration::from_seconds(server_timeout_seconds_);

  try {
    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
      move_group_node_, planning_group_, std::shared_ptr<tf2_ros::Buffer>(), timeout);
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create MoveGroupInterface for group '%s': %s",
      planning_group_.c_str(), ex.what());
    move_group_executor_->cancel();
    if (move_group_spin_thread_.joinable()) {
      move_group_spin_thread_.join();
    }
    move_group_executor_->remove_node(move_group_node_);
    move_group_executor_.reset();
    move_group_node_.reset();
    return;
  }

  move_group_->setPoseReferenceFrame(default_frame_);
  move_group_->setEndEffectorLink(ee_link_);
  move_group_->setMaxVelocityScalingFactor(max_velocity_scaling_);
  move_group_->setMaxAccelerationScalingFactor(max_acceleration_scaling_);
  move_group_->setPlanningTime(planning_time_limit_);
  move_group_->setGoalPositionTolerance(goal_position_tolerance_);
  move_group_->setGoalOrientationTolerance(goal_orientation_tolerance_);

  if (!move_group_->startStateMonitor(5.0)) {
    RCLCPP_WARN(this->get_logger(), "Failed to start state monitor for MoveGroupInterface");
  }

  auto initial_state = move_group_->getCurrentState(5.0);
  if (!initial_state) {
    RCLCPP_WARN(this->get_logger(), "No current robot state received within timeout. Planning requests may fail.");
  }

  ready_ = true;
  RCLCPP_INFO(this->get_logger(), "PoseGoalBridge ready: arm=%s group=%s ee=%s frame=%s",
    arm_.c_str(), planning_group_.c_str(), ee_link_.c_str(), default_frame_.c_str());
}

void PoseGoalBridge::goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!ready_ || !move_group_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "MoveGroupInterface not ready yet; ignoring goal pose.");
    return;
  }

  if (!move_group_->getCurrentState(2.0)) {
    publish_result(false, "Current robot state unavailable", 0.0, 0.0, 0);
    RCLCPP_ERROR(this->get_logger(), "Current state unavailable before planning. Check joint state stream.");
    return;
  }

  std::lock_guard<std::mutex> guard(planning_mutex_);

  geometry_msgs::msg::PoseStamped target = *msg;
  if (target.header.frame_id.empty()) {
    target.header.frame_id = default_frame_;
  }

  move_group_->setPoseReferenceFrame(target.header.frame_id);
  move_group_->setEndEffectorLink(ee_link_);
  move_group_->setMaxVelocityScalingFactor(max_velocity_scaling_);
  move_group_->setMaxAccelerationScalingFactor(max_acceleration_scaling_);
  move_group_->setPlanningTime(planning_time_limit_);
  move_group_->setGoalPositionTolerance(goal_position_tolerance_);
  move_group_->setGoalOrientationTolerance(goal_orientation_tolerance_);
  move_group_->setStartStateToCurrentState();
  move_group_->clearPoseTargets();
  move_group_->setPoseTarget(target.pose, ee_link_);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto plan_error = move_group_->plan(plan);

  if (plan_error != moveit::core::MoveItErrorCode::SUCCESS) {
    if (use_approximate_ik_) {
      move_group_->clearPoseTargets();
      const bool approx_target_ok = move_group_->setApproximateJointValueTarget(target.pose, ee_link_);
      if (!approx_target_ok) {
        std::ostringstream reason;
        reason << "Approximate IK failed (" << moveit::core::error_code_to_string(plan_error) << ")";
        publish_result(false, reason.str(), 0.0, 0.0, 0);
        return;
      }
      plan_error = move_group_->plan(plan);
    }

    if (plan_error != moveit::core::MoveItErrorCode::SUCCESS) {
      std::ostringstream reason;
      reason << "Planning failed (" << moveit::core::error_code_to_string(plan_error) << ")";
      publish_result(false, reason.str(), 0.0, 0.0, 0);
      move_group_->clearPoseTargets();
      return;
    }
  }

  const auto & points = plan.trajectory_.joint_trajectory.points;
  if (points.empty()) {
    publish_result(false, "Empty trajectory returned", plan.planning_time_, 0.0, 0);
    move_group_->clearPoseTargets();
    return;
  }

  double trajectory_duration = 0.0;
  if (!points.empty()) {
    const auto & last_point = points.back();
    trajectory_duration = rclcpp::Duration(last_point.time_from_start).seconds();
  }

  if (!execute_) {
    publish_result(true, "Plan computed (execute disabled)", plan.planning_time_, trajectory_duration, points.size());
    move_group_->clearPoseTargets();
    return;
  }

  auto exec_result = move_group_->execute(plan);
  if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
    std::ostringstream reason;
    reason << "Execution failed (" << moveit::core::error_code_to_string(exec_result) << ")";
    publish_result(false, reason.str(), plan.planning_time_, trajectory_duration, points.size());
    move_group_->clearPoseTargets();
    return;
  }

  publish_result(true, "Motion executed", plan.planning_time_, trajectory_duration, points.size());
  move_group_->clearPoseTargets();
}

void PoseGoalBridge::publish_result(bool success, const std::string & reason, double planning_time,
  double trajectory_duration, std::size_t waypoint_count)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed, std::ios::floatfield);
  oss.precision(3);
  oss << (success ? "SUCCESS" : "FAIL") << ": " << reason;
  if (planning_time > 0.0) {
    oss << ", plan_time=" << planning_time << "s";
  }
  if (trajectory_duration > 0.0) {
    oss << ", trajectory_duration=" << trajectory_duration << "s";
  }
  if (waypoint_count > 0) {
    oss << ", waypoints=" << waypoint_count;
  }

  const auto summary = oss.str();
  if (success) {
    RCLCPP_INFO(this->get_logger(), "%s", summary.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(), "%s", summary.c_str());
  }

  if (result_pub_) {
    std_msgs::msg::String msg;
    msg.data = summary;
    result_pub_->publish(msg);
  }
}

}  // namespace openarm_cartesian_control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(false);
  auto node = std::make_shared<openarm_cartesian_control::PoseGoalBridge>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
