#include <functional>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include "openarm_cartesian_control/srv/get_pose.hpp"

namespace openarm_cartesian_control
{

class TfProbeNode : public rclcpp::Node
{
public:
  explicit TfProbeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("tf_probe", options)
  {
    lookup_timeout_ = this->declare_parameter<double>("lookup_timeout", 0.2);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    service_ = this->create_service<openarm_cartesian_control::srv::GetPose>(
      "get_pose",
      std::bind(&TfProbeNode::handle_request, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "TF probe service ready (timeout=%.3fs)", lookup_timeout_);
  }

private:
  void handle_request(const std::shared_ptr<openarm_cartesian_control::srv::GetPose::Request> request,
    std::shared_ptr<openarm_cartesian_control::srv::GetPose::Response> response)
  {
    const std::string & base_frame = request->base_frame;
    const std::string & tip_frame = request->tip_frame;

    if (base_frame.empty() || tip_frame.empty()) {
      response->success = false;
      response->message = "Base and tip frames must be provided.";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    try {
      const auto timeout = tf2::durationFromSec(lookup_timeout_);
      const auto transform = tf_buffer_->lookupTransform(
        base_frame, tip_frame, tf2::TimePointZero, timeout);

      response->pose.header.stamp = transform.header.stamp;
      response->pose.header.frame_id = transform.header.frame_id;
      response->pose.pose.position.x = transform.transform.translation.x;
      response->pose.pose.position.y = transform.transform.translation.y;
      response->pose.pose.position.z = transform.transform.translation.z;
      response->pose.pose.orientation = transform.transform.rotation;
      response->success = true;
      response->message.clear();

      RCLCPP_DEBUG(this->get_logger(), "TF %s -> %s | xyz=(%.3f, %.3f, %.3f)",
        base_frame.c_str(), tip_frame.c_str(),
        response->pose.pose.position.x,
        response->pose.pose.position.y,
        response->pose.pose.position.z);
    } catch (const tf2::TransformException & ex) {
      response->success = false;
      response->message = ex.what();
      RCLCPP_WARN(this->get_logger(), "TF lookup failed (%s -> %s): %s",
        base_frame.c_str(), tip_frame.c_str(), ex.what());
    }
  }

  double lookup_timeout_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Service<openarm_cartesian_control::srv::GetPose>::SharedPtr service_;
};

}  // namespace openarm_cartesian_control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(false);
  auto node = std::make_shared<openarm_cartesian_control::TfProbeNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
