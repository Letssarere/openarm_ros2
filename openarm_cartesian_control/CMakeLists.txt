cmake_minimum_required(VERSION 3.10)
project(openarm_cartesian_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  rclcpp
  std_msgs
  tf2_ros
  tf2_geometry_msgs
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetPose.srv"
  DEPENDENCIES geometry_msgs
)

add_executable(pose_goal_bridge src/pose_goal_bridge.cpp)
target_include_directories(pose_goal_bridge PRIVATE include)
ament_target_dependencies(pose_goal_bridge ${THIS_PACKAGE_INCLUDE_DEPENDS})

add_executable(tf_probe_node src/tf_probe_node.cpp)
target_include_directories(tf_probe_node PRIVATE include)
ament_target_dependencies(tf_probe_node ${THIS_PACKAGE_INCLUDE_DEPENDS})

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(tf_probe_node ${cpp_typesupport_target})

install(TARGETS pose_goal_bridge tf_probe_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_export_dependencies(rosidl_default_runtime)

ament_package()
