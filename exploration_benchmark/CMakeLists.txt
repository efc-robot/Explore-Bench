cmake_minimum_required(VERSION 3.0.2)
project(exploration_benchmark)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cartographer_ros_msgs
  geometry_msgs
  message_generation
  message_runtime
  nav_msgs
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
  message_filters
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
)

catkin_package( 
CATKIN_DEPENDS actionlib_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(rrt_node src/rrt_node_tf.cpp src/functions.cpp src/mtrand.cpp)
target_link_libraries(rrt_node ${catkin_LIBRARIES})

add_executable(mmpf_node src/mmpf_node.cpp)
target_link_libraries(mmpf_node ${catkin_LIBRARIES})

add_executable(cost_node src/cost_node.cpp)
target_link_libraries(cost_node ${catkin_LIBRARIES})
