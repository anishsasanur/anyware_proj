ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true rgb_camera.color_profile:=1920x1080x30

ros2 launch planning planning.launch.py

colcon build; source install/setup.bash

ros2 run planning dis

ros2 run ur7e_utils enable_comms

ros2 run ur7e_utils reset_state; ros2 run ur7e_utils tuck



numpy                                1.26.4
object-recognition-msgs              2.0.0
octomap-msgs                         2.0.1
open3d                               0.19.0
opencv-contrib-python                4.6.0.66
opencv-python                        4.6.0.66