ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true rgb_camera.color_profile:=1920x1080x30

ros2 launch planning lab7_bringup.launch.py

colcon build; source install/setup.bash;

ros2 run planning main

ros2 run ur7e_utils enable_comms
ros2 run ur7e_utils reset_state; ros2 run ur7e_utils tuck