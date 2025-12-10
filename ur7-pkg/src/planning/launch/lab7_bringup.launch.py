from launch.events import Shutdown
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, EmitEvent, ExecuteProcess
import os

def generate_launch_description():
    # RealSense (include rs_launch.py)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )

    # Block detection node
    block_detection_node = Node(
        package='perception',
        executable='block_detection',
        name='block_detection',
        output='screen'
    )

    # # GUI to robot node
    # gui_to_robot_node = Node(
    #     package='planning',
    #     executable='gui_to_robot',
    #     name='gui_to_robot',
    #     output='screen'
    # )

    # ArUco recognition
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros2_aruco'),
                'launch',
                'aruco_recognition.launch.py'
            )
        )
    )

    # IK node
    ik_node = Node (
        package='planning',
        executable='ik',
        name='ik_node',
        output='screen'
    )

    # Transform cube pose node
    tcp_node = Node(
        package='planning',
        executable='tcp',
        name='tcp_node',
        output='screen'
    )

    # Planning TF node
    planning_tf_node = Node(
        package='planning',
        executable='tf',
        name='tf_node',
        output='screen'
    )

    # Static TF: base_link -> world
    static_base_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_world',
        arguments=['0','0','0','0','0','0','1','base_link','world'],
        output='screen',
    )

    # MoveIt 
    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="false")

    # Path to the MoveIt launch file
    moveit_launch_file = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py"
    )

    # Includes the MoveIt launch description
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz
        }.items(),
    )

    # Path to the Rviz config file
    rviz_config_file = os.path.join(
        get_package_share_directory('perception'),
        'config',
        'lab7.rviz'
    )
    
    # Rviz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else []
    )

    # -------------------------
    # Global shutdown on any process exit
    # -------------------------
    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='SOMETHING BONKED'))]
        )
    )
    
    return LaunchDescription([
        realsense_launch,
        aruco_launch,
        block_detection_node,
        # gui_to_robot_node,
        ik_node,
        tcp_node,
        planning_tf_node,
        static_base_world,
        moveit_launch,
        rviz_node,
        # shutdown_on_any_exit
    ])
