# ROS Libraries
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from visualization_msgs.msg import MarkerArray
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from scipy.spatial.transform import Rotation as R
import numpy as np

from planning.ik import IKPlanner


class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        # Subscribe to all block centers from detection
        self.blocks_sub = self.create_subscription(
            MarkerArray, 
            '/block_centers', 
            self.blocks_callback, 
            1
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            1
        )

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # TF setup for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None
        self.blocks_processed = False

        self.ik_planner = IKPlanner()

        self.job_queue = []  # Entries should be of type either JointState or String('toggle_grip')

        # Compute the rotated quaternion (90 degrees about z-axis)
        # Start with default orientation (pointing down): qy=1.0
        base_rot = R.from_quat([0.0, 1.0, 0.0, 0.0])  # [qx, qy, qz, qw]
        # 90 degree rotation about z-axis
        z_rot = R.from_euler('z', 90, degrees=True)
        # Compose rotations
        combined_rot = z_rot * base_rot
        # Get quaternion [qx, qy, qz, qw]
        self.rotated_quat = combined_rot.as_quat()
        
        self.get_logger().info(f"Using rotated quaternion: qx={self.rotated_quat[0]:.3f}, "
                             f"qy={self.rotated_quat[1]:.3f}, qz={self.rotated_quat[2]:.3f}, "
                             f"qw={self.rotated_quat[3]:.3f}")

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def blocks_callback(self, marker_array: MarkerArray):
        """Process all detected blocks and select the highest one"""
        if self.blocks_processed:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        if len(marker_array.markers) == 0:
            self.get_logger().info("No blocks detected")
            return

        self.get_logger().info(f"Received {len(marker_array.markers)} blocks")

        # Transform all block positions to base_link frame
        blocks_in_base = []
        for marker in marker_array.markers:
            try:
                # Create PointStamped from marker position
                point_stamped = PointStamped()
                point_stamped.header = marker.header
                point_stamped.point = marker.pose.position

                # Transform to base_link
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    marker.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                point_in_base = do_transform_point(point_stamped, transform)

                blocks_in_base.append({
                    'id': marker.id,
                    'x': point_in_base.point.x,
                    'y': point_in_base.point.y,
                    'z': point_in_base.point.z
                })

                self.get_logger().info(
                    f"Block {marker.id} in base_link: "
                    f"x={point_in_base.point.x:.3f}, "
                    f"y={point_in_base.point.y:.3f}, "
                    f"z={point_in_base.point.z:.3f}"
                )

            except Exception as e:
                self.get_logger().error(f"Failed to transform block {marker.id}: {str(e)}")
                continue

        if len(blocks_in_base) == 0:
            self.get_logger().error("No blocks successfully transformed")
            return

        # Find block with maximum z-coordinate (height in base_link frame)
        highest_block = max(blocks_in_base, key=lambda b: b['z'])
        
        self.get_logger().info(
            f"Selected highest block (ID {highest_block['id']}) at height z={highest_block['z']:.3f}m"
        )

        # Set the cube pose and mark as processed
        self.cube_pose = highest_block
        self.blocks_processed = True

        # Plan and execute the grasp
        end_pose = {"x": highest_block["x"] + 0.4, "y": highest_block["y"] - 0.035, "z": highest_block["z"] + 0.185}
        self.plan_grasp_sequence(highest_block, end_pose)

    def plan_grasp_sequence(self, cube_pose_dict, end_pose_dict):
        """Plan the grasp sequence for the selected cube"""
        x = cube_pose_dict['x']
        y = cube_pose_dict['y']
        z = cube_pose_dict['z']
        current_state = self.joint_state

        x_offset = 0.01
        y_offset = -0.03

        self.get_logger().info(f"Planning grasp for cube at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Extract rotated quaternion components
        qx, qy, qz, qw = self.rotated_quat

        # 1) Move to Pre-Grasp Position (gripper above the cube) with rotated orientation
        pre_grasp_js = self.ik_planner.compute_ik(
            current_state, 
            x + x_offset, 
            y + y_offset, 
            z + 0.25,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        
        if pre_grasp_js is None:
            self.get_logger().error("Failed IK for pre-grasp")
            return
        self.job_queue.append(pre_grasp_js)
        current_state = pre_grasp_js
        self.get_logger().info("Added pre-grasp position to queue")

        # 2) Move to Grasp Position (lower the gripper to the cube) with rotated orientation
        grasp_js = self.ik_planner.compute_ik(
            current_state, 
            x + x_offset,
            y + y_offset, 
            z + 0.15,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        
        if grasp_js is None:
            self.get_logger().error("Failed IK for grasp")
            return
        self.job_queue.append(grasp_js)
        current_state = grasp_js
        self.get_logger().info("Added grasp position to queue")

        # 3) Close the gripper
        self.job_queue.append('toggle_grip')
        self.get_logger().info("Added gripper close to queue")

        # 4) Move back to Pre-Grasp Position
        pre_grasp_js = self.ik_planner.compute_ik(
            current_state, 
            x + x_offset, 
            y + y_offset, 
            z + 0.25,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        
        if pre_grasp_js is None:
            self.get_logger().error("Failed IK for pre-grasp")
            return
        self.job_queue.append(pre_grasp_js)
        current_state = pre_grasp_js
        self.get_logger().info("Added pre-grasp position to queue")

        # 5) Move to release Position (0.4m in +x direction) with rotated orientation
        release_js = self.ik_planner.compute_ik(
            current_state,
            end_pose_dict['x'],
            end_pose_dict['y'],
            end_pose_dict['z'],
            qx=qx, qy=qy, qz=qz, qw=qwss
        )
        
        if release_js is None:
            self.get_logger().error("Failed IK for release position")
            return
        self.job_queue.append(release_js)
        current_state = release_js
        self.get_logger().info("Added release position to queue")

        # 6) Release the gripper
        self.job_queue.append('toggle_grip')
        self.get_logger().info("Added gripper open to queue")

        # Start execution
        self.execute_jobs()

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return

            self.get_logger().info("Planned to position")
            self._execute_joint_trajectory(traj.joint_trajectory)
            
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by controller')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing trajectory...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()