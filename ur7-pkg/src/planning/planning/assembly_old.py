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
import json

from planning.ik import IKPlanner


class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        FinalBlockPositionsJSON = json.load(open('FinalBlockPositions.json'))
        self.FinalBlockPositions = FinalBlockPositionsJSON

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

    def move_block(self, start_pose: dict, end_pose: dict):
        """
        High-level primitive:
        Picks a block at start_pose and places it at end_pose.
        Both poses must be in base_link frame.
        """

        if self.joint_state is None:
            self.get_logger().error("No joint state available")
            return

        current_state = self.joint_state

        sx, sy, sz = start_pose["x"], start_pose["y"], start_pose["z"]
        fx, fy, fz = end_pose["x"], end_pose["y"], end_pose["z"]

        self.get_logger().info(
            f"Moving block from ({sx:.3f},{sy:.3f},{sz:.3f}) "
            f"to ({fx:.3f},{fy:.3f},{fz:.3f})"
        )

        # -------------------------
        # 1) Pre-grasp ABOVE start
        # -------------------------
        pre_grasp_js = self.ik_planner.compute_ik(
            current_state,
            sx - 0.0067,
            sy - 0.0367,
            sz + 0.267
        )
        if pre_grasp_js is None:
            self.get_logger().error("Pre-grasp IK failed")
            return

        self.job_queue.append(pre_grasp_js)
        current_state = pre_grasp_js

        # -------------------------
        # 2) Grasp pose
        # -------------------------
        grasp_js = self.ik_planner.compute_ik(
            current_state,
            sx - 0.0067,
            sy - 0.0367,
            sz + 0.1567
        )
        if grasp_js is None:
            self.get_logger().error("Grasp IK failed")
            return

        self.job_queue.append(grasp_js)
        current_state = grasp_js

        # -------------------------
        # 3) Close gripper
        # -------------------------
        self.job_queue.append("toggle_grip")

        # -------------------------
        # 4) Lift back to pre-grasp
        # -------------------------
        self.job_queue.append(pre_grasp_js)
        current_state = pre_grasp_js

        # -------------------------
        # 5) Move ABOVE target
        # -------------------------
        pre_place_js = self.ik_planner.compute_ik(
            current_state,
            fx - 0.0067,
            fy - 0.0367,
            fz + 0.267
        )
        if pre_place_js is None:
            self.get_logger().error("Pre-place IK failed")
            return

        self.job_queue.append(pre_place_js)
        current_state = pre_place_js

        # -------------------------
        # 6) Lower to place pose
        # -------------------------
        place_js = self.ik_planner.compute_ik(
            current_state,
            fx - 0.0067,
            fy - 0.0367,
            fz + 0.1567
        )
        if place_js is None:
            self.get_logger().error("Place IK failed")
            return

        self.job_queue.append(place_js)
        current_state = place_js

        # -------------------------
        # 7) Open gripper
        # -------------------------
        self.job_queue.append("toggle_grip")

        # -------------------------
        # 8) Retreat back up
        # -------------------------
        self.job_queue.append(pre_place_js)

        # Start execution if queue was previously empty
        if len(self.job_queue) == 1:
            self.execute_jobs()

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
        left_block = min(blocks_in_base, key=lambda b: b['x'])
        
        self.get_logger().info(
            f"Selected leftmost block (ID {left_block['id']}) at height z={left_block['z']:.3f}m"
        )

        # Set the cube pose and mark as processed
        self.cube_pose = left_block
        self.blocks_processed = True

        # Plan and execute the grasp
        self.move_block(left_block, self.FinalBlockPositions[0])
        self.FinalBlockPositions.pop(0)

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