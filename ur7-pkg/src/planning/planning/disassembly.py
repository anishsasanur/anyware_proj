import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped 
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
from tf2_geometry_msgs import do_transform_point
from scipy.spatial.transform import Rotation as R
from control_msgs.action import FollowJointTrajectory
from planning.ik import IKPlanner

class Disassembly(Node):
    def __init__(self):
        super().__init__("cube_grasp")

        # Subscriber for block_centers
        self.block_centers_sub = self.create_subscription(
            MarkerArray,
            "/block_centers",
            self.block_centers_callback,
            10
        )
        # Subscriber for joint_states
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )
        # Clients
        self.action_cli = ActionClient(
            self, FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.gripper_cli = self.create_client(Trigger, "/toggle_gripper")

        # TF Cache
        self.tf_cached = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Planning
        self.joint_state = None
        self.home_joint_state = None
        self.ik_planner = IKPlanner()

        # Execution
        self.job_queue = []
        self.is_executing = False

        # Wrist pointing down ([qx,  qy,  qz,  qw ])
        base_rot = R.from_quat([0.0, 1.0, 0.0, 0.0])
        # 90 degree rotation for z:
        z_90_rot = R.from_euler("z", 90, degrees=True)
        # Target wrist rotation
        self.wrist_rot = (z_90_rot * base_rot).as_quat()


    def joint_state_callback(self, msg: JointState):
        if self.home_joint_state is None:
            self.home_joint_state = msg
        self.joint_state = msg


    def block_centers_callback(self, marker_array: MarkerArray):
        if self.is_executing or len(self.job_queue) != 0:
            print("still executing...")
            return

        if self.joint_state is None:
            print("no joint state yet")
            return

        if len(marker_array.markers) == 0:
            print("no blocks received")
            return

        print(f"received {len(marker_array.markers)} blocks")

        if self.tf_cached is None:
            try:
                frame_id = marker_array.markers[0].header.frame_id
                self.tf_cached = self.tf_buffer.lookup_transform(
                    "base_link",
                    frame_id,
                    rclpy.time.Time()
                )
            except Exception as e:
                print(f"ERROR: no transform to base_link")
                return

        # Transform all block centers to base_link
        blocks_in_base = []
        for marker in marker_array.markers:
            try:
                point = PointStamped()
                point.header = marker.header
                point.point = marker.pose.position

                point_in_base = do_transform_point(point, self.tf_cached)
                x, y, z = point_in_base.point.x, point_in_base.point.y, point_in_base.point.z

                with open("block_centers.txt", "a") as f:
                    f.write(f"{marker.id}: {x:.5f} {y:.5f} {z:.5f}\n")

                blocks_in_base.append({"id": marker.id, "x": x, "y": y, "z": z})

            except Exception as e:
                print(f"ERROR: failed to transform block {marker.id}: \n{e}")
                continue

        if len(blocks_in_base) == 0:
            print("ERROR: failed to transform any blocks")
            return
        print(f"transformed {len(blocks_in_base)} blocks to base_link frame")

        # Find block with highest z-coordinate
        highest_block = max(blocks_in_base, key=lambda b: b["z"])
        
        print(f"Found highest block {highest_block['id']}")

        with open("block_centers.txt", "a") as f:
            f.write(f"\n")

        return
        self.plan_grasp(highest_block)
        self.execute_grasp()


    def plan_grasp(self, block_pose: dict):
        x_offset = -0.022
        y_offset = -0.022
        z_offset = 0.175

        x = block_pose["x"]
        y = block_pose["y"]
        z = block_pose["z"]
        
        print(f"Planning grasp for block at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Target wrist rotation
        qx, qy, qz, qw = self.wrist_rot

        # Move above block
        pre_grasp_pose = self.ik_planner.compute_ik(
            self.joint_state, 
            x + x_offset,
            y + y_offset,
            z + z_offset + 0.1,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if pre_grasp_pose is None:
            print("ERROR: failed to plan pre_grasp_pose")
            return
        self.job_queue.append(pre_grasp_pose)

        # Move onto block
        grasp_pose = self.ik_planner.compute_ik(
            pre_grasp_pose,  # Plan from pre-grasp pose
            x + x_offset,
            y + y_offset,
            z + z_offset,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if grasp_pose is None:
            print("ERROR: failed to plan grasp_pose")
            self.job_queue.clear()
            return
        self.job_queue.append(grasp_pose)
        
        # Close gripper
        self.job_queue.append("toggle_grip")

        # Move onto block
        lift_pose = self.ik_planner.compute_ik(
            grasp_pose,
            x + x_offset,
            y + y_offset,
            z + z_offset + 0.1,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if lift_pose is None:
            print("ERROR: failed to plan lift_pose")
            self.job_queue.clear()
            return
        self.job_queue.append(lift_pose)

        # Move to drop pose
        drop_pose = self.ik_planner.compute_ik(
            lift_pose,
            x + 0.420,
            y - 0.069,
            z + 0.420,
        )
        if drop_pose is None:
            print("ERROR: failed to plan drop_pose")
            self.job_queue.clear()
            return
        
        self.job_queue.append(drop_pose)
        self.job_queue.append("toggle_grip")
        
        # Return to home
        self.job_queue.append(self.home_joint_state)
        self.joint_state = self.home_joint_state


    def execute_grasp(self):
        if len(self.job_queue) == 0:
            print("all jobs complete, ready for next block")
            self.is_executing = False
            return
        
        self.is_executing = True
        print(f"Executing job queue: {len(self.job_queue)} jobs")
        job = self.job_queue.pop(0)

        if isinstance(job, JointState):
            traj = self.ik_planner.plan_to_joints(job)
            if traj is None:
                print("ERROR: failed to execute plan")
                self.job_queue.clear()
                self.is_executing = False
                return
            self._execute_joint_trajectory(traj.joint_trajectory)
        
        elif job == "toggle_grip":
            self._toggle_gripper()
        else:
            print(f"ERROR: unknown job type {job}")
            self.is_executing = False


    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            print("ERROR: gripper unavailable")
            self.job_queue.clear()
            self.is_executing = False
            rclpy.shutdown()
            return
        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.execute_grasp()

    def _execute_joint_trajectory(self, joint_traj):
        self.action_cli.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        send_future = self.action_cli.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("ERROR: goal rejected by controller")
            self.job_queue.clear()
            self.is_executing = False
            rclpy.shutdown()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.execute_grasp()
        except Exception as e:
            print(f"ERROR: execution failed: \n{e}")
            self.job_queue.clear()
            self.is_executing = False


def main(args=None):
    rclpy.init(args=args)
    node = Disassembly()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == "__main__":
    main()