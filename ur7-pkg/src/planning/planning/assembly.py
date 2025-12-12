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
            1
        )
        # Subscriber for joint_states
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            1
        )
        # Clients
        self.action_cli = ActionClient(
            self, FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.gripper_cli = self.create_client(Trigger, "/toggle_gripper")

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Joint state and IK
        self.joint_state = None
        self.home_joint_state = None
        
        self.ik_planner = IKPlanner()
        self.job_queue = [] # types: JointState or "toggle_grip"

        # Wrist pointing down ([qx,  qy,  qz,  qw ])
        base_rot = R.from_quat([0.0, 1.0, 0.0, 0.0])
        # 90 degree rotation for z:
        z_90_rot = R.from_euler("z", 90, degrees=True)
        # Tgt wrist rotation
        self.wrist_rot = (z_90_rot * base_rot).as_quat()
        
        print("[Disassembly] node initalized")


    def joint_state_callback(self, msg: JointState):
        if self.joint_state is None:
            self.home_joint_state = msg
        self.joint_state = msg


    def block_centers_callback(self, marker_array: MarkerArray):
        if len(self.job_queue) != 0:
            return

        if self.joint_state is None:
            print("[Disassembly] no joint state yet")
            return

        if len(marker_array.markers) == 0:
            print("[Disassembly] no blocks received")
            return

        print(f"[Disassembly] received {len(marker_array.markers)} blocks")

        # Transform all block centers to base_link frame
        blocks_in_base = []
        for marker in marker_array.markers:
            try:
                point = PointStamped()
                point.header = marker.header
                point.point = marker.pose.position

                transform = self.tf_buffer.lookup_transform(
                    "base_link",
                    marker.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                point_in_base = do_transform_point(point, transform)

                blocks_in_base.append({
                    "id": marker.id,
                    "x": point_in_base.point.x,
                    "y": point_in_base.point.y,
                    "z": point_in_base.point.z
                })
            except Exception as e:
                print(f"ERROR: [Disassembly] failed to transform block {marker.id}: \n{e}")
                continue

        if len(blocks_in_base) == 0:
            print("ERROR: [Disassembly] failed to transform any blocks")
            return
        print(f"[Disassembly] transformed {len(blocks_in_base)} blocks to base_link frame")

        # Find block with highest z-coordinate
        leftmost_block = max(blocks_in_base, key=lambda b: b["x"])
        
        print(f"Found highest block {leftmost_block["id"]}")
        
        self.plan_grasp(leftmost_block, {"x": 0.5, "y": 0.0, "z": 0.1})
        self.execute_grasp()


    def plan_grasp(self, block_pose: dict, place_pose: dict):
        
        bx, by, bz = block_pose["x"], block_pose["y"], block_pose["z"]
        px, py, pz = place_pose["x"], place_pose["y"], place_pose["z"]

        curr_pose = self.joint_state
        qx, qy, qz, qw = self.wrist_rot

        # Pre-grasp
        pre_grasp = self.ik_planner.compute_ik(
            curr_pose, bx+0.01, by-0.03, bz+0.25,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if pre_grasp is None:
            return
        self.job_queue.append(pre_grasp)
        curr_pose = pre_grasp

        # Grasp
        grasp = self.ik_planner.compute_ik(
            curr_pose, bx+0.01, by-0.03, bz+0.15,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if grasp is None:
            return
        self.job_queue.append(grasp)
        self.job_queue.append("toggle_grip")

        # Lift
        self.job_queue.append(pre_grasp)
        curr_pose = pre_grasp

        # Pre-drop
        pre_drop = self.ik_planner.compute_ik(
            curr_pose, px, py, pz+0.20,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if pre_drop is None:
            return
        self.job_queue.append(pre_drop)
        curr_pose = pre_drop

        # Drop
        drop = self.ik_planner.compute_ik(
            curr_pose, px, py, pz,
            qx=qx, qy=qy, qz=qz, qw=qw
        )
        if drop is None:
            return
        self.job_queue.append(drop)
        self.job_queue.append("toggle_grip")

        # Home
        self.job_queue.append(self.home_joint_state)


    def execute_grasp(self):
        if len(self.job_queue) == 0:
            print("[Disassembly] all jobs complete")
            # rclpy.shutdown()
            return
        print(f"[Disassembly] Executing job queue: {len(self.job_queue)} jobs")
        job = self.job_queue.pop(0)

        if isinstance(job, JointState):
            traj = self.ik_planner.plan_to_joints(job)
            if traj is None:
                print("ERROR: [Disassembly] failed to execute plan")
                return
            self._execute_joint_trajectory(traj.joint_trajectory)
        
        elif job == "toggle_grip":
            self._toggle_gripper()
        else:
            print(f"ERROR: [Disassembly] unknown job type {job}")


    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            print("ERROR: [Disassembly] gripper unavailable")
            rclpy.shutdown()
            return
        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.execute_grasp()

    def _execute_joint_trajectory(self, joint_traj):
        print("[Disassembly] waiting for controller action server...")
        self.action_cli.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        print("[Disassembly] sending trajectory to controller...")
        send_future = self.action_cli.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("ERROR: [Disassembly] goal rejected by controller")
            rclpy.shutdown()
            return
        print("[Disassembly] executing trajectory...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            print("[Disassembly] execution complete.")
            self.execute_grasp()
        except Exception as e:
            print(f"ERROR: [Disassembly] execution failed: \n{e}")


def main(args=None):
    rclpy.init(args=args)
    node = Disassembly()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == "__main__":
    main()