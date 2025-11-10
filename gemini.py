import pybullet as p
import pybullet_data
from PIL import Image
import numpy as np
import os

BOX_ROWS = 6
BOX_COLS = 7
BOX_SIZE = 0.67
CAM_DIST = 3.67
BOX_COLOR = [0.7, 0.5, 0.3, 1]

CAM_WIDTH = 1333
CAM_HEIGHT = 1000
CAM_FOV = 67
CAM_NEAR = 0.1
CAM_FAR = CAM_DIST + BOX_SIZE * 2 # Ensure cam can see past the wall

ANGLE_RANGE = 0.067 # Max angle camera, boxes get randomly perturbed
NUM_SAMPLES = 67
DATA_FOLDER = "box_data"
BOX_TEXTURE = "box_light.jpg"


class BoxWallSim:
    def __init__(self):
        self.client = p.connect(p.GUI) 

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAM_FOV, aspect=(CAM_WIDTH / CAM_HEIGHT), nearVal=CAM_NEAR, farVal=CAM_FAR
        )
        self.robot_id = None
        self.tool_link_index = None
        self.home_joint_angles = None
        self.robot_joint_indices = []
        self.box_initial_poses = {}
        self.box_ids = []
        
        self.depth_dir = os.path.join(DATA_FOLDER, "depth")
        self.rgb_np_dir = os.path.join(DATA_FOLDER, "rgb_np")
        self.rgb_img_dir = os.path.join(DATA_FOLDER, "rgb_img")

        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.rgb_np_dir, exist_ok=True)
        os.makedirs(self.rgb_img_dir, exist_ok=True)


    def load_scene(self):
        robot_start_pos = [0, 0, 0]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
            robot_start_pos, robot_start_orn, useFixedBase=True)
        
        self._find_robot_joints()
        self._create_boxes_wall()
        self._aim_robot_at_wall()


    def _find_robot_joints(self):
        tool_link_name = "panda_link8"
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            link_name = info[12].decode('utf-8')
            
            if joint_type == p.JOINT_REVOLUTE and info[8] < info[9]:
                self.robot_joint_indices.append(i)
            if link_name == tool_link_name:
                self.tool_link_index = i
                
        if self.tool_link_index is None:
            raise Exception(f"ERROR {tool_link_name} tool link not found.")


    def _create_boxes_wall(self):
        base_orn = p.getQuaternionFromEuler([0, 0, 0])
        texture_id = None

        box_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[BOX_SIZE / 2] * 3
        )
        if os.path.exists(BOX_TEXTURE):
            print(f"Found {BOX_TEXTURE} for box texture.")
            texture_id = p.loadTexture(BOX_TEXTURE)

            box_visual_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName="cube.obj",
                meshScale=[BOX_SIZE] * 3)
        else:
            print(f"ERROR {BOX_TEXTURE} not found, using constant box color.")
            box_visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[BOX_SIZE / 2] * 3,
                rgbaColor=BOX_COLOR)
        
        for i in range(BOX_ROWS):
            for j in range(BOX_COLS):
                pos_x = CAM_DIST
                pos_y = (j - (BOX_COLS - 1) / 2) * BOX_SIZE
                pos_z = (i + 0.5) * BOX_SIZE
                base_pos = [pos_x, pos_y, pos_z]
                
                box_id = p.createMultiBody(
                    baseMass=0.1, 
                    baseCollisionShapeIndex=box_collision_shape,
                    baseVisualShapeIndex=box_visual_shape,
                    basePosition=base_pos,
                    baseOrientation=base_orn
                )
                self.box_initial_poses[box_id] = (base_pos, base_orn)
                
                if texture_id is not None:
                    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                self.box_ids.append(box_id)


    def _aim_robot_at_wall(self):
        wall_center_z = (BOX_ROWS / 2) * BOX_SIZE
        
        # Places the wrist 0.5m in front of the robot
        target_pos = [0.5, 0, wall_center_z]
        
        # Points the flange (Z-axis) towards the wall
        target_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])
        
        all_joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.tool_link_index,
            targetPosition=target_pos,
            targetOrientation=target_orn
        )
        self.home_joint_angles = list(all_joint_angles[:len(self.robot_joint_indices)])
        self.set_arm_pose(self.home_joint_angles)


    def set_arm_pose(self, joint_angles):
        for i, joint_index in enumerate(self.robot_joint_indices):
            p.resetJointState(self.robot_id, joint_index, joint_angles[i])
        p.stepSimulation()


    def get_camera_image(self):
        tool_state = p.getLinkState(self.robot_id, self.tool_link_index)
        tool_pos, tool_orn = tool_state[0], tool_state[1]
        rot_matrix = p.getMatrixFromQuaternion(tool_orn)
        
        # Camera "forward" is the link's Z-axis
        forward_vec = np.array(rot_matrix[6:9])
        
        # Camera "up" is the link's negative X-axis
        up_vec = -np.array(rot_matrix[0:3])
        
        cam_center = np.array(tool_pos) + forward_vec * 0.1
        cam_target = cam_center + forward_vec * 1.0
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_center,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vec
        )
        img_data = p.getCameraImage(
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_raw = img_data[2]
        depth_buffer = img_data[3]
        seg_mask = img_data[4]

        rgb = np.reshape(rgb_raw, (CAM_HEIGHT, CAM_WIDTH, 4))[:, :, :3]
        depth = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * depth_buffer)
        return rgb, depth, seg_mask


    def collect_data(self):
        print(f"Start samples = {NUM_SAMPLES}")
        sample_count = 1
        
        while sample_count <= NUM_SAMPLES:
            for box_id in self.box_ids:
                initial_pos, initial_orn = self.box_initial_poses[box_id]
                
                pos_perturb = np.random.uniform(-0.02, 0.02, 3)
                orn_perturb_euler = np.random.uniform(-ANGLE_RANGE, ANGLE_RANGE, 3)
                orn_perturb_quat = p.getQuaternionFromEuler(orn_perturb_euler)
                
                new_pos, new_orn = p.multiplyTransforms(
                    initial_pos, initial_orn, pos_perturb, orn_perturb_quat
                )
                p.resetBasePositionAndOrientation(box_id, new_pos, new_orn)

            pose_perturb = np.random.uniform(-ANGLE_RANGE, ANGLE_RANGE, len(self.home_joint_angles))
            target_pose = [a + p for a, p in zip(self.home_joint_angles, pose_perturb)]
            self.set_arm_pose(target_pose)
            
            rgb, depth, seg_mask = self.get_camera_image()
            try:
                depth_filename = os.path.join(self.depth_dir, f"depth_{sample_count:04d}.npy")
                np.save(depth_filename, depth)

                rgb_np_filename = os.path.join(self.rgb_np_dir, f"rgb_np_{sample_count:04d}.npy")
                np.save(rgb_np_filename, rgb)
                
                rgb_img_filename = os.path.join(self.rgb_img_dir, f"rgb_img_{sample_count:04d}.png")
                img = Image.fromarray(rgb)
                img.save(rgb_img_filename)

                print(f"Saved sample {sample_count:04d}")
                sample_count += 1

            except Exception as e:
                print(f"ERROR saving data: {e}")


if __name__ == "__main__":
    sim = BoxWallSim()
    sim.load_scene()
    sim.collect_data()
    try:
        while True:
            p.stepSimulation() 
    except KeyboardInterrupt:
        p.disconnect()