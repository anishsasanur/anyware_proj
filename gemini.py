import pybullet as p
import pybullet_data
from PIL import Image
import numpy as np
import os

BOX_ROWS = 6
BOX_COLS = 7
BOX_SIZE = 0.67
CAM_DIST = 3.67

BOX_COLOR = [0.7, 0.5, 0.3, 1] # Cardboard color
TAPE_SPEC = [0.9, 0.9, 0.9] # Shiny white/silver

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FOV = 67
CAM_NEAR = 0.1
CAM_FAR = CAM_DIST + BOX_SIZE * 2 # Ensure cam can see past the wall

ANGLE_RANGE = 0.067 # Max angle camera, boxes get randomly perturbed
NUM_SAMPLES = 67
DATA_FOLDER = "box_data"


class BoxWallSim:
    def __init__(self):
        self.client = p.connect(p.GUI) 

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        self.aspect = CAM_WIDTH / CAM_HEIGHT
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAM_FOV, aspect=self.aspect, nearVal=CAM_NEAR, farVal=CAM_FAR
        )
        self.robot_id = None
        self.eef_link_index = None
        self.robot_joint_indices = []
        self.box_ids = []
        self.home_joint_angles = None
        self.box_initial_poses = {}
        
        self.depth_dir = os.path.join(DATA_FOLDER, "depth")
        self.rgb_np_dir = os.path.join(DATA_FOLDER, "rgb_np")
        self.rgb_png_dir = os.path.join(DATA_FOLDER, "rgb_png")

        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.rgb_np_dir, exist_ok=True)
        os.makedirs(self.rgb_png_dir, exist_ok=True)


    def load_scene(self):
        p.loadURDF("plane.urdf")
        
        robot_start_pos = [0, 0, 0]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", robot_start_pos, robot_start_orn, useFixedBase=True)
        
        self._find_robot_joints()
        self._create_box_wall()
        self._aim_robot_at_wall()


    def _find_robot_joints(self):
        eef_link_name = "panda_link8"
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            link_name = info[12].decode('utf-8')
            
            if joint_type == p.JOINT_REVOLUTE and info[8] < info[9]:
                self.robot_joint_indices.append(i)
                
            if link_name == eef_link_name:
                self.eef_link_index = i
                
        if self.eef_link_index is None:
            raise Exception(f"Could not find link: {eef_link_name}")
            
        print(f"Found {len(self.robot_joint_indices)} controllable joints.")
        print(f"Found EEF link '{eef_link_name}' at index {self.eef_link_index}")


    def _create_box_wall(self):
        base_orn = p.getQuaternionFromEuler([0, 0, 0])

        box_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[BOX_SIZE / 2] * 3,
            rgbaColor=BOX_COLOR
        )
        box_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[BOX_SIZE / 2] * 3
        )
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
                p.changeVisualShape(box_id, -1, specularColor=TAPE_SPEC)
                self.box_ids.append(box_id)


    def _aim_robot_at_wall(self):
        wall_center_z = (BOX_ROWS / 2) * BOX_SIZE
        
        # Places the wrist 0.5m in front of the robot
        target_pos = [0.5, 0, wall_center_z]
        
        # Points the flange (Z-axis) towards the wall
        target_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])
        
        all_joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.eef_link_index,
            targetPosition=target_pos,
            targetOrientation=target_orn
        )
        self.home_joint_angles = list(all_joint_angles[:len(self.robot_joint_indices)])
        self.set_arm_pose(self.home_joint_angles)


    def set_arm_pose(self, joint_angles):
        for i, joint_index in enumerate(self.robot_joint_indices):
            p.resetJointState(self.robot_id, joint_index, joint_angles[i])
        p.stepSimulation()


    def get_camera_image(self, light_params):
        eef_state = p.getLinkState(self.robot_id, self.eef_link_index)
        eef_pos, eef_orn = eef_state[0], eef_state[1]
        rot_matrix = p.getMatrixFromQuaternion(eef_orn)
        
        # Camera "forward" is the link's Z-axis
        forward_vec = np.array(rot_matrix[6:9])
        
        # Camera "up" is the link's negative X-axis
        up_vec = -np.array(rot_matrix[0:3])
        
        cam_center = np.array(eef_pos) + forward_vec * 0.1
        cam_target = cam_center + forward_vec * 1.0
        light_direction = cam_target - cam_center
        
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
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            lightDirection=light_direction,
            **light_params 
        )
        return self._process_image_data(img_data)


    def _process_image_data(self, img_data):
        rgb_raw = img_data[2]
        depth_buffer = img_data[3]
        seg_mask = img_data[4]

        rgb = np.reshape(rgb_raw, (CAM_HEIGHT, CAM_WIDTH, 4))[:, :, :3]
        depth = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * depth_buffer)
        return rgb, depth, seg_mask


    def check_visibility(self, seg_mask):
        visible_object_ids = np.unique(seg_mask)
        visible_boxes = 0

        for box_id in self.box_ids:
            if box_id in visible_object_ids:
                visible_boxes += 1
        return visible_boxes == BOX_ROWS * BOX_COLS


    def collect_data(self):
        print(f"Start samples = {NUM_SAMPLES}")
        sample_count = 1
        
        while sample_count <= NUM_SAMPLES:
            light_params = {
                "lightAmbientCoeff": np.random.uniform(0.0, 0.1),
                "lightDiffuseCoeff": np.random.uniform(0.7, 1.0),
                "lightSpecularCoeff": np.random.uniform(0.5, 1.0),
            }
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
            
            rgb, depth, seg_mask = self.get_camera_image(light_params)
            
            if not self.check_visibility(seg_mask):
                print("*skip sample, not all boxes are visible.")
                self.set_arm_pose(self.home_joint_angles)
                continue

            try:
                depth_filename = os.path.join(self.depth_dir, f"depth_{sample_count:04d}.npy")
                np.save(depth_filename, depth)

                rgb_np_filename = os.path.join(self.rgb_np_dir, f"rgb_np_{sample_count:04d}.npy")
                np.save(rgb_np_filename, rgb)
                
                rgb_png_filename = os.path.join(self.rgb_png_dir, f"rgb_png_{sample_count:04d}.png")
                img = Image.fromarray(rgb)
                img.save(rgb_png_filename)

                print(f"Saved sample {sample_count:04d}")
                sample_count += 1

            except Exception as e:
                print(f"Error saving data: {e}")


if __name__ == "__main__":
    sim = BoxWallSim()
    sim.load_scene()
    sim.collect_data()
    try:
        while True:
            p.stepSimulation() 
    except KeyboardInterrupt:
        p.disconnect()