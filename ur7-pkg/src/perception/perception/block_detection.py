import io
import cv2
import rclpy
import base64
import requests
import numpy as np
import open3d as o3d
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

SAM_URL = "https://ff956770fe91.ngrok-free.app/segment"

class Detection(Node):
    def __init__(self):
        super().__init__("realsense_pc_subscriber")
        
        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy, self.ds = [None] * 5
        # Subscriber for camera_info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/depth/camera_info",
            self.camera_info_callback,
            1
        )
        # Subscribers for images
        self.depth_image_sub = self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
            self.depth_image_callback,
            1
        )
        self.color_image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_rect_raw",
            self.color_image_callback,
            1
        )
        # Publishers
        self.block_centers_pub = self.create_publisher(MarkerArray, "/block_centers", 10)
        self.binary_masks_pub  = self.create_publisher(Image, "/binary_masks", 10)

        # State vars
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        
        # Detects every {refresh_rate} seconds
        self.refresh_rate = 2
        self.create_timer(self.refresh_rate, self.block_centers_callback)
        
        print("[Detection] node initalized")


    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        print(f"[Detection] camera intrinsics fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def depth_image_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    def color_image_callback(self, msg: Image):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")


    def block_centers_callback(self):
        if self.depth_image is None or self.color_image:
            print("[Detection]: no depth or color image yet")
            return
        block_centers = []

        # Input data for SAM
        _, color_array = cv2.imencode(".jpg", self.color_image)
        io_bytes = io.BytesIO(color_array)
        files = {"image": ("image.jpg", io_bytes, "image/jpeg")}
        data  = {"prompt": "Square cubes"}
        
        # Output data of SAM
        response = requests.post(SAM_URL, files=files, data=data)
        if response.status_code != 200:
            print(f"ERROR: [Detection] SAM status code {response.status_code} \n{response.text}")
            return
        
        sam_data = response.json()
        print(f"[Detection] SAM returned {sam_data["count"]} masks")
        
        # All masks for Rviz
        all_masks = np.zeros_like(self.color_image)
        
        for mask in sam_data["masks"]:
            mask_bytes = base64.b64decode(mask["mask_base64"])
            mask_numpy = np.frombuffer(mask_bytes, np.uint8)
            mask_image = cv2.imdecode(mask_numpy, cv2.IMREAD_UNCHANGED)
            
            # Create binary mask
            if mask_image.max() > 1:
                mask_binary = (mask_image > 127).astype(np.uint8) * 255
            else:
                mask_binary = mask_image.astype(np.uint8) * 255
            
            # Masked depth image
            if mask_binary.shape != self.depth_image.shape:
                mask_binary = cv2.resize(
                    mask_binary, 
                    (self.depth_image.shape[1], self.depth_image.shape[0])
                )
            depth_masked = np.where(mask_binary > 0, self.depth_image, 0)
            
            # Store block center
            pcd, center = self.depth_to_pointcloud(depth_masked)
            self.block_centers.append(center)
            
            # Color mask for viz
            hue = int((mask["index"] * 180) / max(sam_data["count"], 1))
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            
            # Resize mask to RGB
            mask_rgb = mask_binary
            if len(mask_binary.shape) == 2:
                mask_rgb = cv2.resize(
                    mask_binary,
                (self.color_image.shape[1], self.color_image.shape[0])
            )
            # Apply colored mask
            mask_channel = np.stack([mask_rgb, mask_rgb, mask_rgb], axis=-1)
            mask_color = np.where(mask_channel > 0, color, [0, 0, 0]).astype(np.uint8)
            all_masks = cv2.addWeighted(all_masks, 1.0, mask_color, 0.7, 0)
        
        # Publish masks, centers
        self.publish_binary_masks(all_masks)
        self.publish_block_centers(block_centers)

        print(f"[Detection] published {sam_data["count"]} masks to /binary_masks")


    def depth_to_pointcloud(self, depth: Image):
        h, w = depth.shape
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        depth_image = o3d.geometry.Image(depth.astype(np.uint16))
        
        pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic,
            depth_scale=1.0/self.ds, # Convert to m
            depth_trunc=1.0   # Max depth in meters
        )
        # Remove invalid / near origin points
        points = np.asarray(pointcloud.points)
        valid_mask = (points[:, 2] > 0.01)
        
        if np.sum(valid_mask) == 0:
            print("ERROR: [Detection] invalid point cloud")
            return
        pcd_filtered = pointcloud.select_by_index(np.where(valid_mask)[0])
        
        # Computes pcd center
        points_filtered = np.asarray(pcd_filtered.points)
        center = np.mean(points_filtered, axis=0)
        
        print(f"[Detection] found {len(points_filtered)} points with center ({center})")
        return pcd_filtered, center


    def publish_binary_masks(self, all_masks: Image):
        msg = self.bridge.cv2_to_imgmsg(all_masks, encoding="passthrough")
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.binary_masks_pub.publish(msg)


    def publish_block_centers(self, block_centers: list):
        marker_array = MarkerArray()
        
        for id, block in enumerate(block_centers):
            marker = Marker()
            marker.header.frame_id = "camera_depth_optical_frame"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = id
            marker.ns = "block_centers"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(block["center"][0])
            marker.pose.position.y = float(block["center"][1])
            marker.pose.position.z = float(block["center"][2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.lifetime.sec = self.refresh_rate
            marker_array.markers.append(marker)

        self.block_centers_pub.publish(marker_array)
        print(f"[Detection] published {len(marker_array.markers)} markers to /block_centers")


def main(args=None):
    rclpy.init(args=args)
    node = Detection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()