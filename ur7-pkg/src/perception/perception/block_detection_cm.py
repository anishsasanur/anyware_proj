import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
import requests
import base64
import io
import cv2

URL = "https://73a0df306f87.ngrok-free.app/segment"


class BlockDetectionNode(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')
        self.bridge = CvBridge()
        
        # Default camera intrinsics (will be overwritten)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 0.001
        
        # Subscribers for camera depth and color images
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.img_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',
            self.image_callback,
            10
        )
        
        # Actual camera intrinsics (overwrites defaults)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/block_centers', 10)
        self.binary_masks_pub = self.create_publisher(Image, '/binary_masks', 10)
        
        # Detects block centers every couple of seconds
        self.refresh_rate = 2
        self.create_timer(self.refresh_rate, self.block_centers_callback)
        
        # State variables
        self.depth_image = None
        self.rgb_image = None
        self.blocks = []
        self.K_set = False
        
        print("Successfully started block detection node")


    def camera_info_callback(self, msg):
        if not self.K_set:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.K_set = True
            print(f"Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def image_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


    def masked_depth_to_pointcloud(self, masked_depth):
        h, w = masked_depth.shape
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy
        )
        
        # Create open3d object from depth image
        depth_o3d = o3d.geometry.Image(masked_depth.astype(np.uint16))
        
        # Create point cloud from open3d object
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            depth_scale=1.0/self.depth_scale, # Convert to meters
            depth_trunc=1.0 # Max depth in meters
        )
        
        # Filter out invalid/near origin points
        points = np.asarray(pcd.points)
        valid_mask = (points[:, 2] > 0.01)
        
        if np.sum(valid_mask) == 0:
            self.get_logger().error("No valid points in point cloud")
            return pcd, np.array([0, 0, 0])
        
        # Filter point cloud
        pcd_filtered = pcd.select_by_index(np.where(valid_mask)[0])
        
        # Compute the center of the point cloud
        points_filtered = np.asarray(pcd_filtered.points)
        center = np.mean(points_filtered, axis=0)
        
        print(f"{len(points_filtered)} points, center at ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        
        return pcd_filtered, center


    def filter_image_by_hsv(self, img_rgb, hue_center, hue_width=15):
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        h_min = max(0, hue_center - hue_width)
        h_max = min(179, hue_center + hue_width)

        lower_bound = np.array([h_min, 50, 50], dtype=np.uint8)
        upper_bound = np.array([h_max, 255, 255], dtype=np.uint8)
        
        return cv2.inRange(img_hsv, lower_bound, upper_bound)


    def block_centers_callback(self):
        if self.depth_image is None or self.rgb_image is None:
            return

        self.blocks = []
        img = self.rgb_image
        
        # Visualization image (starts black)
        combined_mask_vis = np.zeros_like(img)

        # Define the colors we want to find
        targets = [
            {'hue': 60,  'vis_color': (0, 255, 0), "variance": 10},   # Green
            {'hue': 120, 'vis_color': (255, 0, 0), "variance": 10},   # Blue
            {'hue': 15, 'vis_color': (0, 165, 255), "variance": 10}   # Orange
        ]
        found_count = 0

        for target in targets:
            # 2. Generate Mask based on Color
            mask = self.filter_image_by_hsv(img, target['hue'], target["variance"])
            
            # 3. Apply mask to depth image
            # Resize mask if depth/color resolutions differ
            if mask.shape != self.depth_image.shape:
                mask = cv2.resize(mask, (self.depth_image.shape[1], self.depth_image.shape[0]))
            
            # Create masked depth image (everything outside mask becomes 0 depth)
            masked_depth = np.where(mask > 0, self.depth_image, 0)
            
            # 4. Get 3D Center of this entire color blob
            pcd, center = self.masked_depth_to_pointcloud(masked_depth)

            # 5. Store Block Info
            self.blocks.append({
                'index': target['index'],
                'pointcloud': pcd,
                'center': center,
                'mask': mask
            })
            found_count += 1

            # 6. Add to Visualization
            # Paint the found pixels with their specific color (vis_color)
            color_layer = np.zeros_like(img)
            color_layer[mask > 0] = target['vis_color']
            
            # Add this layer to the combined visualization
            combined_mask_vis = cv2.addWeighted(combined_mask_vis, 1.0, color_layer, 1.0, 0.0)

        # Publish masks and centers
        self.publish_binary_masks(combined_mask_vis)
        print(f"Published {found_count} masks to /binary_masks")
        self.publish_block_centers()


    def publish_binary_masks(self, masks):
        mask_msg = self.bridge.cv2_to_imgmsg(masks, encoding='passthrough')
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        mask_msg.header.frame_id = "camera_color_optical_frame"
        self.binary_masks_pub.publish(mask_msg)


    def publish_block_centers(self):
        marker_array = MarkerArray()
        
        for block in self.blocks:
            marker = Marker()
            marker.header.frame_id = "camera_depth_optical_frame"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "block_centers"
            marker.id = block['index']
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(block['center'][0])
            marker.pose.position.y = float(block['center'][1])
            marker.pose.position.z = float(block['center'][2])
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

            print(f"Block {block['index']}, center at ({block['center'][0]:.3f}, {block['center'][1]:.3f}, {block['center'][2]:.3f})")

        self.marker_pub.publish(marker_array)
        print(f"Published {len(marker_array.markers)} markers to /block_centers")


def main(args=None):
    rclpy.init(args=args)
    node = BlockDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
