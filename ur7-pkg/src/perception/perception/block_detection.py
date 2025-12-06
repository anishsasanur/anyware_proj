import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
import requests
import base64
import io
import cv2

URL = "https://4c9d058633bf.ngrok-free.app/segment"

class RealSensePCSubscriber(Node):
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
            Image,
            '/camera/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publisher for block center visualizing markers
        self.marker_pub = self.create_publisher(MarkerArray, '/block_centers', 10)
        
        # Detects block centers every couple of seconds
        self.create_timer(0.5, self.block_centers_callback)
        
        # State variables
        self.depth_image = None
        self.rgb_image = None
        self.blocks = []
        self.K_set = False
        
        self.get_logger().info("Successfully started block detection node.")
    
    def camera_info_callback(self, msg):
        if not self.K_set:
            # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.K_set = True
            self.get_logger().info(
                f"Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
            )
    
    def depth_callback(self, msg):
        if self.depth_image is None:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Got depth image of shape {self.depth_image.shape}")
    
    def image_callback(self, msg):
        if self.rgb_image is None:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Got rgb image of shape {self.rgb_image.shape}")
    
    def masked_depth_to_pointcloud(self, masked_depth):
        h, w = masked_depth.shape
        
        # Create Open3D camera intrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy
        )
        
        # Convert depth to Open3D format (uint16 or float)
        # Ensure depth is in millimeters for RealSense
        depth_o3d = o3d.geometry.Image(masked_depth.astype(np.uint16))
        
        # Create point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            depth_scale=1.0/self.depth_scale,  # Convert to meters
            depth_trunc=3.0  # Maximum depth in meters
        )
        
        # Filter out zero points (where mask was 0)
        # These will be at origin or have invalid coordinates
        points = np.asarray(pcd.points)
        
        # Remove points at origin or with z=0 (invalid depth)
        valid_mask = (points[:, 2] > 0.01)  # z > 1cm
        
        if np.sum(valid_mask) == 0:
            self.get_logger().warn("No valid points in point cloud!")
            return pcd, np.array([0, 0, 0])
        
        # Filter point cloud
        pcd_filtered = pcd.select_by_index(np.where(valid_mask)[0])
        
        # Compute center of the point cloud
        points_filtered = np.asarray(pcd_filtered.points)
        center = np.mean(points_filtered, axis=0)
        
        self.get_logger().info(
            f"Point cloud: {len(points_filtered)} points, center at ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
        )
        
        return pcd_filtered, center
    
    def block_centers_callback(self):
        if not (self.depth_image is None) and not (self.rgb_image is None) and not self.blocks:
            # Call SAM to get the masks
            _, im_arr = cv2.imencode('.jpg', self.rgb_image)
            io_buf = io.BytesIO(im_arr)
            files = {"image": ("image.jpg", io_buf, "image/jpeg")}
            data = {"prompt": "Blocks"}
            
            response = requests.post(URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                self.get_logger().info(f"Success! Received {result['count']} masks.")
                
                for mask_data in result['masks']:
                    index = mask_data['index']
                    b64_data = mask_data['mask_base64']
                    
                    # Decode mask
                    mask_bytes = base64.b64decode(b64_data)
                    nparr = np.frombuffer(mask_bytes, np.uint8)
                    mask_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    
                    # Ensure mask is binary (0 or 255)
                    if mask_img.max() > 1:
                        mask_binary = (mask_img > 127).astype(np.uint8) * 255
                    else:
                        mask_binary = mask_img.astype(np.uint8) * 255
                    
                    # Apply mask to depth image
                    # Ensure dimensions match
                    if mask_binary.shape != self.depth_image.shape:
                        mask_binary = cv2.resize(
                            mask_binary, 
                            (self.depth_image.shape[1], self.depth_image.shape[0])
                        )
                    
                    masked_depth = np.where(mask_binary > 0, self.depth_image, 0)
                    
                    # Convert to point cloud and get center
                    pcd, center = self.masked_depth_to_pointcloud(masked_depth)
                    
                    # Store block information
                    block_info = {
                        'index': index,
                        'pointcloud': pcd,
                        'center': center,
                        'mask': mask_binary
                    }
                    self.blocks.append(block_info)
                    
                    self.get_logger().info(
                        f"Block {index}: Center at ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m"
                    )
                
                # Publish visualization markers
                self.publish_block_centers()
                
            else:
                self.get_logger().error(f"Error: Status code {response.status_code}")
                self.get_logger().error(response.text)
    
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
            
            # Position
            marker.pose.position.x = float(block['center'][0])
            marker.pose.position.y = float(block['center'][1])
            marker.pose.position.z = float(block['center'][2])
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03
            
            # Color (red)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.lifetime.sec = 0  # Persistent
            
            marker_array.markers.append(marker)
            
            self.get_logger().info(
                f"Block {block['index']}: "
                f"Position = ({block['center'][0]:.3f}, {block['center'][1]:.3f}, {block['center'][2]:.3f}) m, "
                f"Points = {len(block['pointcloud'].points)}"
            )
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers to /block_centers")

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
