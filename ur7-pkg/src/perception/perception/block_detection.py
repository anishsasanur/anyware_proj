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

URL = "https://ff956770fe91.ngrok-free.app/segment"


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


    def block_centers_callback(self):
        if not (self.depth_image is None) and not (self.rgb_image is None):
            self.blocks = []

            # Call SAM to get the masks
            _, im_arr = cv2.imencode('.jpg', self.rgb_image)
            io_buf = io.BytesIO(im_arr)
            files = {"image": ("image.jpg", io_buf, "image/jpeg")}
            data = {"prompt": "Square cubes"}
            
            response = requests.post(URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"SAM returned {result['count']} masks")
                
                # Create combined mask visualization
                combined_mask = np.zeros_like(self.rgb_image)
                
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
                    
                    # Add to combined mask with different colors for each block
                    # Create color for this mask (use HSV colormap)
                    color_hue = int((index * 180) / max(result['count'], 1))
                    color = cv2.cvtColor(np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                    
                    # Resize mask to RGB image size if needed
                    if len(mask_binary.shape) == 2:
                        mask_binary_rgb = cv2.resize(mask_binary, 
                                                     (self.rgb_image.shape[1], self.rgb_image.shape[0]))
                    else:
                        mask_binary_rgb = mask_binary
                    
                    # Apply colored mask
                    mask_3channel = np.stack([mask_binary_rgb, mask_binary_rgb, mask_binary_rgb], axis=-1)
                    colored_mask = np.where(mask_3channel > 0, color, [0, 0, 0]).astype(np.uint8)
                    combined_mask = cv2.addWeighted(combined_mask, 1.0, colored_mask, 0.7, 0)
                
                # Publish masks and centers
                self.publish_binary_masks(combined_mask)
                print(f"Published {result['count']} masks to /binary_masks")
                self.publish_block_centers()

            else:
                self.get_logger().error(f"Error, status code {response.status_code}\n{response.text}")


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

            marker.id = block['index']
            marker.ns = "block_centers"
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
