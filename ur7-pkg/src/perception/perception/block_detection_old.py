import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PointStamped
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image
import numpy as np
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import open3d as o3d
from cv_bridge import CvBridge
import requests
import base64
import io
import cv2

URL = "https://16a7d56f030a.ngrok-free.app/segment"

class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')

        self.bridge = CvBridge()

        # Subscribers
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

        # Timers
        self.create_timer(5, self.block_points_callback)

        # State Variables
        self.depth_image = None
        self.rgb_image = None
        self.blocks = []

        self.get_logger().info("Successfully started block detection node.")

    def depth_callback(self, msg):
        if self.depth_image is None:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Got depth image of shape {self.depth_image.shape}")
    
    def image_callback(self, msg):
        if self.rgb_image is None:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Got rgb image of shape {self.rgb_image.shape}")

    def block_points_callback(self):
        if not (self.depth_image is None) and not (self.rgb_image is None) and not self.blocks:
            # TODO: Call sam3 to get the masks
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
                    
                    # Decode and save
                    mask_data = base64.b64decode(b64_data)
                    nparr = np.frombuffer(mask_data, np.uint8)
                    mask_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    
                    masked_depth = mask_img * self.depth_image
                    # TODO: Convert to open3d depth image
                    o3d_masked_depth = ...
                    # pcd = o3d.geometry.create_point_cloud_from_depth_image()
            else:
                self.get_logger().error(f"Error: Status code {response.status_code}")
                self.get_logger().error(response.text)
                
                

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    