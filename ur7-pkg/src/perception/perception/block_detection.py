import io
import cv2
import rclpy
import base64
import requests
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

SAM_URL = "https://ff956770fe91.ngrok-free.app/segment"


class BlockDetection(Node):
    def __init__(self):
        super().__init__("block_detection")

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_scale = 0.001

        # State vars
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None

        # Period in s
        self.detection_period = 2.0
        self.create_timer(self.detection_period, self.block_detection)

        # Subscribers
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            1,
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_image_callback,
            1,
        )
        self.create_subscription(
            Image,
            "/camera/camera/color/image_rect_raw",
            self.color_image_callback,
            1,
        )
        # Publishers
        self.block_centers_pub = self.create_publisher(MarkerArray, "/block_centers", 10)
        self.binary_masks_pub  = self.create_publisher(Image, "/binary_masks", 10)


    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_image_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def color_image_callback(self, msg: Image):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def block_detection(self):
        if (self.depth_image is None
            or self.color_image is None
            or self.fx is None
        ): return

        # Input data for SAM
        _, jpg = cv2.imencode(".jpg", self.color_image)
        files = {"image": ("image.jpg", io.BytesIO(jpg), "image/jpeg")}
        data = {"prompt": "square cube"}

        # Output data of SAM
        response = requests.post(SAM_URL, files=files, data=data)
        if response.status_code != 200:
            self.get_logger().error(f"SAM request failed: {response}")
            return
        sam_data = response.json()

        # All centers, masks
        block_centers = []
        masks_stacked = np.zeros_like(self.color_image)

        for mask in sam_data["masks"]:
            # Get binary mask
            mask_bytes = base64.b64decode(mask["mask_base64"])
            mask_numpy = np.frombuffer(mask_bytes, np.uint8)
            mask_image = cv2.imdecode(mask_numpy, cv2.IMREAD_GRAYSCALE)
            mask_binary = (mask_image > 127).astype(np.uint8)

            # Compute center
            center = self.mask_to_center_3d(mask_binary)
            if center is not None:
                block_centers.append(center)

            # Visualization
            color = np.random.randint(0, 255, size=3)
            mask_rgb = np.dstack([mask_binary * color[i] for i in range(3)])
            masks_stacked = cv2.addWeighted(masks_stacked, 1.0, mask_rgb, 0.6, 0)

        # Publishers
        self.publish_masks(masks_stacked)
        self.publish_markers(block_centers)
        print(f"Published {len(sam_data)} masks to /binary_masks")


    def mask_to_center_3d(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        Z = self.depth_image[ys, xs]  # meters (32FC1)
        valid = Z > 0.01
        if not np.any(valid):
            return None

        xs, ys, Z = xs[valid], ys[valid], Z[valid]
        Z *= self.depth_scale

        X = (xs - self.cx) * Z / self.fx
        Y = (ys - self.cy) * Z / self.fy

        points = np.vstack((X, Y, Z)).T
        return np.mean(points, axis=0)


    def publish_masks(self, masks):
        msg = self.bridge.cv2_to_imgmsg(masks, "bgr8")
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.binary_masks_pub.publish(msg)


    def publish_markers(self, centers):
        marker_array = MarkerArray()

        for id, center in enumerate(centers):
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = id
            marker.ns = "blocks"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = marker.scale.y = marker.scale.z = 0.03
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.lifetime.sec = int(self.detection_period)
            marker_array.markers.append(marker)

        self.block_centers_pub.publish(marker_array)
        print(f"Published {len(marker_array)} centers to /block_centers")


def main(args=None):
    rclpy.init(args=args)
    node = BlockDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()