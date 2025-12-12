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

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_scale = 0.001

        # ROS
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None

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

        self.marker_pub = self.create_publisher(MarkerArray, "/block_centers", 10)
        self.mask_pub = self.create_publisher(Image, "/binary_masks", 10)

        self.timer_period = 2.0
        self.create_timer(self.timer_period, self.process)


    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_image_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def color_image_callback(self, msg: Image):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def process(self):
        if (
            self.depth_image is None
            or self.color_image is None
            or self.fx is None
        ):
            return

        # --- Send image to SAM ---
        _, jpg = cv2.imencode(".jpg", self.color_image)
        files = {"image": ("image.jpg", io.BytesIO(jpg), "image/jpeg")}
        data = {"prompt": "square cube"}

        response = requests.post(SAM_URL, files=files, data=data)
        if response.status_code != 200:
            self.get_logger().error("SAM request failed")
            return

        sam_data = response.json()

        block_centers = []
        all_masks = np.zeros_like(self.color_image)

        for mask in sam_data["masks"]:
            # Decode mask
            mask_bytes = base64.b64decode(mask["mask_base64"])
            mask_np = np.frombuffer(mask_bytes, np.uint8)
            mask_img = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

            mask_binary = (mask_img > 127).astype(np.uint8)

            # Compute 3D center
            center = self.mask_to_center_3d(mask_binary)
            if center is not None:
                block_centers.append(center)

            # Visualization
            color = np.random.randint(0, 255, size=3)
            mask_rgb = np.dstack([mask_binary * color[i] for i in range(3)])
            all_masks = cv2.addWeighted(all_masks, 1.0, mask_rgb, 0.6, 0)

        self.publish_masks(all_masks)
        self.publish_markers(block_centers)


    def mask_to_center_3d(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        Z = self.depth_image[ys, xs]  # meters (32FC1)
        valid = Z > 0.01
        if not np.any(valid):
            return None

        xs, ys, Z = xs[valid], ys[valid], Z[valid]

        X = (xs - self.cx) * Z / self.fx
        Y = (ys - self.cy) * Z / self.fy

        points = np.vstack((X, Y, Z)).T
        return np.mean(points, axis=0)


    def publish_masks(self, masks):
        msg = self.bridge.cv2_to_imgmsg(masks, "bgr8")
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(msg)

    def publish_markers(self, centers):
        marker_array = MarkerArray()

        for i, c in enumerate(centers):
            m = Marker()
            m.header.frame_id = "camera_color_optical_frame"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "blocks"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = float(c[0]) * self.depth_scale
            m.pose.position.y = float(c[1]) * self.depth_scale
            m.pose.position.z = float(c[2]) * self.depth_scale
            m.pose.orientation.w = 1.0

            m.scale.x = m.scale.y = m.scale.z = 0.03
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0

            m.lifetime.sec = int(self.timer_period)
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = BlockDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()