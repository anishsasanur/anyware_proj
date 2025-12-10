import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json
import os

# GUI coordinate system configuration
BLOCK_WIDTH = 0.067  # meters in real world
ORIGIN_OFFSET = [0.0, 0.67, 0.0]  # [x, y, z] offset from robot base_link to GUI origin

class GuiToRobotNode(Node):
    def __init__(self):
        super().__init__('gui_to_robot')
        
        self.marker_pub = self.create_publisher(MarkerArray, '/planned_centers', 10)
        
        # Publish planned centers every couple of seconds (same as block_detection)
        self.refresh_rate = 2
        self.create_timer(self.refresh_rate, self.publish_planned_centers_callback)
        
        self.get_logger().info('GUI to Robot node started. Publishing from block_plan.json')
    
    def publish_planned_centers_callback(self):
        """Load positions from JSON and publish as markers"""
        try:
            if not os.path.exists('block_plan.json'):
                return
            
            with open('block_plan.json', 'r') as f:
                positions = json.load(f)
            
            if not positions:
                return
            
            marker_array = MarkerArray()
            
            for pos in positions:
                # Convert GUI coordinates to robot frame
                x_gui = pos['x']
                y_gui = pos['y']
                
                # Transform: GUI uses grid, robot uses m
                x_robot = ORIGIN_OFFSET[0] + x_gui * BLOCK_WIDTH
                y_robot = ORIGIN_OFFSET[1]
                z_robot = ORIGIN_OFFSET[2] + y_gui * BLOCK_WIDTH  # y in GUI = z in real world
                
                # Create marker
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "planned_centers"
                marker.id = pos['block'] - 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = x_robot
                marker.pose.position.y = y_robot
                marker.pose.position.z = z_robot
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                marker.lifetime.sec = self.refresh_rate
                marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
            self.get_logger().info(f'Published {len(marker_array.markers)} planned positions')
            
        except Exception as e:
            self.get_logger().error(f'Error loading positions: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GuiToRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
