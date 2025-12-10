import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json
import threading
import time

# GUI coordinate system configuration
BLOCK_WIDTH = 0.067  # meters in real world
ORIGIN_OFFSET = [0.0, 0.67, 0.0]  # [x, y, z] offset from robot base_link to GUI origin

class GuiToRobotNode(Node):
    def __init__(self):
        super().__init__('gui_to_robot')
        
        self.marker_pub = self.create_publisher(MarkerArray, '/planned_centers', 10)
        
        # Start file watcher thread
        self.running = True
        self.watch_thread = threading.Thread(target=self.watch_positions_file, daemon=True)
        self.watch_thread.start()
        
        self.get_logger().info('GUI to Robot node started. Watching square_positions.json')
    
    def watch_positions_file(self):
        """Watch for changes to square_positions.json"""
        last_mtime = 0
        
        while self.running:
            try:
                import os
                if os.path.exists('square_positions.json'):
                    mtime = os.path.getmtime('square_positions.json')
                    if mtime > last_mtime:
                        last_mtime = mtime
                        self.load_and_publish()
                        
            except Exception as e:
                self.get_logger().error(f'Error watching file: {e}')
            
            time.sleep(0.5)
    
    def load_and_publish(self):
        """Load positions from JSON and publish as markers"""
        try:
            with open('square_positions.json', 'r') as f:
                positions = json.load(f)
            
            if not positions:
                return
            
            marker_array = MarkerArray()
            
            for pos in positions:
                # Convert GUI coordinates to robot frame
                x_gui = pos['x']
                y_gui = pos['y']
                
                # Transform: GUI uses square_size units, robot uses meters
                x_robot = ORIGIN_OFFSET[0] + x_gui * BLOCK_WIDTH
                y_robot = ORIGIN_OFFSET[1] + y_gui * BLOCK_WIDTH
                z_robot = ORIGIN_OFFSET[2] + pos['y'] * BLOCK_WIDTH  # y in GUI = height
                
                # Create marker
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "planned_centers"
                marker.id = pos['square'] - 1
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
                
                marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
            self.get_logger().info(f'Published {len(marker_array.markers)} planned positions')
            
        except Exception as e:
            self.get_logger().error(f'Error loading positions: {e}')
    
    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GuiToRobotNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()