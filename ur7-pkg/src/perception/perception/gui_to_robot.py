import os
import json
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

BLOCK_WIDTH  = 0.03           # meters in real world
BUILD_ORIGIN = [0.04, 0.80, -0.063] # offset from robot frame to GUI origin
# 0: 0.04092 0.67033 -0.24158

class GuiToRobot(Node):
    def __init__(self):
        super().__init__("gui_to_robot")
        
        # Publisher
        self.planned_centers_pub = self.create_publisher(MarkerArray, "/planned_centers", 1)
        
        # Publish every {refresh_rate} seconds
        self.refresh_rate = 2
        self.create_timer(self.refresh_rate, self.publish_centers)
        
        print("GuiToRobot node initalized")

    ## Converts centers from GUI frame to robot frame
    def publish_centers(self):
        if not os.path.exists("planned_centers.json"):
            return
        with open("planned_centers.json", "r") as f:
            planned_centers = json.load(f)
        if not planned_centers:
            return
        
        marker_array = MarkerArray()

        block_positions = []
        
        for id, block in enumerate(planned_centers):
            x_gui = block["x"]
            y_gui = block["y"]
            
            x_robot = BUILD_ORIGIN[0] + x_gui * BLOCK_WIDTH # x for robot == x for gui
            y_robot = BUILD_ORIGIN[1]                       # y for robot is planar
            z_robot = BUILD_ORIGIN[2] + y_gui * BLOCK_WIDTH # z for robot == y for GUI
            
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = id
            marker.ns = "planned_centers"
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

            block_pos = {
                "x" : x_robot,
                "y" : y_robot,
                "z" : z_robot
            }
            block_positions.append(block_pos)
            

            marker.lifetime.sec = self.refresh_rate
            marker_array.markers.append(marker)
        
        with open("final_pos.json", "w") as f:
            json.dump(block_positions, f)
        print(f"Positions saved to pfinal_pos.json")
        
        self.planned_centers_pub.publish(marker_array)
        print(f"published {len(marker_array.markers)} markers to planned_centers")

def main(args=None):
    rclpy.init(args=args)
    node = GuiToRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()