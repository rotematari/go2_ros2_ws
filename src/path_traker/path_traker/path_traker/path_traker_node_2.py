import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy ,HistoryPolicy
from geometry_msgs.msg import PoseStamped

from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration
from nav2_msgs.action import FollowPath,FollowWaypoints,NavigateToPose
from std_msgs.msg import String
# Add these imports at the top of your file

#!/usr/bin/env python3
class PathTrakerNode(Node):
    def __init__(self):
        super().__init__('path_traker_node')
        self.get_logger().info("Path Traker Node started.")
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        # Initialize publishers, subscribers, etc. as needed.
        self.subscription = self.create_subscription(Path, '/path', self.path_callback, qos_profile)
        self.botton_sub = self.create_subscription(String, '/botton_output', self.button_callback, qos_profile)
        # self.goal_pose_sub = self.create_subscription(PoseStamped, '/my_goal_pose', self.goal_pose_callback, qos_profile)
        
        self.navogator = BasicNavigator()
        self.path_to_execute = Path()
        self.moving = False
        # self.follow_path_client = ActionClient(self, FollowPath, 'follow_path')
        # self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # self.path_callback()
    def button_callback(self, msg: String):
        self.get_logger().info(f"Received message: {msg.data}")
        if msg.data == "move":
            self.get_logger().info("move robot.")
            if self.path_to_execute is None:
                self.get_logger().error("No path to execute.")
                return

            self.navogator.followPath(self.path_to_execute, controller_id="FollowPath", goal_checker_id="general_goal_checker")
            
    def path_callback(self, msg: Path):
        
        self.path_to_execute = msg
        self.get_logger().info("Received path.")
        
    def goal_pose_callback(self, msg : PoseStamped):
        # self.get_logger().info(f"Received goal pose: {msg}")
        self.get_logger().info(f"\033[94mReceived goal pose\033[0m")
        user_input = input("Please verify the goal pose in RViz. Enter '1' to approve, or press enter to cancel: ")
        if user_input.strip() != "1":
            self.get_logger().info("Goal pose execution cancelled by user.")
            return
        
        # self.goToPose(msg)
        self.navogator.goToPose(msg)
        self.get_logger().info("Goal Pose reached.")
        # Start the navigation
        

def main(args=None):
    rclpy.init(args=args)
    node = PathTrakerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()