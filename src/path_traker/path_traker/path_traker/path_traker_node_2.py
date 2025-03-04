import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy ,HistoryPolicy
from geometry_msgs.msg import PoseStamped

from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration
from nav2_msgs.action import FollowPath,FollowWaypoints,NavigateToPose
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
        # self.goal_pose_sub = self.create_subscription(PoseStamped, '/my_goal_pose', self.goal_pose_callback, qos_profile)
        
        self.navogator = BasicNavigator()

        # self.follow_path_client = ActionClient(self, FollowPath, 'follow_path')
        # self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # self.path_callback()
        
    def path_callback(self, msg: Path):
        self.get_logger().info(f"Received path with {len(msg.poses)} poses in frame {msg.header.frame_id}")
        
        # Transform path to map frame if it's not already in map frame
        if msg.header.frame_id != "map":
            transformed_path = self.transform_path_to_map(msg)
            if transformed_path is None:
                self.get_logger().error("Path transformation failed. Cannot execute path.")
                return
            path_to_execute = transformed_path
        else:
            path_to_execute = msg
        
        user_input = input("Please verify the path in RViz. Enter '1' to approve, or press enter to cancel: ")
        if user_input.strip() != "1":
            self.get_logger().info("Path execution cancelled by user.")
            return
    
        self.navogator.followPath(path_to_execute, controller_id="FollowPath", goal_checker_id="general_goal_checker")
        self.get_logger().info("Path Traker Node finished.")
        
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