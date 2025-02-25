import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy ,HistoryPolicy
from geometry_msgs.msg import PoseStamped

from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration
from nav2_msgs.action import FollowPath,FollowWaypoints,NavigateToPose
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
        self.navogator = BasicNavigator()
        # self.follow_path_client = ActionClient(self, FollowPath, 'follow_path')
        # self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # self.path_callback()
        
    def path_callback(self, msg : Path):
        self.get_logger().info(f"Received path with {len(msg.poses)} poses")
        # user_input = input("Please verify the path in RViz. Enter '1' to approve, or press enter to cancel: ")
        # if user_input.strip() != "1":
        #     self.get_logger().info("Path execution cancelled by user.")
        #     return
        # self.followPath(msg)
        self.navogator.followPath(msg)
        self.get_logger().info("Path Traker Node finished.")
        # self.goToPose(msg.poses[0])
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