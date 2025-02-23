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
        self.subscription = self.create_subscription(Path, 'path', self.path_callback, qos_profile)
        
        self.follow_path_client = ActionClient(self, FollowPath, 'follow_path')
        # self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')


        # self.path_callback()
        
    def path_callback(self, msg : Path):
        self.get_logger().info(f"Received path with {len(msg.poses)} poses")
        # self.navigator.clearWaypoints()
        # self.navigator.setInitialPose(msg.poses[0])
        # Process the received path message
        
        self.followPath(msg)
        self.get_logger().info("Path Traker Node finished.")
        # self.goToPose(msg.poses[0])
        # Start the navigation
    def followPath(self, path, controller_id='FollowPath', goal_checker_id='general_goal_checker'):
        """Send a `FollowPath` action request."""
        self.get_logger().debug("Waiting for 'FollowPath' action server")
        while not self.follow_path_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("'FollowPath' action server not available, waiting...")

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = controller_id
        goal_msg.goal_checker_id = goal_checker_id

        self.get_logger().info('Executing path...')
        send_goal_future = self.follow_path_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.get_logger().error('Follow path was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    def goToPose(self, pose, behavior_tree='nav2_compute_path_to_pose_action_bt_node'):
        """Send a `NavToPose` action request."""
        self.get_logger().debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.get_logger().error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return    
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