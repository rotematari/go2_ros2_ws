import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math

class GoalPosePublisher(Node):
    def __init__(self):
        super().__init__('goal_pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        timer_period = 1.0  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.timer_callback()
        self.get_logger().info('Goal Pose Publisher has been started.')

    def timer_callback(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.pose.position.x = 1.0
        msg.pose.position.y = 2.0
        msg.pose.position.z = 0.0

        # Convert yaw to quaternion (for 90 degrees rotation)
        yaw = math.radians(90)
        qx, qy, qz, qw = self.yaw_to_quaternion(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published goal pose: {msg}')

    def yaw_to_quaternion(self, yaw):
        # Helper function to convert a yaw (in radians) into a quaternion
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qx, qy, qz, qw

def main(args=None):
    rclpy.init(args=args)
    node = GoalPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()