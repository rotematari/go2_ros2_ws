import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from DubinsPath.dubins_path_planner import plan_dubins_path
import math
from scipy.spatial.transform import Rotation as R
import numpy as np

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        self.subscription = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_pose_callback,
            10
        )
        self.publisher = self.create_publisher(Path, '/path', 10)
        self.path = Path()
        self.path.header.frame_id = 'base_footprint'
        self.get_logger().info('Path Planner Node started.')

    def goal_pose_callback(self, msg: PoseStamped):
        self.get_logger().info('Received a new goal pose.')
        
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        goal_yaw = self.quaternion_to_yaw(msg.pose.orientation)
        self.get_logger().info(f'Goal pose: ({goal_x}, {goal_y}, {goal_yaw})')
        self.plan(goal_x, goal_y, goal_yaw)
        # Update path header time and publish the updated path
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.path)
        self.get_logger().info(f'Published path with {len(self.path.poses)} poses.')
    
    def quaternion_to_yaw(self, quaternion):
        # Assuming your quaternion is in [x, y, z, w] format
        q = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        r = R.from_quat(q)

        # 'zyx' convention returns Euler angles in order: yaw, pitch, roll
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        return yaw
    def yaw_to_quaternion(self, yaw):
        # Create a rotation from the yaw angle (around z-axis)
        r = R.from_euler('z', yaw)
        # Get the quaternion (in [x, y, z, w] order)
        q = r.as_quat()
        return q
    def plan(self, g_x, g_y, g_yaw):
        
        path_x, path_y, path_yaw, mode, lengths = plan_dubins_path( 0, 0, 0,
                                                                    g_x, g_y, g_yaw,
                                                                    curvature=20,
                                                                    step_size=5)
        
        for x, y, yaw in zip(path_x, path_y, path_yaw):
            self.get_logger().info(f'Pose: x={x}, y={y}, yaw={yaw}')
        # Plan a Dubins path
        self.get_logger().info(f'Planned a Dubins path with {len(path_x)} poses.')
        self.path.poses = []
        for x, y, yaw in zip(path_x, path_y, path_yaw):
            pose = PoseStamped()
            pose.header.frame_id = 'base_footprint'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            q = self.yaw_to_quaternion(yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            self.path.poses.append(pose)
    def destroy_node(self):
        self.get_logger().info('Destroying node...')
        super().destroy_node()
        
    
def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()