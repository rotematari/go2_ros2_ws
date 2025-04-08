import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from DubinsPath.dubins_path_planner import plan_dubins_path
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import TransformStamped
import time
class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        self.global_frame = "map"
        self.robot_frame = "base_footprint"
        self.subscription = self.create_subscription(
            PoseStamped,
            'my_goal_pose',
            self.goal_pose_callback,
            10
        )
        self.publisher = self.create_publisher(Path, '/path', 10)
        self.path = Path()
        self.path.header.frame_id = self.global_frame
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('Path Planner Node started.')

    def goal_pose_callback(self, msg: PoseStamped):
        # self.get_logger().info('Received a new goal pose.')
        
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        goal_yaw = self.quaternion_to_yaw(msg.pose.orientation)
        # self.get_logger().info(f'Goal pose: ({goal_x}, {goal_y}, {goal_yaw})')
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
        
        # for x, y, yaw in zip(path_x, path_y, path_yaw):
            # self.get_logger().info(f'Pose: x={x}, y={y}, yaw={yaw}')
        # Plan a Dubins path
        # self.get_logger().info(f'Planned a Dubins path with {len(path_x)} poses.')
        self.path.poses = []
        for x, y, yaw in zip(path_x, path_y, path_yaw):
            
            pose = PoseStamped()
            pose.header.frame_id = self.robot_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            q = self.yaw_to_quaternion(yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            transformed_pose = PoseStamped()
            transformed_pose = self.transform_pose_to_map(pose)
            self.path.poses.append(transformed_pose)
            
    
    def transform_pose_to_map(self, pose : PoseStamped):
    
        # self.get_logger().info(f"Transforming pose from {pose.header.frame_id} to map frame")
        # Create a new path object that will contain the transformed poses
        transformed_path = Path()
        transformed_path.header.stamp = self.get_clock().now().to_msg()
        transformed_path.header.frame_id = self.global_frame  # Set the target frame
        
        # Wait for the transform to become available
        try:
            # Give some time for the TF tree to be populated
            # time.sleep(0.5)
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,      # Target frame
                self.robot_frame,       # Source frame
                rclpy.time.Time(),      # Use latest available transform
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            
            # Transform each pose in the path
            # Transform the pose
            transformed_pose = PoseStamped()
            transformed_pose.pose = do_transform_pose(pose.pose, transform)
            transformed_pose.header.frame_id = self.global_frame
            transformed_pose.header.stamp = self.get_clock().now().to_msg()
                
                # Add the transformed pose to the new path
            transformed_path.poses.append(transformed_pose)
                
            # self.get_logger().info(f"Successfully transformed pose to map frame")
            return transformed_pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to transform path: {str(e)}")
            return None
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