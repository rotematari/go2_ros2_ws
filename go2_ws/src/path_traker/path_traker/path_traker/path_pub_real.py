import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy ,HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import rclpy.time
from std_msgs.msg import Header
import math
from tf2_ros import Buffer, TransformListener
import tf2_ros
from rclpy.duration import Duration


#!/usr/bin/env python3

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_pub')
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("robot_frame", "base_footprint")
        
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.robot_frame = self.get_parameter("robot_frame").get_parameter_value().string_value
        
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        
        # Publisher for PATH message
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        # Subscriber for Odometry messages
        # self.odom_sub = self.create_subscription(
        #     Odometry,
        #     self.odom_topic,
        #     self.odom_callback,
        #     qos_profile
        # )
        # Create a Path message to accumulate poses
        # tf 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.path_msg = Path()
        self.path_msg.header = Header()
        self.path_msg.header.frame_id = self.robot_frame  # Change to appropriate frame if needed

        
        
        # self.get_robot_frame()

        

        self.get_logger().info('PathPublisher node has been started.')
        self.pub_path()
        
    def odom_callback(self, msg: Odometry):
        # self.get_logger().info('Received odometry message.')
        # self.current_odom = msg
        self.current_pose.header.stamp = msg.header.stamp
        self.current_pose.pose = msg.pose.pose
        self.get_logger().debug('new odom.')
        

    def get_robot_frame(self):
        # Wait until the transform is available
        delay = Duration(seconds=0.3)
        timeout = 5.0  # max wait time in seconds
        start_time = self.get_clock().now() - delay
        while not self.tf_buffer.can_transform('odom', 'base_link', tf2_ros.Time()):
            if (self.get_clock().now() - start_time).nanoseconds * 1e-9 > timeout:
                self.get_logger().warn("Timeout waiting for transform from odom to %s" % self.robot_frame)
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
            start_time = self.get_clock().now()-delay
        try:
            self.get_logger().info('Getting current pose from TF')
            # Lookup transform from odom to base_link
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.robot_frame,
                source_frame='odom', 
                time=tf2_ros.Time(),
                timeout=Duration(seconds=1.0))

            # Create PoseStamped message
            self.current_pose =  PoseStamped()
            self.current_pose.header = Header()
            self.current_pose.header.stamp = self.get_clock().now().to_msg()
            self.current_pose.header.frame_id = self.robot_frame
            

            # Set position from TF transform
            self.current_pose.pose.position.x = transform.transform.translation.x
            self.current_pose.pose.position.y = transform.transform.translation.y
            self.current_pose.pose.position.z = transform.transform.translation.z

            # Set orientation from TF transform
            self.current_pose.pose.orientation = transform.transform.rotation
            
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f'Could not get transform: {e}')
    
    def pub_path(self):
        self.get_logger().info('Publishing path from robot frame: %s' % self.robot_frame)
        self.current_pose =  PoseStamped()
        self.current_pose.header = Header()
        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        self.current_pose.header.frame_id = self.robot_frame
        
        self.path_msg.poses.append(self.current_pose)

        def compute_sine_path(self, num_points=20, step=0.5, amplitude=0.5, frequency=1.0):
            sine_path = []
            # Starting from the current pose
            x0 = self.current_pose.pose.position.x
            y0 = self.current_pose.pose.position.y
            q = self.current_pose.pose.orientation
            sine_path.append(self.current_pose)
            # Compute the current yaw from quaternion
            theta0 = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            for i in range(1, num_points + 1):
                x_local = i * step
                y_local = amplitude * math.sin(frequency * x_local)
                # Rotate the local point to the global frame
                x_global = x0 + math.cos(theta0) * x_local - math.sin(theta0) * y_local
                y_global = y0 + math.sin(theta0) * x_local + math.cos(theta0) * y_local
                # Compute derivative to get the tangent for orientation
                dy_dx = amplitude * frequency * math.cos(frequency * x_local)
                yaw = theta0 + math.atan2(dy_dx, 1.0)
                # Create a new PoseStamped for this point
                pose = PoseStamped()
                pose.header = Header()
                pose.header.stamp = self.current_pose.header.stamp
                pose.header.frame_id = self.current_pose.header.frame_id
                pose.pose.position.x = x_global
                pose.pose.position.y = y_global
                pose.pose.position.z = self.current_pose.pose.position.z
                half_yaw = yaw / 2.0
                pose.pose.orientation.w = math.cos(half_yaw)
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = math.sin(half_yaw)
                sine_path.append(pose)
            return sine_path

        def compute_straight_path(self,num_points=20,step=0.5):
            straight_path = []
            # x0 = self.current_pose.pose.position.x 
            x0 = 0
            for i in range(1,num_points+1):
                x_local = i * step
                
                x_global = x0 +x_local
                
                pose = PoseStamped()
                pose.header = Header()
                pose.header.stamp = self.current_pose.header.stamp
                pose.header.frame_id = self.current_pose.header.frame_id
                pose.pose.position.x = x_global
                
                straight_path.append(pose)
            return straight_path
        # sine_poses = compute_sine_path(self)
        # self.path_msg.poses.extend(sine_poses)
        # straight_poses = compute_straight_path(self,5,0.1)
        # self.path_msg.poses.extend(straight_poses)
        self.path_pub.publish(self.path_msg)
        self.get_logger().info('Published path with %d poses.' % len(self.path_msg.poses))
def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()