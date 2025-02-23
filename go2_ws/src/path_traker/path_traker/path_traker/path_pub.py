import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy ,HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import math
from tf2_ros import Buffer, TransformListener
import tf2_ros
#!/usr/bin/env python3

class PathPublisher(Node):
    def __init__(self):
        """
        Initialize the PathPublisher node.
        This constructor sets up the ROS node to subscribe to odometry messages and publish an accumulated path.
        It performs the following actions:
            - Declares ROS parameters for the odometry topic ("odom_topic") and the path topic ("path_topic").
            - Retrieves the parameter values for the topics.
            - Configures a Quality of Service (QoS) profile with keep-last history (depth 10) and reliable delivery.
            - Creates a publisher for the Path message on the specified path topic.
            - Sets up a subscriber for the Odometry messages on the specified odometry topic using the QoS profile.
            - Initializes a Path message with a header frame set to "odom" to accumulate poses.
            - Initializes a PoseStamped message with a header frame set to "odom".
        No parameters are taken and no value is returned.
        """
        super().__init__('path_pub')
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/path")
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        self.get_logger().info('Subscribing to Odometry messages on %s' % self.odom_topic)
        # Publisher for PATH message
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        # Subscriber for Odometry messages
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile
        )
        # Create a Path message to accumulate poses
        self.path_msg = Path()
        self.path_msg.header = Header()
        self.path_msg.header.frame_id = "odom"  # Change to appropriate frame if needed

        self.current_pose = PoseStamped()
        self.current_pose.header = Header()
        self.current_pose.header.frame_id = "odom"
        

        self.get_logger().info('PathPublisher node has been started.')
        self.pub_path()
        
    def odom_callback(self, msg: Odometry):
        """
        Callback function for processing incoming odometry messages.

        This function updates the current pose information based on the received Odometry message.
        It synchronizes the header timestamp of the current_pose with that of the message and replaces
        the pose data with the pose data from the message.

        Args:
            msg (Odometry): The odometry message containing the latest position and orientation data.
        """
        # self.get_logger().info('Received odometry message.')
        # self.current_odom = msg
        self.current_pose.header.stamp = msg.header.stamp
        self.current_pose.pose = msg.pose.pose
        self.get_logger().debug('new odom.')

    def pub_path(self):
        """
        Publishes a path message constructed from the current pose and additional computed poses.
        This method performs the following steps:
        1. Appends the current pose to the path message.
        2. Defines two nested functions to generate more poses:
            - compute_sine_path: Generates a sine wave path based on parameters such as number of points,
              step size, amplitude, and frequency. (Note: Its usage is currently commented out.)
            - compute_straight_path: Generates a straight-line path by incrementing the x-position for a given
              number of points and step size.
        3. Calls the compute_straight_path function with specified parameters to obtain additional poses.
        4. Extends the path message with the generated straight-line poses.
        5. Publishes the updated path message using the path publisher.
        6. Logs an informational message indicating the number of poses published.
        Note:
             - Assumes that attributes like self.current_pose, self.path_msg, and self.path_pub are properly
                initialized and contain valid data.
             - The nested functions are defined within pub_path for local use, encapsulating the path calculation logic.
        """
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
            x0 = self.current_pose.pose.position.x 
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
        straight_poses = compute_straight_path(self,1,0.1)
        self.path_msg.poses.extend(straight_poses)
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