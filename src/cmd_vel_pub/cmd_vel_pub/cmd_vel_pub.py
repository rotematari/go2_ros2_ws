#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf2_ros


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')

        self.start_time = self.get_clock().now().nanoseconds/1e9
        # Create a tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # self.odom_first = Odometry()
        qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE
            )
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        # self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        # self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.odom_first_flag = False

        self.moved_flag = False

        self.duration = 0.5  # seconds
        


    def timer_callback(self):
        msg = Twist()
        curent_time = self.get_clock().now().nanoseconds/1e9    
        if not self.moved_flag:
            # Set linear and angular velocities here
            msg.linear.x = 0.5
            msg.angular.z = 0.0
            self.publisher_.publish(msg)
            self.moved_flag = True
            self.get_logger().info('Publishing cmd_vel')
        else:
            if (curent_time - self.start_time) > self.duration:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.publisher_.publish(msg)
                self.get_logger().info('Stop cmd_vel')
                self.start_time = time.time()
    
    
    def odom_callback(self, msg):
        self.get_logger().info('Received odom message')
        self.odom_first = msg
        self.get_logger().info("Odom: %s" % self.odom_first.pose.pose.position.x)

def main(args=None):
    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher()
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()