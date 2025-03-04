#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String



class PromptPublisher(Node):
    def __init__(self):
        super().__init__('prompt_publisher')
        self.publisher_ = self.create_publisher(String, 'prompt_topic', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.published = False

    def timer_callback(self):
        user_input = input("Enter prompt: ")
        msg = String()
        msg.data = user_input
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = PromptPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
