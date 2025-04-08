#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from Speak2Act.whisper_model import *
import whisper

class PromptPublisher(Node):
    def __init__(self):
        super().__init__('prompt_publisher')
        self.publisher_ = self.create_publisher(String, 'prompt_topic', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.published = False
        self.model = whisper.load_model("base")
    def timer_callback(self):
        self.get_logger().info('Give command.')
        
        
        input_command = speech_to_text(model=self.model)
        # user_input = input("Enter prompt: ")
        msg = String()
        msg.data = input_command
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Published: "{msg.data}"')



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
