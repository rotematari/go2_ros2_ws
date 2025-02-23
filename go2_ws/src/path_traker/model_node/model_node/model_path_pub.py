#!/home/admina/learn/path_traker/.venv/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from Speak2Act.DP_train2 import DiffusionPolicy, TextEncoder, NoiseScheduler
from Speak2Act.scripts.transformer_for_diffusion import TransformerForDiffusion
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
import os
share_dir = get_package_share_directory('model_node')

class PathTrackerNode(Node):
    def __init__(self):
        super().__init__('path_tracker_node')
        
        # Subscriber: listen to prompt_topic for String messages
        self.subscription = self.create_subscription(
            String,
            'prompt_topic',
            self.prompt_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Publisher: publish PATH messages
        self.publisher = self.create_publisher(Path, 'path', 10)
        self.get_logger().info("PathTrackerNode has been started.")
        self.model_init()
        self.get_logger().info("Model has been initialized.")
    
    def model_init(self):
        config_path = os.path.join(share_dir,"Speak2Act", "config.yaml")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        weights_path = self.config["checkpoint"]
        noise_steps = 50
        self.num_denoise_steps = 50

        action_dim = 3  # Example for 6DoF actions
        horizon_steps = 200

        from transformers import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_name = "bert-base-uncased"

        text_encoder = TextEncoder(model_name=model_name).to(self.config["device"])
        noise_scheduler = NoiseScheduler(steps=noise_steps)

        transformer_model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon_steps,
            n_obs_steps=50,
            cond_dim=768,  # hidden size
            n_layer=6,
            n_head=4,
            n_emb=128
        )
        transformer_model.to(self.config["device"])

        self.diffusion_policy = DiffusionPolicy(transformer_model, text_encoder,
                                           noise_scheduler, action_dim)

        self.diffusion_policy.load_checkpoint(weights_path)
        
    def run_model(self, command: String):
        
        ncoded_command = self.tokenizer(command, padding='max_length', truncation=True,
                                   return_tensors='pt', max_length=self.config["max_length"])

        ncoded_command_ids = ncoded_command['input_ids'].to(self.config["device"])
        ncoded_command_att = ncoded_command['attention_mask'].to(torch.bool).to(self.config["device"])

        predicted_trajectory = self.diffusion_policy.sample_actions(ncoded_command_ids,
                                                 num_steps=self.num_denoise_steps,
                                                 attention_mask=ncoded_command_att)
        return predicted_trajectory
    def prompt_callback(self, msg: String):
        self.get_logger().info(f"Received prompt: {msg.data}")

        # Run the model
        predicted_trajectory = self.run_model(msg.data)
        self.get_logger().info(f"Model output: {predicted_trajectory}")
        
        # Create and populate a Path message
        path_msg = Path()
        path_msg.header.frame_id = "base_link"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path = []
        for trajectory_pose in predicted_trajectory:
            pose = PoseStamped()
            pose.header.frame_id = "base_link"
            pose.header.stamp = self.get_clock().now().to_msg()
            
            #position
            pose.pose.position.x = float(trajectory_pose[0])
            pose.pose.position.y = float(trajectory_pose[1])
            # orientation
            
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = float(trajectory_pose[2])
            pose.pose.orientation.w = float(trajectory_pose[3])
            
            
            path.append(pose)
        path_msg.poses.extend(path)
        # Populate path_msg fields as needed, for example:
        # path_msg.header.stamp = self.get_clock().now().to_msg()
        # path_msg.header.frame_id = "map"
        # Add other fields initialization here based on your PATH msg definition

        # Publish the PATH message
        self.publisher.publish(path_msg)
        self.get_logger().info("Published PATH message.")

def main(args=None):
    rclpy.init(args=args)
    node = PathTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()