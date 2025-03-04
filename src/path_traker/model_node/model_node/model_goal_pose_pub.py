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
# Add tf2 imports for transform handling
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import geometry_msgs.msg

share_dir = get_package_share_directory('model_node')

class PathTrackerNode(Node):
    def __init__(self):
        super().__init__('goal_pose_pub_node')
        self.frame_id = "base_footprint"
        self.target_frame = "map"  # Target frame for transformation
        
        # Create tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriber: listen to prompt_topic for String messages
        self.subscription = self.create_subscription(
            String,
            'prompt_topic',
            self.prompt_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(
            PoseStamped,
            'my_goal_pose',
            10
        )
        self.get_logger().info("GoalPosePubNode has been started.")
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

    def transform_pose_to_map(self, pose):
        """
        Transform a PoseStamped from base_footprint frame to map frame
        """
        try:
            # Look up the transform from base_footprint to map
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Apply the transform to the pose
            transformed_pose = do_transform_pose(pose, transform)
            self.get_logger().info(f"Successfully transformed pose to {self.target_frame} frame")
            return transformed_pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'Could not transform pose: {str(e)}')
            return pose  # Return original pose if transformation fails
    
    def prompt_callback(self, msg: String):
        self.get_logger().info(f"Received prompt: {msg.data}")

        # Run the model
        predicted_trajectory = self.run_model(msg.data)
        self.get_logger().info(f"Model output: {predicted_trajectory}")
        predicted_goal_pose = predicted_trajectory[-1]
        self.get_logger().info(f"Predicted goal pose: {predicted_goal_pose}")
        
        # Create and populate a Path message

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        
        #position
        pose.pose.position.x = float(predicted_goal_pose[0])
        pose.pose.position.y = float(predicted_goal_pose[1])
        pose.pose.position.z = 0.0
        # orientation
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(predicted_goal_pose[2])
        pose.pose.orientation.w = float(predicted_goal_pose[3])

        # Transform the pose from base_footprint to map frame
        # transformed_pose = PoseStamped()
        # transformed_pose.header.frame_id = self.target_frame
        # transformed_pose.header.stamp = self.get_clock().now().to_msg()
        # transformed_pose.pose = self.transform_pose_to_map(pose.pose)

        # self.get_logger().info(f"Publishing pose in {self.target_frame} frame: "
        #                       f"position=({transformed_pose.pose.position.x}, {transformed_pose.pose.position.y}), "
        #                       f"orientation=({transformed_pose.pose.orientation.x}, {transformed_pose.pose.orientation.y}, "
        #                       f"{transformed_pose.pose.orientation.z}, {transformed_pose.pose.orientation.w})")
                              
        self.publisher_.publish(pose)
        self.get_logger().info("Published transformed goal pose.")

    
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