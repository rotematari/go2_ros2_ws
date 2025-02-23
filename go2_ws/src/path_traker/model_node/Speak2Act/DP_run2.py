from DP_train2 import DiffusionPolicy, TextEncoder, NoiseScheduler
from scripts.transformer_for_diffusion import TransformerForDiffusion
from utils import quaternion_to_yaw
import matplotlib.pyplot as plt
import numpy as np
from configs import get_parser
import torch
from pynput import keyboard
import threading
import sys
#!/usr/bin/env python3
running = True  # Flag to control the while loop
command = None  # Global variable to store user input


def on_press(key):
    global running
    if key == keyboard.Key.esc:
        print("\nESC pressed. Simulation shutting down.")
        running = False  # Set the flag to False to exit the loop
        return False  # Stops the listener


def get_input():
    """Function to handle input in a separate thread"""
    global command, running
    while running:
        try:
            user_input = input("Enter Your Command: ")  # Blocking call
            if not running:  # If ESC was pressed while waiting for input, exit
                break
            command = user_input
        except EOFError:
            break


def run_diffusion(config, command, num_denoise_steps, model):

    ncoded_command = tokenizer(command, padding='max_length', truncation=True,
                                    return_tensors='pt', max_length=config.max_length)

    ncoded_command_ids = ncoded_command['input_ids'].to(config.device)
    ncoded_command_att = ncoded_command['attention_mask'].to(torch.bool).to(config.device)

    predicted_actions = model.sample_actions(ncoded_command_ids,
                                                        num_steps=num_denoise_steps,
                                                        attention_mask=ncoded_command_att)

    return predicted_actions


def plot_trajectory(results, command=None):
    if results.dim() == 3:
        results = results.squeeze(0)
    x_coords = results[:, 0].cpu()
    y_coords = results[:, 1].cpu()
    quaternions = results[:, 2:].cpu()

    # Extract yaw from quaternion
    yaws = [quaternion_to_yaw(qz, qw) for qz, qw in quaternions]

    x_subsampled = x_coords
    y_subsampled = y_coords
    yaws_subsampled = yaws

    dx = np.cos(yaws_subsampled)
    dy = np.sin(yaws_subsampled)

    # Create figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Trajectory Plot
    ax[0].plot(x_coords, y_coords, 'bo-', label='Trajectory')
    ax[0].quiver(x_subsampled, y_subsampled, dx, dy,
                 angles='xy', scale_units='xy',
                 scale=1, color='r', label='Orientation')
    ax[0].set_xlabel('X [meters]')
    ax[0].set_ylabel('Y [meters]')
    ax[0].set_title('Trajectory with Orientation')
    ax[0].grid(True)
    ax[0].axis('equal')
    ax[0].legend()

    # Polar Plot for Orientation
    ax[1] = plt.subplot(122, projection='polar')
    ax[1].scatter(yaws, np.ones_like(yaws), c='r', label='Orientations')
    ax[1].set_title('Orientations (Yaw)')
    ax[1].set_yticks([])
    ax[1].legend()

    # Add overall title
    plt.suptitle(command if command else 'Trajectory Visualization')

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = get_parser()
    args_config = parser.parse_args()
    weights_path = "checkpoint/BERT/DP1/02_13_2025/24_net_Thu_Feb_13_19_30_51_2025.pt"

    noise_steps = 50
    num_denoise_steps = 50
    tokenizer_type = 'BERT' # GPT; BERT

    action_dim = 3  # Example for 6DoF actions
    horizon_steps = 200

    running = True  # Flag to control the while loop

    # Don't touch this
    # -----------------
    model_name = None
    if tokenizer_type == 'GPT':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model_name = "gpt2"
    if tokenizer_type == 'BERT':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_name = "bert-base-uncased"


    text_encoder = TextEncoder(model_name=model_name).to(args_config.device)
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
    transformer_model.to(args_config.device)

    diffusion_policy = DiffusionPolicy(transformer_model, text_encoder,
                                       noise_scheduler, action_dim)

    diffusion_policy.load_checkpoint(weights_path)

    # ----------------------------------------------------------------

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start input thread
    input_thread = threading.Thread(target=get_input, daemon=True)
    input_thread.start()

    while running:
        if command:  # Process command only when entered
            predicted_trajectory = run_diffusion(args_config, command, num_denoise_steps, diffusion_policy)
            plot_trajectory(predicted_trajectory, command=command)
            command = None  # Reset command

    print("Program exited.")