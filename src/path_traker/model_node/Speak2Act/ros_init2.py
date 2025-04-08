#!/usr/bin/env python3

from dp.DP_train3 import DiffusionPolicy, TextEncoder, NoiseScheduler
from dp.scripts.transformer_for_diffusion2 import TransformerForDiffusion
import torch
import yaml


class DP:
    def __init__(self):
        try:
            with open("/home/roblab20/ROSProjects/instruct_robot_ws/src/go_teleop/scripts/dp/config2.yaml", "r") as file:
                self.config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print("YAML error:", e)
        weights_path = self.config["checkpoint"]
        noise_steps = 50
        self.num_denoise_steps = 50

        action_dim = 3
        horizon_steps = 1

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

    def run(self, command):
        ncoded_command = self.tokenizer(command, padding='max_length', truncation=True,
                                   return_tensors='pt', max_length=self.config["max_length"])

        ncoded_command_ids = ncoded_command['input_ids'].to(self.config["device"])
        ncoded_command_att = ncoded_command['attention_mask'].to(torch.bool).to(self.config["device"])

        predicted_trajectory = self.diffusion_policy.sample_actions(ncoded_command_ids,
                                                 num_steps=self.num_denoise_steps,
                                                 attention_mask=ncoded_command_att)
        return predicted_trajectory
    
