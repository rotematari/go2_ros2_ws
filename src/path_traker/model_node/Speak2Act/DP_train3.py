import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Get text embeddings
        if self.model_name == "gpt2":
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            output = outputs.last_hidden_state
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            output = outputs.last_hidden_state
        return output


class NoiseScheduler:
    def __init__(self, steps, beta_start=0.1, beta_end=0.02):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_noise_level(self, step):
        return self.alpha_cumprod[step]


class DiffusionPolicy:
    def __init__(self,
                 model, text_encoder, noise_scheduler, action_dim,
                 lr=1e-4,
                 betas=(0.9, 0.999),
                 weight_decay=0.0,
                 checkpoint=None
                 ):
        self.model = model
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # betas=betas, weight_decay=weight_decay
        self.loss_fn = nn.MSELoss()
        self.eos_loss_fn = nn.BCEWithLogitsLoss()
        self.action_dim = action_dim
        self.sigmoid_ = nn.Sigmoid()

        if checkpoint is not None:
            print(f"🔄 Loading checkpoint from {checkpoint}")
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_path):
        """Load pre-trained model weights, optimizer, and scheduler, along with epoch number"""
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ Optimizer state restored.")

        if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✅ Scheduler state restored.")

        # 🔹 Restore last epoch
        self.start_epoch = checkpoint.get("epoch", 0)  # Default to 0 if not found
        print(f"✅ Resuming training from epoch {self.start_epoch}")

    def train_step(self, commands_ids, actions, action_mask=None, attention_mask=None, eos=None):
        self.model.train()
        commands_ids = self.text_encoder(commands_ids, attention_mask)

        step = torch.randint(0, self.noise_scheduler.steps, (actions.shape[0],))
        noise_level = self.noise_scheduler.get_noise_level(step).view(-1, 1, 1).to(actions.device)
        noise = torch.randn_like(actions)
        noisy_actions = actions * torch.sqrt(noise_level) + noise * torch.sqrt(1 - noise_level)

        predicted_noise = self.model(
            sample=noisy_actions,
            timestep=step.to(actions.device),
            cond=commands_ids,
            eos=False
        )

        traj_loss = self.loss_fn(
            predicted_noise,
            noise
        )

        loss = traj_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def validation_step(self, commands_ids, actions, action_mask=None, attention_mask=None, eos=None):
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad():
            commands_ids = self.text_encoder(commands_ids, attention_mask)

            step = torch.randint(0, self.noise_scheduler.steps, (actions.shape[0],))
            noise_level = self.noise_scheduler.get_noise_level(step).view(-1, 1, 1).to(actions.device)
            noise = torch.randn_like(actions)
            noisy_actions = actions * torch.sqrt(noise_level) + noise * torch.sqrt(1 - noise_level)

            predicted_noise= self.model(
                sample=noisy_actions,
                timestep=step.to(actions.device),
                cond=commands_ids,
                eos=False
            )

            traj_loss = self.loss_fn(predicted_noise,noise)
            loss = traj_loss

            pred = (noisy_actions - torch.sqrt(1 - noise_level) * predicted_noise) / torch.sqrt(noise_level)

            pred_pos = pred[:, :, :2].squeeze(1)
            action_pos = actions[:, :, :2].squeeze(1)
            differences = pred_pos - action_pos.float()
            norms = torch.norm(differences, dim=1)
            mean_err_in_m = norms.mean()

        return loss.item(), mean_err_in_m, self.model

    def sample_actions(self,commands_ids, num_steps=50, num_actions=1,attention_mask=None):
        print("🔮 Sampling actions")
        commands_ids = self.text_encoder(commands_ids, attention_mask)
        batch_size = commands_ids.shape[0]
        device = commands_ids.device
        print(f"🔮 Sampling actions for {batch_size} commands")
        self.model.eval()

        with torch.no_grad():

            x_t = torch.randn((batch_size, num_actions, 3), device=device)

            # for step in range(num_steps):
            for step in reversed(
                    range(num_steps)):  # num_steps should be equal to self.noise_scheduler.steps, during the training
                step_tensor = torch.full((batch_size,), step, device=device, dtype=torch.long)

                predicted_noise= self.model(
                    sample=x_t,
                    timestep=step_tensor,
                    cond=commands_ids,
                    eos=False
                )

                x_t_minus_1 = (1 / torch.sqrt(self.noise_scheduler.alphas[step])) * (
                        x_t - ((1 - self.noise_scheduler.alphas[step]) / torch.sqrt(
                    1 - self.noise_scheduler.alpha_cumprod[step])) * predicted_noise
                )

                x_t = x_t_minus_1

            predicted_actions = x_t

            q_w = torch.cos(predicted_actions[:, :, -1]/2).unsqueeze(-1)  # Compute q_w
            q_z = torch.sin(predicted_actions[:, :, -1]/2).unsqueeze(-1)  # Compute q_z

            # Stack them back into the original shape
            quaternion_reconstructed = torch.cat([q_z, q_w], dim=-1)

            predicted_actions = torch.cat([predicted_actions[:, :, :2], quaternion_reconstructed], dim=-1)
            
            return predicted_actions
