# #!/home/admina/learn/path_traker/.venv/bin/python3
# import numpy as np
# import os
import torch
import torch.nn as nn
import torch.optim as optim
# import transformers
from transformers import AutoModel
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from torch.optim.lr_scheduler import StepLR
# from configs import get_parser
# from Speak2Act.dataloader2 import data_loaders
# from transformers import AutoModel
# from scripts.transformer_for_diffusion import TransformerForDiffusion
from tqdm import tqdm
import copy
# from utils import save_net, save_epoch_summary


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
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

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

        # Forward pass with masking

        predicted_noise, eos_vector = self.model(
            sample=noisy_actions,
            timestep=step.to(actions.device),
            cond=commands_ids,
            eos=True
        )

        predicted_noise = predicted_noise[action_mask.bool()]
        traj_loss = self.loss_fn(
            predicted_noise,
            noise[action_mask.bool()]
        )

        eos_loss = self.eos_loss_fn(eos_vector, eos)
        # print(eos_loss)

        loss = traj_loss + eos_loss

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

            predicted_noise, eos_vector = self.model(
                sample=noisy_actions,
                timestep=step.to(actions.device),
                cond=commands_ids,
                eos=True
            )

            if action_mask is not None:
                # predicted_noise = predicted_noise[action_mask.bool()]

                traj_loss = self.loss_fn(
                    predicted_noise[action_mask.bool()],
                    noise[action_mask.bool()]
                )
                eos_loss = self.eos_loss_fn(eos_vector, eos)
                loss = traj_loss + eos_loss

                # pred = noisy_actions - predicted_noise

                pred = (noisy_actions - torch.sqrt(1 - noise_level) * predicted_noise) / torch.sqrt(noise_level)

                pred_pos = pred[action_mask.bool()][:, :2]
                action_pos = actions[action_mask.bool()][:, :2]
                differences = pred_pos - action_pos.float()
                norms = torch.norm(differences, dim=1)
                mean_err_in_m = norms.mean()

        eos_prediction = (eos_vector[0] > 0.5).int()
        prediction = torch.cat((pred[0], eos_prediction), dim=1)
        return loss.item(), mean_err_in_m, prediction, self.model

    def sample_actions(self,commands_ids, num_steps=50, num_actions=200,attention_mask=None):

        commands_ids = self.text_encoder(commands_ids, attention_mask)
        batch_size = commands_ids.shape[0]
        device = commands_ids.device

        self.model.eval()

        with torch.no_grad():
            eos_vector_list = []

            x_t = torch.randn((batch_size, num_actions, 3), device=device)

            # for step in range(num_steps):
            for step in reversed(
                    range(num_steps)):  # num_steps should be equal to self.noise_scheduler.steps, during the training
                step_tensor = torch.full((batch_size,), step, device=device, dtype=torch.long)

                predicted_noise, eos_vector = self.model(
                    sample=x_t,
                    timestep=step_tensor,
                    cond=commands_ids,
                    eos=True
                )

                x_t_minus_1 = (1 / torch.sqrt(self.noise_scheduler.alphas[step])) * (
                        x_t - ((1 - self.noise_scheduler.alphas[step]) / torch.sqrt(
                    1 - self.noise_scheduler.alpha_cumprod[step])) * predicted_noise
                )

                x_t = x_t_minus_1
                eos_vector_list.append(eos_vector)
                # if step==num_steps-1:
                #     eos_vector0 = eos_vector

            eos_vector_avg = torch.mean(torch.stack(eos_vector_list), dim=0)  # Average across all steps
            eos_prediction = (self.sigmoid_(eos_vector_avg) > 0.5).int()
            mask = eos_prediction.squeeze(-1).bool()

            predicted_actions = x_t[mask]

            q_w = torch.cos(predicted_actions[:, -1]/2).unsqueeze(-1)  # Compute q_w
            q_z = torch.sin(predicted_actions[:, -1]/2).unsqueeze(-1)  # Compute q_z

            # Stack them back into the original shape
            quaternion_reconstructed = torch.cat([q_z, q_w], dim=-1)

            predicted_actions = torch.cat([predicted_actions[:, :2], quaternion_reconstructed], dim=-1)
            predicted_actions[:, :2] -= predicted_actions[0, :2]
            # print(predicted_actions)

            return predicted_actions


# # Example usage
# def main(config):
#     action_dim = 3  # Example for 6DoF actions
#     horizon_steps = 200
#     noise_steps = 50  # Default: 50
#     val_error = np.inf
#     checkpoint_path = config.checkpoint if hasattr(config, "checkpoint") else None

#     betas = (config.beta1, config.beta2)
#     w_d = config.weight_decay

#     enc_model_name = None
#     if config.tokenizer_type == 'GPT':
#         enc_model_name = "gpt2"
#     if config.tokenizer_type == 'BERT':
#         enc_model_name = "bert-base-uncased"

#     lr = config.lr
#     text_encoder = TextEncoder(model_name=enc_model_name).to(config.device)
#     noise_scheduler = NoiseScheduler(steps=noise_steps)  # .to(config.device)

#     # Replace DiffusionModel with TransformerForDiffusion
#     transformer_model = TransformerForDiffusion(
#         input_dim=action_dim,
#         output_dim=action_dim,
#         horizon=horizon_steps,
#         n_obs_steps=50,
#         cond_dim=768,  # hidden size
#         n_layer=6,
#         n_head=4,
#         n_emb=128
#     )
#     transformer_model.to(config.device)

#     diffusion_policy = DiffusionPolicy(transformer_model, text_encoder,
#                                        noise_scheduler, action_dim, lr=lr, betas=betas,
#                                        weight_decay=w_d,
#                                        checkpoint=checkpoint_path)

#     # Example dataset
#     train_dataloader, val_dataloader = data_loaders(config)

#     # scheduler = StepLR(diffusion_policy.optimizer, step_size=7, gamma=0.1)

#     start_epoch = diffusion_policy.start_epoch if hasattr(diffusion_policy, "start_epoch") else 0

#     for epoch in range(start_epoch, config.epochs):
#         print(f"🔄 Starting Epoch {epoch + 1}/{config.epochs}")
#         pbar = tqdm(train_dataloader, total=len(train_dataloader))
#         for command_ids, command_att, label, label_mask, eos in pbar:
#             torch.cuda.empty_cache()
#             command_ids = command_ids.to(config.device)
#             command_att = command_att.to(torch.bool).to(config.device)
#             eos = eos.to(config.device)

#             label = label.to(config.device).to(torch.float32)
#             label_mask = label_mask.to(config.device)

#             loss = diffusion_policy.train_step(command_ids, label, action_mask=label_mask,
#                                                attention_mask=command_att, eos=eos)  # , eos=eos
#             pbar.set_postfix({'Epoch': epoch + 1,
#                               'Training Loss': loss})

#         val_losses = []
#         errors_in_meters = []
#         v_pbar = tqdm(val_dataloader, total=len(val_dataloader))
#         for command_ids, command_att, label, label_mask, eos in v_pbar:
#             torch.cuda.empty_cache()
#             command_ids = command_ids.to(config.device)
#             command_att = command_att.to(torch.bool).to(config.device)
#             eos = eos.to(config.device)

#             label = label.to(config.device).to(torch.float32)
#             label_mask = label_mask.to(config.device)

#             val_loss, err_in_m, predicted_actions, model = diffusion_policy.validation_step(command_ids, label,
#                                                                                             action_mask=label_mask,
#                                                                                             attention_mask=command_att,
#                                                                                             eos=eos)  # eos=eos,

#             val_losses.append(val_loss)
#             errors_in_meters.append(err_in_m.item())
#             v_pbar.set_postfix({'Epoch': epoch + 1,
#                                 'Validation Loss': val_loss})
#             del command_ids, command_att, label, label_mask, eos

#         epoch_val_loss = sum(val_losses) / len(val_losses)

#         errors_in_meters_arr = np.array(errors_in_meters)

#         mean_error_meters = np.mean(errors_in_meters_arr)
#         median_error_meters = np.median(errors_in_meters_arr)
#         std_error_meters = np.std(errors_in_meters_arr)

#         epoch_summary = (f"\nFinished epoch {epoch + 1}\n"
#                          f"Validation loss: {epoch_val_loss}\n"
#                          f"Mean loss in meters: {mean_error_meters}\n"
#                          f"Median Error in meters: {median_error_meters}\n"
#                          f"Standard Deviation in meters: {std_error_meters}"
#                          f"\nMax error in meters: {max(errors_in_meters)}\n"
#                          "\nDP_train2\n")

#         if epoch_val_loss < val_error:
#             val_error = epoch_val_loss
#             model_path = os.path.join(config.saveM_path, config.model_name)
#             # best_weights = copy.deepcopy(model.state_dict())
#             best_weights = {
#                 "epoch": epoch + 1,  # Save the next epoch to continue from
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": diffusion_policy.optimizer.state_dict(),
#                 # "scheduler_state_dict": scheduler.state_dict(),
#             }
#             best_weights_path = save_net(model_path, best_weights, str(epoch + 1))
#             save_epoch_summary(config, best_weights_path, epoch_summary)
#             print(f"model {best_weights_path} saved ")

#         print(epoch_summary)

        # scheduler.step()


# if __name__ == "__main__":
#     parser = get_parser()
#     args_config = parser.parse_args()
#     main(args_config)
