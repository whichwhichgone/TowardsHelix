"""
action_model.py

"""
from action_model.models import DiT
from action_model import create_diffusion
from . import gaussian_diffusion as gd
from .conditional_flow_matching import ConditionalFlowMatcher as CFM
import torch
from torch import nn

# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

# Create ActionModel
class ActionModel(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 100,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.flow_matching = CFM(sigma=0.0)
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z, context, context_mask):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z, context=context, context_mask=context_mask)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(timestep_respacing = "ddim"+str(ddim_step), 
                                               noise_schedule = self.noise_schedule,
                                               diffusion_steps = self.diffusion_steps, 
                                               sigma_small = True, 
                                               learn_sigma = False
                                               )
        return self.ddim_diffusion

    # flow matching loss
    def cfm_loss(self, x_action, z_cognition, context, context_mask):
        """
        Compute the CFM loss for the ActionModel.
        Args:
            x_action: Ground truth action tensor.
            z_cognition: Cognition tensor.
            context: Context from VLM.
            context_mask: Context mask from VLM.
        Returns:
            loss: Computed CFM loss.
        """

        x_source = torch.randn_like(x_action)
        timestep, x_t, u_t = self.flow_matching.sample_location_and_conditional_flow(x0=x_source, x1=x_action)
        v_t = self.net(x_t, timestep, z_cognition, context=context, context_mask=context_mask)

        loss = ((v_t - u_t) ** 2).mean()
        return loss