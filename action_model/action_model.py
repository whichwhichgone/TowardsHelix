"""
action_model.py

"""
from action_model.models import DiT
from action_model import create_diffusion
from action_model.obs_encoder import ObsImgEncoder
from . import gaussian_diffusion as gd
from .conditional_flow_matching import ConditionalFlowMatcher as CFM
import torch
from torch import nn
from torchvision.models import resnet18

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
        self.flow_matching = CFM(sigma=0.0)
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

        resnet_scene = resnet18(weights=None)
        resnet_left = resnet18(weights=None)
        resnet_right = resnet18(weights=None)

        # image encoder for action model
        self.scene_encoder = ObsImgEncoder(backbone_model=resnet_scene, input_shape=(3, 224, 224), output_dim=4096, use_group_norm=True)
        self.left_encoder = ObsImgEncoder(backbone_model=resnet_left, input_shape=(3, 224, 224), output_dim=4096, use_group_norm=True)
        self.right_encoder = ObsImgEncoder(backbone_model=resnet_right, input_shape=(3, 224, 224), output_dim=4096, use_group_norm=True)

    # Given condition z and ground truth token x, compute loss
    def loss(self, x_action, z_cognition, s_state, p_image, d_depth, repeated_diffusion_steps=None, action_masks=None):
        # sample random noise and timestep
        noise = torch.randn_like(x_action)
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x_action.size(0),), device= x_action.device)

        p_scene, p_left, p_right = p_image['scene'], p_image['left'], p_image['right']
        p_scene = self.scene_encoder(p_scene).unsqueeze(1)
        p_left = self.left_encoder(p_left).unsqueeze(1)
        p_right = self.right_encoder(p_right).unsqueeze(1)
        p = torch.cat([p_scene, p_left, p_right], dim=1)
        assert z_cognition.shape[0] % p.shape[0] == 0
        p = p.repeat(z_cognition.shape[0] // p.shape[0], 1, 1)
        z = torch.cat([p, z_cognition], dim=1)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x_action, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z)

        assert noise_pred.shape == noise.shape == x_t.shape
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
    
    def cfm_loss(self, x_action, z_cognition, s_state, p_image, d_depth, repeated_diffusion_steps=None, action_masks=None):
        """
        Compute the CFM loss for the ActionModel.
        Args:
            x_action: Ground truth action tensor.
            z_cognition: Cognition tensor.
            s_state: State tensor.
            p_image: Dictionary containing preprocessed images.
            d_depth: Depth tensor.
            repeated_diffusion_steps: Optional, repeated diffusion steps.
            action_masks: Optional, action masks.
        Returns:
            loss: Computed CFM loss.
        """
        p_scene, p_left, p_right = p_image['scene'], p_image['left'], p_image['right']
        p_scene = self.scene_encoder(p_scene).unsqueeze(1)
        p_left = self.left_encoder(p_left).unsqueeze(1)
        p_right = self.right_encoder(p_right).unsqueeze(1)
        p = torch.cat([p_scene, p_left, p_right], dim=1)
        assert z_cognition.shape[0] % p.shape[0] == 0

        # x_action has repeated elements, so the source noise should be repeated as well
        x_source = torch.randn(p.shape[0], x_action.shape[1], x_action.shape[2], dtype=x_action.dtype, device=x_action.device)
        x_source = x_source.repeat(z_cognition.shape[0] // p.shape[0], 1, 1)
        p = p.repeat(z_cognition.shape[0] // p.shape[0], 1, 1)
        z = torch.cat([p, z_cognition], dim=1)

        timestep, x_t, u_t = self.flow_matching.sample_location_and_conditional_flow(x0=x_source, x1=x_action)
        v_t = self.net(x_t, timestep, z)

        loss = ((v_t - u_t) ** 2).mean()
        return loss