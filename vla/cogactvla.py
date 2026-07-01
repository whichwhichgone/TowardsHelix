"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torchdiffeq
import torch.nn as nn
import numpy as np
from PIL.Image import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast

import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from action_model.action_model import ActionModel
from action_model.models import DiT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class CogACT(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = 'DiT-B',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        e2e_vlm: bool = False,
        lm_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.e2e_vlm = e2e_vlm
        self.action_model = ActionModel(
            model_type = action_model_type, 
            token_size = token_size, 
            in_channels = action_dim, 
            future_action_window_size = future_action_window_size, 
            past_action_window_size = past_action_window_size
            )
        self.state_proj = nn.Sequential(
            nn.Linear(15, token_size, bias=False),
            nn.Dropout(p=0.8),
        )
        hidden_size = vlm.vlm_backbone.embed_dim if e2e_vlm else vlm.llm_backbone.embed_dim
        self.hidden_proj = nn.Linear(hidden_size, self.action_model.net.hidden_size, bias=False)
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion', 'state_proj', 'hidden_proj']
        else:
            self.all_module_keys = ['action_model', 'state_proj', 'hidden_proj']
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        # Diffusion head is always trainable
        self._trainable_module_keys = ['action_model', 'state_proj', 'hidden_proj']
        self.norm_stats = norm_stats
        self.lm_loss_weight = lm_loss_weight
        self.use_cfm = True

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        if self.e2e_vlm:
            return None  # End-to-end VLMs don't have separate vision backbone
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def _build_vlm_kwargs(
        self,
        input_ids, attention_mask, pixel_values, pixel_utils, labels,
        inputs_embeds, past_key_values, use_cache, output_attentions,
        output_hidden_states, return_dict, image_grid_thw,
    ) -> dict:
        # form the base kwargs that are common to both e2e and non-e2e VLMs
        base = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.e2e_vlm:
            if "qwen" in self.vlm.model_id.lower():
                base["image_grid_thw"] = image_grid_thw
        else:
            base["pixel_utils"] = pixel_utils
        return base

    def _project_vlm_hidden_states_for_dit(self, hidden_states):
        num_dit_blocks = len(self.action_model.net.blocks)
        assert len(hidden_states) >= num_dit_blocks, (
            f"VLM returned {len(hidden_states)} hidden states, but the action DiT has "
            f"{num_dit_blocks} blocks."
        )
        hidden_features = [self.hidden_proj(hidden_state) for hidden_state in hidden_states[-num_dit_blocks:]]
        return torch.stack(hidden_features, dim=1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_utils: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks = None,
        state = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        
        # assemble VLM kwargs based on VLM type; all forward inputs to be passed to self.vlm() must go through here.
        vlm_kwargs = self._build_vlm_kwargs(
            input_ids, attention_mask, pixel_values, pixel_utils, labels,
            inputs_embeds, past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, image_grid_thw,
        )
        vlm_kwargs["output_hidden_states"] = True
        output: CausalLMOutputWithPast = self.vlm(**vlm_kwargs)

        # Extract the last DiT-depth hidden states and their corresponding attention masks.
        # Keep only the assistant response tokens (incl. latent actions) available to DiT cross-attention.
        fused_attention_mask = ~output["fused_attention_mask"].bool()
        leading_ignore = labels.eq(IGNORE_INDEX).long().cumprod(dim=1).bool()
        assert leading_ignore.shape[1] == fused_attention_mask.shape[1], (
            "Expected labels and fused_attention_mask to have the same sequence length, "
            f"but got labels length {leading_ignore.shape[1]} and fused_attention_mask length {fused_attention_mask.shape[1]}."
        )
        fused_attention_mask = fused_attention_mask | leading_ignore

        # extract the cognition feature
        state_features = self.state_proj(state)                                                 # [B, 1, D]
        hidden_features = self._project_vlm_hidden_states_for_dit(output.hidden_states)

        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        state_features_repeated = state_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]
        hidden_features_repeated = hidden_features.repeat(repeated_diffusion_steps, 1, 1, 1)
        hidden_mask_repeated = fused_attention_mask.repeat(repeated_diffusion_steps, 1)

        # Action model forward and compute loss
        if self.use_cfm:
            action_loss = self.action_model.cfm_loss(actions_repeated, state_features_repeated, context=hidden_features_repeated, context_mask=hidden_mask_repeated)
        else:
            action_loss = self.action_model.loss(actions_repeated, state_features_repeated, context=hidden_features_repeated, context_mask=hidden_mask_repeated)

        loss = action_loss
        has_lm_targets = labels.ne(IGNORE_INDEX).any()
        if self.lm_loss_weight > 0 and output.loss is not None and has_lm_targets:
            loss = loss + self.lm_loss_weight * output.loss
        return loss, output, action_loss

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        if self.e2e_vlm:
            # For end-to-end VLMs, use the VLM's own wrapping policy
            vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()
            prismatic_fsdp_wrapping_policy = partial(
                _module_wrap_policy,
                module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
            )
            return partial(
                _or_policy,
                policies=[vlm_fsdp_wrapping_policy, prismatic_fsdp_wrapping_policy],
            )
        else:
            vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
            llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

            # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
            prismatic_fsdp_wrapping_policy = partial(
                _module_wrap_policy,
                module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
            )

            # Return union (_or_) over constituent policies
            #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
            #            automatically be folded into the root VLM FSDP instance.
            return partial(
                _or_policy,
                policies=[
                    vision_fsdp_wrapping_policy,
                    llm_fsdp_wrapping_policy,
                    prismatic_fsdp_wrapping_policy,
                ],
            )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        lm_loss_weight: float = 1.0,
        **kwargs,
    ) -> CogACT:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        
        # Load llm_backbone with shape mismatch tolerance (e.g., resized token embeddings)
        llm_ckpt = model_state_dict["llm_backbone"]
        llm_current = vlm.llm_backbone.state_dict()
        llm_filtered = {k: v for k, v in llm_ckpt.items() if k in llm_current and llm_current[k].shape == v.shape}
        vlm.llm_backbone.load_state_dict(llm_filtered, strict=False)
        
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize CogACT
        cogact = CogACT(vlm,
                        token_size = vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        lm_loss_weight = lm_loss_weight,
                        )
        # Load State projector from Checkpoint
        if "state_proj" in model_state_dict:
            try:
                cogact.state_proj.load_state_dict(model_state_dict["state_proj"])
                overwatch.info("Successfully loaded state_proj weights from the pretrained checkpoint.")
            except Exception as e:
                overwatch.warning(f"Failed to load state_proj weights: {e}. Initializing with random weights.")
        
        # Load Hidden projector from Checkpoint
        if "hidden_proj" in model_state_dict:
            try:
                cogact.hidden_proj.load_state_dict(model_state_dict["hidden_proj"])
                overwatch.info("Successfully loaded hidden_proj weights from the pretrained checkpoint.")
            except Exception as e:
                overwatch.warning(f"Failed to load hidden_proj weights: {e}. Initializing with random weights.")

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            try:
                cogact.action_model.load_state_dict(model_state_dict["action_model"])
                overwatch.info("Successfully loaded ActionModel weights from the pretrained checkpoint.")
            except Exception as e:
                overwatch.warning(f"Failed to load ActionModel weights: {e}. Initializing with random weights.")

            if "ema_diffusion" in model_state_dict and use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")
        return cogact

    @classmethod
    def from_pretrained_end_to_end(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vlm_backbone,  # VLMBackbone instance (e.g., Qwen3VLBackbone)
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "end-to-end",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        lm_loss_weight: float = 1.0,
        **kwargs,
    ) -> "CogACT":
        """
        Load CogACT from pretrained checkpoint using an end-to-end VLM backbone.
        
        This method handles VLM models like Qwen3-VL that have integrated
        vision and language components.
        """
        from prismatic.models.vlms import EndToEndVLM
        
        # Create EndToEndVLM wrapper
        vlm = EndToEndVLM(
            model_id,
            vlm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint if exists
        if pretrained_checkpoint is not None and pretrained_checkpoint.exists():
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
            
            # Load VLM backbone weights
            if "vlm_backbone" in model_state_dict:
                vlm.vlm_backbone.load_state_dict(model_state_dict["vlm_backbone"])
                overwatch.info("Loaded vlm_backbone weights from checkpoint.")

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Get token size from VLM backbone
        token_size = vlm_backbone.embed_dim

        # Initialize CogACT
        cogact = cls(
            vlm,
            action_model_type=action_model_type,
            token_size=token_size,
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            use_ema=use_ema,
            norm_stats=norm_stats,
            e2e_vlm=True,
            lm_loss_weight=lm_loss_weight,
        )

        # Load action model and projector weights from checkpoint
        if pretrained_checkpoint is not None and pretrained_checkpoint.exists():
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
            
            if "state_proj" in model_state_dict:
                try:
                    cogact.state_proj.load_state_dict(model_state_dict["state_proj"])
                    overwatch.info("Successfully loaded state_proj weights from the pretrained checkpoint.")
                except Exception as e:
                    overwatch.warning(f"Failed to load state_proj weights: {e}. Initializing with random weights.")
            
            if "hidden_proj" in model_state_dict:
                try:
                    cogact.hidden_proj.load_state_dict(model_state_dict["hidden_proj"])
                    overwatch.info("Successfully loaded hidden_proj weights from the pretrained checkpoint.")
                except Exception as e:
                    overwatch.warning(f"Failed to load hidden_proj weights: {e}. Initializing with random weights.")

            if "action_model" in model_state_dict:
                try:
                    cogact.action_model.load_state_dict(model_state_dict["action_model"])
                    overwatch.info("Successfully loaded ActionModel weights from the pretrained checkpoint.")
                except Exception as e:
                    overwatch.warning(f"Failed to load ActionModel weights: {e}. Initializing with random weights.")

                if "ema_diffusion" in model_state_dict and use_ema:
                    cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
                elif use_ema:
                    cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])
            else:
                overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")

        return cogact        

    @torch.inference_mode()
    def predict_action(
        self, image: Image,
        utils: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        robot_obs = None,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        robot_obs_tensor = torch.Tensor(robot_obs).unsqueeze(0)

        if self.e2e_vlm:
            # ------------------------------------------------------------------ #
            # End-to-end VLM inference (e.g., Qwen3-VL)                          #
            # ------------------------------------------------------------------ #
            prompt_builder_fn = self.vlm.vlm_backbone.prompt_builder_fn
            autocast_dtype = self.vlm.vlm_backbone.half_precision_dtype

            img_scene, image_hand_left = image["scene"], image["left"]
            img_obs = [img_scene, image_hand_left]            
            mm_utils = img_obs + utils

            # Build multimodal instruction string with <oe> placeholders
            oe_placeholders = "<oe>" * len(img_obs)
            oe_lang = f"Given the observation {oe_placeholders}; what action should the robot take to {instruction}?"

            # Interleave text tokens and image tokens into a content list
            content = []
            parts = oe_lang.split("<oe>")
            for i, token in enumerate(parts):
                if token:
                    content.append({"type": "text", "text": token})
                if i < len(parts) - 1:
                    content.append({"type": "image", "image": mm_utils[i]})

            # Construct the messages for the model
            messages = [{"role": "user", "content": content}]

            # Tokenize with add_generation_prompt=True to include the assistant header
            # so the model can autoregressively generate latent action text.
            inputs = prompt_builder_fn(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )

            prompt_without_assistant = prompt_builder_fn(
                messages, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt"
            )
            assistant_header_start = prompt_without_assistant["input_ids"].shape[1]

            input_ids = inputs["input_ids"].to(self.vlm.device)
            attention_mask = inputs["attention_mask"].to(self.vlm.device)
            pixel_values = inputs["pixel_values"].to(self.vlm.device)
            image_grid_thw = inputs["image_grid_thw"].to(self.vlm.device)
            robot_obs_proj = self.state_proj(robot_obs_tensor.to(self.vlm.device))

            # Step 1: Autoregressively generate latent action + action tokens.
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
                gen_output = self.vlm.vlm_backbone.vlm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=512,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            full_input_ids = gen_output.sequences

            # Step 2: Re-run a full teacher-forced forward pass over prompt + generated tokens.
            full_attention_mask = torch.ones_like(full_input_ids, dtype=torch.bool, device=self.vlm.device)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
                output = self.vlm.forward(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    image_grid_thw=image_grid_thw,
                )

            # Step 3: Mask out user prompt while keeping the assistant header and response visible.
            fused_attention_mask = ~output["fused_attention_mask"].bool()
            fused_attention_mask[:, :assistant_header_start] = True

        else:
            # ------------------------------------------------------------------ #
            # Non-end-to-end VLM inference (PrismaticVLM)                        #
            # ------------------------------------------------------------------ #
            image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
            special_tokens = {"additional_special_tokens": ["<oe>"]}
            tokenizer.add_special_tokens(special_tokens)

            # Build VLA Prompt
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
            if isinstance(tokenizer, LlamaTokenizerFast):
                # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
                #       insert it to match the inputs seen at training time. The empty token is at index 29871.
                #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
                )
            else:
                raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

            # Preprocess Image
            image_scene, image_hand_left = image["scene"], image["left"]
            pixel_values_scene = image_transform(image_scene)
            if isinstance(pixel_values_scene, torch.Tensor):
                pixel_values_scene = pixel_values_scene[None, ...].to(self.vlm.device)
            elif isinstance(pixel_values_scene, dict):
                pixel_values_scene = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values_scene.items()}
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_scene)}")

            pixel_values_left = image_transform(image_hand_left)
            if isinstance(pixel_values_left, torch.Tensor):
                pixel_values_left = pixel_values_left[None, ...].to(self.vlm.device)
            elif isinstance(pixel_values_left, dict):
                pixel_values_left = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values_left.items()}
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_left)}")

            pixel_values = {"scene" : pixel_values_scene, "left" : pixel_values_left}
            pixel_utils = []
            for util_image in utils:
                util_image = image_transform(util_image)
                util_image = {k: v[None, ...].to(self.vlm.device) for k, v in util_image.items()}
                pixel_utils.append(util_image)

            keys = pixel_utils[0].keys()
            pixel_utils = {k: torch.cat([d[k] for d in pixel_utils], dim=0) for k in keys}
            pixel_utils = [pixel_utils]    # add the batch dimension

            autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
            labels = torch.ones_like(input_ids).to(self.vlm.device) * IGNORE_INDEX
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(self.vlm.device)
            robot_obs_proj = self.state_proj(robot_obs_tensor.to(self.vlm.device))

            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
                output = self.vlm.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    pixel_utils=pixel_utils,
                    labels=labels,
                    output_hidden_states=True,
                    **kwargs
                )

            fused_attention_mask = ~output["fused_attention_mask"]

        # ------------------------------------------------------------------ #
        # Shared: project hidden states and run diffusion action sampling     #
        # ------------------------------------------------------------------ #
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = 1

        robot_obs_proj = robot_obs_proj.unsqueeze(1).to(model_dtype)   # [B, 1, D]
        cognition_features = robot_obs_proj                             # [B, 1, D]
        hidden_features = self._project_vlm_hidden_states_for_dit(output.hidden_states).to(model_dtype)

        using_cfg = cfg_scale > 1.0

        # Sample random noise
        noise = torch.randn(
            B, self.future_action_window_size + 1, self.action_model.in_channels,
            device=cognition_features.device,
        ).to(model_dtype)

        # Setup classifier-free guidance
        if not self.use_cfm:
            if using_cfg:
                noise = torch.cat([noise, noise], 0)
                uncondition = self.action_model.net.z_embedder.uncondition
                uncondition = uncondition.unsqueeze(0).expand(B, cognition_features.shape[1], -1)
                z = torch.cat([cognition_features, uncondition], 0)
                hidden_features = torch.cat([hidden_features, hidden_features], 0)
                fused_attention_mask = torch.cat([fused_attention_mask, fused_attention_mask], 0)
                model_kwargs = dict(z=z, cfg_scale=cfg_scale, context=hidden_features, context_mask=fused_attention_mask)
                sample_fn = self.action_model.net.forward_with_cfg
            else:
                model_kwargs = dict(z=cognition_features, context=hidden_features, context_mask=fused_attention_mask)
                sample_fn = self.action_model.net.forward

            # DDIM or DDPM sampling
            if use_ddim and num_ddim_steps is not None:
                if self.action_model.ddim_diffusion is None:
                    self.action_model.create_ddim(ddim_step=num_ddim_steps)
                samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                    sample_fn, noise.shape, noise,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=cognition_features.device, eta=0.0,
                )
            else:
                samples = self.action_model.diffusion.p_sample_loop(
                    sample_fn, noise.shape, noise,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=cognition_features.device,
                )

            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            normalized_actions = samples[0].cpu().numpy()
        else:
            # CFM: integrate the learned vector field from t=0 (noise) to t=1 (action)
            if using_cfg:
                overwatch.warning("cfg_scale > 1.0 is ignored when using CFM sampling.")
            samples = torchdiffeq.odeint(
                lambda t, x: self.action_model.net.forward(
                    x, t.view(1), cognition_features, hidden_features, fused_attention_mask
                ),
                noise,
                torch.linspace(0, 1, num_ddim_steps + 1, device=cognition_features.device),
                method='rk4',
            )
            normalized_actions = samples[-1][0].cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @torch.inference_mode()
    def predict_action_batch(
        self, image: List[Image], 
        instruction: List[str], 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference in batch; maps input image and task instruction to continuous action.
        This function is used for batch inference in the simulators.
        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        
        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction[id].lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)), dim=0
            ) # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch :]

        cumulative_sum = attention_mask.cumsum(dim=1)  
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1) #[B, D]

        assert (cognition_features.shape[0], cognition_features.shape[1]) == (B, 4096), "Batch size must be B for action prediction"
        using_cfg = cfg_scale > 1.0


        model_dtype = next(self.action_model.net.parameters()).dtype

        B = cognition_features.shape[0]
        
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,#False, try to set True 
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0)
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,#False, try to set True 
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
