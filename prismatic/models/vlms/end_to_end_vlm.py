"""
end_to_end_vlm.py

PyTorch Module defining an EndToEndVLM, which wraps end-to-end Vision-Language Models
like Qwen3-VL, InternVL, etc. that have integrated vision and language components.

This class provides compatibility with the existing CogACT/VLA training pipeline while
using the VLM's native vision-language processing.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.vlm import VLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class EndToEndVLM(nn.Module, GenerationMixin):
    """
    A VLM implementation for end-to-end Vision-Language Models.
    
    Unlike PrismaticVLM which combines separate Vision and LLM backbones,
    this class wraps models that have integrated vision-language processing
    (e.g., Qwen3-VL, InternVL, LLaVA-Next).
    """

    def __init__(
        self,
        model_id: str,
        vlm_backbone: VLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "end-to-end",
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.model_family = "end-to-end-vlm"
        self.model_id = model_id
        self.vlm_backbone = vlm_backbone
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.arch_specifier = arch_specifier

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vlm_backbone"]
        self.trainable_module_keys = []

        # === GenerationMixin Expected Attributes ===
        self.generation_config = self.vlm_backbone.vlm.generation_config
        self.main_input_name = "input_ids"

        # === Generation Utilities ===
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.vlm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            if len(token_idx_list) == 1:
                self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vlm_backbone: VLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "end-to-end",
        freeze_weights: bool = True,
        **kwargs,
    ) -> "EndToEndVLM":
        """Initialize an EndToEndVLM from a pretrained checkpoint."""
        vlm = cls(
            model_id,
            vlm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint
        if pretrained_checkpoint is not None and pretrained_checkpoint.exists():
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
            if "vlm_backbone" in model_state_dict:
                vlm.vlm_backbone.load_state_dict(model_state_dict["vlm_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        """Get prompt builder for this VLM."""
        prompt_initializer: Type[PromptBuilder] = self.vlm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        Set requires_grad_ on component modules based on training stage.
        
        For end-to-end VLMs, we treat the entire vlm_backbone as a single unit.
        """
        if stage == "align":
            # For end-to-end VLMs, "align" doesn't apply since there's no separate projector
            # We freeze everything
            self.vlm_backbone.requires_grad_(False)
            self.trainable_module_keys = []
            self.vision_backbone_requires_grad = False
            overwatch.info(f"[Frozen] 🥶 =>> VLM Backbone `{self.vlm_backbone.identifier}`", ctx_level=1)

        elif stage == "finetune":
            # Finetune the entire VLM
            self.vlm_backbone.requires_grad_(True)
            self.trainable_module_keys = ["vlm_backbone"]
            self.vlm_backbone.vlm.visual.requires_grad_(False)
            self.vision_backbone_requires_grad = False        # Vision part typically stays frozen
            overwatch.info(f"[TRAINABLE] 🔥 =>> VLM Backbone `{self.vlm_backbone.identifier}`", ctx_level=1)

        elif stage == "full-finetune":
            # Full finetune including vision encoder
            self.vlm_backbone.requires_grad_(True)
            self.trainable_module_keys = ["vlm_backbone"]
            self.vision_backbone_requires_grad = True
            overwatch.info(f"[TRAINABLE] 🔥 =>> VLM Backbone (full) `{self.vlm_backbone.identifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for EndToEndVLM!")

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint if required."""
        if pretrained_checkpoint is not None and pretrained_checkpoint.exists():
            overwatch.info(f"Loading from Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            if "vlm_backbone" in model_state_dict:
                self.vlm_backbone.load_state_dict(model_state_dict["vlm_backbone"])

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy."""
        vlm_fsdp_wrapping_policy = self.vlm_backbone.get_fsdp_wrapping_policy()
        return vlm_fsdp_wrapping_policy

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_utils: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the end-to-end VLM.
        
        This method delegates to the underlying VLM backbone's native forward method,
        which handles vision-language fusion internally.
        """
        # Handle inference with cache
        if input_ids is not None and input_ids.shape[1] == 1 and past_key_values is not None:
            return self.vlm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Standard multimodal forward
        return self.vlm_backbone(
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
            **kwargs,
        )

    # === GenerationMixin Expected Properties & Methods ===
    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def config(self) -> PretrainedConfig:
        return self.vlm_backbone.vlm.config

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.vlm_backbone.vlm._reorder_cache(past_key_values, beam_idx)

    # === Compatibility methods with existing code ===
    @property
    def vision_backbone(self):
        """For compatibility - returns None as there's no separate vision backbone."""
        return None

    @property
    def llm_backbone(self):
        """For compatibility - provides access to tokenizer and other LLM-like properties."""
        return self.vlm_backbone

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Embed input IDs for compatibility with existing code."""
        return self.vlm_backbone.embed_input_ids(input_ids)
