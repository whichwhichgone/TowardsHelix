"""
base_vlm_backbone.py

Abstract class definition of an end-to-end Vision-Language Model (VLM) backbone.
This provides a unified interface for models like Qwen3-VL, InternVL, LLaVA-Next, etc.
that have integrated vision and language components.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class VLMBackbone(nn.Module, ABC):
    """
    Abstract base class for end-to-end Vision-Language Models.
    
    Unlike the separate Vision Backbone + LLM Backbone design, this class
    encapsulates models that have integrated vision encoders and language models
    (e.g., Qwen3-VL, InternVL, LLaVA-Next).
    """

    def __init__(self, vlm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = vlm_backbone_id
        self.vlm: nn.Module = None
        self.tokenizer: PreTrainedTokenizerBase = None
        self.processor = None                                   # Optional: unified processor for some models

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Hidden dimension of the language model component."""
        ...

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Padding token ID."""
        ...

    @property
    @abstractmethod
    def vision_embed_dim(self) -> int:
        """
        Dimension of visual embeddings after projection.
        For end-to-end VLMs, this is typically the same as embed_dim.
        """
        ...

    @abstractmethod
    def get_image_transform(self) -> Callable:
        """Return the image preprocessing transform/processor."""
        ...

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for distributed training."""
        ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        ...

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Embed input token IDs to hidden representations."""
        ...

    @abstractmethod
    def embed_images(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process images and return visual embeddings.
        
        Args:
            pixel_values: Preprocessed image tensor
            **kwargs: Additional arguments (e.g., image_grid_thw for Qwen3-VL)
            
        Returns:
            Visual embeddings tensor
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the VLM.
        
        Supports both:
        1. Direct vision-language forward (pixel_values + input_ids)
        2. Embedding-based forward (inputs_embeds) for compatibility with existing code
        """
        ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        """Return the prompt builder class for this model."""
        ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]:
        """Return the transformer layer class for FSDP wrapping."""
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """Return the preferred half-precision dtype (bf16 or fp16)."""
        ...

    @property
    def requires_projector(self) -> bool:
        """
        Whether this VLM requires an external projector.
        
        Most end-to-end VLMs (Qwen3-VL, InternVL) have built-in projectors,
        so this defaults to False.
        """
        return False

    @property
    def is_end_to_end_vlm(self) -> bool:
        """Flag to indicate this is an end-to-end VLM backbone."""
        return True
