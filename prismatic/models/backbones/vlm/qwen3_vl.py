"""
qwen3_vl.py

Qwen3-VL Vision-Language Model backbone implementation.
"""

from functools import partial
from typing import Callable, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoProcessor, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLVisionBlock,
)

from prismatic.models.backbones.vlm.base_vlm_backbone import VLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Registry for Qwen3-VL models
QWEN3_VL_MODELS = {
    "qwen3-vl-4b-instruct": {
        "vlm_family": "qwen3-vl",
        "hf_hub_path": "Qwen/Qwen3-VL-4B-Instruct",
    },
    "qwen3-vl-8b-instruct": {
        "vlm_family": "qwen3-vl",
        "hf_hub_path": "Qwen/Qwen3-VL-8B-Instruct",
    },
}


class Qwen3VLBackbone(VLMBackbone):
    """
    Qwen3-VL backbone implementation.
    
    Qwen3-VL is an end-to-end VLM with:
    - Built-in vision encoder
    - Built-in vision-language projector
    - Qwen3 language model
    """

    def __init__(
        self,
        vlm_backbone_id: str,
        vlm_max_length: int = 4096,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        local_path: Optional[str] = None,
    ) -> None:
        super().__init__(vlm_backbone_id)
        
        self.vlm_max_length = vlm_max_length
        self.inference_mode = inference_mode
        
        # Get model config
        model_cfg = QWEN3_VL_MODELS[vlm_backbone_id]
        self.vlm_family = model_cfg["vlm_family"]
        
        # Use local path if provided, otherwise use HF hub path
        model_path = local_path if local_path else model_cfg["hf_hub_path"]
        
        # Qwen3-VL specific classes
        self._model_cls = Qwen3VLForConditionalGeneration
        self._transformer_layer_cls = (Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock)

        # Load model
        overwatch.info(
            f"Loading [bold]{self.vlm_family}[/] VLM from [underline]`{model_path}`[/]",
            ctx_level=1
        )
        self.vlm = self._model_cls.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float32,
            attn_implementation="flash_attention_2" if use_flash_attention_2 else "sdpa",
        )
        
        # Set cache mode
        self.vlm.config.use_cache = self.inference_mode
        
        # Load processor (handles both tokenization and image processing)
        overwatch.info(
            f"Loading [bold]{self.vlm_family}[/] Processor via AutoProcessor",
            ctx_level=1
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            token=hf_token,
            padding_side="right",
        )
        self.tokenizer = self.processor.tokenizer

        # Set default image size for Qwen3-VL for compatibility, though it's flexible
        self.default_image_resolution = (3, 224, 224)
        
        # Ensure tokenizer has proper max length
        if hasattr(self.tokenizer, 'model_max_length'):
            self.tokenizer.model_max_length = vlm_max_length

    @property
    def embed_dim(self) -> int:
        """Hidden dimension of the language model."""
        return self.vlm.config.text_config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def vision_embed_dim(self) -> int:
        """
        For Qwen3-VL, visual embeddings are projected to LLM hidden size internally.
        """
        return self.vlm.config.vision_config.out_hidden_size

    def get_image_transform(self) -> Callable:
        """
        Return image preprocessing function.
        
        For Qwen3-VL, we return the processor's image processing capability.
        """
        def transform(image):
            # The processor handles image preprocessing
            return self.processor.image_processor(images=image, return_tensors="pt")
        return transform

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy."""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self._transformer_layer_cls}
        )
        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self.vlm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Embed input token IDs."""
        return self.vlm.model.embed_tokens(input_ids)

    def embed_images(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process images through Qwen3-VL's vision encoder.
        
        Args:
            pixel_values: Preprocessed image tensor
            **kwargs: Additional arguments like image_grid_thw
            
        Returns:
            Visual embeddings
        """
        image_grid_thw = kwargs.get("image_grid_thw", None)
        
        # Get visual features from the vision model
        visual_outputs = self.vlm.visual(
            pixel_values,
            grid_thw=image_grid_thw,
        )
        return visual_outputs

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
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through Qwen3-VL.
        """

        out = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        out["fused_attention_mask"] = attention_mask
        return out

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        """Return Qwen3 prompt builder."""
        return self.processor.apply_chat_template

    @property
    def transformer_layer_cls(self) -> tuple:
        """Return transformer layer class for FSDP."""
        return self._transformer_layer_cls

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Qwen3-VL uses bfloat16."""
        return torch.bfloat16

    @property
    def requires_projector(self) -> bool:
        """Qwen3-VL has built-in projector."""
        return False

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        """Return modules for last-layer finetuning."""
        return (
            self.vlm.model.embed_tokens,
            self.vlm.model.layers[-1],
            self.vlm.lm_head,
        )
