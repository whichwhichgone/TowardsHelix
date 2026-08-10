"""
vla.py

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
model configuration thereof. A given VLA model (`policy`) configures the following attributes:
    - Data Mixture (e.g., Bridge, OXE_MAGIC_SOUP, etc.)
    - Base VLM from Prismatic Registry (e.g., `prism-dinosiglip+7b`)
    - VLA Model Architecture / Parameters (e.g., freeze vision encoder, last layer finetuning)
    - Training / Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int]                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # Optional partial fine-tuning for end-to-end LLM backbones
    trainable_last_llm_layers: Optional[int] = None # Train only the last N LLM decoder layers when set

    # fmt: on


# === OpenVLA Training Configurations ===


# = [8 GPU] Fast Iteration =>> SigLIP 224px + Bridge =
@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = "siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


# === CogACT-VLA Pretraining Configs ===

@dataclass
class Exp_CogACT_OXE_Magic_Soup_Plus_Minus(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+oxe+diffusion"
    base_vlm: Union[str, Path] = "/zhaowei/models/prismatic-vlms/prism-dinosiglip-224px+7b"

    # data_mix: str = "oxe_magic_soup_plus"
    data_mix: str = "oxe_magic_soup_plus_minus"
    shuffle_buffer_size: int = 1_000
    freeze_vision_backbone: bool = True
    expected_world_size: int = 16
    global_batch_size: int = 256
    per_device_batch_size: int = 16
    max_grad_norm: float = 1.0
    learning_rate: float = 2e-5

    epochs: int = 100


# === Qwen3-VL Based VLA Configs ===

@dataclass
class Exp_Qwen3VL_4B_OXE(VLAConfig):
    vla_id: str = "qwen3-vl-4b+oxe+diffusion"
    base_vlm: Union[str, Path] = "qwen3-vl-4b-instruct"

    # For end-to-end VLMs, these control whether to freeze the VLM's internal components
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "oxe_magic_soup_plus_minus"
    shuffle_buffer_size: int = 1_000

    # Optimization Parameters
    epochs: int = 100
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 64
    per_device_batch_size: int = 8  # Smaller batch for 4B model

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


@dataclass
class Exp_Qwen3VL_4B_Bridge(Exp_Qwen3VL_4B_OXE):
    """VLA config for Qwen3-VL-4B finetuning on Bridge dataset."""
    vla_id: str = "qwen3-vl-4b+bridge+diffusion"
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 1_000
    epochs: int = 1000


@dataclass
class Exp_Qwen3VL_8B_OXE(VLAConfig):
    vla_id: str = "qwen3-vl-8b+oxe+diffusion"
    base_vlm: Union[str, Path] = "qwen3-vl-8b-instruct"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "oxe_magic_soup_plus_minus"
    shuffle_buffer_size: int = 1_000

    # Optimization Parameters
    epochs: int = 100
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 64
    per_device_batch_size: int = 8

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    # Sanity Check Configurations =>> BridgeV2
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge

    # === CogACT-VLA Pretraining Configs ===
    EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS = Exp_CogACT_OXE_Magic_Soup_Plus_Minus

    # === Qwen3-VL VLA Configs ===
    QWEN3VL_4B_OXE = Exp_Qwen3VL_4B_OXE
    QWEN3VL_4B_BRIDGE = Exp_Qwen3VL_4B_Bridge
    QWEN3VL_8B_OXE = Exp_Qwen3VL_8B_OXE

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
