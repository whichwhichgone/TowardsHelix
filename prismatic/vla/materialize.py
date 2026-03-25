"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.

Two entry points:
  - get_vla_dataset_and_collator       — for Prismatic (non-e2e) VLMs
  - get_vla_dataset_and_collator_e2e   — for end-to-end VLMs (Qwen-VL, etc.)
"""

from pathlib import Path
from typing import Callable, Tuple, Type, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollator, PaddedCollatorOe, PaddedCollatorOeQwenVL3
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSBatchTransformOe, RLDSDataset, RLDSBatchTransformOeQwenVL3


def _build_rlds_dataset(
    data_root_dir: Path,
    data_mix: str,
    batch_transform: RLDSBatchTransform,
    default_image_resolution: Tuple[int, int, int],
    shuffle_buffer_size: int,
    train: bool,
    episodic: bool,
    future_action_window_size: int,
    past_action_window_size: int,
    image_aug: bool,
    load_all_data_for_training: bool,
) -> Dataset:
    """Shared helper to construct the RLDS Iterable Dataset."""
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    return cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
        image_aug=image_aug,
        load_all_data_for_training=load_all_data_for_training,
    )


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,
    load_all_data_for_training: bool = True,
    base_action_tokenizer: PreTrainedTokenizerBase = None,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollator]:
    """Initialize RLDS Dataset for Prismatic (non-e2e) VLMs.

    Uses RLDSBatchTransformOe + PaddedCollatorOe.
    """
    action_tokenizer = ActionTokenizer(base_action_tokenizer) if base_action_tokenizer is not None else None

    batch_transform = RLDSBatchTransformOe(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorOe(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    dataset = _build_rlds_dataset(
        data_root_dir, data_mix, batch_transform, default_image_resolution,
        shuffle_buffer_size, train, episodic, future_action_window_size,
        past_action_window_size, image_aug, load_all_data_for_training,
    )

    return dataset, action_tokenizer, collator


def get_vla_dataset_and_collator_e2e(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Union[Type[PromptBuilder], Callable],
    default_image_resolution: Tuple[int, int, int],
    e2e_vlm_name: str,
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,
    load_all_data_for_training: bool = True,
    base_action_tokenizer: PreTrainedTokenizerBase = None,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollator]:
    """Initialize RLDS Dataset for end-to-end VLMs.

    Dispatches to the appropriate RLDSBatchTransform and PaddedCollator based on
    ``e2e_vlm_name``.  Currently supported families:

    * ``"qwen"`` — RLDSBatchTransformOeQwenVL3 + PaddedCollatorOeQwenVL3
    """
    action_tokenizer = ActionTokenizer(base_action_tokenizer) if base_action_tokenizer is not None else None

    name_lower = e2e_vlm_name.lower()

    if "qwen" in name_lower:
        batch_transform = RLDSBatchTransformOeQwenVL3(
            action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
        )
        collator = PaddedCollatorOeQwenVL3(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
        )
    else:
        raise ValueError(
            f"Unsupported end-to-end VLM '{e2e_vlm_name}'. "
            "Please add a corresponding RLDSBatchTransform and PaddedCollator for this model."
        )

    dataset = _build_rlds_dataset(
        data_root_dir, data_mix, batch_transform, default_image_resolution,
        shuffle_buffer_size, train, episodic, future_action_window_size,
        past_action_window_size, image_aug, load_all_data_for_training,
    )

    return dataset, action_tokenizer, collator