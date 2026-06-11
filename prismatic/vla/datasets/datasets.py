"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Type
import random
import io
import re
import base64
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from transformers import AutoProcessor

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""

        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img_scene = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_left = Image.fromarray(rlds_batch["observation"]["image_secondary"][0])
        img_right = Image.fromarray(rlds_batch["observation"]["image_wrist"][0])
        if debug := False:
            img_debug_path = Path("imgs_debug")
            img_debug_path.mkdir(parents=True, exist_ok=True)
            img_scene.save(img_debug_path / "train_img_scene.png")
            img_left.save(img_debug_path / "train_img_left.png")
            img_right.save(img_debug_path / "train_img_right.png")

        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        # If action tokenizer is not used, we don't add the action to the chat answer
        if self.action_tokenizer is None:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]
        else:
            # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values_scene = self.image_transform(img_scene)
        pixel_values_left = self.image_transform(img_left)
        pixel_values_right = self.image_transform(img_right)
        pixel_values = {
            "scene" : pixel_values_scene,
            "left" : pixel_values_left,
            "right" : pixel_values_right,
        }

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        if self.action_tokenizer is None:
            labels[: -1] = IGNORE_INDEX
        else:
            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(len(action) + 1)] = IGNORE_INDEX

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=action, action_masks=action_mask)


@dataclass
class RLDSBatchTransformOe(RLDSBatchTransform):

    def __post_init__(self):
        special_tokens = {"additional_special_tokens": ["<oe>"]}
        num_added = self.base_tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens to the tokenizer.")

    def decode_utils(self, byte_utils):
        img_utils = []

        # byte_utils is a JSON string of base64-encoded PNG images
        raw = byte_utils.decode()
        if raw:
            encoded_list = json.loads(raw)
            for b64_str in encoded_list:
                img_data = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((224, 224))              # Resize to the default size for utils images
                if debug := False:
                    img_debug_path = Path("imgs_debug")
                    img_debug_path.mkdir(parents=True, exist_ok=True)
                    img.save(img_debug_path / f"train_util_{len(img_utils)}.png")
                img_utils.append(img)

        if not img_utils:
            # add a fake image placeholder for supporting fsdp
            random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            random_image = Image.fromarray(random_array)
            img_utils.append(random_image)
        return img_utils

    def get_oe_lang(self, rlds_batch):
        """Get the unified language instructions for oe-vla models."""
        if not rlds_batch["mm_task"]:
            raise ValueError(
                "mm_task is empty! OE-prompt datasets require mm_task to contain "
                "mm_instruction and mm_utils; make sure load_oe_prompt=True is set."
            )
        mm_instruction = rlds_batch["mm_task"]["mm_instruction"].decode().lower()
        mm_utils = rlds_batch["mm_task"]["mm_utils"]
        mm_utils = self.decode_utils(mm_utils)
        return mm_instruction, mm_utils

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""

        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img_scene = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_left = Image.fromarray(rlds_batch["observation"]["image_secondary"][0])
        img_right = Image.fromarray(rlds_batch["observation"]["image_wrist"][0])

        if debug := False:
            img_debug_path = Path("imgs_debug")
            img_debug_path.mkdir(parents=True, exist_ok=True)
            img_scene.save(img_debug_path / "train_img_scene.png")
            img_left.save(img_debug_path / "train_img_left.png")
            img_right.save(img_debug_path / "train_img_right.png")

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        oe_lang, mm_utils = self.get_oe_lang(rlds_batch)
        pixel_utils = [self.image_transform(util) for util in mm_utils]

        # If action tokenizer is not used, we don't add the action to the chat answer
        if self.action_tokenizer is None:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {oe_lang}?"},
                {"from": "gpt", "value": ""},
            ]
        else:
            # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {oe_lang}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values_scene = self.image_transform(img_scene)
        pixel_values_left = self.image_transform(img_left)
        pixel_values_right = self.image_transform(img_right)
        pixel_values = {
            "scene" : pixel_values_scene,
            "left" : pixel_values_left,
            "right" : pixel_values_right,
        }

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        if self.action_tokenizer is None:
            labels[: -1] = IGNORE_INDEX
        else:
            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(len(action) + 1)] = IGNORE_INDEX

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        state = rlds_batch["observation"]["proprio"]
        state = torch.tensor(state, dtype=torch.float32)
        return dict(pixel_values=pixel_values, pixel_utils=pixel_utils, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=action, action_masks=action_mask, state=state)


@dataclass
class RLDSBatchTransformOeQwenVL3(RLDSBatchTransformOe):

    def __post_init__(self):
        # QwenVL3 consider all the images same, no need to add special token for utils
        # This should not be removed even do nothing to rewrite the function
        pass

    def process_latent_action(self, latent_action):
        """Convert latent_action (3, 16, 2) array into a formatted string.

        Coordinates are normalized to Qwen3's 1000×1000 grounding space
        (divide by image size 200, then multiply by 1000).

        Format: [(x1,y1) (x2,y2) ... (x16,y16)];[(...)];[(...)]
        () wraps a coordinate pair, [] wraps one timestep, ; separates timesteps.
        """
        arr = np.array(latent_action, dtype=np.float32)
        arr = arr / 200.0 * 1000.0
        arr = np.round(arr)
        timesteps = []
        for t in range(arr.shape[0]):
            points = " ".join(f"({int(p[0])},{int(p[1])})" for p in arr[t])
            timesteps.append(f"[{points}]")
        return ";".join(timesteps)

    def oelang_to_qwen_input(self, obs_imgs, oe_lang, mm_utils, action, latent_action=None):
        # Replace <oe> tokens with actual image references in the content list
        view_nums = len(obs_imgs)
        oe_placeholders = "<oe>" * view_nums
        oe_lang = f"Given the observation {oe_placeholders}; what action should the robot take to {oe_lang}?"
        mm_utils = obs_imgs + mm_utils
        content = []

        parts = oe_lang.split("<oe>")
        for i, token in enumerate(parts):
            if token:
                content.append({"type": "text", "text": token})
            if i < len(parts) - 1:
                content.append({"type": "image", "image": mm_utils[i]})

        # Build assistant text: latent_action (if present) + action tokens
        assistant_text = ""
        if latent_action is not None:
            assistant_text += latent_action
        if self.action_tokenizer is not None:
            assistant_text += self.action_tokenizer(action)

        assistant_content = [{"type": "text", "text": assistant_text}] if assistant_text else []

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_content},
        ]

        inputs = self.prompt_builder_fn(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )

        # Count assistant tokens for label masking
        assistant_inputs = self.prompt_builder_fn(
            [{"role": "assistant", "content": assistant_content}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        assistant_token_count = len(assistant_inputs["input_ids"][0])

        return inputs, assistant_token_count


    def __call__(self, rlds_batch):
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img_scene = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_left = Image.fromarray(rlds_batch["observation"]["image_secondary"][0])
        img_right = Image.fromarray(rlds_batch["observation"]["image_wrist"][0])

        if debug := False:
            img_debug_path = Path("imgs_debug")
            img_debug_path.mkdir(parents=True, exist_ok=True)
            img_scene.save(img_debug_path / "train_img_scene.png")
            img_left.save(img_debug_path / "train_img_left.png")
            img_right.save(img_debug_path / "train_img_right.png")

        oe_lang, mm_utils = self.get_oe_lang(rlds_batch)

        # Extract and process latent action
        latent_action_raw = rlds_batch["mm_task"].get("latent_action", None)
        latent_action = None
        if latent_action_raw is not None:
            latent_action = self.process_latent_action(latent_action_raw)
            latent_action_raw = np.array(latent_action_raw, dtype=np.float32)

        # Tokenize using QwenVL3's processor
        # For single arm setting (default use the right arm), the left image and right image are the same
        obs_imgs = [img_scene, img_left]
        qwen_input, assistant_token_count = self.oelang_to_qwen_input(obs_imgs, oe_lang, mm_utils, action, latent_action)
        input_ids = qwen_input["input_ids"]                                                 # shape: [1, seq_len]
        labels = input_ids.detach().clone()                                                 # shape: [1, seq_len]
        pixel_values = qwen_input["pixel_values"]
        image_grid_thw = qwen_input["image_grid_thw"]

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        labels[0, :-assistant_token_count] = IGNORE_INDEX

        state = rlds_batch["observation"]["proprio"]
        state = torch.tensor(state, dtype=torch.float32)
        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dataset_name=dataset_name,
            actions=action,
            action_masks=action_mask,
            state=state,
            debug_img=img_scene,
            debug_latent_action=latent_action_raw,
        )


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        future_action_window_size: int = 0,
        past_action_window_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
        load_all_data_for_training: bool = True,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]
        random.shuffle(mixture_spec)  # Shuffle to more diverse dataset across distributed processes

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary", "secondary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            load_oe_prompt=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=past_action_window_size + 1,                                    # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                                        # Skip trajectories without language labels
                #goal_relabeling_strategy="uniform",                                        # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            load_all_data_for_training=load_all_data_for_training,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            load_all_data_for_training=rlds_config["load_all_data_for_training"],
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
