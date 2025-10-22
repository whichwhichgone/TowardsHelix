from tqdm import tqdm
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages
import tensorflow as tf
import tensorflow_datasets as tfds
import re
import numpy as np
from PIL import Image

import draccus
import yaml
from dataclasses import dataclass
from pathlib import Path
from functools import partial
import multiprocessing


def get_npy_episodes(dataset):
    if isinstance(dataset, list) and isinstance(dataset[0], tf.data.Dataset):
        total = []
        for item in dataset:
            retrieved_episodes = retrieve_episodes(item)
            total.extend(retrieved_episodes)
    elif isinstance(dataset, tf.data.Dataset):
        total = retrieve_episodes(dataset)
    else:
        raise TypeError("Input dataset must be a list or a tf.data.Dataset")
    return total


def retrieve_episodes(dataset):
    episodes = []
    for traj in tqdm(dataset, desc="Retrieve episodes", total=len(dataset)):
        steps = []
        for step in traj["steps"]:
            language_instruction = step["language_instruction"].numpy().decode("utf-8")
            state = step["observation"]["state"].numpy()
            image = step["observation"]["image"].numpy()
            wrist_image = step["observation"]["wrist_image"].numpy()
            action = step["action"].numpy()
            steps.append(
                {
                    "observation": {
                        "image": image,
                        "wrist_image": wrist_image,
                        "state": state,
                    },
                    "language_instruction": language_instruction,
                    "action": action,
                }
            )
        episodes.append(steps)
    return episodes


def process_episode(
    episodes,
    object_text2mmins=None,
    object_mmins2utils=None,
    object_utils=None,
    ocr_text2mmins=None,
    ocr_mmins2utils=None,
    ocr_utils=None,
    target_path=None,
):
    worker_id = multiprocessing.current_process().name.split("-")[-1]
    worker_id = int(worker_id) if worker_id.isdigit() else 0

    # the process applied to each episode
    for episode_index, episode in tqdm(
        enumerate(episodes),
        desc=f"Worker {worker_id}",
        total=len(episodes),
    ):
        for step in episode:
            text_instruction = step["language_instruction"]
            random_value = np.random.uniform(0, 1)
            if random_value < 0.2:
                # reserve the original data
                task_type = "plain"
                mm_instruction = None
                mm_utils = None
            elif random_value < 0.4:
                task_type = "object"
                mm_instruction, mm_utils = get_mm_utils_object(
                    text_instruction,
                    object_text2mmins,
                    object_mmins2utils,
                    Path(object_utils),
                )
            elif random_value < 0.6:
                task_type = "ocr"
                mm_instruction, mm_utils = get_mm_utils_ocr(
                    text_instruction,
                    ocr_text2mmins,
                    ocr_mmins2utils,
                    Path(ocr_utils),
                )
            elif random_value < 0.8:
                task_type = "vgr"
                mm_instruction, mm_utils = get_mm_utils_vgr(episode)
            else:
                task_type = "vdl"
                mm_instruction, mm_utils = get_mm_utils_vdl(episode)

            step["task_type"] = task_type
            step["mm_instruction"] = mm_instruction
            step["mm_utils"] = mm_utils

        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        np.save(
            target_path / f"episode_{worker_id}_{episode_index:06d}.npy",
            episode,
            allow_pickle=True,
        )

    return episodes


def extract_placeholders(mm_instruction):
    # Extract placeholders from the instruction using regex
    pattern = r"<([^>]+)>"
    placeholders = re.findall(pattern, mm_instruction)
    return placeholders


def get_mm_utils_object(text_instruction, text2mmins, mmins2utils, utils_folder):

    if text_instruction in text2mmins:
        mm_instruction = text2mmins[text_instruction]
        mm_placeholders = extract_placeholders(mm_instruction)
        mm_utils = []
        for item in mm_placeholders:
            if item not in mmins2utils:
                raise ValueError(f"Placeholder '{item}' not found in map file")
            mm_utils.append(str(utils_folder / mmins2utils[item]))
    else:
        raise ValueError(
            f"'{text_instruction}' not found in resources, check consistency of resources"
        )

    # keep the original text instruction if this instruction is still not multimodal
    if not mm_utils:
        mm_instruction, loaded_utils = None, None
    else:
        # load images
        # mm_utils are image paths, load them as numpy arrays
        loaded_utils = []
        for img_path in mm_utils:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_np = np.array(image)
            loaded_utils.append(image_np)

    return mm_instruction, loaded_utils


def get_mm_utils_ocr(text_instruction, text2mmins, mmins2utils, utils_folder):
    NUM_STYLES = 768
    style_id = np.random.randint(0, NUM_STYLES)

    if text_instruction in text2mmins:
        mm_instruction = text2mmins[text_instruction]
        mm_utils = []
        if text_instruction in mmins2utils:
            mm_utils.append(utils_folder / (mmins2utils[text_instruction] + f"_style{style_id}.png"))
        else:
            raise ValueError(f"'{text_instruction}' not found in map file")
    else:
        raise ValueError(
            f"'{text_instruction}' not found in resources, check consistency of resources"
        )

    loaded_utils = []
    for img_path in mm_utils:
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        loaded_utils.append(image_np)
    return mm_instruction, loaded_utils


def get_mm_utils_vgr(episode):
    # for the visual goal reaching task and the video demonstration learning task,
    # we will use the specific instruction format
    mm_instruction = "<goal image>"
    goal_image = episode[-1]["observation"]["image"]
    mm_utils = [goal_image]
    return mm_instruction, mm_utils


def get_mm_utils_vdl(episode):
    # for the visual goal reaching task and the video demonstration learning task,
    # we will use the specific instruction format
    mm_instruction = "<imitation video>"
    traj_len = len(episode)

    # Uniformly sample 2 indices from (0, traj_len - 1) excluding endpoints
    if traj_len > 3:
        mid_indices = np.linspace(1.0, traj_len - 2, 4)
        mid_indices = np.round(mid_indices).astype(np.int32)
        indices = np.concatenate([[0], mid_indices[1:3], [traj_len - 1]], axis=0)
    else:
        # If traj_len <= 3, just use available indices (may duplicate)
        indices = np.arange(traj_len)
        pad_width = 4 - len(indices)
        if pad_width > 0:
            indices = np.pad(indices, (0, pad_width), constant_values=traj_len - 1)

    mm_utils = []
    for index in indices:
        mm_utils.append(episode[index]["observation"]["image"])
    return mm_instruction, mm_utils


@dataclass
class DataConfig:
    dataset_path: str = "/liujinxin/dataset/calvin/calvin_abc2d_rlds"
    debug_mode: bool = False

    object_text2mmins: str = (
        "/liujinxin/zhaowei/CogACT/vla/calvin_res/calvin_object.yaml"
    )
    object_mmins2utils: str = (
        "/liujinxin/zhaowei/CogACT/vla/calvin_res/calvin_object_map.yaml"
    )
    object_utils: str = "/liujinxin/zhaowei/CogACT/vla/calvin_res/utils/calvin_object"

    ocr_text2mmins: str = "/liujinxin/zhaowei/CogACT/vla/calvin_res/calvin_ocr.yaml"
    ocr_mmins2utils: str = (
        "/liujinxin/zhaowei/CogACT/vla/calvin_res/calvin_ocr_map.yaml"
    )
    ocr_utils: str = "/liujinxin/zhaowei/CogACT/vla/calvin_res/utils/calvin_ocr"

    # target directory for processed episodes
    target_path: str = (
        "/liujinxin/zhaowei/rlds_dataset_builder/npy_files/processed_oe_calvin"
    )


@draccus.wrap()
def main(cfg: DataConfig):
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # cuda with multiprocessing may crush
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    raw_dataset_builder = tfds.builder("example_dataset", data_dir=dataset_path)
    raw_dataset = raw_dataset_builder.as_dataset(split="train", shuffle_files=False)

    with open(cfg.object_text2mmins, "r") as f:
        object_text2mmins = yaml.safe_load(f)
    with open(cfg.object_mmins2utils, "r") as f:
        object_mmins2utils = yaml.safe_load(f)
    with open(cfg.ocr_text2mmins, "r") as f:
        ocr_text2mmins = yaml.safe_load(f)
    with open(cfg.ocr_mmins2utils, "r") as f:
        ocr_mmins2utils = yaml.safe_load(f)

    episode_partial = partial(
        process_episode,
        object_text2mmins=object_text2mmins,
        object_mmins2utils=object_mmins2utils,
        object_utils=cfg.object_utils,
        ocr_text2mmins=ocr_text2mmins,
        ocr_mmins2utils=ocr_mmins2utils,
        ocr_utils=cfg.ocr_utils,
        target_path=cfg.target_path,
    )

    if cfg.debug_mode:
        raw_dataset = raw_dataset.take(100)
        episodes = get_npy_episodes(raw_dataset)
        results = episode_partial(episodes)
        print(f"Debug mode: {len(results)} episodes have been saved")
    else:
        episodes = get_npy_episodes(raw_dataset)
        num_workers = min(multiprocessing.cpu_count(), 16)
        episodes = [episodes[i::num_workers] for i in range(num_workers)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(episode_partial, episodes)
        total = [item for sub_episodes in results for item in sub_episodes]
        print(f"{len(total)} episodes have been saved")


if __name__ == "__main__":
    main()
