from PIL import Image
from vla import load_vla
import torch

from flask import Flask, jsonify, request, Response
import argparse
import os
import socket
import io
import json
import numpy as np
from functools import partial
from pathlib import Path

from prismatic.models.vlms.prismatic import PrismaticVLM
from action_model.action_model import ActionModel
from prismatic.overwatch import initialize_overwatch
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.conf import ModelConfig
from transformers import LlamaTokenizerFast

import threading
from collections import deque
import torchvision.transforms as transforms
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import time


VISION_IMAGE_SIZE = 224

class AsynchronousBuffer:
    def __init__(self, maxsize=10):
        assert maxsize > 0, "maxsize must be greater than 0"
        self.maxsize = maxsize
        self._queue = deque()
        self._lock = threading.Lock()
    
    def put(self, item):
        with self._lock:
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                self._queue.popleft()
            self._queue.append(item)
    
    def get(self):
        with self._lock:
            if len(self._queue) == 0:
                raise QueueEmpty("Queue is empty")
            return self._queue.popleft()
    
    def peek(self):
        # return the latest item in the queue without removing it
        with self._lock:
            if len(self._queue) == 0:
                raise QueueEmpty("Queue is empty")
            return self._queue[-1]
    
    def size(self):
        with self._lock:
            return len(self._queue)

    def empty(self):
        with self._lock:
            return len(self._queue) == 0


class VLAServer:
    def __init__(self, args):
        model_id_or_path = args.model_id_or_path
        hf_token = args.hf_token
        load_for_training = args.load_for_training
        self.hidden_buffer = AsynchronousBuffer(maxsize=10)
        self.image_task_buffer = AsynchronousBuffer(maxsize=10)
       
        self.vlm_device = torch.device(f"cuda:{args.vlm_cuda}")
        self.action_device = torch.device(f"cuda:{args.action_cuda}")

        self.vlm = self.get_system2(model_id_or_path, hf_token, load_for_training)
        self.action_model, self.norm_stats = self.get_system1(model_id_or_path, args)
        self.is_begin = True
        self.vlm_loop_running = False

        self.thread_vlm_once = threading.Thread(target=self.vlm_worker_once)
        self.thread_vlm_loop = threading.Thread(target=self.vlm_worker_loop)

    def forward_system2(self, instruction, image):
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

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
        image_scene, image_hand_left, image_hand_right = image["scene"], image["left"], image["right"]
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
        
        pixel_values_right = image_transform(image_hand_right)
        if isinstance(pixel_values_right, torch.Tensor):
            pixel_values_right = pixel_values_right[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values_right, dict):
            pixel_values_right = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values_right.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values_right)}")
        pixel_values = {"scene" : pixel_values_scene, "left" : pixel_values_left, "right" : pixel_values_right}

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
            )
            # fmt: on

        # Extract cognition feature
        cognition_features = output.hidden_states[0][-1][:,-1,:]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (1,4096), "Batch size must be 1 for action prediction"
        return cognition_features

    def get_system2(self, model_id_or_path, hf_token, load_for_training):
        # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
        #   checkpoint `.pt` file, rather than the top-level run directory!
        if os.path.isfile(model_id_or_path):
            overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

            # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
            assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
            run_dir = checkpoint_pt.parents[1]

            # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
            config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
            assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
            assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"
        
        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            vla_cfg = json.load(f)["vla"]
            model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()
        
        # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
        #   =>> Print Minimal Config
        overwatch.info(
            f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
            f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
            f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
            f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
            f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
        )

        # Load Vision Backbone
        overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            model_cfg.vision_backbone_id,
            model_cfg.image_resize_strategy,
        )

         # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
        overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            model_cfg.llm_backbone_id,
            llm_max_length=model_cfg.llm_max_length,
            hf_token=hf_token,
            inference_mode=not load_for_training,
        )

        # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
        overwatch.info(f"Loading VLM of VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_cfg.model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=True,
            arch_specifier=model_cfg.arch_specifier,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(model_id_or_path, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        assert load_for_training == False, "load_for_training must be False when deploying"
        if not load_for_training:
            vlm.requires_grad_(False)
            vlm.eval()
        return vlm.to(self.vlm_device)
    
    def get_system1(self, model_id_or_path, args):
        model_state_dict = torch.load(model_id_or_path, map_location="cpu")["model"]
        dataset_statistics_json = Path(model_id_or_path).parent.parent / "dataset_statistics.json"

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)

        action_model = ActionModel(model_type = args.action_model_type,
                                            token_size = args.token_size, 
                                            in_channels = args.in_channels, 
                                            future_action_window_size = args.future_action_window_size, 
                                            past_action_window_size = args.past_action_window_size)

        action_model.load_state_dict(model_state_dict["action_model"])
        action_model.requires_grad_(False)
        action_model.eval()
        return action_model.to(self.action_device), norm_stats
    
    def forward_system1(self, cognition_features, image_all, unnorm_key='ur5e_benchmark_v3_with_depth', cfg_scale=1.5, use_ddim=True, num_ddim_steps=10):
        image_scene, image_hand_left, image_hand_right = image_all["scene"], image_all["left"], image_all["right"]
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype).to(self.action_device)  # [B, 1, D]

        # add vision for action model
        image_scene = transforms.ToTensor()(np.array(image_scene)).unsqueeze(0).to(model_dtype).to(cognition_features.device)
        image_hand_left = transforms.ToTensor()(np.array(image_hand_left)).unsqueeze(0).to(model_dtype).to(cognition_features.device)
        image_hand_right = transforms.ToTensor()(np.array(image_hand_right)).unsqueeze(0).to(model_dtype).to(cognition_features.device)
        p_scene = self.action_model.scene_encoder(image_scene).unsqueeze(1)
        p_left = self.action_model.left_encoder(image_hand_left).unsqueeze(1)
        p_right = self.action_model.right_encoder(image_hand_right).unsqueeze(1)
        p = torch.cat([p_scene, p_left, p_right], dim=1)
        cognition_features = torch.cat([p, cognition_features], dim=1)

        # Sample random noise
        using_cfg = cfg_scale > 1.0
        noise = torch.randn(B, self.action_model.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(cognition_features.shape[0], cognition_features.shape[1], -1) #[B, 1, D]
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
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.3, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None and len(norm_stats) != 1:
            raise ValueError(
                f"Your model was trained on more than one dataset. "
                f"Please pass a `unnorm_key` from the following options to choose the statistics used for "
                f"de-normalizing actions: {norm_stats.keys()}"
            )

        # If None, grab the (singular) dataset in `norm_stats` to use as `unnorm_key`
        unnorm_key = unnorm_key if unnorm_key is not None else next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {norm_stats.keys()}"
            )

        return unnorm_key
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def vlm_worker_once(self):
        item = self.image_task_buffer.peek()
        task, image_all = item["task"], item["image_all"]
        cognition_features = self.forward_system2(task, image_all)
        self.hidden_buffer.put(cognition_features)
    
    def vlm_worker_loop(self):
        while True:
            item = self.image_task_buffer.peek()
            task, image_all = item["task"], item["image_all"]
            start_time = time.time()
            time.sleep(0.1)
            cognition_features = self.forward_system2(task, image_all)
            end_time = time.time()
            overwatch.info(f"VLM forward_system2 prediction time: {end_time - start_time:.4f} seconds")
            self.hidden_buffer.put(cognition_features)

    def compose_input(
        self, img_scene, img_hand_left, img_hand_right, instruction, debug=True
    ):
        img_scene = Image.fromarray(img_scene)
        img_hand_left = Image.fromarray(img_hand_left)
        img_hand_right = Image.fromarray(img_hand_right)
        image_all = {
            "scene" : img_scene,
            "left" : img_hand_left,
            "right" : img_hand_right,
        }

        if debug:
            # images for final input
            img_scene.save(Path("/liujinxin/code/CogACT_speedup/imgs_debug/eval_scene.png"))
            img_hand_left.save(Path("/liujinxin/code/CogACT_speedup/imgs_debug/eval_img_hand_left.png"))
            img_hand_right.save(Path("/liujinxin/code/CogACT_speedup/imgs_debug/eval_img_hand_right.png"))
        return image_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default= "/liujinxin/code/CogACT_speedup/logs/ur5e_benchmark_v3_with_depth_asynchronous_0608_1500--image_aug/checkpoints/step-004000-epoch-04-loss=0.0258.pt"
        )
    parser.add_argument(
        "--load-for-training",
        action="store_true",
        help="Load the model for training (default: False)",
    )
    parser.add_argument(
        "--action-model-type",
        type=str,
        default="DiT-B",
        help="Action model type (default: DiT-B)",
    )
    parser.add_argument(
        "--future-action-window-size",
        type=int,
        default=15,
        help="Future action window size (default: 15)",
    )
    parser.add_argument(
        "--past-action-window-size",
        type=int,
        default=0,
        help="Past action window size (default: 0)",
    )
    parser.add_argument(
        "--token-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=7,
        help="Action dim"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=".hf_token",
        help="HF token"
    )
    parser.add_argument(
        "--vlm-cuda",
        type=int,
        default=0,
        help="VLM cuda"
    )
    parser.add_argument(
        "--action-cuda",
        type=int,
        default=1,
        help="action cuda"
    )
    parser.add_argument("--port", type=int, default=9002, help="Port number for flask server")
    args = parser.parse_args()
    overwatch = initialize_overwatch(__name__)

    # Start the server (Flask)
    flask_app = Flask(__name__)
    vla_robot = VLAServer(args)

    # Define the route for remote requests
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            img_scene = np.frombuffer(request.files["img_scene"].read(), dtype=np.uint8)
            img_scene = img_scene.reshape((VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, 3))
            img_hand_left = np.frombuffer(request.files["img_hand_left"].read(), dtype=np.uint8)
            img_hand_left = img_hand_left.reshape((VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, 3))
            img_hand_right = np.frombuffer(request.files["img_hand_right"].read(), dtype=np.uint8)
            img_hand_right = img_hand_right.reshape((VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, 3))

            # instructions and robot_obs for final input
            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            image_all = vla_robot.compose_input(img_scene, img_hand_left, img_hand_right, instruction)

            if vla_robot.is_begin:
                vla_robot.is_begin = False
                vla_robot.image_task_buffer.put({"task": instruction, "image_all": image_all})
                vla_robot.thread_vlm_once.start()
                vla_robot.thread_vlm_once.join()
                cognition_features = vla_robot.hidden_buffer.peek()
                actions, _ = vla_robot.forward_system1(cognition_features, image_all)
            else:
                vla_robot.image_task_buffer.put({"task": instruction, "image_all": image_all})
                if not vla_robot.vlm_loop_running:
                    vla_robot.thread_vlm_loop.start()
                    vla_robot.vlm_loop_running = True
                cognition_features = vla_robot.hidden_buffer.peek()
                start_time = time.time()
                actions, _ = vla_robot.forward_system1(cognition_features, image_all)
                end_time = time.time()
                overwatch.info(f"Action model prediction time: {end_time - start_time:.4f} seconds")

            return jsonify(actions.tolist())

    # Run the server
    flask_app.run(host="0.0.0.0", port=args.port)
