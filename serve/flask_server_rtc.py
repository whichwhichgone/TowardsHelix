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


VISION_IMAGE_SIZE = 224


class VLAServer:
    def __init__(self, args):
        model_path = os.path.expanduser(args.model_path)

        # Load the model
        self.vla = load_vla(
            model_id_or_path=model_path,
            load_for_training=args.load_for_training,
            action_model_type=args.action_model_type,
            future_action_window_size=args.future_action_window_size,
        )

    def compose_input(
        self, img_scene, img_hand_left, img_hand_right, instruction, debug=False
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
            img_scene.save(Path("./imgs_debug") / "eval_scene.png")
            img_hand_left.save(Path("./imgs_debug") / "eval_img_hand_left.png")
            img_hand_right.save(Path("./imgs_debug") / "eval_img_hand_right.png")
        return image_all
    
    def generate_action_rtc(self, instruction, image_all, action_part_prev, action_exec_s, delay):
        """
        Generate action with real-time chunking flow policies.

        Args:
            instruction (str): Task instruction string
            image_all (dict): Dictionary containing scene and hand images
                - scene (np.ndarray): Scene image, shape (224, 224, 3)
                - left (np.ndarray): Left hand camera image, shape (224, 224, 3)
                - right (np.ndarray): Right hand camera image, shape (224, 224, 3)
            action_part_prev (list): Previous action sequence, shape (16, 7)
            action_exec_s (str): Executed actions between previous inference and current inference
            delay (str): Number of actions elapsed during the inference procedure 

        Returns:
            np.ndarray: Generated action sequence with shape (16, 7)
        """

        with torch.inference_mode():
            self.vla.to('cuda:0').eval()
            if action_part_prev is not None:
                action_part_prev = np.asarray(action_part_prev)
                action_exec_s = int(action_exec_s)
                delay = int(delay)

            actions, _ = self.vla.predict_action_with_cfm(
                image_all,
                instruction,
                unnorm_key='ur5e_benchmark_v4_with_depth',
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
                previous_actions=action_part_prev,
                action_exec_s=action_exec_s,
                delay=delay,
            )
        # np.ndarray and its shape is (16, 7)
        return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/liujinxin/code/CogACT_speedup/logs/ur5e_benchmark_v4_with_depth_flowmatching_06191508_zw--image_aug/checkpoints/step-010000-epoch-17-loss=0.0876.pt",
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
    parser.add_argument("--port", type=int, default=9002, help="Port number for flask server")
    args = parser.parse_args()


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
            action_part_prev = content["action_part_prev"]
            action_exec_s = content["s"]
            delay = content["delay"]

            # compose the input
            image_all = vla_robot.compose_input(img_scene, img_hand_left, img_hand_right, instruction)
            action = vla_robot.generate_action_rtc(instruction, image_all, action_part_prev, action_exec_s, delay)
            return jsonify(action.tolist())

    # Run the server
    flask_app.run(host="0.0.0.0", port=args.port)
