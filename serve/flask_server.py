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
            img_scene.save(Path("./imgs_debug") / "eval_scene.png")
            img_hand_left.save(Path("./imgs_debug") / "eval_img_hand_left.png")
            img_hand_right.save(Path("./imgs_debug") / "eval_img_hand_right.png")
        return image_all

    def generate_action(self, instruction, image_all):
        with torch.inference_mode():
            self.vla.to('cuda:0').eval()
            actions, _ = self.vla.predict_action(
                image_all,
                instruction,
                unnorm_key='example_dataset',
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
            )
        
        # np.ndarray and its shape is (16, 7)
        return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/liujinxin/zhaowei/CogACT/logs/calvin_cogact/checkpoints/step-033600-epoch-01-loss=0.0411.pt",
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
        default=9,
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
            img_static = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8)
            img_static = img_static.reshape((200, 200, 3))
            img_gripper = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8)
            img_gripper = img_gripper.reshape((84, 84, 3))

            # instructions and robot_obs for final input
            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]

            # compose the input, use only a single view for calvin
            image_all = vla_robot.compose_input(img_static, img_static, img_static, instruction)
            action = vla_robot.generate_action(instruction, image_all).reshape(-1)
            return jsonify(action.tolist())

    # Run the server
    flask_app.run(host="0.0.0.0", port=args.port)
