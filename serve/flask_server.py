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
import re
from functools import partial
from pathlib import Path


class VLAServer:
    def __init__(self, args):
        model_path = os.path.expanduser(args.model_path)
        self.instruction_type = None
        self.unnorm_key = args.unnorm_key

        # Load the model
        self.vla = load_vla(
            model_id_or_path=model_path,
            load_for_training=args.load_for_training,
            action_model_type=args.action_model_type,
            future_action_window_size=args.future_action_window_size,
        )
        self.prompt_builder_fn = self.vla.vlm.vlm_backbone.prompt_builder_fn

    def compose_input(
        self,
        img_static,
        img_gripper,
        instruction,
        obj_image=None,
        goal_img_static=None,
        video_static=None,
        ins_image=None,
        debug=False,
    ):
        if debug:
            img_static.save("imgs_debug/eval_scene.png")
            img_gripper.save("imgs_debug/eval_gripper.png")

        img_static = img_static.resize((224, 224))
        img_gripper = img_gripper.resize((224, 224))

        def replace_all_placeholders(mm_instruction, replacement="<oe>"):
            pattern = r"<[^>]+>"
            return re.sub(pattern, replacement, mm_instruction)

        if self.instruction_type == "text":
            oe_lang = instruction.lower()
            # To be consistent with forward() in the training stage,
            # the random image is used to support FSDP training.
            random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            random_image = Image.fromarray(random_array)
            img_utils = [random_image]
        elif self.instruction_type == "mmins":
            oe_lang = replace_all_placeholders(instruction.lower())
            if len(obj_image) > 0:
                img_utils = obj_image
            else:
                random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                random_image = Image.fromarray(random_array)
                img_utils = [random_image]
        elif self.instruction_type == "goal_image":
            oe_lang = "reach the goal state <oe>"
            assert goal_img_static is not None, "goal_img_static should be provided for goal_image instruction type!"
            img_utils = goal_img_static
        elif self.instruction_type == "imitation_video":
            oe_lang = "learn the video demo: <oe>,<oe>,<oe>,<oe>"
            assert video_static is not None, "video_static should be provided for imitation_video instruction type!"
            img_utils = video_static
        elif self.instruction_type == "ins_image":
            oe_lang = "follow the command in <oe>"
            assert ins_image is not None, "ins_image should be provided for ins_image instruction type!"
            img_utils = ins_image
        else:
            oe_lang = instruction
            img_utils = []

        img_obs = {"scene": img_static, "left": img_gripper}
        return img_obs, img_utils, oe_lang

    def generate_action(self, img_obs, img_utils, oe_lang, robot_obs):
        with torch.inference_mode():
            self.vla.to("cuda:0").eval()
            actions, _ = self.vla.predict_action(
                img_obs,
                img_utils,
                oe_lang,
                unnorm_key=self.unnorm_key,
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
                robot_obs=robot_obs,
            )

        # np.ndarray and its shape is (16, 7)
        return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/zhaowei/workspace/CogACT/logs/calvin_abc2d_oe_baseline/checkpoints/step-016744-epoch-02-loss=0.0619.pt",
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
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default="calvin_abc2d_oe",
        help="Unnormalization key for dataset statistics",
    )
    parser.add_argument(
        "--debug",
        action="store_false",
        help="Enable debug mode (default: False)",
    )
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
            img_static = Image.fromarray(img_static)
            img_gripper = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8)
            img_gripper = img_gripper.reshape((84, 84, 3))
            img_gripper = Image.fromarray(img_gripper)

            # instructions and robot_obs for final input
            content = json.loads(request.files["json"].read())
            instruction = content["instruction"]
            robot_obs = content["robot_obs"]

            instruction_type_file = request.files.get("subtask_instruction_type")
            if instruction_type_file is not None:
                instruction_type = json.loads(instruction_type_file.read())["instruction_type"]
                assert instruction_type in [
                    "text",
                    "mmins",
                    "goal_image",
                    "imitation_video",
                    "ins_image",
                ], "Wrong instruction_type!"
                vla_robot.instruction_type = instruction_type

            obj_img_data, goal_img_static, video_static_list, ins_image = None, None, None, None
            if vla_robot.instruction_type == "mmins":
                object_goal_list = re.findall(r"<(.*?)>", instruction)
                obj_img_data = []
                if len(object_goal_list) > 0:
                    obj_img_size = json.loads(request.files["obj_image_size"].read())
                    for obj in object_goal_list:
                        img_obj = np.frombuffer(request.files[f"obj_{obj.replace(' ', '_')}"].read(), dtype=np.uint8)
                        img_obj = img_obj.reshape(
                            (
                                obj_img_size[f"obj_{obj.replace(' ', '_')}"][0],
                                obj_img_size[f"obj_{obj.replace(' ', '_')}"][1],
                                3,
                            )
                        )
                        if args.debug:
                            Image.fromarray(img_obj).save(f"imgs_debug/debug_obj_{obj.replace(' ', '_')}.png", "PNG")
                        obj_img_data.append(Image.fromarray(img_obj))
                elif len(object_goal_list) == 0:
                    print("No object images provided for mmins instruction.")
            elif vla_robot.instruction_type == "goal_image":
                goal_img_static = np.frombuffer(request.files["goal_img_static"].read(), dtype=np.uint8)
                goal_img_static = goal_img_static.reshape((200, 200, 3))
                goal_img_static = Image.fromarray(goal_img_static)
                if args.debug:
                    goal_img_static.save("imgs_debug/debug_goal_img_static.png", "PNG")
                goal_img_static = [goal_img_static]
            elif vla_robot.instruction_type == "imitation_video":
                video_static = np.frombuffer(request.files["imitation_video_static"].read(), dtype=np.uint8).reshape(
                    (-1, 200, 200, 3)
                )
                video_static_list = [
                    Image.fromarray(video_static[i].reshape((200, 200, 3))) for i in range(video_static.shape[0])
                ]
                if args.debug:
                    for i in range(len(video_static_list)):
                        video_static_list[i].save(f"imgs_debug/debug_video_static_{i}.png", "PNG")
            elif vla_robot.instruction_type == "ins_image":
                ins_image = np.frombuffer(request.files["ins_image"].read(), dtype=np.uint8).reshape((334, 334, 3))
                ins_image = Image.fromarray(ins_image)
                if args.debug:
                    ins_image.save("imgs_debug/debug_ins_image.png", "PNG")
                ins_image = [ins_image]
            else:
                print(f"No additional image data used, instruction: {instruction}")

            # compose the input
            img_obs, img_utils, oe_lang = vla_robot.compose_input(
                img_static,
                img_gripper,
                instruction,
                obj_image=obj_img_data,
                goal_img_static=goal_img_static,
                video_static=video_static_list,
                ins_image=ins_image,
                debug=args.debug,
            )

            robot_obs_norm = os.path.dirname(os.path.dirname(args.model_path)) + "/dataset_statistics.json"
            with open(robot_obs_norm, "r") as f:
                norm_stats = json.load(f)
            robot_obs_low = np.array(norm_stats[args.unnorm_key]["proprio"]["q01"])
            robot_obs_high = np.array(norm_stats[args.unnorm_key]["proprio"]["q99"])
            robot_obs = np.array(robot_obs)
            robot_obs = np.clip(2 * (robot_obs - robot_obs_low) / (robot_obs_high - robot_obs_low + 1e-8) - 1, -1, 1)
            action = vla_robot.generate_action(img_obs, img_utils, oe_lang, robot_obs)
            return jsonify(action.tolist())

    # Run the server
    flask_app.run(host="0.0.0.0", port=args.port)
