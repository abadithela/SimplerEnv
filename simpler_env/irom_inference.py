"""
Apurva Badithela (1/30/25)
Simple script for real-to-sim eval using the prepackaged visual matching setup in ManiSkill2.
Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
        --ckpt-path ./checkpoints/rt_1_tf_trained_for_000400120  --task google_robot_pick_coke_can  --logging-root ./results_simple_eval/  --n-trajs 10
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy octo-small \
        --ckpt-path None --task widowx_spoon_on_towel  --logging-root ./results_simple_eval/  --n-trajs 10
"""

import argparse
import os
import random
import string
import mediapy as media
import numpy as np
import tensorflow as tf
import json
import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from pdb import set_trace as st
from simlogger import SimLogger
import shutil
import math
import itertools
import yaml
# ============================================================================================ # 
EXP_DIR = "/home/apurva/software/RapidEvalPPI/experiments/results_simple_random_eval"

# ============================================================================================ # 
# Utility Functions 
def save_experiment_script(logging_dir):
    """
    Copies the script to the experiment folder for record-keeping.
    
    :param script_path: Path to the script you want to save (e.g., 'train.py')
    :param experiment_folder: Folder where the experiment results are stored
    """
    script_path = os.path.abspath(__file__)  # Get current script path
    os.makedirs(logging_dir, exist_ok=True)  # Ensure folder exists
    shutil.copy(script_path, os.path.join(logging_dir, os.path.basename(script_path)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="rt1", choices=["rt1", "octo-base", "octo-small", "openvla"])
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--task",
        default="irom_widowx_carrot_on_plate",
        choices=ENVIRONMENTS,
    )
    parser.add_argument("--logging-root", type=str, default=f"{EXP_DIR}/results_simple_random_eval")
    parser.add_argument("--tf-memory-limit", type=int, default=3072)
    parser.add_argument("--n-trajs", type=int, default=10)
    parser.add_argument("--dirname-end", type=str, default=None)
    args = parser.parse_args()

    if args.policy in ["octo-base", "octo-small", "openvla"]:
        if args.ckpt_path in [None, "None"] or "rt_1_x" in args.ckpt_path:
            args.ckpt_path = args.policy
        if args.ckpt_path[-1] == "/":
            args.ckpt_path = args.ckpt_path[:-1]

    save_experiment_script(logging_dir=args.logging_root) # Save a copy at the root of the logging directory

    if args.dirname_end is not None:
        logging_dir = os.path.join(args.logging_root, args.task, args.policy, os.path.basename(args.ckpt_path), args.dirname_end)
    else:
        logging_dir = os.path.join(args.logging_root, args.task, args.policy, os.path.basename(args.ckpt_path))

    # Trial ID for accounting for variations.
    sim_id = ''.join(random.choices(string.ascii_letters + string.digits, k=4)) 
    while os.path.exists(os.path.join(logging_dir, sim_id)):
        sim_id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    logging_dir = os.path.join(logging_dir, sim_id)
    os.makedirs(logging_dir)
    return args, logging_dir

def get_light_kwargs(is_dir_light = True, is_ambient_light = True, dir_light_position=[-0.5, 0, -math.sqrt(3)/2], dir_light_color=[0.3, 0.3, 0.3], ambient_light_color=[0.3, 0.3, 0.3], shadow=True, dir_light_scale=10):
    # Define your lighting parameters
    light_kwargs = {
        "is_dir_light": is_dir_light,
        "is_ambient_light": is_ambient_light,
        "dir_light_position": dir_light_position,
        "dir_light_color": dir_light_color,
        "ambient_light_color": ambient_light_color,
        "shadow": shadow,
        "shadow_map_size": 2048,
        "dir_light_scale": dir_light_scale
    }
    
    return light_kwargs

# Gripper initial poses:
def get_init_qpos():
    gripper_init_qpos = "/home/apurva/software/RapidEvalPPI/hardware/6cm_up/robot_init_qpos.json"
    with open(gripper_init_qpos, "r") as f:
        robot_init_qpos = json.load(f)
    return robot_init_qpos

# Process the hardware init_qpos
def process_qpos(hw_qpos):
    qpos = [hw_qpos[0], hw_qpos[1], hw_qpos[2], hw_qpos[3], hw_qpos[4], hw_qpos[5], hw_qpos[7], -hw_qpos[8]]
    return qpos

# ============================================================================================ # 
# Build Policy
def build_policy(args, sim_log):
    init_rng = 0 # Seed for diffusion based policies
    if "google_robot" in args.task:
        policy_setup = "google_robot"
    elif "widowx" in args.task:
        policy_setup = "widowx_bridge"
    else:
        raise NotImplementedError()

    if args.policy == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference

        model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)

    elif "octo" in args.policy:
        from simpler_env.policies.octo.octo_model import OctoInference
        model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=init_rng)
        sim_log.add_entry("policy_init_rng", init_rng)
    elif "openvla" in args.policy:
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        model = OpenVLAInference(policy_setup=policy_setup)
    else:
        raise NotImplementedError()
    return model

# ============================================================================================ # 
# Run Inference
def run_inference(env, model, logging_dir, **light_kwargs):
    success_arr = []
    num_each_config = 10
    init_configs = ["center" for k in range(num_each_config)]
    init_configs.extend(["left" for k in range(num_each_config)])
    init_configs.extend(["right" for k in range(num_each_config)])
    robot_init_qpos = get_init_qpos()
    ep_id = 1
    env.add_lighting_params(**light_kwargs)

    for init_config in init_configs:
        if init_config == "left":
            xy_config = np.array([env.carrot_left, env.plate])
        elif init_config == "right":
            xy_config = np.array([env.carrot_right, env.plate])
        else:
            xy_config = np.array([env.carrot_center, env.plate])
        qpos = random.choice(robot_init_qpos[init_config])
        qpos = np.array(process_qpos(qpos))
        
        env_reset_options = {"obj_init_options":{"init_xys": xy_config}, "robot_init_options": {"qpos": qpos}, "reconfigure": True} # Reconfigure sets lighting conditions
        obs, reset_info = env.reset(options=env_reset_options)
        instruction = env.get_language_instruction()
        # for long-horizon environments, we check if the current subtask is the final subtask
        is_final_subtask = env.is_final_subtask() 

        model.reset(instruction)
        print(instruction)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = model.step(image, instruction)
            
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
            print(timestep, info)
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                # update instruction for long horizon tasks
                instruction = new_instruction
                print(instruction)
            is_final_subtask = env.is_final_subtask() 
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        episode_stats = info.get("episode_stats", {})
        success_arr.append(success)
        print(f"Episode {ep_id} success: {success}")
        media.write_video(f"{logging_dir}/normal_episode_{ep_id}_success_{success}.mp4", images, fps=1)
        ep_id += 1
        

    print(
        "**Overall Success**",
        np.mean(success_arr),
        f"({np.sum(success_arr)}/{len(success_arr)})",
    )
    return np.mean(success_arr)

# ============================================================================================ # 
# Build environment and start logger
def experiment(args, logging_dir, is_dir_light, is_ambient_light, dir_light_position, dir_light_color, ambient_light_color, shadow, dir_light_scale,enable_raytracing):
    sim_log = SimLogger(folder=logging_dir)
    sim_log.add_args(args)
    kwargs = get_light_kwargs(is_dir_light = is_dir_light, is_ambient_light = is_ambient_light, dir_light_position=dir_light_position, dir_light_color=dir_light_color, ambient_light_color=ambient_light_color, shadow=shadow, dir_light_scale=dir_light_scale)
    sim_log.log_lighting(is_dir_light, is_ambient_light, dir_light_position, dir_light_color, ambient_light_color, shadow, dir_light_scale)
    
    additional_env_build_kwargs = dict()
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        # ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    
    sim_log.add_entry("raytracing", enable_raytracing)
    kwargs["rgb_overlay_cameras"] = ["3rd_view_camera"]

    env = simpler_env.make(args.task, **additional_env_build_kwargs)

    # Reset lighting conditions

    # env = simpler_env.make(args.task, **additional_env_build_kwargs, **kwargs)
    sim_log.log_env_vars(env) # Log environment variables
    sim_log.log_robot_setup(env)

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    model = build_policy(args, sim_log) # Build model
    success_rate = run_inference(env, model, logging_dir, **kwargs)
    sim_log.add_entry("success_rate", "%.2f" % round(success_rate, 2))
    sim_log.save_to_yaml()

# ============================================================================================ # 
# Main experiment:
def run_experiments():
    DIR_LIGHT_COLOR_VAL = [0.3, 0.5]
    AMB_LIGHT_COLOR_VAL = [0.5, 1.0]

    is_dir_light_vals = [True]
    is_amb_light_vals = [True]
    enable_rt_vals = [True, False]
    enable_shadow_vals = [True, False]
    dir_light_pose_vals = [[-0.5, 0, -math.sqrt(3)/2]]
    dir_light_color_vals =  [[v,v,v] for v in DIR_LIGHT_COLOR_VAL]
    amb_light_color_vals = [[v,v,v] for v in AMB_LIGHT_COLOR_VAL]
    dir_light_scale_vals = [5.0]

    for is_dir_light, is_ambient_light, dir_light_position, dir_light_color, ambient_light_color, shadow, dir_light_scale,enable_raytracing in itertools.product(
        is_dir_light_vals, is_amb_light_vals, dir_light_pose_vals, dir_light_color_vals, amb_light_color_vals, enable_shadow_vals, dir_light_scale_vals,enable_rt_vals):
        args, logging_dir = get_args() 
        experiment(args, logging_dir,is_dir_light, is_ambient_light, dir_light_position, dir_light_color, ambient_light_color, shadow, dir_light_scale,enable_raytracing)

def find_max_success_rates(base_dir):
    max_rate = -float('inf')
    max_folder = None
    
    for root, _, files in os.walk(base_dir):
        if 'sim_log.yaml' in files:
            yaml_path = os.path.join(root, 'sim_log.yaml')
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'success_rate' in data:
                        success_rate = float(data['success_rate'])
                        if success_rate > max_rate:
                            max_rate = success_rate
                            max_folder = root
                    else:
                        print(f"Warning: 'success_rate' not found in {yaml_path}")
            except Exception as e:
                print(f"Error reading {yaml_path}: {e}")
    
    return max_folder, max_rate


def param_sweep():
    policy = "openvla"

    if policy =="octo-base":
        base_directory = f"{EXP_DIR}/irom_widowx_carrot_on_plate/octo-base/octo-base"
    elif policy =="octo-small":
        base_directory = f"{EXP_DIR}/irom_widowx_carrot_on_plate/octo-small/octo-small"
    elif policy =="openvla":
        base_directory = f"{EXP_DIR}/irom_widowx_carrot_on_plate/openvla/openvla"
    folder, max_success_rate = find_max_success_rates(base_directory)
    print(f"Highest Success Rate: {max_success_rate}, Folder: {folder}")
        
if __name__=="__main__":
    # run_experiments()
    param_sweep()