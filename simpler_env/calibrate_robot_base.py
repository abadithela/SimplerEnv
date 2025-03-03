# Script to calibrate robot base

import argparse
import os
import random
import string
from transforms3d.euler import euler2axangle, euler2quat, quat2euler

import mediapy as media
import numpy as np
import tensorflow as tf
import json
import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from pdb import set_trace as st
import shutil
import math
import itertools
import yaml
from PIL import Image
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env import ENVIRONMENTS
import matplotlib.animation as animation

def get_args(folder=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="octo-base", choices=["rt1", "octo-base", "octo-small"])
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="octo-base",
    )

    parser.add_argument(
        "--task",
        default="irom_widowx_carrot_on_plate",
        choices=ENVIRONMENTS,
    )
    if folder:
        parser.add_argument("--logging-root", type=str, default=folder)
    else:
        parser.add_argument("--logging-root", type=str, default="./results_simple_random_eval")
    parser.add_argument("--tf-memory-limit", type=int, default=3072)
    args = parser.parse_args()
    return args

def process_qpos(hw_qpos):
    qpos = [hw_qpos[0], hw_qpos[1], hw_qpos[2], hw_qpos[3], hw_qpos[4], hw_qpos[5], hw_qpos[7], -1*hw_qpos[8]]
    return qpos

def main_qpos(init_qpos, recorded_traj,recorded_traj_qpos,recorded_traj_dir):
    exp_length = len(recorded_traj)
    def get_tcp_pose_at_robot_base():
        tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        xyz = tcp_pose_at_robot_base.p.tolist()
        rot_quat = tcp_pose_at_robot_base.q.tolist()
        rot_euler = quat2euler(rot_quat)
        ee_pose = [*xyz, *rot_euler]
        return ee_pose

    args = get_args(folder=recorded_traj_dir)
    os.makedirs(args.logging_root, exist_ok=True)

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

    # build environment
    env = simpler_env.make(args.task)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)
    print("qpos", env.agent.robot.get_qpos())
    init_qpos = process_qpos(init_qpos)
    
    env_reset_options = {}

    init_rot_quat = Pose(q=[0, 0, 0, 1]).q
    if env.robot_uid == "irom_widowx":
        # Check how I can pass in init_qpos
        env_reset_options = {
            "obj_init_options": {},
            "robot_init_options": {
                "init_xy": [0.185,0.22],
                'init_height': env.scene_table_height + 0.045,
                "init_rot_quat": init_rot_quat,
                "qpos": np.array(init_qpos)
            },
        }
    env_reset_options["obj_init_options"]["init_xys"] = np.array([env.carrot_center, env.plate]) - np.array([[0.10,0.01], [0,0]])
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image] # should just be sleep.
    images = []
    
    robot_qpos = [(env.agent.robot.get_qpos()).tolist()]
    robot_qpos_real_minus_sim = [(np.array(init_qpos) - np.array(robot_qpos[0])).tolist()]
    print("Reset info:", info)
    print("robot pose", env.agent.robot.pose)
    print("qpos", env.agent.robot.get_qpos())
    
    robot_ee_pos =[get_tcp_pose_at_robot_base()]
    #### Go over trajectory:
    timestep = 1
    while timestep < exp_length:
        hw_qpos = process_qpos(recorded_traj_qpos[str(timestep)]) # Hardware qpos is 0 indexed.

        print("hardware qpos", hw_qpos)
        # env.agent.robot.set_qpos(hw_qpos)
        env.agent.reset(hw_qpos)
        obs = env.get_obs()
        image = (obs["image"]["3rd_view_camera"]["Color"][..., :-1] * 255).astype(np.uint8)
        images.append(image)
        # obs = env.get_obs() # Does not return rgb for some reason
        im = Image.fromarray(image)
        im.save(os.path.join(recorded_traj_dir, f"direct_qpos_Sim_IMG_{timestep}.jpeg"))
        
        timestep += 1
        qpos = env.agent.robot.get_qpos()
        log_qpos(qpos.tolist())
        robot_qpos_real_minus_sim.append((np.array(hw_qpos) - np.array(qpos)).tolist())
        robot_qpos.append(qpos.tolist())
        eepos = get_tcp_pose_at_robot_base()

        robot_ee_pos.append(eepos)
        
    with open(os.path.join(recorded_traj_dir, "sim_ee_traj_direct_qpos.json"), "w") as f:
        json.dump(robot_ee_pos, f)

    with open(os.path.join(recorded_traj_dir, "direct_qpos_cmd_sim_traj.json"), "w") as f:
        json.dump(robot_qpos, f)
    with open(os.path.join(recorded_traj_dir, "direct_hw_qpos_traj_error.json"), "w") as f:
        json.dump(robot_qpos_real_minus_sim, f)
    media.write_video(f"{args.logging_root}/recorded_qpos_traj.mp4", images, fps=1)