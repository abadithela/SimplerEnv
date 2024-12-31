"""
# Apurva Badithela
# 12/13/24
# A script to command the WidowX robot to go to particular waypoints to check if the robot is setup correctly

Simple script for real-to-sim eval using the prepackaged visual matching setup in ManiSkill2.
Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/command_robot_traj.py --policy octo  --logging-root ./results_compare_traj/  
"""

import argparse
import os
import json
import wandb
import copy
from transforms3d.euler import euler2axangle, euler2quat 
import pickle as pkl
import mediapy as media
import numpy as np
import tensorflow as tf
from sapien.core import Pose
import gymnasium as gym
import sys
import simpler_env
from PIL import Image
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env import ENVIRONMENTS

from pdb import set_trace as st

# =======================================================================
# Parser and logging dir setup
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

# =======================================================
# Main function to control the robot:
def main_qpos(init_qpos, recorded_traj,recorded_traj_qpos,recorded_traj_dir):
    exp_length = len(recorded_traj)

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
                "init_xy": [0.245,0.22],
                'init_height': env.scene_table_height + 0.04,
                "init_rot_quat": init_rot_quat,
                "qpos": np.array(init_qpos)
            },
        }
    env_reset_options["obj_init_options"]["episode_id"] = 0
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image] # should just be sleep.
    images = []
    
    robot_qpos = [(env.agent.robot.get_qpos()).tolist()]
    robot_qpos_real_minus_sim = [(np.array(init_qpos) - np.array(robot_qpos[0])).tolist()]
    print("Reset info:", info)
    print("robot pose", env.agent.robot.pose)
    print("qpos", env.agent.robot.get_qpos())
    
    #### Go over trajectory:
    timestep = 1
    while timestep < exp_length:
        hw_qpos = process_qpos(recorded_traj_qpos[str(timestep)]) # Hardware qpos is 0 indexed.
        print("hardware qpos", hw_qpos)
        env_reset_options = {"obj_init_options": {},
            "robot_init_options": {
            "init_xy": [0.245,0.22],
            'init_height': env.scene_table_height + 0.04,
            "init_rot_quat": init_rot_quat,
            "qpos": np.array(hw_qpos)
            },
        }
        obs, info = env.reset(options=env_reset_options)
        # obs = env.get_obs() # Does not return rgb for some reason
        
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        im = Image.fromarray(image)
        im.save(os.path.join(recorded_traj_dir, f"direct_qpos_Sim_IMG_{timestep}.jpeg"))
        
        timestep += 1
        qpos = env.agent.robot.get_qpos()
        robot_qpos_real_minus_sim.append((np.array(hw_qpos) - np.array(qpos)).tolist())
        robot_qpos.append(qpos.tolist())

    with open(os.path.join(recorded_traj_dir, "direct_qpos_cmd_sim_traj.json"), "w") as f:
        json.dump(robot_qpos, f)
    with open(os.path.join(recorded_traj_dir, "direct_hw_qpos_traj_error.json"), "w") as f:
        json.dump(robot_qpos_real_minus_sim, f)
    media.write_video(f"{args.logging_root}/recorded_qpos_traj.mp4", images, fps=1)

# =======================================================
# Main function to control the robot:
def main(init_qpos, recorded_traj,recorded_traj_qpos,recorded_traj_dir):
    exp_length = len(recorded_traj)

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
                "init_xy": [0.245,0.22],
                'init_height': env.scene_table_height + 0.04,
                "init_rot_quat": init_rot_quat,
                "qpos": np.array(init_qpos)
            },
        }
    env_reset_options["obj_init_options"]["episode_id"] = 0
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image] # should just be sleep.
    im = Image.fromarray(image)
    im.save(os.path.join(recorded_traj_dir, "Sim_IMG_init.jpeg"))
    
    robot_qpos = [(env.agent.robot.get_qpos()).tolist()]
    robot_qpos_real_minus_sim = [(np.array(init_qpos) - np.array(robot_qpos[0])).tolist()]
    print("Reset info:", info)
    print("robot pose", env.agent.robot.pose)
    print("qpos", env.agent.robot.get_qpos())
    
    #### Go over trajectory:
    timestep = 0
    while timestep < exp_length-2:
        recorded_action=recorded_traj[str(timestep+1)]
        print("hardware action", recorded_action)
        action = process_action(recorded_action)
        
        # Debug:
        # toy_action = [0,-0.02,0,0,0,0,0]
        # action = process_action(toy_action)
        print("action", action)
        
        obs, reward, success, truncated, info = env.step(np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),)
        
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        im = Image.fromarray(image)
        im.save(os.path.join(recorded_traj_dir, f"Sim_IMG_{timestep}.jpeg"))

        hw_qpos = process_qpos(recorded_traj_qpos[str(timestep)]) # Hardware qpos is 0 indexed.
        log_qpos(hw_qpos, logname="real")

        timestep += 1
        qpos = env.agent.robot.get_qpos()
        log_qpos(qpos.tolist())

        err_qpos = np.array(hw_qpos) - np.array(qpos)
        log_qpos(err_qpos, logname="error")

        norm_err_qpos= np.linalg.norm(err_qpos)
        wandb.log({"err_qpos": norm_err_qpos})

        robot_qpos_real_minus_sim.append(err_qpos.tolist())
        print("ground truth qpos", hw_qpos)
        print("qpos", qpos)
        print("error qpos", norm_err_qpos)
        
        robot_qpos.append(qpos.tolist())
        print("reward", reward)
        print("info", info)
    with open(os.path.join(recorded_traj_dir, "sim_traj.json"), "w") as f:
        json.dump(robot_qpos, f)
    with open(os.path.join(recorded_traj_dir, "hw_qpos_traj_error.json"), "w") as f:
        json.dump(robot_qpos_real_minus_sim, f)
    media.write_video(f"{args.logging_root}/recorded_traj.mp4", images, fps=1)

# =======================================================
# Sim instructions hardware
def main_sim_rollout(init_qpos, recorded_traj_dir):
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

    # Model and policy:
    policy_setup = "widowx_bridge"
    from simpler_env.policies.octo.octo_model import OctoInference
    
    model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=0)

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
            "obj_init_options": {"init_xys": np.array([[-0.1597,0.1905],[-0.1397,0.3496]]), "init_rot_quats": np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),},
            "robot_init_options": {
                "init_xy": [0.245,0.22],
                'init_height': env.scene_table_height + 0.04,
                "init_rot_quat": init_rot_quat,
                "qpos": np.array(init_qpos)
            },
        }
    
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image] # should just be sleep.
    im = Image.fromarray(image)
    im.save(os.path.join(recorded_traj_dir, "SimRollout_img_init.jpeg"))
    robot_qpos = [(env.agent.robot.get_qpos()).tolist()]
    print("Reset info:", info)
    print("robot pose", env.agent.robot.pose)
    print("qpos", env.agent.robot.get_qpos())
    
    
    # Setup
    instruction = env.get_language_instruction()
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    model.reset(instruction)
    print(instruction)
    
    #### Go over trajectory:
    success_arr = []
    timestep = 0
    predicted_terminated, success, truncated = False, False, False
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

        im = Image.fromarray(image)
        im.save(os.path.join(recorded_traj_dir, f"SimRollout_img_{timestep}.jpeg"))
        
        timestep += 1
        qpos = env.agent.robot.get_qpos()
        log_qpos(qpos.tolist())
        
        robot_qpos.append(qpos.tolist())
        print("reward", reward)
        print("info", info)

    episode_stats = info.get("episode_stats", {})
    success_arr.append(success)
    print(f"Episode 0 success: {success}")
    media.write_video(f"{args.logging_root}/sim_rollout_success_{success}.mp4", images, fps=1)


# =======================================================
# Helper functions
# Process qpos
# qpos reading from the robot includes: [waist, shoulder, elbow, forearm, wrist angle, wrist rotate, gripper, left finger, right finger]
# Simpler qpos requires: [waist, shoulder, elbow, forearm, wrist angle, wrist rotate, left finger, right finger]
def process_qpos(hw_qpos):
    qpos = [hw_qpos[0], hw_qpos[1], hw_qpos[2], hw_qpos[3], hw_qpos[4], hw_qpos[5], hw_qpos[7], hw_qpos[8]]
    return qpos

def log_qpos(qpos, logname="sim"):
    if logname:
        links = ["waist", "shoulder", "elbow", "forearm", "wrist angle", "wrist rotate", "left finger", "right finger"]
        keys = [logname+" "+link_name for link_name in links]
        for k in range(len(qpos)):
            key = keys[k]
            wandb.log({key:qpos[k]})
    return qpos

# Process action from Octo to OctoInference model:
def process_action(hw_action):
    action_scale=1.0
    raw_action = {
        "world_vector": np.array(hw_action[:3]),
        "rotation_delta": np.array(hw_action[3:6]),
        "open_gripper": np.array(hw_action[6:7]),  # range [0, 1]; 1 = open; 0 = close
    }
    # process raw_action to obtain the action to be sent to the maniskill2 environment
    action = {}
    action["world_vector"] = raw_action["world_vector"] * action_scale
    action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
    roll, pitch, yaw = action_rotation_delta
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action_rotation_axangle = action_rotation_ax * action_rotation_angle
    action["rot_axangle"] = action_rotation_axangle * action_scale

    action["gripper"] = (2.0 * (raw_action["open_gripper"] > 0.5) - 1.0)  # binarize gripper action to 1 (open) and -1 (close)
    action["terminate_episode"] = np.array([0.0])

    wandb.log({"act_x": action["world_vector"][0]})
    wandb.log({"act_y": action["world_vector"][1]})
    wandb.log({"act_z": action["world_vector"][2]})
    wandb.log({"act_roll": action["rot_axangle"][0]})
    wandb.log({"act_pitch": action["rot_axangle"][1]})
    wandb.log({"act_yaw": action["rot_axangle"][2]})
    wandb.log({"act_gripper": action["gripper"]})
    return action

# =======================================================
# Main function to control the robot:
if __name__ == "__main__":
    trial = "trial_1_corrected"
    recorded_traj_dir = f"/home/apurva/software/RapidEvalPPI/SimplerEnv/recorded_trajectories/{trial}"
    traj_log = os.path.join(recorded_traj_dir, "log.json")
    actions_log = os.path.join(recorded_traj_dir, "actions.pkl")
    image_log = os.path.join(recorded_traj_dir, "images.pkl")
    with open(traj_log, "r") as f:
        traj_info = json.load(f)
    # with open(actions_log,"r") as f:
    #     actions_info = pkl.load(f)
    st()
    wandb.login()
    problem_data = dict()
    problem_data = copy.deepcopy(traj_info)
    problem_data["trial"] = trial
    problem_data
    run = wandb.init(
        project=f"match_sim_real",
        # Track hyperparameters and run metadata
        config=problem_data, reinit=True
    )
    
    init_qpos, recorded_traj_actions, recorded_traj_qpos = traj_info["init_qpos"], traj_info["traj_act"], traj_info["qpos_traj"]
    init_qpos_above_carrot = recorded_traj_qpos['0']

    ee_state_cmd = traj_info["ee_state_cmd"]
    ee_traj = traj_info["traj_ee_state"]
    for k in ee_state_cmd.keys():
        ee_error_state = traj_info["traj_err"][k]
        wandb.log({"ee_state_err": np.linalg.norm(ee_error_state[:-1])})
    
    
    main(init_qpos_above_carrot, recorded_traj_actions, recorded_traj_qpos,recorded_traj_dir)
    # main_sim_rollout(init_qpos_above_carrot, recorded_traj_dir)
    wandb.finish()