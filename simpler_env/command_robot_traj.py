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
from transforms3d.euler import euler2axangle, euler2quat, quat2euler
import pickle as pkl
import mediapy as media
import numpy as np
import tensorflow as tf
from sapien.core import Pose
import gymnasium as gym
import sys
import simpler_env
import matplotlib.pyplot as plt
if os.path.exists("/home/apurva/software/presentation.mplstyle"):
    plt.style.use("/home/apurva/software/presentation.mplstyle") # Use default matplotlib style
from PIL import Image
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env import ENVIRONMENTS
import matplotlib.animation as animation
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
                "init_xy": [0.194,0.191],
                'init_height': env.scene_table_height + 0.035,
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
        log_qpos(hw_qpos, logname="real")

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

# =======================================================
# Main function to control the robot:
def main(carrot_pose, init_qpos, recorded_traj,recorded_traj_qpos,recorded_traj_dir, debug=False, init_step=0):
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
                "init_xy": [0.195,0.191], #[0.185,0.215], #[0.245,0.22]
                'init_height': env.scene_table_height + 0.035,
                "init_rot_quat": init_rot_quat,
                "qpos": np.array(init_qpos)
            },
        }
    
    if carrot_pose == "middle":
        env_reset_options["obj_init_options"]["init_xys"] = np.array([env.carrot_center, env.plate]) - np.array([[0.02,0], [0.02,0]])
    if carrot_pose == "left":
        env_reset_options["obj_init_options"]["init_xys"] = np.array([env.carrot_left, env.plate]) - np.array([[0.02,0], [0.02,0]])
    if carrot_pose == "right":
        env_reset_options["obj_init_options"]["init_xys"] = np.array([env.carrot_right, env.plate]) - np.array([[0.02,0], [0.02,0]])
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image] # should just be sleep.
    im = Image.fromarray(image)
    im.save(os.path.join(recorded_traj_dir, "Sim_IMG_0.jpeg"))
    
    qpos = env.agent.robot.get_qpos()
    log_qpos(qpos.tolist())
    
    robot_qpos = [(env.agent.robot.get_qpos()).tolist()]
    sim_robot_qpos_cmds = [[float(qcmd) for qcmd in list(env.agent.get_qpos_command)]]

    robot_qpos_real_minus_sim = [(np.array(init_qpos) - np.array(robot_qpos[0])).tolist()]
    print("Reset info:", info)
    print("robot pose", env.agent.robot.pose)
    print("qpos", env.agent.robot.get_qpos())
    
    #### Go over trajectory:
    timestep = init_step+1
    while timestep <= exp_length-1:
        recorded_action=recorded_traj[str(timestep)]
        print("hardware action", recorded_action)
        action = process_action(recorded_action)
        
        # Debug:
        # if debug: 
        #     if timestep <= 5:
        #         toy_action = [0,-0.01, 0, 0,0,0,0] # Applying just a yaw command
        #     else: 
        #         toy_action = [0,0,0.0,0, 0.0, -0.01,0] # Applying just a yaw command
        #     action = process_action(toy_action)
        #     print("action", action)
        
        obs, reward, success, truncated, info = env.step(np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),)
        
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        im = Image.fromarray(image)
        im.save(os.path.join(recorded_traj_dir, f"Sim_IMG_{timestep}.jpeg"))

        hw_qpos = process_qpos(recorded_traj_qpos[str(timestep)]) # Hardware qpos is 0 indexed.
        log_qpos(hw_qpos, logname="real")

        sim_cmd_qpos = env.agent.get_qpos_command
        qpos = env.agent.robot.get_qpos()
        log_qpos(qpos.tolist())

        timestep += 1

        err_qpos_arm = np.array(hw_qpos[:-2]) - np.array(qpos[:-2])
        log_qpos(err_qpos_arm, logname="error", links=["waist", "shoulder", "elbow", "forearm", "wrist angle", "wrist rotate"])

        norm_err_qpos= np.linalg.norm(err_qpos_arm)
        wandb.log({"err_qpos": norm_err_qpos})

        robot_qpos_real_minus_sim.append(err_qpos_arm.tolist())
        print("ground truth qpos", hw_qpos)
        print("qpos", qpos)
        print("error qpos", norm_err_qpos)
        
        robot_qpos.append(qpos.tolist())
        sim_robot_qpos_cmds.append([float(qcmd) for qcmd in list(env.agent.get_qpos_command)])
        print("reward", reward)
        print("info", info)

    with open(os.path.join(recorded_traj_dir, "sim_traj.json"), "w") as f:
        json.dump(robot_qpos, f)
    with open(os.path.join(recorded_traj_dir, "sim_qpos_cmds_traj.json"), "w") as f:
        json.dump(sim_robot_qpos_cmds, f)
    with open(os.path.join(recorded_traj_dir, "hw_qpos_traj_error.json"), "w") as f:
        json.dump(robot_qpos_real_minus_sim, f)

    if debug:
        media.write_video(f"{args.logging_root}/debug_traj.mp4", images, fps=1)
    else:
        media.write_video(f"{args.logging_root}/recorded_traj.mp4", images, fps=1)

def main_ee(init_qpos, recorded_traj_actions, recorded_ee_traj, recorded_traj_dir):

    def get_tcp_pose_at_robot_base():
        tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
        xyz = tcp_pose_at_robot_base.p.tolist()
        rot_quat = tcp_pose_at_robot_base.q.tolist()
        rot_euler = quat2euler(rot_quat)
        ee_pose = [*xyz, *rot_euler]
        return ee_pose
    
    exp_length = len(recorded_traj_actions)
    args = get_args(folder=recorded_traj_dir)
    os.makedirs(args.logging_root, exist_ok=True)

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],)

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
                "init_xy": [0.194,0.191],
                'init_height': env.scene_table_height + 0.035,
                "init_rot_quat": init_rot_quat,
                "qpos": np.array(init_qpos)
            },
        }
    
    env_reset_options["obj_init_options"]["init_xys"] = np.array([env.carrot_center, env.plate]) - np.array([[0.02,0], [0.02,0]])
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image] # should just be sleep.
    im = Image.fromarray(image)
    im.save(os.path.join(recorded_traj_dir, "Sim_IMG_init.jpeg"))
    
    robot_ee_pos = [get_tcp_pose_at_robot_base()]
    
    #### Go over trajectory:
    timestep = 0
    while timestep <= exp_length-2:
        recorded_action=recorded_traj_actions[str(timestep+1)]
        print("hardware action", recorded_action)
        action = process_action(recorded_action)
        
        # Debug:
        # if debug: 
        #     if timestep <= 5:
        #         toy_action = [0,0,-0.01,0,0,-0.01,0] # Applying just a yaw command
        #     else: 
        #         toy_action = [0,0,0,0, 0,-0.01,0] # Applying just a yaw command
        #     action = process_action(toy_action)
        #     print("action", action)
    
        obs, reward, success, truncated, info = env.step(np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),)
        st()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        im = Image.fromarray(image)
        im.save(os.path.join(recorded_traj_dir, f"Sim_IMG_{timestep}.jpeg"))

        hw_ee_pos = recorded_ee_traj[str(timestep)] # Hardware qpos is 0 indexed.

        timestep += 1
        eepos = get_tcp_pose_at_robot_base()

        robot_ee_pos.append(eepos)
        
    with open(os.path.join(recorded_traj_dir, "sim_ee_traj.json"), "w") as f:
        json.dump(robot_ee_pos, f)

    if debug:
        media.write_video(f"{args.logging_root}/debug_ee_traj.mp4", images, fps=1)
    else:
        media.write_video(f"{args.logging_root}/recorded_ee_traj.mp4", images, fps=1)

def plot_qpos(recorded_traj_dir, recorded_qpos, qpos_cmd =None, sim_qpos_fn=None, init_step=0):
    if sim_qpos_fn is None:
        sim_qpos_fn = "sim_traj"
        sim_qpos_cmd_fn  = "sim_qpos_cmds_traj"

    with open(os.path.join(recorded_traj_dir, sim_qpos_fn+".json"), "r") as f:
        sim_qpos = json.load(f)
    hw_qpos = [process_qpos(recorded_traj_qpos[key]) for key in recorded_traj_qpos.keys()]
    hw_qpos = hw_qpos[init_step:]

    if qpos_cmd is not None:
        hw_qpos_cmd = [process_qpos(qpos_cmd[key]) for key in qpos_cmd.keys()]
        hw_qpos_cmd = hw_qpos_cmd[init_step:]

        with open(os.path.join(recorded_traj_dir, sim_qpos_cmd_fn+".json"), "r") as f:
            sim_qpos_cmd = json.load(f)

    joints = ["waist", "shoulder", "elbow", "forearm", "wrist angle", "wrist rotate", "left finger", "right finger"]
    fig_folder = f"{recorded_traj_dir}/action_replay"

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    for k, name in enumerate(joints):
        sim_joint = [sim_qpos_k[k] for sim_qpos_k in sim_qpos]
        real_joint = [hw_qpos_k[k] for hw_qpos_k in hw_qpos]
        
        # Plotting the lists
        plt.figure()
        plt.plot(sim_joint, marker='o', label='sim')
        plt.plot(real_joint, marker='s', label='real')
        
        plt.title(f"Joint Angle: {name} (rad)")

        if qpos_cmd is not None:
            if name != "left finger" and name!="right finger":
                real_cmd = [hw_qpos_cmd_k[k] for hw_qpos_cmd_k in hw_qpos_cmd]
                plt.plot(real_cmd, marker='^', label='real_cmd')

                sim_cmd = [sim_qpos_cmd_k[k] for sim_qpos_cmd_k in sim_qpos_cmd]
                plt.plot(sim_cmd, marker='s', label='sim_cmd')
        plt.legend()
        plt.savefig(f"{fig_folder}/joint_{name}_{sim_qpos_fn}.pdf", bbox_inches='tight')

def plot_eepos(recorded_traj_dir, recorded_ee_traj, cmd_ee_traj, sim_fn = None, init_step=0):
    '''
    TODO: Fix EE Pos
    '''
    if sim_fn is None:
        sim_fn = "sim_ee_traj"
    with open(os.path.join(recorded_traj_dir, sim_fn+".json"), "r") as f:
        sim_ee_pose = json.load(f)
    hw_ee_pose = [recorded_ee_traj[key] for key in recorded_ee_traj.keys()]
    hw_ee_pose = hw_ee_pose[init_step:]

    hw_cmd_ee_pose = [cmd_ee_traj[key] for key in cmd_ee_traj.keys()]
    hw_cmd_ee_pose = hw_cmd_ee_pose[init_step:]

    coords = ["x", "y", "z", "roll", "pitch", "yaw"]
    fig_folder = f"{recorded_traj_dir}/action_replay"

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    for k, name in enumerate(coords):
        sim_ee = [sim_eepos_k[k] for sim_eepos_k in sim_ee_pose]
        real_ee = [hw_eepos_k[k] for hw_eepos_k in hw_ee_pose]
        real_cmd_ee = [hw_cmd_eepos_k[k] for hw_cmd_eepos_k in hw_cmd_ee_pose]
        
        # Plotting the lists
        plt.figure()
        plt.plot(sim_ee, marker='o', label='sim')
        plt.plot(real_ee, marker='s', label='real')
        plt.plot(real_cmd_ee, marker='^', alpha=0.5, label='real_cmd')
        # x_cmd = list(range(1, len(real_cmd_ee) + 1))
        # plt.plot(x_cmd, real_cmd_ee, marker='^', label='real_cmd')
        plt.legend()
        if name in ["x", "y", "z"]:
            plt.title(f"EE Pose: {name} (in m)")
        elif name in ["roll", "pitch", "yaw"]:
            plt.title(f"EE Pose: {name} (in rad)")
        plt.savefig(f"{fig_folder}/ee_{name}_{sim_fn}.pdf", bbox_inches='tight')

# =======================================================
# Sim instructions hardware
def main_sim_rollout(init_qpos, recorded_traj_dir, debug=True):
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
    
    # Action history
    recorded_actions = []

    #### Go over trajectory:
    success_arr = []
    timestep = 0
    predicted_terminated, success, truncated = False, False, False
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, instruction)
        curr_action = [*action["world_vector"], *action["rot_axangle"], *action["gripper"]]
        recorded_actions.append(curr_action)

        # Debug:
        if debug: 
            mod_action = [0,0,0, 0, 0, action['rot_axangle'][2], 0]
            action = process_action(mod_action)
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

    with open(os.path.join(recorded_traj_dir, "sim_actions.json"), "w") as f:
        json.dump(recorded_actions, f)


# =======================================================
# Helper functions
# Process qpos
# qpos reading from the robot includes: [waist, shoulder, elbow, forearm, wrist angle, wrist rotate, gripper, left finger, right finger]
# Simpler qpos requires: [waist, shoulder, elbow, forearm, wrist angle, wrist rotate, left finger, right finger]
def process_qpos(hw_qpos):
    if len(hw_qpos) > 6:
        qpos = [hw_qpos[0], hw_qpos[1], hw_qpos[2], hw_qpos[3], hw_qpos[4], hw_qpos[5], hw_qpos[7], -1*hw_qpos[8]]
    else:
        qpos = [hw_qpos[0], hw_qpos[1], hw_qpos[2], hw_qpos[3], hw_qpos[4], hw_qpos[5]] # No processing needed
    return qpos

def log_qpos(qpos, logname="sim", links=["waist", "shoulder", "elbow", "forearm", "wrist angle", "wrist rotate", "left finger", "right finger"]):
    if logname:
        keys = [logname+" "+link_name for link_name in links]
        for k in range(len(qpos)):
            key = keys[k]
            wandb.log({key:qpos[k]})
    return qpos

def log_eepos(eepos, logname="sim", links=["x", "y", "z", "roll", "pitch", "yaw"]):
    if logname:
        keys = [logname+" "+link_name for link_name in links]
        for k in range(len(eepos)):
            key = keys[k]
            wandb.log({key:eepos[k]})
    return eepos

# Record animation
def record_real_animation(image_folder):
    # Get sorted list of images
    image_names = [f"Original{k}.jpg" for k in range(0, 64)]
    image_files = [os.path.join(image_folder, im_file) for im_file in image_names if os.path.exists(os.path.join(image_folder, im_file))]

    # Create a figure
    fig, ax = plt.subplots()
    ax.axis('off')  # Remove axes

    # Load first image to initialize the plot
    img = Image.open(image_files[0])
    im = ax.imshow(img)

    # Update function for animation
    def update(frame):
        img = Image.open(image_files[frame])
        im.set_data(img)
        return [im]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(image_files), interval=200, blit=True)

    # Save the animation
    ani.save(os.path.join(image_folder, "real_animation.mp4"), fps=1, extra_args=['-vcodec', 'libx264'])

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

    # Policy outputs roll, pitch, yaw of EE in Space/Base Frame
    roll, pitch, yaw = action_rotation_delta # 
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action_rotation_axangle = action_rotation_ax * action_rotation_angle
    action["rot_axangle"] = action_rotation_axangle * action_scale

    action["gripper"] = (2.0 * (raw_action["open_gripper"] > 0.5) - 1.0)  # binarize gripper action to 1 (open) and -1 (close)
    action["terminate_episode"] = np.array([0.0])

    return action

#=======================================================
# Utility functions:
def rpy_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def angular_distance(rpy1, rpy2):
    R1 = rpy_to_rotation_matrix(*rpy1)
    R2 = rpy_to_rotation_matrix(*rpy2)
    
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    theta = np.arccos((trace - 1) / 2)
    
    return theta

# =======================================================
# Main function to control the robot:
# Trial 70 and trial 110 in fixed policies
if __name__ == "__main__":
    trial = "trial_27"
    recorded_traj_dir = f"/home/apurva/software/RapidEvalPPI/hardware/random_no_grasp_no_contact/{trial}"
    traj_log = os.path.join(recorded_traj_dir, "log.json")
    actions_log = os.path.join(recorded_traj_dir, "actions.pkl")
    image_log = os.path.join(recorded_traj_dir, "images.pkl")
    with open(traj_log, "r") as f:
        traj_info = json.load(f)
    
    init_qpos, recorded_traj_actions, recorded_traj_qpos = traj_info["init_qpos"], traj_info["traj_act"], traj_info["qpos_traj"]
    
    exp = "replay"
    debug = False
    if exp == "replay":
        project = f"match_sim_real_action_replay_{trial}"
    elif exp == "qpos":
        project = f"match_real_qpos_{trial}"
    elif exp == "ee":
        project = "ee"
    elif exp == "rollout":
        project = "sim_rollout"

    wandb.login()
    problem_data = dict()
    problem_data = copy.deepcopy(traj_info)
    problem_data["trial"] = trial
    problem_data
    run = wandb.init(project=f"{project}",config=problem_data, reinit=True)

    # Recorded actions at time "t" (are applied on robot state at t-1) result in the recorded qpos at time "t". 
    # Ignore action["0"] and directly set the robot to be above the carrot at qpos["0"]
    recorded_ee_traj = traj_info["traj_ee_state"]
    
    init_qpos_above_carrot = recorded_traj_qpos['0']
    init_ee_above_carrot = recorded_ee_traj['0']

    ee_state_cmd = traj_info["ee_state_cmd"]
    qpos_cmd = traj_info["qpos_cmd"]
    ee_traj = traj_info["traj_ee_state"]
    for k in ee_state_cmd.keys():
        ee_error_state = traj_info["traj_err"][k]
        cmd_state = np.array(ee_state_cmd[k])
        act_state = np.array(ee_traj[k])
        position_error = np.linalg.norm(cmd_state[0:3] - act_state[0:3])
        angle_error = angular_distance(cmd_state[3:6], act_state[3:6])
        gripper_error =  cmd_state[6] - act_state[6]

    if exp == "replay":
        try:
            carrot_pose = traj_info['carrot_pos']
        except:
            carrot_pose = traj_info["initial_pos"]
        main(carrot_pose, init_qpos_above_carrot, recorded_traj_actions, recorded_traj_qpos,recorded_traj_dir, debug=debug)
        plot_qpos(recorded_traj_dir, recorded_traj_qpos, qpos_cmd)
    elif exp == "rollout":
        main_sim_rollout(init_qpos_above_carrot,recorded_traj_dir, debug=debug)
    elif exp == "ee":
        main_ee(init_qpos_above_carrot, recorded_traj_actions, recorded_ee_traj, recorded_traj_dir)
        plot_eepos(recorded_traj_dir, recorded_ee_traj,ee_state_cmd)
    elif exp == "qpos":
        main_qpos(init_qpos_above_carrot, recorded_traj_actions, recorded_traj_qpos,recorded_traj_dir)
        plot_qpos(recorded_traj_dir, recorded_traj_qpos, sim_qpos_fn="direct_qpos_cmd_sim_traj")
        plot_eepos(recorded_traj_dir, recorded_ee_traj,ee_state_cmd, sim_fn="sim_ee_traj_direct_qpos")

    # Record real animation:
    record_real_animation(image_folder=recorded_traj_dir)
    wandb.finish()

 
    