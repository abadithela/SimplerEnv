import os
from pdb import set_trace as st
import yaml
import datetime
from abc import ABC, abstractmethod

class Log:
    def __init__(self, folder):
        self.folder = folder
        self.data = {"results_dir": self.folder}
        now = datetime.datetime.now()
        date_str = now.strftime("%m-%d-%Y")
        time_str = now.strftime("%H-%M-%S")
        self.data["timestamp"] = {"date": date_str,"time": time_str}

    def save_to_yaml(self, filename="sim_log.yaml"):
        """Saves the results dictionary to a YAML file."""
        with open(os.path.join(self.folder,filename), "w") as file:
            yaml.dump(self.data, file, default_flow_style=False)

    def add_entry(self, name, val):
        self.data[name] = val

class SimLogger(Log):
    """Concrete implementation for simulation logging."""
    def add_args(self, args):
        """ Adding experiment args """
        for k,v in args.__dict__.items():
            self.data[k] = v

    def log_env_vars(self, env):
        self.data["xy_center"] = {"desc": "Lower left red button position from lower left table corner", "pos": env.get_xy_center().tolist()}
        # self.data["carrot"] = {"model_id":env._source_obj_name, "scale": env._carrot_scale, "left": env.carrot_left, "center": env.carrot_center, "right": env.carrot_right}
        # self.data["plate"] = {"model_id":env._target_obj_name, "scale": env._plate_scale, "left": env.plate}
        carrot_left, carrot_center, carrot_right = env.get_carrot()
        
        self.data["carrot"] = { "left":carrot_left.tolist(), "center": carrot_center.tolist(), "right": carrot_right.tolist()}
        self.data["plate"] = {"left": env.get_plate().tolist()}

        # self.data["obj_quat_configs"] = env._quat_configs
        self.data["instruction"] = env.get_language_instruction()
        # env_conditions = env._setup_prepackaged_env_init_config()
        # self.data["control_params"] = {"sim_freq": env_conditions['sim_freq'], "control_freq": env_conditions['control_freq'], "control_mode": env_conditions["control_mode"]}
        # self.data["overlay"] = env_conditions["rgb_overlay_path"]
        self.data["max_episode_steps"] = 64

    def log_robot_setup(self, env):
        self.data["robot_init_options"] = {"init_xy": env.robot_init_xy, "base_init_height": env.robot_init_height, "base_init_quat": env.robot_init_quat}

    # Log Robot Texture

    # Log Ray Tracing

    # Lighting Conditions
    def log_lighting(self, is_dir_light, is_ambient_light, dir_light_position, dir_light_color, ambient_light_color,  shadow, dir_light_scale):
        self.data["lighting"] = {"is_dir_light":is_dir_light, "is_ambient_light":is_ambient_light, "dir_light_position":dir_light_position, "dir_light_color":dir_light_color, "ambient_light_color":ambient_light_color, "shadow":shadow, "dir_light_scale":dir_light_scale}
    

