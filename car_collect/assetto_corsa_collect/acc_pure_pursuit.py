from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_dynamics import ASSETTO_CORSA_ASSETS_DIR
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import random
import time
import math
import datetime
import argparse
import subprocess
import atexit
from tqdm import tqdm
from car_dataset import CarDataset


from car_dynamics.envs.assetto_corsa.assetto_corsa_gym.AssettoCorsaEnv import assettoCorsa #capital C in assetto"C"orsa!!
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController

import sys
import pandas as pd
import glob as glob
from omegaconf import OmegaConf

import faulthandler
faulthandler.enable()

STARTED_XBOXDRV = False
BIN_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "vel_x", "vel_y", "vel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "throttle", "steer",
]

def xboxdrv_running():
    return subprocess.run(["pgrep", "-x", "xboxdrv"], stdout=subprocess.DEVNULL).returncode == 0

def stop_started_xboxdrv():
    if STARTED_XBOXDRV:
        subprocess.run(["sudo", "-n", "pkill", "xboxdrv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_xboxdrv():
    global STARTED_XBOXDRV
    if xboxdrv_running():
        return
    subprocess.run([
        "sudo", "-b", "xboxdrv", "--daemon", "--silent", "--mimic-xpad",
        "--type", "xbox360", "--dbus", "disabled",
    ], check=True)
    STARTED_XBOXDRV = True
    atexit.register(stop_started_xboxdrv)
    time.sleep(1.0)

def find_xbox_event():
    from evdev import InputDevice, list_devices

    for path in list_devices():
        device = InputDevice(path)
        name = device.name.lower()
        if "x-box 360" in name or "xbox 360" in name or "x-box" in name:
            return path
    raise RuntimeError("Could not find virtual Xbox controller. Is xboxdrv running?")

def send_gear_up(device_path=None):
    from evdev import InputDevice, ecodes

    device = InputDevice(device_path or find_xbox_event())
    device.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
    device.write(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)
    time.sleep(0.15)
    device.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
    device.write(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)

def get_next_bin_path(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for name in os.listdir(output_dir):
        stem, ext = os.path.splitext(name)
        if ext == ".bin" and stem.isdigit():
            existing.append(int(stem))
    next_idx = max(existing, default=-1) + 1
    return os.path.join(output_dir, f"{next_idx:06d}.bin")

def dataset_to_bin_array(dataset):
    logs = dataset.data_logs
    arr = np.column_stack([
        logs["xpos_x"],
        logs["xpos_y"],
        logs["xpos_z"],
        logs["xori_w"],
        logs["xori_x"],
        logs["xori_y"],
        logs["xori_z"],
        logs["xvel_x"],
        logs["xvel_y"],
        logs["xvel_z"],
        logs["avel_x"],
        logs["avel_y"],
        logs["avel_z"],
        logs["throttle"],
        logs["steer"],
    ]).astype(np.float32, copy=False)
    assert arr.shape[1] == len(BIN_COLUMNS), arr.shape
    return arr

def rotate_body_to_world(quat_wxyz, vec_body):
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return R.from_quat(quat_xyzw).apply(vec_body)

def log_data(dataset, env, controller, action):
        dataset.data_logs["xpos_x"].append(env.state["world_position_x"])
        dataset.data_logs["xpos_y"].append(env.state["world_position_y"])
        dataset.data_logs["xpos_z"].append(env.state["world_position_z"])
        #log orientation
        roll, pitch, yaw = env.state["roll"], env.state["pitch"], env.state["yaw"]
        quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        dataset.data_logs["xori_w"].append(quat_wxyz[0])
        dataset.data_logs["xori_x"].append(quat_wxyz[1])
        dataset.data_logs["xori_y"].append(quat_wxyz[2])
        dataset.data_logs["xori_z"].append(quat_wxyz[3])
        #log linear velocity
        local_vel = np.array([
            env.state["local_velocity_x"],
            env.state["local_velocity_y"],
            env.state["local_velocity_z"],
        ])
        world_vel = rotate_body_to_world(quat_wxyz, local_vel)
        dataset.data_logs["xvel_x"].append(world_vel[0])
        dataset.data_logs["xvel_y"].append(world_vel[1])
        dataset.data_logs["xvel_z"].append(world_vel[2])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.state["accelX"])
        dataset.data_logs["xacc_y"].append(env.state["accelY"])
        dataset.data_logs["xacc_z"].append(0)
        #log angular velocity
        local_ang_vel = np.array([
            env.state["angular_velocity_x"],
            env.state["angular_velocity_y"],
            env.state["angular_velocity_z"],
        ])
        world_ang_vel = rotate_body_to_world(quat_wxyz, local_ang_vel)
        dataset.data_logs["avel_x"].append(world_ang_vel[0])
        dataset.data_logs["avel_y"].append(world_ang_vel[1])  
        dataset.data_logs["avel_z"].append(world_ang_vel[2])

        dataset.data_logs["traj_x"].append(controller.target_pos[0])
        dataset.data_logs["traj_y"].append(controller.target_pos[1])

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1] * controller.max_steering)

def rollout(id, simend, debug_plots, datadir, lower_vel, upper_vel, max_steering, lookahead, kp, kd, steer_sign, steer_gain, xbox_event, no_offtrack_termination):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
        datefmt='%Y-%m-%d %H:%M:%S',  # Format of the timestamp
    )

    tic = time.time()

    dataset = CarDataset()
    config = OmegaConf.load(os.path.join(ASSETTO_CORSA_ASSETS_DIR, "config.yml"))
    if no_offtrack_termination:
        config.AssettoCorsa.enable_out_of_track_termination = False
    env = assettoCorsa.make_ac_env(cfg=config, work_dir="output")
    if hasattr(env, "env"):
        env = env.env

    static_info = env.client.simulation_management.get_static_info()
    ac_mod_config = env.client.simulation_management.get_config()

    logger.info("Static info:")
    for i in static_info:
        logger.info(f"{i}: {static_info[i]}")
    logger.info("AC Mod config:")
    for i in ac_mod_config:
        logger.info(f"{i}: {ac_mod_config[i]}")

    env.reset()
    env.client.controls.set_controls(steer=0, acc=-1, brake=-1, enable_gear_shift=True, shift_up=True)
    env.client.respond_to_server()
    time.sleep(0.2)
    env.client.controls.set_controls(steer=0, acc=-1, brake=-1)
    env.client.respond_to_server()
    send_gear_up(xbox_event)
    time.sleep(0.2)

    # set the simulator
    dataset.car_params["sim"] = "assetto_corsa_"+ env.car_name #env.name
    # set the wheelbase
    dataset.car_params["wheelbase"] = env.wheelbase
    # generate a new mass
    dataset.car_params["mass"] = None #env.generate_new_mass()
    # generate a new com
    dataset.car_params["com"] = None #env.generate_new_com()
    # generate a new friction
    dataset.car_params["friction"] = None#env.generate_new_friction()
    # generate new max throttle
    dataset.car_params["max_throttle"] = None#env.generate_new_max_throttle()
    # generate new delay
    dataset.car_params["delay"] = None#env.generate_new_delay()
    # generate new max steering
    dataset.car_params["max_steer"] = None#env.generate_new_max_steering()
    # generate new steering bias
    dataset.car_params["steer_bias"] = None#env.generate_new_steering_bias()

    print("Car Params", dataset.car_params)

    #wenli where did you get the lower vel from?
    ppcontrol = AltPurePursuitController({
        'wheelbase': dataset.car_params["wheelbase"], 
        'totaltime': simend,
        'lowervel': lower_vel,
        'uppervel': upper_vel,
        'max_steering': max_steering,
    })
    ppcontrol.lookahead_distance = lookahead

    controller = ppcontrol #all_controllers[np.random.choice([0, 1])]
    trajectory = np.array([env.ref_lap.df["pos_x"], env.ref_lap.df["pos_y"]]).T

    last_err_vel = 0.
    is_terminate = False 
    clipped = 0
    actions = []
    vels = []
    for t in range(simend):
        start = time.time()
        action = controller.get_control(t, env, trajectory)
        if controller.name == "pure_pursuit":

            target_vel = action[0]

            action[0] = kp * (target_vel - env.lin_vel[0]) + kd * ((target_vel - env.lin_vel[0]) - last_err_vel)
            action[1] = steer_sign * steer_gain * action[1]
            
            # print(action[0])
            vels.append(env.lin_vel[0])
            last_err_vel = target_vel - env.lin_vel[0]
        else:
            raise ValueError(f"Unknown Controller: {controller.name}")

        # action = [0., 0.] #throttle. steer
        print("time taken:", time.time() - start)
        action = np.clip(action, -1, 1)
        # actions.append(action[0]) 
        log_data(dataset, env, controller, action)

        obs, reward, done, info = env.step(np.array(action))

        # check if the robot screwed up
        #TODO Fix when adding slope changes to the world.
        if done:
            is_terminate = True
            break
    
    if dataset.data_logs["xpos_x"]:
        if is_terminate:
            dataset.data_logs["lap_end"][-1] = 1
        output_dir = os.path.join(datadir, env.track_name, env.car_name)
        filepath = get_next_bin_path(output_dir)
        bin_data = dataset_to_bin_array(dataset)
        bin_data.tofile(filepath)
        print("Saved Data to:", filepath)
        print("Bin shape:", bin_data.shape, "columns:", BIN_COLUMNS)

    if debug_plots:
        actions = np.array(actions)
        vels = np.array(vels)
        # print(actions.shape)
        # plt.plot(actions, label = "actual command")
        # plt.plot(actions[:, 1], label = "steering")
        plt.plot(ppcontrol.target_velocities, label="target vel")
        plt.plot(vels, label = "actual vel")
        plt.legend()
        plt.show()
        plt.plot(dataset.data_logs["traj_x"], dataset.data_logs["traj_y"], label='Trajectory')
        plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='Car Position')
        plt.show()
        
        dataset.reset_logs()
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)

    env.recover_car()
    env.close()
    return not is_terminate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--simend", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--data-dir", default=os.path.join(CAR_FOUNDATION_DATA_DIR, "assetto_corsa_sim_debugging"))
    parser.add_argument("--lower-vel", type=float, default=3.0)
    parser.add_argument("--upper-vel", type=float, default=8.0)
    parser.add_argument("--max-steering", type=float, default=0.61)
    parser.add_argument("--lookahead", type=float, default=8.0)
    parser.add_argument("--kp", type=float, default=0.25)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--steer-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--steer-gain", type=float, default=1.0)
    parser.add_argument("--debug-plots", action="store_true")
    parser.add_argument("--xbox-event", default=None, help="Virtual Xbox event path, e.g. /dev/input/event21")
    parser.add_argument("--no-start-xboxdrv", action="store_true")
    parser.add_argument("--no-offtrack-termination", action="store_true",
                        help="Disable episode termination when Assetto Corsa reports the car off track")
    args = parser.parse_args()

    if not args.no_start_xboxdrv:
        ensure_xboxdrv()
    xbox_event = args.xbox_event or find_xbox_event()
    print(f"Using virtual Xbox controller: {xbox_event}")

    debug_plots = args.debug_plots
    simend = args.simend
    episodes = args.episodes
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    num_success = 0
    start = time.time()
    
    for i in range(episodes):
        ret = rollout(
            i, simend, debug_plots, data_dir,
            args.lower_vel, args.upper_vel, args.max_steering, args.lookahead,
            args.kp, args.kd, args.steer_sign, args.steer_gain, xbox_event,
            args.no_offtrack_termination,
        )
        print(f"Episode {i} Complete")
        if ret:
            num_success += 1
    
    dur = time.time() - start
    print(f"Success Rate: {num_success}/{episodes}")
    print(f"Time Elapsed:, {dur}")   
