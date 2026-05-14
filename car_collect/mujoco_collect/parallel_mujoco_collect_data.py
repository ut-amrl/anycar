from car_foundation import CAR_FOUNDATION_DATA_DIR
import argparse
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import ray
import time
import datetime
from tqdm import tqdm
from car_dynamics.envs.mujoco_sim.cam_utils import *
from car_dataset import CarDataset
from car_planner.track_generation import change_track
# from car_planner.track_generation_realistic import change_track
from car_dynamics import MUJOCO_MODEL_DIR

from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar
from car_dynamics.controllers_torch import AltPurePursuitController

import faulthandler
faulthandler.enable()

BIN_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "vel_x", "vel_y", "vel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "throttle", "steer",
]

# Tunable collection defaults. These cover the rollout length, open-loop
# command generator, and randomized vehicle parameters used by this file.
DEFAULT_SIMEND = 5000
DEFAULT_EPISODES = 5000
DEFAULT_CONTROL_MODE = "open-loop"
DEFAULT_SEED = 0
DEFAULT_CHANGE_PARAMETERS = False

DEFAULT_TRACK_MIN_SCALE = 1
DEFAULT_TRACK_MAX_SCALE = 5
DEFAULT_PURE_PURSUIT_LOWER_VEL = 0.7
DEFAULT_PURE_PURSUIT_UPPER_VEL = 2.0
DEFAULT_PURE_PURSUIT_MIN_KP = 6.0
DEFAULT_PURE_PURSUIT_MAX_KP = 10.0
DEFAULT_PURE_PURSUIT_MIN_KD = 0.5
DEFAULT_PURE_PURSUIT_MAX_KD = 1.5

DEFAULT_OPEN_LOOP_MIN_VEL = -2.0
DEFAULT_OPEN_LOOP_MAX_VEL = 5.0
DEFAULT_OPEN_LOOP_VEL_SEGMENTS = 8
DEFAULT_OPEN_LOOP_STEER_STEP_STD = 0.03
DEFAULT_OPEN_LOOP_STEER_DAMPING = 0.985

DEFAULT_WHEELBASE = 0.2965
DEFAULT_MASS_RANGE = (1.0, 15.0)
DEFAULT_COM_CENTER = np.array([-0.02323112, -0.00007926, 0.09058852], dtype=np.float32)
DEFAULT_COM_DELTA = 0.05
DEFAULT_FRICTION_RANGE = (0.5, 1.5)
DEFAULT_MAX_THROTTLE_RANGE = (2.0, 10.0)
DEFAULT_MAX_STEER_RANGE = (0.15, 0.5)
DEFAULT_STEER_BIAS_RANGE = (0.0, 0.03)
DEFAULT_DELAY_RANGE = (0, 6)


def sample_uniform_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def sample_int_range(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.uniform(lo, hi))


def generate_bezier_velocity_commands(
    simend: int,
    min_vel: float,
    max_vel: float,
    num_segments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    commands = np.zeros(simend, dtype=np.float32)
    num_segments = max(1, int(num_segments))
    span = max_vel - min_vel
    segment_edges = np.linspace(0, simend, num_segments + 1, dtype=np.int64)
    knots = rng.uniform(min_vel, max_vel, size=num_segments + 1)

    for segment_id in range(num_segments):
        start = int(segment_edges[segment_id])
        end = int(segment_edges[segment_id + 1])
        if end <= start:
            continue

        p0 = knots[segment_id]
        p3 = knots[segment_id + 1]
        p1 = np.clip(p0 + rng.uniform(-0.5, 0.5) * span, min_vel, max_vel)
        p2 = np.clip(p3 + rng.uniform(-0.5, 0.5) * span, min_vel, max_vel)
        u = np.linspace(0.0, 1.0, end - start, endpoint=False)
        commands[start:end] = (
            (1.0 - u) ** 3 * p0
            + 3.0 * (1.0 - u) ** 2 * u * p1
            + 3.0 * (1.0 - u) * u ** 2 * p2
            + u ** 3 * p3
        )

    commands[-1] = knots[-1]
    return commands


def generate_random_walk_steer_commands(
    simend: int,
    step_std: float,
    damping: float,
    rng: np.random.Generator,
) -> np.ndarray:
    commands = np.zeros(simend, dtype=np.float32)
    damping = float(np.clip(damping, 0.0, 1.0))
    steer = rng.uniform(-0.2, 0.2)

    for t in range(simend):
        steer = damping * steer + rng.normal(0.0, step_std)
        steer = float(np.clip(steer, -1.0, 1.0))
        commands[t] = steer

    return commands


def sample_car_params(env: MuJoCoCar, args: argparse.Namespace,
                      rng: np.random.Generator) -> dict:
    friction = sample_uniform_range(rng, args.friction_min, args.friction_max)
    com_center = np.array(args.com_center, dtype=np.float32)
    return {
        "sim": env.name,
        "wheelbase": args.wheelbase,
        "mass": sample_uniform_range(rng, args.mass_min, args.mass_max),
        "com": rng.uniform(com_center - args.com_delta, com_center + args.com_delta),
        "friction": np.array([friction, 0.005, 0.0001]),
        "max_throttle": sample_uniform_range(rng, args.max_throttle_min, args.max_throttle_max),
        "delay": sample_int_range(rng, args.delay_min, args.delay_max),
        "max_steer": sample_uniform_range(rng, args.max_steer_min, args.max_steer_max),
        "steer_bias": sample_uniform_range(rng, args.steer_bias_min, args.steer_bias_max),
    }


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

def log_data(dataset, env, action0_to_log, steer_to_log, target_pos=None):
        """Log SE(3) state plus the two collector action channels.

        In pure-pursuit mode, action0_to_log is target velocity in m/s.
        In open-loop mode, action0_to_log is target velocity in m/s.
        steer_to_log is normalized steering in [-1, 1].
        """
        dataset.data_logs["xpos_x"].append(env.world.pose[0])
        dataset.data_logs["xpos_y"].append(env.world.pose[1])
        dataset.data_logs["xpos_z"].append(env.world.pose[2])
        #log orientation
        quat_wxyz = env.world.orientation
        dataset.data_logs["xori_w"].append(quat_wxyz[0])
        dataset.data_logs["xori_x"].append(quat_wxyz[1])
        dataset.data_logs["xori_y"].append(quat_wxyz[2])
        dataset.data_logs["xori_z"].append(quat_wxyz[3])
        #log linear velocity
        world_vel = rotate_body_to_world(quat_wxyz, env.world.lin_vel)
        dataset.data_logs["xvel_x"].append(world_vel[0])
        dataset.data_logs["xvel_y"].append(world_vel[1])
        dataset.data_logs["xvel_z"].append(world_vel[2])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.world.lin_acc[0])
        dataset.data_logs["xacc_y"].append(env.world.lin_acc[1])
        dataset.data_logs["xacc_z"].append(env.world.lin_acc[2])
        #log angular velocity
        world_ang_vel = rotate_body_to_world(quat_wxyz, env.world.ang_vel)
        dataset.data_logs["avel_x"].append(world_ang_vel[0])
        dataset.data_logs["avel_y"].append(world_ang_vel[1])
        dataset.data_logs["avel_z"].append(world_ang_vel[2])

        if target_pos is None:
            target_pos = (np.nan, np.nan)
        dataset.data_logs["traj_x"].append(target_pos[0])
        dataset.data_logs["traj_y"].append(target_pos[1])

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action0_to_log)
        dataset.data_logs["steer"].append(steer_to_log)

# @ray.remote
def rollout(id, simend, render, debug_plots, datadir, args):
    tic = time.time()
    rng = np.random.default_rng(args.seed + id)

    dataset = CarDataset()
    env = MuJoCoCar({'is_render': render}) # create the simulation environment
    env.reset()

    dataset.car_params.update(sample_car_params(env, args, rng))

    print("Car Params", dataset.car_params)

    #choose to change parameters or not
    env.world.change_parameters(dataset.car_params, change=args.change_parameters)

    direction = rng.choice([-1, 1])
    scale = sample_int_range(rng, args.track_min_scale, args.track_max_scale)

    ppcontrol = AltPurePursuitController({
        'wheelbase': dataset.car_params["wheelbase"], 
        'totaltime': simend,
        'lowervel': args.pure_pursuit_lower_vel,
        'uppervel': args.pure_pursuit_upper_vel,
        'max_steering': env.world.max_steer    
    })

    controller = ppcontrol
    trajectory = change_track(scale, direction)
    target_velocity_commands = generate_bezier_velocity_commands(
        simend,
        args.open_loop_min_vel,
        args.open_loop_max_vel,
        args.open_loop_vel_segments,
        rng,
    )
    steer_commands = generate_random_walk_steer_commands(
        simend,
        args.open_loop_steer_step_std,
        args.open_loop_steer_damping,
        rng,
    )

    kp = sample_uniform_range(rng, args.pure_pursuit_min_kp, args.pure_pursuit_max_kp)
    kd = sample_uniform_range(rng, args.pure_pursuit_min_kd, args.pure_pursuit_max_kd)
    
    last_err_vel = 0.
    is_terminate = False 
    actions = []
    for t in tqdm(range(simend)):

        if args.control_mode == "open-loop":
            target_vel = float(target_velocity_commands[t])
            err_vel = target_vel - env.world.lin_vel[0]
            throttle = kp * err_vel + kd * (err_vel - last_err_vel)
            throttle /= env.world.max_throttle
            last_err_vel = err_vel
            action = np.array([throttle, steer_commands[t]], dtype=np.float32)
            action0_to_log = target_vel
            steer_to_log = float(np.clip(action[1], -1.0, 1.0))
            target_pos = None
        else:
            action = controller.get_control(t, env.world, trajectory)
            if controller.name != "pure_pursuit":
                raise ValueError(f"Unknown Controller: {controller.name}")

            target_vel = action[0]              # m/s velocity setpoint from pursuit
            target_steer_norm = action[1]       # [-1, 1] normalized steer setpoint

            action[0] = kp * (target_vel - env.world.lin_vel[0]) + kd * ((target_vel - env.world.lin_vel[0]) - last_err_vel)

            action[0] /= env.world.max_throttle # normalize to -1, 1

            # print(action[0])
            last_err_vel = target_vel - env.world.lin_vel[0]
            action0_to_log = float(target_vel)
            steer_to_log = float(np.clip(target_steer_norm, -1.0, 1.0))
            target_pos = controller.target_pos

        #TODO Fix this, the action is getting clipped like 25% of the time
        action = np.clip(action, env.action_space.low, env.action_space.high)
        log_data(dataset, env, action0_to_log, steer_to_log, target_pos)
        actions.append(action.copy())

        obs, reward, done, info = env.step(np.array(action))

        # check if the robot screwed up
        #TODO Fix when adding slope changes to the world.
        if np.abs(env.world.rpy[0]) > 0.05 or np.abs(env.world.rpy[1]) > 0.05:
            is_terminate = True
            break
    
    if not is_terminate:
        dataset.data_logs["lap_end"][-1] = 1 
        now = datetime.datetime.now().isoformat(timespec='milliseconds')
        file_name = "log_" + str(id) + '_' + str(now) + ".bin"
        filepath = os.path.join(datadir, file_name)

        bin_data = dataset_to_bin_array(dataset)
        bin_data.tofile(filepath)

        print("Saved Data to:", filepath)
        print("Bin shape:", bin_data.shape, "columns:", BIN_COLUMNS)

        if debug_plots:
            actions = np.array(actions)
            print(actions.shape)
            plt.plot(actions, label = "actual command")
            # plt.plot(actions[:, 1], label = "steering")
            if args.control_mode == "pure-pursuit":
                plt.plot(ppcontrol.target_velocities, label="targetvel")
            plt.legend()
            plt.show()
            plt.plot(dataset.data_logs["traj_x"], dataset.data_logs["traj_y"], label='Trajectory')
            plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='Car Position')
            plt.show()
        
        dataset.reset_logs()
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)
    return not is_terminate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--simend", type=int, default=DEFAULT_SIMEND)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--debug-plots", action="store_true")
    parser.add_argument("--control-mode", choices=("open-loop", "pure-pursuit"), default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--change-parameters", action="store_true", default=DEFAULT_CHANGE_PARAMETERS)
    parser.add_argument("--track-min-scale", type=int, default=DEFAULT_TRACK_MIN_SCALE)
    parser.add_argument("--track-max-scale", type=int, default=DEFAULT_TRACK_MAX_SCALE)
    parser.add_argument("--pure-pursuit-lower-vel", type=float, default=DEFAULT_PURE_PURSUIT_LOWER_VEL)
    parser.add_argument("--pure-pursuit-upper-vel", type=float, default=DEFAULT_PURE_PURSUIT_UPPER_VEL)
    parser.add_argument("--pure-pursuit-min-kp", type=float, default=DEFAULT_PURE_PURSUIT_MIN_KP)
    parser.add_argument("--pure-pursuit-max-kp", type=float, default=DEFAULT_PURE_PURSUIT_MAX_KP)
    parser.add_argument("--pure-pursuit-min-kd", type=float, default=DEFAULT_PURE_PURSUIT_MIN_KD)
    parser.add_argument("--pure-pursuit-max-kd", type=float, default=DEFAULT_PURE_PURSUIT_MAX_KD)
    parser.add_argument(
        "--open-loop-min-vel",
        dest="open_loop_min_vel", type=float, default=DEFAULT_OPEN_LOOP_MIN_VEL,
    )
    parser.add_argument(
        "--open-loop-max-vel",
        dest="open_loop_max_vel", type=float, default=DEFAULT_OPEN_LOOP_MAX_VEL,
    )
    parser.add_argument(
        "--open-loop-vel-segments",
        dest="open_loop_vel_segments", type=int, default=DEFAULT_OPEN_LOOP_VEL_SEGMENTS,
    )
    parser.add_argument("--open-loop-steer-step-std", type=float, default=DEFAULT_OPEN_LOOP_STEER_STEP_STD)
    parser.add_argument("--open-loop-steer-damping", type=float, default=DEFAULT_OPEN_LOOP_STEER_DAMPING)
    parser.add_argument("--wheelbase", type=float, default=DEFAULT_WHEELBASE)
    parser.add_argument("--mass-min", type=float, default=DEFAULT_MASS_RANGE[0])
    parser.add_argument("--mass-max", type=float, default=DEFAULT_MASS_RANGE[1])
    parser.add_argument("--com-center", type=float, nargs=3, default=DEFAULT_COM_CENTER.tolist())
    parser.add_argument("--com-delta", type=float, default=DEFAULT_COM_DELTA)
    parser.add_argument("--friction-min", type=float, default=DEFAULT_FRICTION_RANGE[0])
    parser.add_argument("--friction-max", type=float, default=DEFAULT_FRICTION_RANGE[1])
    parser.add_argument("--max-throttle-min", type=float, default=DEFAULT_MAX_THROTTLE_RANGE[0])
    parser.add_argument("--max-throttle-max", type=float, default=DEFAULT_MAX_THROTTLE_RANGE[1])
    parser.add_argument("--max-steer-min", type=float, default=DEFAULT_MAX_STEER_RANGE[0])
    parser.add_argument("--max-steer-max", type=float, default=DEFAULT_MAX_STEER_RANGE[1])
    parser.add_argument("--steer-bias-min", type=float, default=DEFAULT_STEER_BIAS_RANGE[0])
    parser.add_argument("--steer-bias-max", type=float, default=DEFAULT_STEER_BIAS_RANGE[1])
    parser.add_argument("--delay-min", type=int, default=DEFAULT_DELAY_RANGE[0])
    parser.add_argument("--delay-max", type=int, default=DEFAULT_DELAY_RANGE[1])
    args = parser.parse_args()
    if args.open_loop_min_vel > args.open_loop_max_vel:
        raise ValueError("--open-loop-min-vel must be <= --open-loop-max-vel")
    if args.track_min_scale >= args.track_max_scale:
        raise ValueError("--track-min-scale must be < --track-max-scale")
    for min_name, max_name in (
        ("mass_min", "mass_max"),
        ("friction_min", "friction_max"),
        ("max_throttle_min", "max_throttle_max"),
        ("max_steer_min", "max_steer_max"),
        ("steer_bias_min", "steer_bias_max"),
        ("delay_min", "delay_max"),
        ("pure_pursuit_min_kp", "pure_pursuit_max_kp"),
        ("pure_pursuit_min_kd", "pure_pursuit_max_kd"),
    ):
        if getattr(args, min_name) > getattr(args, max_name):
            raise ValueError(f"--{min_name.replace('_', '-')} must be <= --{max_name.replace('_', '-')}")

    render = not args.no_render
    debug_plots = args.debug_plots
    simend = args.simend
    episodes = args.episodes
    data_dir = args.data_dir or os.path.join(CAR_FOUNDATION_DATA_DIR, "mujoco_sim_debugging")
    os.makedirs(data_dir, exist_ok=True)

    num_success = 0
    start = time.time()
    if render or debug_plots:
        for i in range(episodes):
            ret = rollout(i, simend, render, debug_plots, data_dir, args)
            print(f"Episode {i} Complete")
            if ret:
                num_success += 1
        
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")   
    else:
        assert(render == False and debug_plots == False)
        # Let's start Ray
        ray.init()
        rollout_remote = ray.remote(rollout)
        ret = []
        for i in range(episodes):
            ret.append(rollout_remote.remote(i, simend, render, debug_plots, data_dir, args))
            print(f"Episode {i} Complete")
        output = ray.get(ret)
        # print(output)
        for ifsuccess in output:
            if ifsuccess:
                num_success+=1
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")    
