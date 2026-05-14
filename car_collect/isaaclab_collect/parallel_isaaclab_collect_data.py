"""Parallel Isaac Lab data collection for the F1Tenth USD.

This script intentionally does not import Isaac Sim/Lab simulation modules until
after AppLauncher starts the simulator.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
for package_dir in ("car_dynamics", "car_foundation", "car_planner", "car_dataset", "car_collect"):
    path = str(REPO_ROOT / package_dir)
    if path not in sys.path:
        sys.path.insert(0, path)

from car_dynamics import ISAAC_ASSETS_DIR
from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_planner.track_generation import change_track


BIN_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "vel_x", "vel_y", "vel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "throttle", "steer",
]

# Tunable collection defaults. Keep these near the top so data-generation
# sweeps can change one block instead of hunting through parser definitions.
DEFAULT_NUM_ENVS = 16
DEFAULT_EPISODES = 1
DEFAULT_SIMEND = 2000
DEFAULT_USD_NAME = "F1Tenth_lecar.usd"
DEFAULT_CONTROL_MODE = "open-loop"
DEFAULT_DT = 0.02
DEFAULT_PHYSICS_DT = 0.01
DEFAULT_ENV_SPACING = 0.0

DEFAULT_TRACK_POINTS = 400
DEFAULT_LOOKAHEAD = 4.0
DEFAULT_LOWER_VEL = 0.7
DEFAULT_UPPER_VEL = 2.0
DEFAULT_MIN_TRACK_SCALE = 1
DEFAULT_MAX_TRACK_SCALE = 5

DEFAULT_MIN_KP = 6.0
DEFAULT_MAX_KP = 10.0
DEFAULT_MIN_KD = 0.5
DEFAULT_MAX_KD = 1.5

DEFAULT_WHEELBASE = 0.32
DEFAULT_MAX_THROTTLE = 1.0
DEFAULT_MAX_THROTTLE_RANGE = (DEFAULT_MAX_THROTTLE, DEFAULT_MAX_THROTTLE)
DEFAULT_MAX_STEER = 0.5
DEFAULT_MAX_STEER_RANGE = (DEFAULT_MAX_STEER, DEFAULT_MAX_STEER)
DEFAULT_STEER_BIAS = 0.0
DEFAULT_STEER_BIAS_RANGE = (DEFAULT_STEER_BIAS, DEFAULT_STEER_BIAS)
DEFAULT_GROUND_STATIC_FRICTION = 0.7
DEFAULT_GROUND_DYNAMIC_FRICTION = 0.7
DEFAULT_WHEEL_STATIC_FRICTION = 1.0
DEFAULT_WHEEL_STATIC_FRICTION_RANGE = (DEFAULT_WHEEL_STATIC_FRICTION, DEFAULT_WHEEL_STATIC_FRICTION)
DEFAULT_WHEEL_DYNAMIC_FRICTION = 1.0
DEFAULT_WHEEL_DYNAMIC_FRICTION_RANGE = (DEFAULT_WHEEL_DYNAMIC_FRICTION, DEFAULT_WHEEL_DYNAMIC_FRICTION)

DEFAULT_SPEED_FILTER_ALPHA = 0.25
DEFAULT_ACTION_FILTER_ALPHA = 0.25
DEFAULT_OPEN_LOOP_MIN_VEL = -2.0
DEFAULT_OPEN_LOOP_MAX_VEL = 5.0
DEFAULT_OPEN_LOOP_VEL_SEGMENTS = 8
DEFAULT_OPEN_LOOP_STEER_STEP_STD = 0.015
DEFAULT_OPEN_LOOP_STEER_DAMPING = 0.985
DEFAULT_THROTTLE_NOISE = 0.0
DEFAULT_STEER_NOISE = 0.0
DEFAULT_SEED = 0

DRIVE_JOINTS = [
    "Wheel__Knuckle__Front_Left",
    "Wheel__Knuckle__Front_Right",
    "Wheel__Upright__Rear_Left",
    "Wheel__Upright__Rear_Right",
]
STEER_JOINTS = [
    "Knuckle__Upright__Front_Left",
    "Knuckle__Upright__Front_Right",
]


def resample_track(track: np.ndarray, num_points: int) -> np.ndarray:
    closed = np.vstack([track, track[0]])
    seg = np.diff(closed, axis=0)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    samples = np.linspace(0.0, dist[-1], num_points, endpoint=False)
    x = np.interp(samples, dist, closed[:, 0])
    y = np.interp(samples, dist, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def generate_target_velocities(totaltime: int, lowervel: float, uppervel: float) -> np.ndarray:
    samples = max(2, totaltime // 100)
    mean = lowervel + (uppervel - lowervel) * 0.5
    std_dev = (uppervel - lowervel) / 4.0
    sampled_vels = np.random.normal(mean, std_dev, samples)
    sampled_vels = np.clip(sampled_vels, lowervel, uppervel)
    x = np.linspace(0, totaltime, samples)
    spline = make_interp_spline(x, sampled_vels, k=min(3, samples - 1))
    x_total = np.arange(0, totaltime)
    return np.clip(spline(x_total), lowervel, uppervel).astype(np.float32)


def generate_bezier_velocity_commands(
    num_envs: int,
    simend: int,
    min_vel: float | np.ndarray,
    max_vel: float | np.ndarray,
    num_segments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    commands = np.zeros((num_envs, simend), dtype=np.float32)
    num_segments = max(1, num_segments)
    min_vel = np.broadcast_to(np.asarray(min_vel, dtype=np.float32), (num_envs,))
    max_vel = np.broadcast_to(np.asarray(max_vel, dtype=np.float32), (num_envs,))
    segment_edges = np.linspace(0, simend, num_segments + 1, dtype=np.int64)

    for env_id in range(num_envs):
        lo = float(min_vel[env_id])
        hi = float(max_vel[env_id])
        span = hi - lo
        knots = rng.uniform(lo, hi, size=num_segments + 1)
        for segment_id in range(num_segments):
            start = int(segment_edges[segment_id])
            end = int(segment_edges[segment_id + 1])
            if end <= start:
                continue

            p0 = knots[segment_id]
            p3 = knots[segment_id + 1]
            p1 = np.clip(p0 + rng.uniform(-0.5, 0.5) * span, lo, hi)
            p2 = np.clip(p3 + rng.uniform(-0.5, 0.5) * span, lo, hi)
            u = np.linspace(0.0, 1.0, end - start, endpoint=False)
            curve = (
                (1.0 - u) ** 3 * p0
                + 3.0 * (1.0 - u) ** 2 * u * p1
                + 3.0 * (1.0 - u) * u ** 2 * p2
                + u ** 3 * p3
            )
            commands[env_id, start:end] = curve

        commands[env_id, -1] = knots[-1]

    return commands


def generate_random_walk_steer_commands(
    num_envs: int,
    simend: int,
    max_steer: float | np.ndarray,
    step_std: float,
    damping: float,
    rng: np.random.Generator,
) -> np.ndarray:
    commands = np.zeros((num_envs, simend), dtype=np.float32)
    damping = float(np.clip(damping, 0.0, 1.0))
    max_steer = np.broadcast_to(np.asarray(max_steer, dtype=np.float32), (num_envs,))

    for env_id in range(num_envs):
        limit = float(max_steer[env_id])
        steer = rng.uniform(-0.2 * limit, 0.2 * limit)
        for t in range(simend):
            steer = damping * steer + rng.normal(0.0, step_std)
            steer = float(np.clip(steer, -limit, limit))
            commands[env_id, t] = steer

    return commands


def sample_uniform_np(num_envs: int, lo: float, hi: float,
                      rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lo, hi, size=num_envs).astype(np.float32)


def maybe_apply_legacy_fixed_arg(args: argparse.Namespace, scalar_name: str,
                                 min_name: str, max_name: str,
                                 default_scalar: float) -> None:
    """Keep legacy scalar flags meaningful when range flags are untouched."""
    if not np.isclose(getattr(args, scalar_name), default_scalar):
        default_range = (default_scalar, default_scalar)
        current_range = (getattr(args, min_name), getattr(args, max_name))
        if np.allclose(current_range, default_range):
            value = getattr(args, scalar_name)
            setattr(args, min_name, value)
            setattr(args, max_name, value)


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
    quat[:, 0] = torch.cos(0.5 * yaw)
    quat[:, 3] = torch.sin(0.5 * yaw)
    return quat


def quat_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pure_pursuit_actions(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    forward_speed: torch.Tensor,
    tracks_w: torch.Tensor,
    progress_idx: torch.Tensor,
    max_throttle: torch.Tensor,
    max_steer: torch.Tensor,
    steer_bias: torch.Tensor,
    lookahead: float,
    wheelbase: float,
    target_vel: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    last_err_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs, num_track, _ = tracks_w.shape
    yaw = quat_yaw_wxyz(quat_w)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    search_window = max(2, num_track // 4)
    target = torch.empty((num_envs, 2), device=pos_w.device, dtype=torch.float32)
    new_progress_idx = progress_idx.clone()
    offsets = torch.arange(search_window, device=pos_w.device, dtype=torch.float32)

    for env_id in range(num_envs):
        start_index = torch.floor(progress_idx[env_id]).to(torch.long)
        segment_index = start_index + offsets.to(torch.long)
        curr_index = segment_index % num_track
        next_index = (segment_index + 1) % num_track
        curr_point = tracks_w[env_id, curr_index, :2]
        next_point = tracks_w[env_id, next_index, :2]
        segment = next_point - curr_point
        rel_start = curr_point - pos_w[env_id, :2]

        a = torch.sum(segment * segment, dim=1)
        b = 2.0 * torch.sum(rel_start * segment, dim=1)
        c = torch.sum(rel_start * rel_start, dim=1) - lookahead * lookahead
        disc = b * b - 4.0 * a * c
        valid_disc = disc > 0.0

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        denom = torch.clamp(2.0 * a, min=1.0e-6)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        candidates_t = torch.stack([t1, t2], dim=1)
        candidates_progress = segment_index[:, None].to(torch.float32) + candidates_t
        valid = (
            valid_disc[:, None]
            & (candidates_t >= 0.0)
            & (candidates_t <= 1.0)
            & (candidates_progress > progress_idx[env_id])
        )

        valid_flat = valid.reshape(-1)
        if torch.any(valid_flat):
            first_valid = torch.nonzero(valid_flat, as_tuple=False)[0, 0]
            segment_slot = first_valid // 2
            candidate_t = candidates_t.reshape(-1)[first_valid]
            target[env_id] = curr_point[segment_slot] + candidate_t * segment[segment_slot]
            new_progress_idx[env_id] = candidates_progress.reshape(-1)[first_valid]
        else:
            fallback_idx = torch.floor(progress_idx[env_id]).to(torch.long) % num_track
            target[env_id] = tracks_w[env_id, fallback_idx, :2]

    target_vec_w = target[:, :2] - pos_w[:, :2]

    # Rotate target vector into each car's body frame.
    target_x_b = cos_yaw * target_vec_w[:, 0] + sin_yaw * target_vec_w[:, 1]
    target_y_b = -sin_yaw * target_vec_w[:, 0] + cos_yaw * target_vec_w[:, 1]
    alpha = torch.atan2(target_y_b, torch.clamp(target_x_b, min=1.0e-6))
    steer = torch.atan2(2.0 * wheelbase * torch.sin(alpha), torch.full_like(alpha, lookahead))
    steer = torch.clamp(steer, -max_steer, max_steer)

    err_vel = target_vel - forward_speed
    throttle = kp * err_vel + kd * (err_vel - last_err_vel)
    throttle = torch.clamp(throttle, -max_throttle, max_throttle)

    steer = torch.clamp(steer + steer_bias, -max_steer, max_steer)
    return throttle, steer, new_progress_idx, err_vel


def save_episode(datadir: str, episode: int, env_id: int, rows: list[np.ndarray]) -> None:
    if not rows:
        return
    now = datetime.datetime.now().isoformat(timespec="milliseconds")
    file_name = f"log_{episode}_env_{env_id}_{now}.bin"
    os.makedirs(datadir, exist_ok=True)
    data = np.asarray(rows, dtype=np.float32)
    assert data.ndim == 2 and data.shape[1] == len(BIN_COLUMNS), data.shape
    path = os.path.join(datadir, file_name)
    data.tofile(path)
    print(f"Saved {data.shape} to {path}")


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--simend", type=int, default=DEFAULT_SIMEND)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--usd-name", type=str, default=DEFAULT_USD_NAME)
    parser.add_argument("--control-mode", choices=("open-loop", "pure-pursuit"), default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--physics-dt", type=float, default=DEFAULT_PHYSICS_DT)
    parser.add_argument("--env-spacing", type=float, default=DEFAULT_ENV_SPACING)
    parser.add_argument("--track-points", type=int, default=DEFAULT_TRACK_POINTS)
    parser.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--lower-vel", type=float, default=DEFAULT_LOWER_VEL)
    parser.add_argument("--upper-vel", type=float, default=DEFAULT_UPPER_VEL)
    parser.add_argument("--min-track-scale", type=int, default=DEFAULT_MIN_TRACK_SCALE)
    parser.add_argument("--max-track-scale", type=int, default=DEFAULT_MAX_TRACK_SCALE)
    parser.add_argument("--min-kp", type=float, default=DEFAULT_MIN_KP)
    parser.add_argument("--max-kp", type=float, default=DEFAULT_MAX_KP)
    parser.add_argument("--min-kd", type=float, default=DEFAULT_MIN_KD)
    parser.add_argument("--max-kd", type=float, default=DEFAULT_MAX_KD)
    parser.add_argument("--wheelbase", type=float, default=DEFAULT_WHEELBASE)
    parser.add_argument("--max-throttle", type=float, default=DEFAULT_MAX_THROTTLE)
    parser.add_argument("--max-throttle-min", type=float, default=DEFAULT_MAX_THROTTLE_RANGE[0])
    parser.add_argument("--max-throttle-max", type=float, default=DEFAULT_MAX_THROTTLE_RANGE[1])
    parser.add_argument("--max-steer", type=float, default=DEFAULT_MAX_STEER)
    parser.add_argument("--max-steer-min", type=float, default=DEFAULT_MAX_STEER_RANGE[0])
    parser.add_argument("--max-steer-max", type=float, default=DEFAULT_MAX_STEER_RANGE[1])
    parser.add_argument("--steer-bias", type=float, default=DEFAULT_STEER_BIAS)
    parser.add_argument("--steer-bias-min", type=float, default=DEFAULT_STEER_BIAS_RANGE[0])
    parser.add_argument("--steer-bias-max", type=float, default=DEFAULT_STEER_BIAS_RANGE[1])
    parser.add_argument("--ground-static-friction", type=float, default=DEFAULT_GROUND_STATIC_FRICTION)
    parser.add_argument("--ground-dynamic-friction", type=float, default=DEFAULT_GROUND_DYNAMIC_FRICTION)
    parser.add_argument("--wheel-static-friction", type=float, default=DEFAULT_WHEEL_STATIC_FRICTION)
    parser.add_argument("--wheel-static-friction-min", type=float, default=DEFAULT_WHEEL_STATIC_FRICTION_RANGE[0])
    parser.add_argument("--wheel-static-friction-max", type=float, default=DEFAULT_WHEEL_STATIC_FRICTION_RANGE[1])
    parser.add_argument("--wheel-dynamic-friction", type=float, default=DEFAULT_WHEEL_DYNAMIC_FRICTION)
    parser.add_argument("--wheel-dynamic-friction-min", type=float, default=DEFAULT_WHEEL_DYNAMIC_FRICTION_RANGE[0])
    parser.add_argument("--wheel-dynamic-friction-max", type=float, default=DEFAULT_WHEEL_DYNAMIC_FRICTION_RANGE[1])
    parser.add_argument("--speed-filter-alpha", type=float, default=DEFAULT_SPEED_FILTER_ALPHA)
    parser.add_argument("--action-filter-alpha", type=float, default=DEFAULT_ACTION_FILTER_ALPHA)
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
    parser.add_argument("--throttle-noise", type=float, default=DEFAULT_THROTTLE_NOISE)
    parser.add_argument("--steer-noise", type=float, default=DEFAULT_STEER_NOISE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not hasattr(args, "headless"):
        args.headless = True

    maybe_apply_legacy_fixed_arg(
        args, "max_throttle", "max_throttle_min", "max_throttle_max", DEFAULT_MAX_THROTTLE
    )
    maybe_apply_legacy_fixed_arg(
        args, "max_steer", "max_steer_min", "max_steer_max", DEFAULT_MAX_STEER
    )
    maybe_apply_legacy_fixed_arg(
        args, "steer_bias", "steer_bias_min", "steer_bias_max", DEFAULT_STEER_BIAS
    )
    maybe_apply_legacy_fixed_arg(
        args,
        "wheel_static_friction",
        "wheel_static_friction_min",
        "wheel_static_friction_max",
        DEFAULT_WHEEL_STATIC_FRICTION,
    )
    maybe_apply_legacy_fixed_arg(
        args,
        "wheel_dynamic_friction",
        "wheel_dynamic_friction_min",
        "wheel_dynamic_friction_max",
        DEFAULT_WHEEL_DYNAMIC_FRICTION,
    )

    if args.open_loop_min_vel > args.open_loop_max_vel:
        raise ValueError("--open-loop-min-vel must be <= --open-loop-max-vel")
    for min_name, max_name in (
        ("max_throttle_min", "max_throttle_max"),
        ("max_steer_min", "max_steer_max"),
        ("steer_bias_min", "steer_bias_max"),
        ("wheel_static_friction_min", "wheel_static_friction_max"),
        ("wheel_dynamic_friction_min", "wheel_dynamic_friction_max"),
    ):
        if getattr(args, min_name) > getattr(args, max_name):
            raise ValueError(f"--{min_name.replace('_', '-')} must be <= --{max_name.replace('_', '-')}")
    if args.dt < args.physics_dt:
        raise ValueError(f"--dt ({args.dt}) must be >= --physics-dt ({args.physics_dt})")
    decimation = args.dt / args.physics_dt
    if not np.isclose(decimation, round(decimation)):
        raise ValueError(f"--dt must be an integer multiple of --physics-dt, got {args.dt} / {args.physics_dt}")
    args.decimation = int(round(decimation))
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Launching Isaac Lab app: headless={args.headless}, device={args.device}", flush=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("Isaac Lab app launched", flush=True)

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

    device = args.device
    datadir = args.data_dir or os.path.join(CAR_FOUNDATION_DATA_DIR, "isaaclab_sim_debugging")
    usd_path = str(Path(ISAAC_ASSETS_DIR) / args.usd_name)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(usd_path)
    print(f"Using USD: {usd_path}", flush=True)
    print(f"Saving data under: {datadir}", flush=True)

    sim_utils.create_new_stage()
    sim_cfg = sim_utils.SimulationCfg(dt=args.physics_dt, device=device)
    sim_cfg.gravity = (0.0, 0.0, -9.81)
    sim = sim_utils.SimulationContext(sim_cfg)
    ground_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=args.ground_static_friction,
        dynamic_friction=args.ground_dynamic_friction,
        restitution=0.0,
    )
    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=ground_material)
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    print("Building vectorized car scene", flush=True)
    sim.set_camera_view(eye=(6.0, -8.0, 6.0), target=(0.0, 0.0, 0.0))
    scene_cfg = InteractiveSceneCfg(
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        replicate_physics=True,
        filter_collisions=True,
    )
    scene = InteractiveScene(scene_cfg)

    car_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/F1Tenth",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            joint_pos={
                "Shock__Rear_Right": -0.03,
                "Shock__Rear_Left": -0.03,
                "Shock__Front_Right": 0.03,
                "Shock__Front_Left": 0.03,
            },
        ),
        actuators={
            "drive": IdealPDActuatorCfg(
                joint_names_expr=DRIVE_JOINTS,
                effort_limit=max(args.max_throttle, args.max_throttle_max),
                effort_limit_sim=max(args.max_throttle, args.max_throttle_max),
                stiffness=0.0,
                damping=0.0,
            ),
            "steer": ImplicitActuatorCfg(
                joint_names_expr=STEER_JOINTS,
                effort_limit=100.0,
                effort_limit_sim=100.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )
    cars = Articulation(car_cfg)
    scene.clone_environments(copy_from_source=False)
    scene.articulations["cars"] = cars
    origins = scene.env_origins.to(device=sim.device, dtype=torch.float32)
    sim.reset()
    scene.update(sim.cfg.dt)
    wheel_rng = np.random.default_rng(args.seed + 17)
    wheel_static_friction_np = sample_uniform_np(
        args.num_envs,
        args.wheel_static_friction_min,
        args.wheel_static_friction_max,
        wheel_rng,
    )
    wheel_dynamic_friction_np = sample_uniform_np(
        args.num_envs,
        args.wheel_dynamic_friction_min,
        args.wheel_dynamic_friction_max,
        wheel_rng,
    )
    material_props = cars.root_physx_view.get_material_properties()
    wheel_static_friction = torch.as_tensor(
        wheel_static_friction_np, device=material_props.device, dtype=material_props.dtype
    )
    wheel_dynamic_friction = torch.as_tensor(
        wheel_dynamic_friction_np, device=material_props.device, dtype=material_props.dtype
    )
    while wheel_static_friction.ndim < material_props[..., 0].ndim:
        wheel_static_friction = wheel_static_friction.unsqueeze(-1)
        wheel_dynamic_friction = wheel_dynamic_friction.unsqueeze(-1)
    material_props[..., 0] = wheel_static_friction
    material_props[..., 1] = wheel_dynamic_friction
    material_props[..., 2] = 0.0
    cars.root_physx_view.set_material_properties(material_props, torch.arange(args.num_envs, device="cpu"))
    print("Scene reset complete", flush=True)

    drive_ids, _ = cars.find_joints(DRIVE_JOINTS)
    steer_ids, _ = cars.find_joints(STEER_JOINTS)
    drive_ids = torch.as_tensor(drive_ids, device=sim.device, dtype=torch.long)
    steer_ids = torch.as_tensor(steer_ids, device=sim.device, dtype=torch.long)

    env_ids = torch.arange(args.num_envs, device=sim.device)
    rng = torch.Generator(device=sim.device)
    rng.manual_seed(args.seed)

    for episode in range(args.episodes):
        episode_rng = np.random.default_rng(args.seed + 1009 * episode)
        max_throttle_np = sample_uniform_np(
            args.num_envs, args.max_throttle_min, args.max_throttle_max, episode_rng
        )
        max_steer_np = sample_uniform_np(
            args.num_envs, args.max_steer_min, args.max_steer_max, episode_rng
        )
        steer_bias_np = sample_uniform_np(
            args.num_envs, args.steer_bias_min, args.steer_bias_max, episode_rng
        )
        max_throttle = torch.as_tensor(max_throttle_np, device=sim.device)
        max_steer = torch.as_tensor(max_steer_np, device=sim.device)
        steer_bias = torch.as_tensor(steer_bias_np, device=sim.device)

        tracks_np = np.zeros((args.num_envs, args.track_points, 2), dtype=np.float32)
        target_velocities_np = np.zeros((args.num_envs, args.simend), dtype=np.float32)
        kp_np = np.zeros(args.num_envs, dtype=np.float32)
        kd_np = np.zeros(args.num_envs, dtype=np.float32)
        origins_np = origins[:, :2].detach().cpu().numpy()
        for env_id in range(args.num_envs):
            direction = np.random.choice([-1, 1])
            scale = int(np.random.uniform(args.min_track_scale, args.max_track_scale))
            scale = max(args.min_track_scale, scale)
            track = resample_track(change_track(scale, direction), args.track_points)
            track = track - track[0]
            tracks_np[env_id] = track + origins_np[env_id]
            target_velocities_np[env_id] = generate_target_velocities(args.simend, args.lower_vel, args.upper_vel)
            kp_np[env_id] = np.random.uniform(args.min_kp, args.max_kp)
            kd_np[env_id] = np.random.uniform(args.min_kd, args.max_kd)

        tracks_w = torch.as_tensor(tracks_np, device=sim.device)
        target_velocities = torch.as_tensor(target_velocities_np, device=sim.device)
        kp = torch.as_tensor(kp_np, device=sim.device)
        kd = torch.as_tensor(kd_np, device=sim.device)
        if args.control_mode == "open-loop":
            target_velocity_commands = torch.as_tensor(
                generate_bezier_velocity_commands(
                    args.num_envs,
                    args.simend,
                    args.open_loop_min_vel,
                    args.open_loop_max_vel,
                    args.open_loop_vel_segments,
                    episode_rng,
                ),
                device=sim.device,
            )
            steer_commands = torch.as_tensor(
                generate_random_walk_steer_commands(
                    args.num_envs,
                    args.simend,
                    max_steer_np,
                    args.open_loop_steer_step_std,
                    args.open_loop_steer_damping,
                    episode_rng,
                ),
                device=sim.device,
            )

        phase_idx = torch.zeros(args.num_envs, device=sim.device, dtype=torch.long)
        next_phase_idx = (phase_idx + 1) % args.track_points
        start_xy = tracks_w[env_ids, phase_idx]
        next_xy = tracks_w[env_ids, next_phase_idx]
        start_yaw = torch.atan2(next_xy[:, 1] - start_xy[:, 1], next_xy[:, 0] - start_xy[:, 0])

        progress_idx = phase_idx.to(torch.float32)
        last_err_vel = torch.zeros(args.num_envs, device=sim.device)
        filtered_forward_speed = torch.zeros(args.num_envs, device=sim.device)
        filtered_throttle = torch.zeros(args.num_envs, device=sim.device)
        filtered_steer = torch.zeros(args.num_envs, device=sim.device)
        root_state = cars.data.default_root_state.clone()
        root_state[:, 0:2] = start_xy
        root_state[:, 2] = 0.05
        root_state[:, 3:7] = yaw_to_quat_wxyz(start_yaw)
        root_state[:, 7:13] = 0.0
        sim.reset()
        cars.write_root_state_to_sim(root_state)
        cars.write_joint_state_to_sim(
            torch.zeros_like(cars.data.default_joint_pos),
            torch.zeros_like(cars.data.default_joint_vel),
        )
        cars.reset()
        cars.update(sim.cfg.dt)

        logs: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
        active_envs = torch.ones(args.num_envs, device=sim.device, dtype=torch.bool)
        for t in tqdm(range(args.simend), desc=f"episode {episode}"):
            state = cars.data.root_state_w
            pos_w = state[:, 0:3]
            quat_w = state[:, 3:7]
            vel_w = state[:, 7:10]
            angvel_w = state[:, 10:13]
            measured_forward_speed = cars.data.root_link_lin_vel_b[:, 0]
            filtered_forward_speed = (
                args.speed_filter_alpha * measured_forward_speed
                + (1.0 - args.speed_filter_alpha) * filtered_forward_speed
            )

            if args.control_mode == "open-loop":
                target_vel = target_velocity_commands[:, t]
                err_vel = target_vel - filtered_forward_speed
                throttle = kp * err_vel + kd * (err_vel - last_err_vel)
                throttle = torch.clamp(throttle, -max_throttle, max_throttle)
                last_err_vel = err_vel
                steer = steer_commands[:, t]
                target_vel_to_log = target_vel
                target_steer_norm_to_log = torch.clamp(steer / max_steer, -1.0, 1.0)
            else:
                throttle, steer, progress_idx, last_err_vel = pure_pursuit_actions(
                    pos_w,
                    quat_w,
                    filtered_forward_speed,
                    tracks_w,
                    progress_idx,
                    max_throttle,
                    max_steer,
                    steer_bias,
                    lookahead=args.lookahead,
                    wheelbase=args.wheelbase,
                    target_vel=target_velocities[:, t],
                    kp=kp,
                    kd=kd,
                    last_err_vel=last_err_vel,
                )
                # Capture the SETPOINTS (what MPPI/planner emits), pre-filter,
                # pre-noise. The sim is still driven by the PD-output throttle
                # and filtered steer below, but the logged action mirrors what
                # a deployment-time planner would output.
                target_vel_to_log = target_velocities[:, t]                        # m/s
                target_steer_norm_to_log = torch.clamp(steer / max_steer, -1.0, 1.0)

                if args.throttle_noise > 0.0:
                    throttle = throttle + args.throttle_noise * torch.randn(
                        args.num_envs, device=sim.device, generator=rng
                    )
                    throttle = torch.clamp(throttle, -max_throttle, max_throttle)
                if args.steer_noise > 0.0:
                    steer = steer + args.steer_noise * torch.randn(args.num_envs, device=sim.device, generator=rng)
                    steer = torch.clamp(steer, -max_steer, max_steer)
                filtered_throttle = (
                    args.action_filter_alpha * throttle + (1.0 - args.action_filter_alpha) * filtered_throttle
                )
                filtered_steer = args.action_filter_alpha * steer + (1.0 - args.action_filter_alpha) * filtered_steer
                throttle = filtered_throttle
                steer = filtered_steer

            rows_t = torch.cat(
                [
                    pos_w,
                    quat_w,
                    vel_w,
                    angvel_w,
                    target_vel_to_log[:, None],
                    target_steer_norm_to_log[:, None],
                ],
                dim=1,
            )
            finite_rows = torch.isfinite(rows_t).all(dim=1)
            newly_invalid = active_envs & ~finite_rows
            if torch.any(newly_invalid):
                for env_id in torch.nonzero(newly_invalid, as_tuple=False).flatten().detach().cpu().tolist():
                    logs[env_id].clear()
                active_envs = active_envs & finite_rows

            rows = rows_t.detach().cpu().numpy().astype(np.float32)
            for env_id, row in enumerate(rows):
                if bool(active_envs[env_id].item()):
                    logs[env_id].append(row)

            valid_controls = active_envs & torch.isfinite(throttle) & torch.isfinite(steer)
            throttle = torch.where(valid_controls, throttle, torch.zeros_like(throttle))
            steer = torch.where(valid_controls, steer, torch.zeros_like(steer))
            cars.set_joint_effort_target(throttle[:, None].repeat(1, len(drive_ids)), joint_ids=drive_ids)
            cars.set_joint_position_target(steer[:, None].repeat(1, len(steer_ids)), joint_ids=steer_ids)
            for _ in range(args.decimation):
                cars.write_data_to_sim()
                sim.step(render=not args.headless)
                cars.update(sim.cfg.dt)

        for env_id, rows in enumerate(logs):
            save_episode(datadir, episode, env_id, rows)

    print("Closing Isaac Lab app", flush=True)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
