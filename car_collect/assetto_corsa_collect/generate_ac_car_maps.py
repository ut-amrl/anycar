import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from car_dynamics import ASSETTO_CORSA_ASSETS_DIR
from car_dynamics.envs.assetto_corsa.assetto_corsa_gym.AssettoCorsaEnv import assettoCorsa
from car_dynamics.envs.assetto_corsa.assetto_corsa_gym.AssettoCorsaEnv.brake_map import BrakeMap


logger = logging.getLogger(__name__)


def unwrap_env(env):
    return env.env if hasattr(env, "env") else env


def main():
    parser = argparse.ArgumentParser(
        description="Generate brake_map.csv and steer_map.csv for the currently loaded Assetto Corsa car."
    )
    parser.add_argument("--config", default=os.path.join(ASSETTO_CORSA_ASSETS_DIR, "config.yml"))
    parser.add_argument(
        "--output-root",
        default=os.path.join(
            ASSETTO_CORSA_ASSETS_DIR,
            "assetto_corsa_gym",
            "AssettoCorsaConfigs",
            "cars",
        ),
    )
    parser.add_argument("--steps-per-command", type=int, default=25)
    parser.add_argument("--brake-samples", type=int, default=21)
    parser.add_argument("--force", action="store_true", help="Overwrite existing map files.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = OmegaConf.load(args.config)

    client = assettoCorsa.make_client_only(config.AssettoCorsa)
    static_info = client.simulation_management.get_static_info()
    config.AssettoCorsa.car = static_info["CarName"]
    config.AssettoCorsa.track = static_info["TrackFullName"]
    config.AssettoCorsa.enable_out_of_track_termination = False
    config.AssettoCorsa.enable_low_speed_termination = False

    car_dir = Path(args.output_root) / static_info["CarName"]
    brake_map_file = car_dir / "brake_map.csv"
    steer_map_file = car_dir / "steer_map.csv"
    if not args.force and (brake_map_file.exists() or steer_map_file.exists()):
        raise FileExistsError(f"{car_dir} already contains map files. Use --force to overwrite.")
    car_dir.mkdir(parents=True, exist_ok=True)

    env = unwrap_env(assettoCorsa.make_ac_env(cfg=config, work_dir="output"))
    try:
        logger.info("Generating maps for car %s on track %s", static_info["CarName"], static_info["TrackFullName"])

        brake_results = []
        env.reset()
        for brake_cmd in np.linspace(-1, 1, args.brake_samples):
            for _ in range(args.steps_per_command):
                env.step(np.array([0.0, -1.0, brake_cmd]))
            brake_results.append({"brake": brake_cmd, "brakeStatus": env.state["brakeStatus"]})
            logger.info("brake_cmd %.3f -> brakeStatus %.6f", brake_cmd, env.state["brakeStatus"])

        brake_map = BrakeMap(
            [r["brake"] for r in brake_results],
            [r["brakeStatus"] for r in brake_results],
            kind="cubic",
        )
        brake_map.save(brake_map_file)

        steer_results = []
        env.reset()
        for steer_cmd in [-1.0, 1.0]:
            for _ in range(args.steps_per_command):
                env.step(np.array([steer_cmd, -1.0, -1.0]))
            steer_results.append({"x_points": steer_cmd, "y_points": env.state["steerAngle"]})
            logger.info("steer_cmd %.3f -> steerAngle %.6f", steer_cmd, env.state["steerAngle"])

        pd.DataFrame(steer_results).to_csv(steer_map_file, index=False)
        logger.info("Saved %s", brake_map_file)
        logger.info("Saved %s", steer_map_file)
    finally:
        env.close()


if __name__ == "__main__":
    main()
