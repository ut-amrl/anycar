import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from car_dynamics import ASSETTO_CORSA_ASSETS_DIR

AC_GYM_DIR = os.path.join(ASSETTO_CORSA_ASSETS_DIR, "assetto_corsa_gym")
sys.path.append(AC_GYM_DIR)


logger = logging.getLogger(__name__)


def config_block(track_full_name, static_info):
    track_configuration = static_info.get("TrackConfiguration") or ""
    track_length = static_info["TrackLength"]
    lines = [
        f"{track_full_name}:",
        f"  track_name: {static_info['TrackName']}",
        f"  track_configuration: {track_configuration}",
        f"  track_file: {track_full_name}.csv",
        f"  track_grid_file: {track_full_name}_0.1m.pkl",
        f"  ref_lap_file: {track_full_name}-racing_line.csv",
        f"  TrackLength: {track_length}",
    ]
    return "\n".join(lines) + "\n"


def append_track_config(config_path, track_full_name, static_info, force=False):
    with open(config_path) as f:
        tracks_config = yaml.safe_load(f) or {}

    if track_full_name in tracks_config and not force:
        raise ValueError(
            f"{track_full_name} already exists in {config_path}. "
            "Use --force-config-update to append another block anyway."
        )

    block = config_block(track_full_name, static_info)
    with open(config_path, "a") as f:
        f.write("\n" + block)

    return block


def make_client_only(config):
    from AssettoCorsaEnv.ac_client import Client

    config = copy.deepcopy(config)
    config.ego_server_host_name = config.remote_machine_ip
    config.opponents_server_host_name = config.remote_machine_ip
    config.simulation_management_server_host_name = config.remote_machine_ip
    return Client(config)


def main():
    parser = argparse.ArgumentParser(
        description="Export the currently loaded Assetto Corsa track borders and racing line."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            ASSETTO_CORSA_ASSETS_DIR,
            "assetto_corsa_gym",
            "AssettoCorsaConfigs",
            "tracks",
        ),
        help="Directory where <track>.csv and <track>-racing_line.csv will be written.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ASSETTO_CORSA_ASSETS_DIR, "config.yml"),
        help="Assetto Corsa config.yml used to find the simulation management server.",
    )
    parser.add_argument(
        "--tracks-config",
        default=None,
        help="Track config YAML to update. Defaults to <output-dir>/config.yaml.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Append a config.yaml entry for the exported track.",
    )
    parser.add_argument(
        "--force-config-update",
        action="store_true",
        help="Append the config block even if the key already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    client = make_client_only(config.AssettoCorsa)
    track_full_name, track_file, ref_lap_file, static_info = client.export_track_and_racing_line(
        output_path=str(output_dir)
    )

    logger.info("Exported track: %s", track_full_name)
    logger.info("Track CSV: %s", track_file)
    logger.info("Racing line CSV: %s", ref_lap_file)
    logger.info("TrackLength: %s", static_info["TrackLength"])

    block = config_block(track_full_name, static_info)
    if args.update_config:
        tracks_config = args.tracks_config or output_dir / "config.yaml"
        append_track_config(tracks_config, track_full_name, static_info, args.force_config_update)
        logger.info("Appended config block to %s", tracks_config)
    else:
        print("\nAdd this to tracks/config.yaml:\n")
        print(block)


if __name__ == "__main__":
    main()
