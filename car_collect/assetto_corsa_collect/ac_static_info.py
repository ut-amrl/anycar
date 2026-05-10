import argparse
import copy
import os
import sys

from omegaconf import OmegaConf

from car_dynamics import ASSETTO_CORSA_ASSETS_DIR

AC_GYM_DIR = os.path.join(ASSETTO_CORSA_ASSETS_DIR, "assetto_corsa_gym")
sys.path.append(AC_GYM_DIR)


def make_client_only(config):
    from AssettoCorsaEnv.ac_client import Client

    config = copy.deepcopy(config)
    config.ego_server_host_name = config.remote_machine_ip
    config.opponents_server_host_name = config.remote_machine_ip
    config.simulation_management_server_host_name = config.remote_machine_ip
    return Client(config)


def main():
    parser = argparse.ArgumentParser(description="Print static info from the running Assetto Corsa session.")
    parser.add_argument(
        "--config",
        default=os.path.join(ASSETTO_CORSA_ASSETS_DIR, "config.yml"),
        help="Assetto Corsa config.yml used to find the simulation management server.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    client = make_client_only(config.AssettoCorsa)
    static_info = client.simulation_management.get_static_info()

    print(f"CarName: {static_info['CarName']}")
    print(f"TrackFullName: {static_info['TrackFullName']}")
    print(f"TrackLength: {static_info['TrackLength']}")


if __name__ == "__main__":
    main()
