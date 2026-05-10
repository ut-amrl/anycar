import argparse
import time

from evdev import InputDevice, ecodes, list_devices


def find_xbox_event():
    for path in list_devices():
        device = InputDevice(path)
        name = device.name.lower()
        if "x-box 360" in name or "xbox 360" in name or "x-box" in name:
            return path
    raise RuntimeError("Could not find virtual Xbox controller. Is xboxdrv running?")


def tap_button(device_path, hold_time):
    device = InputDevice(device_path)
    device.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 1)
    device.write(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)
    time.sleep(hold_time)
    device.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, 0)
    device.write(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)


def main():
    parser = argparse.ArgumentParser(description="Send Xbox A / gear-up to Assetto Corsa via xboxdrv.")
    parser.add_argument("--device", default=None, help="Input event path, e.g. /dev/input/event21")
    parser.add_argument("--hold-time", type=float, default=0.15)
    args = parser.parse_args()

    device_path = args.device or find_xbox_event()
    tap_button(device_path, args.hold_time)
    print(f"sent gear up on {device_path}")


if __name__ == "__main__":
    main()
