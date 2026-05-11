import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


STATE_ACTION_DIM = 15
COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "vel_x", "vel_y", "vel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "throttle", "steer",
]


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("First-frame quaternion has zero norm")
    w, x, y, z = quat / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_bin(path):
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"{path} is empty")
    if data.size % STATE_ACTION_DIM != 0:
        raise ValueError(
            f"{path} has {data.size} floats, not divisible by {STATE_ACTION_DIM}"
        )
    return data.reshape(-1, STATE_ACTION_DIM)


def set_equal_xy(ax, xy):
    ax.set_aspect("equal", adjustable="box")
    if len(xy) == 0:
        return
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins)
    radius = max(0.5 * span, 1.0)
    pad = 0.1 * radius
    ax.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
    ax.set_ylim(center[1] - radius - pad, center[1] + radius + pad)


def plot_bin(path, save_path=None, show=True):
    data = load_bin(path)
    pos_world = data[:, 0:3]
    quat0_wxyz = data[0, 3:7]

    rot_world_from_body0 = quat_wxyz_to_matrix(quat0_wxyz)
    pos_body0 = (rot_world_from_body0.T @ (pos_world - pos_world[0]).T).T

    frame = np.arange(data.shape[0])
    throttle = data[:, 13]
    steer = data[:, 14]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    world_xy = pos_world[:, 0:2]
    axes[0, 0].plot(world_xy[:, 0], world_xy[:, 1], linewidth=1.5)
    axes[0, 0].scatter(world_xy[0, 0], world_xy[0, 1], c="green", label="start", zorder=3)
    axes[0, 0].scatter(world_xy[-1, 0], world_xy[-1, 1], c="red", label="end", zorder=3)
    axes[0, 0].set_title("World XY")
    axes[0, 0].set_xlabel("world x")
    axes[0, 0].set_ylabel("world y")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    set_equal_xy(axes[0, 0], world_xy)

    body_xy = pos_body0[:, 0:2]
    axes[0, 1].plot(body_xy[:, 0], body_xy[:, 1], linewidth=1.5)
    axes[0, 1].scatter(body_xy[0, 0], body_xy[0, 1], c="green", label="start", zorder=3)
    axes[0, 1].scatter(body_xy[-1, 0], body_xy[-1, 1], c="red", label="end", zorder=3)
    axes[0, 1].arrow(0, 0, 1, 0, width=0.02, length_includes_head=True, color="black")
    axes[0, 1].set_title("First-Frame Body XY")
    axes[0, 1].set_xlabel("body0 x")
    axes[0, 1].set_ylabel("body0 y")
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    set_equal_xy(axes[0, 1], body_xy)

    axes[1, 0].plot(frame, world_xy[:, 0], label="x", linewidth=1.2)
    axes[1, 0].plot(frame, world_xy[:, 1], label="y", linewidth=1.2)
    axes[1, 0].set_title("World X/Y vs Frame")
    axes[1, 0].set_xlabel("frame")
    axes[1, 0].set_ylabel("position")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(frame, body_xy[:, 0], label="x", linewidth=1.2)
    axes[1, 1].plot(frame, body_xy[:, 1], label="y", linewidth=1.2)
    axes[1, 1].set_title("Body0 X/Y vs Frame")
    axes[1, 1].set_xlabel("frame")
    axes[1, 1].set_ylabel("position")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    axes[0, 2].plot(frame, throttle, linewidth=1.2)
    axes[0, 2].set_title("Throttle vs Frame")
    axes[0, 2].set_xlabel("frame")
    axes[0, 2].set_ylabel("throttle")
    axes[0, 2].grid(True)

    axes[1, 2].plot(frame, steer, linewidth=1.2)
    axes[1, 2].set_title("Steer vs Frame")
    axes[1, 2].set_xlabel("frame")
    axes[1, 2].set_ylabel("steer")
    axes[1, 2].grid(True)

    fig.suptitle(f"{os.path.basename(path)}  ({data.shape[0]} frames)")

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Loaded {data.shape[0]} rows x {data.shape[1]} cols from {path}")
    print("Columns:", COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 15D float32 AnyCar/WheeledLab bin trajectory in world XY and first-frame body XY."
    )
    parser.add_argument("bin_file", help="Path to a raw float32 .bin file with 15 columns")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the plot image")
    parser.add_argument("--no-show", action="store_true", help="Save/print only; do not open a plot window")
    args = parser.parse_args()

    plot_bin(args.bin_file, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
