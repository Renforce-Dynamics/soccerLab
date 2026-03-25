"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    python scripts/csv_to_npz.py --robot hi --input_file source/motion/hightorque/hi/cshi_cut_T_pos.csv --input_fps 30 --output_name source/motion/hightorque/hi/npz/hi_cut_T_pos
    python scripts/csv_to_npz.py --robot pi_plus --input_file source/motion/hightorque/pi_plus/csv/pi_plus_kungfu.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/pi_plus_kungfu

    # Usage Examples:
    # For G1 robot:
    python csv_to_npz.py --robot g1 --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_name ./motions/dance1_subject2 --output_fps 50

    # For HI robot:
    python csv_to_npz.py --robot hi --input_file source/motion/hightorque/hi/csv/dance1_subject2.csv --input_fps 30 \
    --frame_range 174 424 --output_name source/motion/hightorque/hi/npz/dance1_subject2 --output_fps 50

    # For PI Plus robot:
    python csv_to_npz.py --robot pi_plus --input_file source/motion/hightorque/pi_plus/csv/dance1_subject2.csv --input_fps 30 \
    --frame_range 174 424 --output_name source/motion/hightorque/pi_plus/npz/dance1_subject2 --output_fps 50

    # For PI Plus Waist Shell robot:
    python csv_to_npz.py --robot pi_plus_waist_shell --input_file source/motion/hightorque/pi_plus_waist_shell/csv/dance1_subject2.csv --input_fps 30 \
    --frame_range 174 424 --output_name source/motion/hightorque/pi_plus_waist_shell/npz/dance1_subject2 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
    parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
    parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
            " loaded."
        ),
    )
    parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
    parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
    parser.add_argument(
        "--robot",
        type=str,
        choices=["g1", "hi", "pi_plus", "pi_plus_waist_shell", "pi_plus_head"],
        required=True,
        help="Robot type: g1 (Unitree G1), hi (Unitree Hi), pi_plus (PI Plus),pi_plus_head",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Skip WandB upload and save NPZ locally only.")
    parser.add_argument("--save_to", type=str, default="/tmp/", help="Path to save the generated npz.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from csv_to_npz_core import run_all

    run_all(args_cli, simulation_app, jobs=None)


if __name__ == "__main__":
    main()
    print("[INFO]: Exiting program...")
