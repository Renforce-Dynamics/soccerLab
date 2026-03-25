"""Batch CSV→NPZ in one Isaac session (serial FK replay, shared scene).

Edit the constants below, then run this script with the same Python/Isaac env as ``csv_to_npz.py``.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Edit these
# ---------------------------------------------------------------------------

# (input_csv_path, output_basename_without_suffix) — same convention as --output_name in csv_to_npz.py
JOBS: list[tuple[str, str]] = [
    # ("/abs/path/to/clip1.csv", "/abs/path/to/out/clip1"),
    # ("/abs/path/to/clip2.csv", "/abs/path/to/out/clip2"),
]

ROBOT = "pi_plus"
INPUT_FPS = 30
OUTPUT_FPS = 50
# Inclusive 1-based frame range, or None for full clip (same semantics as --frame_range)
FRAME_RANGE: tuple[int, int] | None = None

# Passed to AppLauncher (add or override fields your kit expects)
_APP_DEFAULTS: dict = {
    "headless": True,
    # "device": "cuda:0",
}


# ---------------------------------------------------------------------------


def _build_app_args():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(**_APP_DEFAULTS)
    return parser.parse_args([])


if __name__ == "__main__":
    app_ns = _build_app_args()
    launcher = AppLauncher(app_ns)
    simulation_app = launcher.app

    args_cli = SimpleNamespace(
        robot=ROBOT,
        input_fps=INPUT_FPS,
        output_fps=OUTPUT_FPS,
        frame_range=FRAME_RANGE,
        device=app_ns.device,
        input_file=None,
        output_name=None,
    )

    from csv_to_npz_core import run_all

    if not JOBS:
        raise SystemExit("csv_to_npz_batch.py: set JOBS to a non-empty list of (csv_path, output_name) pairs.")

    run_all(args_cli, simulation_app, jobs=JOBS)
    print("[INFO]: Exiting program...")
