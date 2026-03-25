"""Convert GMR pickle motion blobs (misnamed .csv) to text CSV for pi_plus."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Matches source/motion/hightorque/pi_plus/csv/*.csv header order (pi_plus_goal.csv).
_ROOT_COLS = [
    "root pos x",
    "root pos y",
    "root pos z",
    "root rot x",
    "root rot y",
    "root rot z",
    "root rot w",
]
_JOINT_NAMES = [
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "l_wrist_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "r_wrist_joint",
]
_CSV_HEADER = _ROOT_COLS + [j.replace("_joint", "") for j in _JOINT_NAMES]
_EXPECTED_DOF = len(_JOINT_NAMES)


def _is_motion_dict(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    for k in ("root_pos", "root_rot", "dof_pos"):
        if k not in obj:
            return False
    rp, rr, dof = obj["root_pos"], obj["root_rot"], obj["dof_pos"]
    if not all(isinstance(x, np.ndarray) for x in (rp, rr, dof)):
        return False
    if rp.ndim != 2 or rr.ndim != 2 or dof.ndim != 2:
        return False
    if rp.shape[1] != 3 or rr.shape[1] != 4:
        return False
    n = rp.shape[0]
    return rr.shape[0] == n and dof.shape[0] == n


def pickle_motion_to_csv(data: dict, out_path: Path) -> None:
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    if dof_pos.shape[1] != _EXPECTED_DOF:
        raise ValueError(
            f"dof_pos has {dof_pos.shape[1]} columns, expected {_EXPECTED_DOF} (pi_plus)"
        )
    if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError("root_pos, root_rot, dof_pos row counts must match")
    table = np.hstack([root_pos, root_rot, dof_pos])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(_CSV_HEADER) + "\n")
        np.savetxt(f, table, delimiter=",", fmt="%.8f")


def load_pickle_motion(path: Path) -> dict:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not _is_motion_dict(obj):
        raise ValueError(f"not a motion dict with root_pos/root_rot/dof_pos ndarrays: {path}")
    return obj


def convert_file(src: Path, dst: Path) -> None:
    data = load_pickle_motion(src)
    pickle_motion_to_csv(data, dst)
    fps = data.get("fps", "?")
    print(f"[ok] {src} -> {dst}  (frames={root_pos_frames(data)}, fps={fps})")


def root_pos_frames(data: dict) -> int:
    return int(np.asarray(data["root_pos"]).shape[0])


def collect_inputs(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return sorted(path.glob(pattern))
    raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert GMR retarget pickle files (often named *.csv) to text CSV."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Pickle motion file, or directory of files to convert",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (single input file) or output directory (directory input)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob under input directory (default: *.csv)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_text",
        help="Suffix before .csv for auto output names (default: _text)",
    )
    args = parser.parse_args()
    inp = args.input_path.resolve()
    sources = collect_inputs(inp, args.pattern)
    if not sources:
        print(f"[warn] no files matched {args.pattern!r} under {inp}", file=sys.stderr)
        return 1

    if inp.is_file():
        if args.output is None:
            out = inp.with_name(f"{inp.stem}{args.suffix}.csv")
        else:
            out = args.output.resolve()
            if out.is_dir():
                out = out / f"{inp.stem}{args.suffix}.csv"
        try:
            convert_file(inp, out)
        except Exception as e:
            print(f"[fail] {inp}: {e}", file=sys.stderr)
            return 1
        return 0

    # Directory
    out_dir: Path | None = None
    if args.output is not None:
        out_dir = args.output.resolve()
        if out_dir.exists() and not out_dir.is_dir():
            print(f"[error] --output exists but is not a directory: {out_dir}", file=sys.stderr)
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for src in sources:
        if out_dir is not None:
            dst = out_dir / f"{src.stem}{args.suffix}.csv"
        else:
            dst = src.with_name(f"{src.stem}{args.suffix}.csv")
        try:
            convert_file(src, dst)
        except Exception as e:
            print(f"[fail] {src}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
