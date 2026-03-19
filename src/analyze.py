from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _series_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "peak_abs": float(np.max(np.abs(x))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    log_csv = os.path.join(args.log_dir, "log.csv")
    df = pd.read_csv(log_csv)
    out_dir = _ensure_dir(args.out_dir)

    stats: dict = {
        "torque": {},
        "joint_speed": {},
        "contact_force_truth": {},
        "contact_force_est": {},
        "contact_force_error": {},
    }

    # torque & speed stats
    for i in range(12):
        stats["torque"][f"tau_{i}"] = _series_stats(df[f"tau_{i}"].to_numpy())
        stats["joint_speed"][f"qd_{i}"] = _series_stats(df[f"qd_{i}"].to_numpy())

    # contact force stats per leg + error
    for leg in range(4):
        f_t = df[[f"f_truth_{leg}_x", f"f_truth_{leg}_y", f"f_truth_{leg}_z"]].to_numpy()
        f_e = df[[f"f_est_{leg}_x", f"f_est_{leg}_y", f"f_est_{leg}_z"]].to_numpy()
        err = f_e - f_t
        stats["contact_force_truth"][f"leg_{leg}"] = {
            "fx": _series_stats(f_t[:, 0]),
            "fy": _series_stats(f_t[:, 1]),
            "fz": _series_stats(f_t[:, 2]),
            "norm": _series_stats(np.linalg.norm(f_t, axis=1)),
        }
        stats["contact_force_est"][f"leg_{leg}"] = {
            "fx": _series_stats(f_e[:, 0]),
            "fy": _series_stats(f_e[:, 1]),
            "fz": _series_stats(f_e[:, 2]),
            "norm": _series_stats(np.linalg.norm(f_e, axis=1)),
        }
        stats["contact_force_error"][f"leg_{leg}"] = {
            "fx": _series_stats(err[:, 0]),
            "fy": _series_stats(err[:, 1]),
            "fz": _series_stats(err[:, 2]),
            "norm": _series_stats(np.linalg.norm(err, axis=1)),
        }

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    t = df["t"].to_numpy()

    # plot torques
    plt.figure(figsize=(12, 6))
    for i in range(12):
        plt.plot(t, df[f"tau_{i}"], lw=0.8, label=f"tau{i}")
    plt.xlabel("t (s)")
    plt.ylabel("Torque (N·m)")
    plt.title("Joint torques τ(t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "torque.png"), dpi=160)
    plt.close()

    # plot joint speeds
    plt.figure(figsize=(12, 6))
    for i in range(12):
        plt.plot(t, df[f"qd_{i}"], lw=0.8, label=f"qd{i}")
    plt.xlabel("t (s)")
    plt.ylabel("Joint speed (rad/s)")
    plt.title("Joint angular speeds ω(t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "joint_speed.png"), dpi=160)
    plt.close()

    # plot contact force truth vs estimate (fz)
    plt.figure(figsize=(12, 6))
    for leg in range(4):
        plt.plot(t, df[f"f_truth_{leg}_z"], lw=1.0, label=f"truth leg{leg} fz")
        plt.plot(t, df[f"f_est_{leg}_z"], lw=0.9, ls="--", label=f"est leg{leg} fz")
    plt.xlabel("t (s)")
    plt.ylabel("Fz (N)")
    plt.title("Contact normal force: truth vs Jacobian-estimated")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "contact_force_truth_vs_est.png"), dpi=160)
    plt.close()

    # plot contact state estimate vs stance_cmd (as a proxy of truth contact schedule)
    plt.figure(figsize=(12, 6))
    for leg in range(4):
        plt.plot(t, df[f"stance_cmd_{leg}"] + leg * 1.2, lw=1.0, label=f"stance_cmd leg{leg}")
        plt.plot(t, df[f"contact_est_{leg}"] + leg * 1.2, lw=0.9, ls="--", label=f"contact_est leg{leg}")
    plt.xlabel("t (s)")
    plt.ylabel("offseted binary")
    plt.title("Contact state: commanded stance vs estimated")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "contact_state.png"), dpi=160)
    plt.close()

    print(f"Saved plots + stats to: {out_dir}")


if __name__ == "__main__":
    main()

