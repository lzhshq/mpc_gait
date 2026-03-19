from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from .contact_estimation import estimate_contact_forces_from_tau, estimate_contact_state_from_force
from .controller_qp import GaitParams, QPGaitController
from .utils_math import LegIndex, quat_to_rpy


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _joint_info(body_id: int, j: int) -> dict:
    info = p.getJointInfo(body_id, j)
    return {
        "jointIndex": info[0],
        "jointName": info[1].decode("utf-8"),
        "jointType": info[2],
        "qIndex": info[3],
        "uIndex": info[4],
        "flags": info[5],
        "jointDamping": info[6],
        "jointFriction": info[7],
        "jointLowerLimit": info[8],
        "jointUpperLimit": info[9],
        "jointMaxForce": info[10],
        "jointMaxVelocity": info[11],
        "linkName": info[12].decode("utf-8"),
        "jointAxis": info[13],
        "parentFramePos": info[14],
        "parentFrameOrn": info[15],
        "parentIndex": info[16],
    }


def _build_laikago_mapping(body_id: int) -> tuple[list[int], list[int], list[slice], np.ndarray]:
    """
    Returns:
    - joint_indices (12) ordered as [FL(3), FR(3), RL(3), RR(3)]
    - foot_link_indices (4) ordered as [FL, FR, RL, RR]
    - leg_joint_slices: 4 slices into tau_12 vector
    - nominal_foot_pos_base (4,3) from initial link states in base frame (yaw ignored later)
    """
    leg = LegIndex()
    prefixes = {leg.FL: "FL_", leg.FR: "FR_", leg.RL: "RL_", leg.RR: "RR_"}
    joint_names = {i: [] for i in range(4)}
    joint_ids = {i: [] for i in range(4)}
    foot_link = {i: None for i in range(4)}

    num_j = p.getNumJoints(body_id)
    for j in range(num_j):
        ji = _joint_info(body_id, j)
        name = ji["jointName"]
        link = ji["linkName"]
        # motor joints are revolute
        if ji["jointType"] != p.JOINT_REVOLUTE:
            # but we still try to find foot links by name
            for lid, pref in prefixes.items():
                if link.startswith(pref) and ("foot" in link or "toe" in link):
                    foot_link[lid] = j
            continue

        for lid, pref in prefixes.items():
            if name.startswith(pref):
                joint_names[lid].append(name)
                joint_ids[lid].append(j)

    # Sort joints per leg into hip/upper/lower order by keywords
    def sort_leg(names: list[str], ids: list[int]) -> list[int]:
        pairs = list(zip(names, ids))
        def key(n: str) -> int:
            n = n.lower()
            if "hip" in n:
                return 0
            if "upper" in n:
                return 1
            if "lower" in n or "knee" in n:
                return 2
            return 99
        pairs.sort(key=lambda x: key(x[0]))
        return [pid for _n, pid in pairs]

    ordered_joint_indices: list[int] = []
    for lid in (leg.FL, leg.FR, leg.RL, leg.RR):
        ids_sorted = sort_leg(joint_names[lid], joint_ids[lid])
        if len(ids_sorted) != 3:
            raise RuntimeError(f"Expected 3 revolute joints for leg {lid}, got {len(ids_sorted)}: {joint_names[lid]}")
        ordered_joint_indices.extend(ids_sorted)

    # foot link indices: fallback to last link of lower joint if not found
    if any(v is None for v in foot_link.values()):
        # for each leg, find the child link of the last joint (lower)
        for lid, pref in prefixes.items():
            if foot_link[lid] is not None:
                continue
            lower_joint = sort_leg(joint_names[lid], joint_ids[lid])[-1]
            ji = _joint_info(body_id, lower_joint)
            # child link index is the joint index itself in Bullet's convention
            foot_link[lid] = lower_joint

    foot_link_indices = [int(foot_link[lid]) for lid in (leg.FL, leg.FR, leg.RL, leg.RR)]

    leg_joint_slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]

    # nominal foot pos in base frame from initial pose
    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    base_pos = np.array(base_pos, dtype=float)
    base_orn = np.array(base_orn, dtype=float)
    # base frame rotation world->base is R^T
    R = np.array(p.getMatrixFromQuaternion(base_orn), dtype=float).reshape(3, 3)
    nominal = np.zeros((4, 3), dtype=float)
    for i, link_id in enumerate(foot_link_indices):
        ls = p.getLinkState(body_id, link_id, computeForwardKinematics=True)
        foot_w = np.array(ls[4], dtype=float)
        nominal[i] = R.T @ (foot_w - base_pos)

    return ordered_joint_indices, foot_link_indices, leg_joint_slices, nominal


def _get_all_joint_state(body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_j = p.getNumJoints(body_id)
    q = np.zeros(num_j, dtype=float)
    qd = np.zeros(num_j, dtype=float)
    tau = np.zeros(num_j, dtype=float)
    for j in range(num_j):
        st = p.getJointState(body_id, j)
        q[j] = st[0]
        qd[j] = st[1]
        tau[j] = st[3]
    return q, qd, tau


def _foot_truth_contact_force(body_id: int, foot_link_id: int, ground_id: int) -> np.ndarray:
    """
    Sum all contact forces acting on this foot link from ground, in world frame.
    """
    cps = p.getContactPoints(bodyA=body_id, bodyB=ground_id, linkIndexA=foot_link_id)
    f = np.zeros(3, dtype=float)
    for cp in cps:
        # cp fields (Bullet): normalForce, lateralFriction1, lateralFriction2, lateralDir1, lateralDir2, contactNormalOnB
        normal_force = float(cp[9])
        lateral1 = float(cp[10])
        lateral2 = float(cp[12])
        lateral_dir1 = np.array(cp[11], dtype=float)
        lateral_dir2 = np.array(cp[13], dtype=float)
        # normal points from B to A at contact on B
        n_on_b = np.array(cp[7], dtype=float)
        # Force on A from B:
        f += normal_force * n_on_b + lateral1 * lateral_dir1 + lateral2 * lateral_dir2
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", type=int, default=0)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--log_dir", type=str, default="runs/run1")
    ap.add_argument("--terrain", type=str, default="plane", choices=["plane"])
    args = ap.parse_args()

    log_dir = _ensure_dir(args.log_dir)

    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    dt = 1.0 / 240.0
    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=50)

    ground_id = p.loadURDF("plane.urdf")
    # Laikago from pybullet_data/quadruped/laikago/laikago_toes.urdf
    laikago_urdf = os.path.join(pybullet_data.getDataPath(), "quadruped/laikago/laikago_toes.urdf")
    start_pos = [0, 0, 0.48]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(laikago_urdf, start_pos, start_orn, useFixedBase=False)

    # disable default motors (velocity control)
    num_j = p.getNumJoints(robot_id)
    for j in range(num_j):
        ji = p.getJointInfo(robot_id, j)
        if ji[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0)

    joint_indices, foot_link_indices, leg_joint_slices, nominal_foot_base = _build_laikago_mapping(robot_id)

    dyn = p.getDynamicsInfo(robot_id, -1)
    mass = float(dyn[0])
    inertia_diag = np.array(dyn[2], dtype=float)
    inertia_base = np.diag(inertia_diag)

    gait = GaitParams()
    ctrl = QPGaitController(
        dt=dt,
        joint_indices=joint_indices,
        foot_link_indices=foot_link_indices,
        nominal_foot_pos_base=nominal_foot_base,
        mass=mass,
        inertia_base=inertia_base,
        gait=gait,
    )

    # initial foot states
    foot_pos_w0 = []
    for link_id in foot_link_indices:
        ls = p.getLinkState(robot_id, link_id, computeForwardKinematics=True)
        foot_pos_w0.append(ls[4])
    ctrl.reset(0.0, np.array(foot_pos_w0, dtype=float))

    # log
    meta = {
        "dt": dt,
        "duration": args.duration,
        "speed_cmd": args.speed,
        "joint_indices": joint_indices,
        "foot_link_indices": foot_link_indices,
        "gait": asdict(gait),
        "urdf": laikago_urdf,
    }
    with open(os.path.join(log_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    csv_path = os.path.join(log_dir, "log.csv")
    fieldnames = (
        ["t"]
        + [f"base_p_{k}" for k in "xyz"]
        + [f"base_v_{k}" for k in "xyz"]
        + [f"base_rpy_{k}" for k in ("r", "p", "y")]
        + [f"q_{i}" for i in range(12)]
        + [f"qd_{i}" for i in range(12)]
        + [f"tau_{i}" for i in range(12)]
        + [f"stance_cmd_{i}" for i in range(4)]
        + [f"f_truth_{i}_{k}" for i in range(4) for k in "xyz"]
        + [f"f_est_{i}_{k}" for i in range(4) for k in "xyz"]
        + [f"contact_est_{i}" for i in range(4)]
    )
    fp = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()

    steps = int(args.duration / dt)
    v_cmd_xy = np.array([args.speed, 0.0], dtype=float)
    z_cmd = 0.48

    for step in range(steps):
        t = step * dt

        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        base_vel_lin, base_vel_ang = p.getBaseVelocity(robot_id)
        base_pos = np.array(base_pos, dtype=float)
        base_orn = np.array(base_orn, dtype=float)
        base_vel = np.array(base_vel_lin, dtype=float)
        base_omega = np.array(base_vel_ang, dtype=float)
        base_rpy = quat_to_rpy(base_orn)

        q_all, qd_all, tau_all = _get_all_joint_state(robot_id)
        q12 = np.array([q_all[j] for j in joint_indices], dtype=float)
        qd12 = np.array([qd_all[j] for j in joint_indices], dtype=float)
        tau12_meas = np.array([tau_all[j] for j in joint_indices], dtype=float)

        foot_pos_world = np.zeros((4, 3), dtype=float)
        J_foot = np.zeros((4, 3, 12), dtype=float)
        for i, link_id in enumerate(foot_link_indices):
            ls = p.getLinkState(robot_id, link_id, computeForwardKinematics=True, computeLinkVelocity=True)
            foot_pos_world[i] = np.array(ls[4], dtype=float)

            # jacobian in world frame for this link origin
            linJ, _angJ = p.calculateJacobian(
                robot_id,
                link_id,
                localPosition=[0, 0, 0],
                objPositions=q_all.tolist(),
                objVelocities=qd_all.tolist(),
                objAccelerations=[0.0] * len(q_all),
            )
            linJ = np.array(linJ, dtype=float)  # (3, num_joints)
            # reorder/select columns for our 12 motor joints
            J_foot[i] = linJ[:, joint_indices]

        tau_cmd, dbg = ctrl.step(
            base_pos=base_pos,
            base_quat_xyzw=base_orn,
            base_vel=base_vel,
            base_omega=base_omega,
            q=q12,
            qd=qd12,
            foot_pos_world=foot_pos_world,
            foot_jacobians=J_foot,
            v_cmd_xy=v_cmd_xy,
            z_cmd=z_cmd,
        )

        p.setJointMotorControlArray(
            robot_id,
            joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=tau_cmd.tolist(),
        )

        p.stepSimulation()

        # truth contact forces from Bullet
        f_truth = np.zeros((4, 3), dtype=float)
        for i, link_id in enumerate(foot_link_indices):
            f_truth[i] = _foot_truth_contact_force(robot_id, link_id, ground_id)

        # estimated forces / contact state (from measured tau)
        f_est = estimate_contact_forces_from_tau(tau12_meas, J_foot, leg_joint_slices)
        contact_est = estimate_contact_state_from_force(f_est, fz_threshold=20.0)

        row = {
            "t": t,
            "base_p_x": base_pos[0],
            "base_p_y": base_pos[1],
            "base_p_z": base_pos[2],
            "base_v_x": base_vel[0],
            "base_v_y": base_vel[1],
            "base_v_z": base_vel[2],
            "base_rpy_r": base_rpy[0],
            "base_rpy_p": base_rpy[1],
            "base_rpy_y": base_rpy[2],
        }
        row.update({f"q_{i}": q12[i] for i in range(12)})
        row.update({f"qd_{i}": qd12[i] for i in range(12)})
        row.update({f"tau_{i}": tau12_meas[i] for i in range(12)})
        stance_cmd = np.asarray(dbg["stance"], dtype=int).reshape(4)
        row.update({f"stance_cmd_{i}": int(stance_cmd[i]) for i in range(4)})
        for i in range(4):
            for k, ax in enumerate("xyz"):
                row[f"f_truth_{i}_{ax}"] = f_truth[i, k]
                row[f"f_est_{i}_{ax}"] = f_est[i, k]
            row[f"contact_est_{i}"] = int(contact_est[i])

        writer.writerow(row)

    fp.close()
    p.disconnect(cid)
    print(f"Saved log to: {csv_path}")


if __name__ == "__main__":
    main()

