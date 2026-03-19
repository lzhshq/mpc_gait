from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import mujoco.viewer

from .contact_estimation import estimate_contact_forces_from_tau, estimate_contact_state_from_force
from .controller_qp import ControlGains, GaitParams, QPGaitController
from .utils_math import wrap_to_pi


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=float).reshape(4)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def _rpy_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    # Mujoco stores quaternion as w,x,y,z
    w, x, y, z = [float(v) for v in q_wxyz]
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2) if abs(sinp) >= 1 else np.arcsin(sinp)
    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)


def _build_mapping(model: mujoco.MjModel) -> tuple[list[str], list[int], list[int], list[slice], np.ndarray]:
    """
    Returns:
    - motor_joint_names ordered [FL(3), FR(3), RL(3), RR(3)]
    - joint_qpos_adr indices into qpos for these joints (12)
    - joint_dof_adr indices into qvel for these joints (12)
    - leg_joint_slices in tau_12
    - nominal_foot_pos_base (4,3) from initial pose in base frame
    """
    joint_names = [
        "FL_hip_abd", "FL_hip_flex", "FL_knee",
        "FR_hip_abd", "FR_hip_flex", "FR_knee",
        "RL_hip_abd", "RL_hip_flex", "RL_knee",
        "RR_hip_abd", "RR_hip_flex", "RR_knee",
    ]
    qpos_adr = []
    dof_adr = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"Joint not found in model: {jn}")
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        dof_adr.append(int(model.jnt_dofadr[jid]))

    leg_joint_slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]

    # nominal foot position in base frame: use sites
    foot_sites = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sn) for sn in foot_sites]
    if any(s < 0 for s in site_ids):
        raise RuntimeError("Foot sites missing in XML.")

    data0 = mujoco.MjData(model)
    mujoco.mj_forward(model, data0)

    base_pos = data0.qpos[0:3].copy()
    base_quat_wxyz = data0.qpos[3:7].copy()
    # world->base rotation: use mujoco quaternion
    R = np.zeros((3, 3), dtype=float)
    mujoco.mju_quat2Mat(R.ravel(), base_quat_wxyz)
    nominal = np.zeros((4, 3), dtype=float)
    for i, sid in enumerate(site_ids):
        foot_w = data0.site_xpos[sid].copy()
        nominal[i] = R.T @ (foot_w - base_pos)

    return joint_names, qpos_adr, dof_adr, leg_joint_slices, nominal


def _find_a1_foot_geoms(model: mujoco.MjModel) -> list[int]:
    """
    A1 XML foot geoms are unnamed. We identify them as:
    - geom type sphere
    - attached to body {FL_calf, FR_calf, RL_calf, RR_calf}
    If multiple match, pick the sphere with smallest radius (~0.02) and lowest relative z.
    Order returned: [FL, FR, RL, RR]
    """
    calf_bodies = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    bid = {bn: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn) for bn in calf_bodies}
    if any(v < 0 for v in bid.values()):
        missing = [k for k, v in bid.items() if v < 0]
        raise RuntimeError(f"A1 calf bodies not found: {missing}")

    out: list[int] = []
    for bn in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
        body_id = bid[bn]
        candidates = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != int(body_id):
                continue
            if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                continue
            # geom_size[gid,0] is radius for sphere
            r = float(model.geom_size[gid][0])
            # geom_pos is local position in body frame
            zloc = float(model.geom_pos[gid][2])
            candidates.append((abs(r - 0.02), zloc, gid))
        if not candidates:
            raise RuntimeError(f"No sphere geom found on body {bn} (cannot locate foot).")
        # prefer radius near 0.02, then lowest z local (more 'foot-like')
        candidates.sort(key=lambda x: (x[0], x[1]))
        out.append(int(candidates[0][2]))
    return out


def _build_mapping_a1(model: mujoco.MjModel) -> tuple[list[str], list[int], list[int], list[slice], np.ndarray, list[int], int]:
    """
    Returns:
    - joint_names ordered [FL(3), FR(3), RL(3), RR(3)] (12)
    - joint qpos/qvel addresses (12)
    - leg_joint_slices
    - nominal foot positions in trunk frame (4,3)
    - foot geom ids (4) ordered [FL, FR, RL, RR]
    - trunk body id
    """
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    qpos_adr: list[int] = []
    dof_adr: list[int] = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"A1 joint not found: {jn}")
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        dof_adr.append(int(model.jnt_dofadr[jid]))

    leg_joint_slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]

    trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_bid < 0:
        raise RuntimeError("A1 body 'trunk' not found.")

    foot_geom_ids = _find_a1_foot_geoms(model)

    # nominal feet in trunk frame at initial pose
    data0 = mujoco.MjData(model)
    mujoco.mj_forward(model, data0)
    trunk_pos = data0.xpos[trunk_bid].copy()
    trunk_quat_wxyz = data0.xquat[trunk_bid].copy()
    R = np.zeros((3, 3), dtype=float)
    mujoco.mju_quat2Mat(R.ravel(), trunk_quat_wxyz)
    nominal = np.zeros((4, 3), dtype=float)
    for i, gid in enumerate(foot_geom_ids):
        foot_w = data0.geom_xpos[gid].copy()
        nominal[i] = R.T @ (foot_w - trunk_pos)

    return joint_names, qpos_adr, dof_adr, leg_joint_slices, nominal, foot_geom_ids, int(trunk_bid)


def _a1_actuator_order_indices(model: mujoco.MjModel) -> dict[str, int]:
    """
    Returns actuator index by name for A1 scene.
    """
    names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    ]
    out = {}
    for n in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            raise RuntimeError(f"A1 actuator not found: {n}")
        out[n] = int(aid)
    return out

def _foot_truth_contact_force_world(model: mujoco.MjModel, data: mujoco.MjData, foot_geom_id: int) -> np.ndarray:
    """
    Sum all contact forces on a given foot geom (world frame).
    Uses mj_contactForce (contact frame) and transforms to world via contact frame axes.
    """
    f_world = np.zeros(3, dtype=float)
    for ci in range(data.ncon):
        con = data.contact[ci]
        if con.geom1 != foot_geom_id and con.geom2 != foot_geom_id:
            continue
        # 6D force/torque in contact frame
        cf = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, ci, cf)
        # contact frame orientation in world: con.frame is 3x3 row-major, x-axis is normal
        R = np.array(con.frame, dtype=float).reshape(3, 3)
        # MuJoCo returns force on geom1. If foot is geom2, flip the sign.
        f_c = R @ cf[0:3]
        if con.geom1 == foot_geom_id:
            f_world += f_c
        else:
            f_world -= f_c
    return f_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", type=int, default=0, help="1: show viewer (needs GPU/desktop), 0: headless")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--log_dir", type=str, default="runs/run1")
    ap.add_argument("--model", type=str, default=os.path.join("assets", "quadruped.xml"))
    ap.add_argument("--model_preset", type=str, default="simple", choices=["simple", "a1", "a1_torque"])
    ap.add_argument("--controller", type=str, default="auto", choices=["auto", "qp", "joint_trot"])
    ap.add_argument("--trot_thigh0", type=float, default=None, help="joint_trot baseline thigh angle (rad), A1 only")
    ap.add_argument("--trot_knee0", type=float, default=None, help="joint_trot baseline knee angle (rad), A1 only")
    args = ap.parse_args()

    log_dir = _ensure_dir(args.log_dir)

    if args.model_preset == "a1":
        model_path = os.path.join("assets", "mujoco_menagerie", "unitree_a1", "scene.xml")
    elif args.model_preset == "a1_torque":
        model_path = os.path.join("assets", "mujoco_menagerie", "unitree_a1", "scene_torque.xml")
    else:
        model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)

    if args.model_preset in ("a1", "a1_torque"):
        joint_names, qpos_adr, dof_adr, leg_joint_slices, nominal_foot_base, foot_geom_ids, trunk_bid = _build_mapping_a1(model)
        # Reset to keyframe 'home' so the robot starts standing.
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid < 0:
            raise RuntimeError("A1 keyframe 'home' not found.")
        mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)
        a1_act_idx = _a1_actuator_order_indices(model)
        ctrl_home = data.ctrl.copy()
        # baseline joint targets in our controller order (FL,FR,RL,RR)
        q_home = np.array([0.0, 0.9, -1.8] * 4, dtype=float)
        # desired nominal joint pose (controller order FL,FR,RL,RR)
        q_nom = np.array([0.0, 0.9, -1.8] * 4, dtype=float)
    else:
        joint_names, qpos_adr, dof_adr, leg_joint_slices, nominal_foot_base = _build_mapping(model)
        foot_geom_ids = []
        trunk_bid = -1

    # set a slightly crouched initial posture to avoid singularities
    q_nom = np.zeros(12, dtype=float)
    for leg_i in range(4):
        q_nom[3 * leg_i + 0] = 0.0   # hip_abd
        q_nom[3 * leg_i + 1] = 0.65  # hip_flex
        q_nom[3 * leg_i + 2] = -1.30 # knee
    for i in range(12):
        data.qpos[qpos_adr[i]] = q_nom[i]
        data.qvel[dof_adr[i]] = 0.0

    if args.model_preset == "a1_torque":
        # stability-first gait
        gait = GaitParams(period=0.38, duty=0.58, swing_height=0.09, mu=0.8, fz_max=900.0)
    else:
        gait = GaitParams()
    if args.model_preset == "a1":
        gains = ControlGains(
            kp_v_xy=8.0,
            kd_v_xy=1.0,
            kp_z=80.0,
            kd_z=10.0,
            kp_rp=70.0,
            kd_rp=8.0,
            kp_swing=140.0,
            kd_swing=10.0,
            kp_stance=120.0,
            kd_stance=12.0,
            k_stance_anchor=1.2,
            kd_joint=0.6,
        )
    elif args.model_preset == "a1_torque":
        gains = ControlGains(
            kp_v_xy=120.0,
            kd_v_xy=2.5,
            kp_z=120.0,
            kd_z=14.0,
            kp_rp=90.0,
            kd_rp=10.0,
            kp_swing=220.0,
            kd_swing=16.0,
            kp_stance=0.0,
            kd_stance=0.0,
            k_stance_anchor=0.0,
            kd_joint=0.8,
        )
    else:
        gains = None
    if args.model_preset in ("a1", "a1_torque"):
        base_bid = trunk_bid
    else:
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_bid < 0:
            raise RuntimeError("Body 'base' not found in model.")
    total_mass = float(np.sum(model.body_mass))
    ctrl = QPGaitController(
        dt=dt,
        joint_indices=list(range(12)),
        foot_link_indices=[0, 1, 2, 3],  # unused in mujoco version
        nominal_foot_pos_base=nominal_foot_base,
        mass=total_mass,
        inertia_base=np.diag(model.body_inertia[base_bid]),
        gait=gait,
        gains=gains,
    )

    mujoco.mj_forward(model, data)
    if args.model_preset in ("a1", "a1_torque"):
        site_ids = []
        foot_pos_w0 = np.array([data.geom_xpos[gid].copy() for gid in foot_geom_ids], dtype=float)
    else:
        foot_sites = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sn) for sn in foot_sites]
        foot_geoms = ["FL_foot_geom", "FR_foot_geom", "RL_foot_geom", "RR_foot_geom"]
        foot_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gn) for gn in foot_geoms]
        foot_pos_w0 = np.array([data.site_xpos[sid].copy() for sid in site_ids], dtype=float)
    ctrl.reset(0.0, foot_pos_w0)

    meta = {
        "dt": dt,
        "duration": args.duration,
        "speed_cmd": args.speed,
        "joint_names": joint_names,
        "qpos_adr": qpos_adr,
        "dof_adr": dof_adr,
        "gait": asdict(gait),
        "model_xml": model_path,
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

    # Simple model used -X as forward; A1 uses +X forward in standard scene.
    # In these scenes the camera points such that forward is approximately -X.
    v_cmd_xy = np.array([-args.speed, 0.0], dtype=float)
    # Keep a fixed height target (from A1 keyframe home) for robust stance.
    z_cmd = 0.27

    steps = int(args.duration / dt)

    viewer = None
    if args.gui:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print("[GUI] Failed to launch MuJoCo viewer.")
            print("[GUI] Common causes: missing GPU/OpenGL support, running in non-desktop session, or graphics driver issues.")
            print(f"[GUI] Error: {type(e).__name__}: {e}")
            viewer = None

    # For joint_trot straight-line correction
    yaw0 = None
    y0 = None

    for step in range(steps):
        t = step * dt

        mujoco.mj_forward(model, data)

        # stance detection from contact presence (robust to force sign)
        stance_truth = np.zeros(4, dtype=bool)
        if foot_geom_ids:
            for ci in range(data.ncon):
                con = data.contact[ci]
                for li, gid in enumerate(foot_geom_ids):
                    if con.geom1 == gid or con.geom2 == gid:
                        stance_truth[li] = True

        # Use freejoint state for base pose/vel (works for both models).
        base_pos = data.qpos[0:3].copy()
        base_quat_wxyz = data.qpos[3:7].copy()
        base_vel = data.qvel[0:3].copy()
        base_omega = data.qvel[3:6].copy()
        base_rpy = _rpy_from_quat_wxyz(base_quat_wxyz)

        q12 = np.array([data.qpos[qa] for qa in qpos_adr], dtype=float)
        qd12 = np.array([data.qvel[da] for da in dof_adr], dtype=float)

        # foot positions and jacobians
        if args.model_preset in ("a1", "a1_torque"):
            foot_pos_world = np.array([data.geom_xpos[gid].copy() for gid in foot_geom_ids], dtype=float)
            J_foot = np.zeros((4, 3, model.nv), dtype=float)
            for i, gid in enumerate(foot_geom_ids):
                jacp = np.zeros((3, model.nv), dtype=float)
                jacr = np.zeros((3, model.nv), dtype=float)
                mujoco.mj_jacGeom(model, data, jacp, jacr, gid)
                J_foot[i] = jacp
        else:
            foot_pos_world = np.array([data.site_xpos[sid].copy() for sid in site_ids], dtype=float)
            J_foot = np.zeros((4, 3, model.nv), dtype=float)
            for i, sid in enumerate(site_ids):
                jacp = np.zeros((3, model.nv), dtype=float)
                jacr = np.zeros((3, model.nv), dtype=float)
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                J_foot[i] = jacp
        # select columns for our 12 joints (dof_adr)
        J_sel = np.zeros((4, 3, 12), dtype=float)
        for j in range(12):
            J_sel[:, :, j] = J_foot[:, :, dof_adr[j]]

        controller_mode = args.controller
        if controller_mode == "auto":
            controller_mode = "joint_trot" if args.model_preset == "a1" else "qp"

        if controller_mode == "qp":
            # warm-up: stabilize to nominal posture briefly
            if t < 0.6:
                tau_cmd = np.zeros(12, dtype=float)
                dbg = {"stance": np.array([1, 1, 1, 1], dtype=int)}
            else:
                tau_cmd, dbg = ctrl.step(
                    base_pos=base_pos,
                    base_quat_xyzw=_quat_wxyz_to_xyzw(base_quat_wxyz),
                    base_vel=base_vel,
                    base_omega=base_omega,
                    q=q12,
                    qd=qd12,
                    foot_pos_world=foot_pos_world,
                    foot_jacobians=J_sel,
                    v_cmd_xy=v_cmd_xy,
                    z_cmd=z_cmd,
                    stance_override=stance_truth,
                )
                # ramp in QP output to avoid a sudden kick at t=0.6s
                ramp = float(np.clip((t - 0.6) / 0.6, 0.0, 1.0))
                tau_cmd = ramp * tau_cmd
        else:
            # A1-friendly joint-space trot. We drive position actuators directly.
            # This reliably "runs" and still lets us export τ(t)=actuator_force and contact forces.
            # Calibrated gait parameters: map --speed to step amplitude & period.
            vmag = float(abs(args.speed))
            high_speed = vmag >= 0.30
            if high_speed:
                # "Speed=0.4 feels best" profile
                period = float(np.clip(0.50 - 0.18 * vmag, 0.32, 0.52))
                duty = 0.62
                thigh0 = 0.88 if args.trot_thigh0 is None else float(args.trot_thigh0)
                knee0 = -1.55 if args.trot_knee0 is None else float(args.trot_knee0)
                step_amp = float(np.clip(0.12 + 0.42 * vmag, 0.10, 0.35))
                lift_knee = 0.35
                # Reduce outward splay slightly but keep stability
                abd_bias_mag = 0.03
                k_yaw = 1.5
                yaw_lim = 0.4
            else:
                # Low-speed (0.2 m/s) stability profile
                period = float(np.clip(0.56 - 0.08 * vmag, 0.48, 0.56))
                duty = 0.62
                thigh0 = 0.82 if args.trot_thigh0 is None else float(args.trot_thigh0)
                knee0 = -1.52 if args.trot_knee0 is None else float(args.trot_knee0)
                step_amp = float(np.clip(0.08 + 0.40 * vmag, 0.08, 0.18))
                lift_knee = 0.35
                abd_bias_mag = 0.03
                k_yaw = 1.2
                yaw_lim = 0.35

            abd0 = 0.0

            # Smoothly blend from standing (home) to running targets to avoid the initial snap
            ramp_t = 0.6
            alpha = float(np.clip(t / max(1e-6, ramp_t), 0.0, 1.0))
            # Heading correction to reduce route drift.
            # Avoid a start-up "kick" by ramping corrections in smoothly.
            # Steering corrections OFF (user requested): no yaw/vy correction.
            steer_alpha = 0.0
            yaw_c = 0.0
            y_c = 0.0

            def leg_phase(leg_id: int) -> float:
                # 0: FL, 1: FR, 2: RL, 3: RR
                ph = (t % period) / period
                if leg_id in (0, 3):  # FL, RR
                    return ph
                return (ph + 0.5) % 1.0

            q_target = q_home.copy() if args.model_preset == "a1" else q12.copy()
            stance_cmd = np.zeros(4, dtype=int)
            for leg_id in range(4):
                ph = leg_phase(leg_id)
                in_stance = ph < duty
                stance_cmd[leg_id] = 1 if in_stance else 0
                thigh_base = (1.0 - alpha) * 0.9 + alpha * thigh0
                knee_base = (1.0 - alpha) * -1.8 + alpha * knee0
                step_a = alpha * step_amp
                lift_a = alpha * lift_knee
                if in_stance:
                    s = ph / max(1e-6, duty)
                    # sweep backward during stance (propulsion)
                    # (flip sign here if you observe backwards walking)
                    thigh = thigh_base + step_a * (2 * s - 1)
                    knee = knee_base
                else:
                    s = (ph - duty) / max(1e-6, (1.0 - duty))
                    # swing forward with knee lift
                    thigh = thigh_base - step_a * (2 * s - 1)
                    knee = knee_base - lift_a * np.sin(np.pi * s)
                # Stabilize with small fixed abduction; yaw correction left/right opposite
                is_left = leg_id in (0, 2)  # FL, RL
                # Make abduction bias come in more gently to avoid a brief initial steering transient.
                abd_alpha = alpha * alpha
                abd_bias = (abd_bias_mag if is_left else -abd_bias_mag) * abd_alpha
                abd = abd0 + abd_bias + ((yaw_c if is_left else -yaw_c) + y_c) * steer_alpha
                q_target[3 * leg_id + 0] = abd
                q_target[3 * leg_id + 1] = thigh
                q_target[3 * leg_id + 2] = knee

            tau_cmd = np.zeros(12, dtype=float)
            dbg = {"stance": stance_cmd}

        if args.model_preset in ("a1", "a1_torque"):
            # A1 uses position actuators (kp=100, forcerange ~= +/-33.5).
            # We command a small position offset so that actuator produces approx desired torque:
            #   tau ≈ kp_act * (q_target - q)
            # and we read back actual torque from data.actuator_force.
            if controller_mode == "joint_trot":
                # Direct position control (native for A1 model)
                pass
            else:
                kp_act = 100.0
                tau_cmd = np.clip(tau_cmd, -33.0, 33.0)
                q_target = q_home + (tau_cmd / kp_act)

            # Fill data.ctrl in A1 actuator order (FR, FL, RR, RL)
            # Our q12 order is (FL, FR, RL, RR) with 3 joints each.
            if args.model_preset == "a1_torque":
                # torque actuators: ctrl is torque directly (no position targets)
                data.ctrl[:] = 0.0
                if controller_mode == "qp":
                    # Add joint-space PD to hold the standing pose (otherwise torque-only model will collapse).
                    stance_i = np.asarray(dbg.get("stance", [1, 1, 1, 1]), dtype=int).reshape(4)
                    # Smaller hold gains; apply strongly on stance legs, weakly on swing legs.
                    kp_leg = np.array([20.0, 28.0, 36.0], dtype=float)
                    kd_leg = np.array([0.7, 1.0, 1.2], dtype=float)
                    kp_hold = np.zeros(12, dtype=float)
                    kd_hold = np.zeros(12, dtype=float)
                    for leg in range(4):
                        w = 1.0 if stance_i[leg] == 1 else 0.6
                        kp_hold[3 * leg : 3 * leg + 3] = w * kp_leg
                        kd_hold[3 * leg : 3 * leg + 3] = w * kd_leg
                    tau_hold = -kp_hold * (q12 - q_home) - kd_hold * qd12
                    tau_cmd = np.clip(tau_cmd + tau_hold, -33.5, 33.5)
                    # map controller order (FL,FR,RL,RR) to actuator order (FR,FL,RR,RL)
                    data.ctrl[a1_act_idx["FR_hip"]] = float(tau_cmd[3])
                    data.ctrl[a1_act_idx["FR_thigh"]] = float(tau_cmd[4])
                    data.ctrl[a1_act_idx["FR_calf"]] = float(tau_cmd[5])
                    data.ctrl[a1_act_idx["FL_hip"]] = float(tau_cmd[0])
                    data.ctrl[a1_act_idx["FL_thigh"]] = float(tau_cmd[1])
                    data.ctrl[a1_act_idx["FL_calf"]] = float(tau_cmd[2])
                    data.ctrl[a1_act_idx["RR_hip"]] = float(tau_cmd[9])
                    data.ctrl[a1_act_idx["RR_thigh"]] = float(tau_cmd[10])
                    data.ctrl[a1_act_idx["RR_calf"]] = float(tau_cmd[11])
                    data.ctrl[a1_act_idx["RL_hip"]] = float(tau_cmd[6])
                    data.ctrl[a1_act_idx["RL_thigh"]] = float(tau_cmd[7])
                    data.ctrl[a1_act_idx["RL_calf"]] = float(tau_cmd[8])
            else:
                # position actuators: baseline standing + desired targets
                data.ctrl[:] = ctrl_home  # baseline standing target
                # FR indices in our q12: 3..5
                data.ctrl[a1_act_idx["FR_hip"]] = float(q_target[3])
                data.ctrl[a1_act_idx["FR_thigh"]] = float(q_target[4])
                data.ctrl[a1_act_idx["FR_calf"]] = float(q_target[5])
                # FL indices: 0..2
                data.ctrl[a1_act_idx["FL_hip"]] = float(q_target[0])
                data.ctrl[a1_act_idx["FL_thigh"]] = float(q_target[1])
                data.ctrl[a1_act_idx["FL_calf"]] = float(q_target[2])
                # RR indices in our q12: 9..11
                data.ctrl[a1_act_idx["RR_hip"]] = float(q_target[9])
                data.ctrl[a1_act_idx["RR_thigh"]] = float(q_target[10])
                data.ctrl[a1_act_idx["RR_calf"]] = float(q_target[11])
                # RL indices: 6..8
                data.ctrl[a1_act_idx["RL_hip"]] = float(q_target[6])
                data.ctrl[a1_act_idx["RL_thigh"]] = float(q_target[7])
                data.ctrl[a1_act_idx["RL_calf"]] = float(q_target[8])
            data.qfrc_applied[:] = 0.0
        else:
            data.ctrl[:] = tau_cmd

        mujoco.mj_step(model, data)

        # measured torque: use actuator forces if available, else command
        if args.model_preset in ("a1", "a1_torque"):
            # Map actuator_force (A1 order FR,FL,RR,RL) back to our tau_12 order (FL,FR,RL,RR)
            af = np.asarray(data.actuator_force, dtype=float).reshape(model.nu)
            tau_meas = np.zeros(12, dtype=float)
            # FL
            tau_meas[0] = af[a1_act_idx["FL_hip"]]
            tau_meas[1] = af[a1_act_idx["FL_thigh"]]
            tau_meas[2] = af[a1_act_idx["FL_calf"]]
            # FR
            tau_meas[3] = af[a1_act_idx["FR_hip"]]
            tau_meas[4] = af[a1_act_idx["FR_thigh"]]
            tau_meas[5] = af[a1_act_idx["FR_calf"]]
            # RL
            tau_meas[6] = af[a1_act_idx["RL_hip"]]
            tau_meas[7] = af[a1_act_idx["RL_thigh"]]
            tau_meas[8] = af[a1_act_idx["RL_calf"]]
            # RR
            tau_meas[9] = af[a1_act_idx["RR_hip"]]
            tau_meas[10] = af[a1_act_idx["RR_thigh"]]
            tau_meas[11] = af[a1_act_idx["RR_calf"]]
        else:
            tau_meas = tau_cmd.copy()

        # truth after step (logged)
        f_truth = np.zeros((4, 3), dtype=float)
        for i, gid in enumerate(foot_geom_ids):
            f_truth[i] = _foot_truth_contact_force_world(model, data, gid)

        f_est = estimate_contact_forces_from_tau(tau_meas, J_sel, leg_joint_slices)
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
        row.update({f"tau_{i}": tau_meas[i] for i in range(12)})
        stance_cmd = np.asarray(dbg["stance"], dtype=int).reshape(4)
        row.update({f"stance_cmd_{i}": int(stance_cmd[i]) for i in range(4)})
        for i in range(4):
            for k, ax in enumerate("xyz"):
                row[f"f_truth_{i}_{ax}"] = f_truth[i, k]
                row[f"f_est_{i}_{ax}"] = f_est[i, k]
            row[f"contact_est_{i}"] = int(contact_est[i])
        writer.writerow(row)

        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()
    fp.close()
    print(f"Saved log to: {csv_path}")


if __name__ == "__main__":
    main()

