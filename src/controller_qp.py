from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .utils_math import LegIndex, clamp, quat_to_rpy


@dataclass
class GaitParams:
    period: float = 0.5
    duty: float = 0.6
    swing_height: float = 0.10
    mu: float = 0.9
    fz_max: float = 900.0


@dataclass
class ControlGains:
    # body-level tracking (for wrench target)
    kp_v_xy: float = 55.0
    kd_v_xy: float = 1.5
    kp_z: float = 35.0
    kd_z: float = 5.0
    kp_rp: float = 80.0
    kd_rp: float = 8.0
    # swing task-space
    kp_swing: float = 160.0
    kd_swing: float = 12.0
    # stance foot impedance (to "pin" stance feet in world)
    kp_stance: float = 650.0
    kd_stance: float = 35.0
    # stance "ground speed" feedforward: move stance anchor opposite desired body velocity
    k_stance_anchor: float = 6.0
    # joint damping
    kd_joint: float = 0.8


class QPGaitController:
    """
    QP force distribution + simple swing trajectory.

    Outputs: joint torques (12,)
    """

    def __init__(
        self,
        dt: float,
        joint_indices: list[int],
        foot_link_indices: list[int],
        nominal_foot_pos_base: np.ndarray,
        mass: float,
        inertia_base: np.ndarray,
        gait: GaitParams | None = None,
        gains: ControlGains | None = None,
    ):
        self.dt = float(dt)
        self.joint_indices = list(joint_indices)
        self.foot_link_indices = list(foot_link_indices)
        self.nominal_foot_pos_base = np.asarray(nominal_foot_pos_base, dtype=float).reshape(4, 3)
        self.mass = float(mass)
        self.inertia_base = np.asarray(inertia_base, dtype=float).reshape(3, 3)
        self.gait = gait or GaitParams()
        self.gains = gains or ControlGains()
        self.leg = LegIndex()

        self._t = 0.0
        self._swing_start_world = np.zeros((4, 3), dtype=float)
        self._swing_target_world = np.zeros((4, 3), dtype=float)
        self._stance_anchor_world = np.zeros((4, 3), dtype=float)
        self._was_stance = np.zeros(4, dtype=bool)

    def reset(self, t0: float, foot_pos_world: np.ndarray):
        self._t = float(t0)
        self._swing_start_world = np.asarray(foot_pos_world, dtype=float).reshape(4, 3).copy()
        self._swing_target_world = np.asarray(foot_pos_world, dtype=float).reshape(4, 3).copy()
        self._stance_anchor_world = np.asarray(foot_pos_world, dtype=float).reshape(4, 3).copy()
        self._was_stance[:] = True

    def _phase(self) -> float:
        return (self._t % self.gait.period) / self.gait.period

    def _stance_mask(self) -> np.ndarray:
        """
        Trot: (FL, RR) in phase; (FR, RL) opposite.
        """
        ph = self._phase()
        duty = self.gait.duty
        # group A: FL, RR
        stance_A = ph < duty
        # group B: FR, RL is shifted by 0.5
        ph_b = (ph + 0.5) % 1.0
        stance_B = ph_b < duty
        stance = np.zeros(4, dtype=bool)
        stance[self.leg.FL] = stance_A
        stance[self.leg.RR] = stance_A
        stance[self.leg.FR] = stance_B
        stance[self.leg.RL] = stance_B
        return stance

    def _swing_progress(self, leg_id: int) -> float:
        ph = self._phase()
        duty = self.gait.duty
        if leg_id in (self.leg.FL, self.leg.RR):
            # swing when ph in [duty, 1)
            if ph < duty:
                return 0.0
            return (ph - duty) / max(1e-6, (1.0 - duty))
        else:
            ph_b = (ph + 0.5) % 1.0
            if ph_b < duty:
                return 0.0
            return (ph_b - duty) / max(1e-6, (1.0 - duty))

    @staticmethod
    def _smoothstep(s: float) -> float:
        s = clamp(s, 0.0, 1.0)
        return s * s * (3 - 2 * s)

    def _plan_swing_targets(
        self,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
        base_vel: np.ndarray,
        yaw_rate: float,
        v_cmd_xy: np.ndarray,
        foot_pos_world: np.ndarray,
    ):
        """
        Update swing start/targets when a leg transitions stance->swing.
        Target is a simple Raibert-style foothold: nominal + k * v_cmd * (T/2)
        """
        stance = self._stance_mask()
        # nominal in base frame (from URDF stance)
        # we ignore roll/pitch for foothold and use yaw only
        yaw = float(base_rpy[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        nominal_world = base_pos[None, :] + (R_yaw @ self.nominal_foot_pos_base.T).T

        # Raibert-style foothold: nominal + feedforward + velocity error feedback
        step_time = self.gait.period * (1.0 - self.gait.duty)
        # conservative foothold (stability first)
        k_ff = 1.15
        k_fb = 0.25
        v_xy = base_vel[:2]
        vel_term = (k_ff * v_cmd_xy + k_fb * (v_cmd_xy - v_xy)) * step_time
        vel_term = np.array([vel_term[0], vel_term[1], 0.0], dtype=float)
        yaw_term = np.array([0.0, 0.0, 0.0], dtype=float)  # keep simple

        for i in range(4):
            if (not stance[i]) and self._was_stance[i]:
                self._swing_start_world[i] = foot_pos_world[i].copy()
                self._swing_target_world[i] = (nominal_world[i] + vel_term + yaw_term).copy()
                self._swing_target_world[i][2] = foot_pos_world[i][2]  # keep landing height
            # swing -> stance: refresh stance anchor at touch-down moment
            if stance[i] and (not self._was_stance[i]):
                self._stance_anchor_world[i] = foot_pos_world[i].copy()

        self._was_stance = stance.copy()

    def _desired_wrench(
        self,
        base_pos: np.ndarray,
        base_rpy: np.ndarray,
        base_vel: np.ndarray,
        base_omega: np.ndarray,
        v_cmd_xy: np.ndarray,
        z_cmd: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns desired (F_world, tau_world) for centroidal force distribution.
        """
        g = 9.81
        kp_v = self.gains.kp_v_xy
        kd_v = self.gains.kd_v_xy

        a_xy = kp_v * (v_cmd_xy - base_vel[:2]) - kd_v * base_vel[:2]
        a_xy = np.clip(a_xy, -12.0, 12.0)

        # height hold
        kpz, kdz = self.gains.kp_z, self.gains.kd_z
        az = kpz * (z_cmd - base_pos[2]) - kdz * base_vel[2]
        az = float(clamp(az, -3.0, 3.0))

        F = np.array([self.mass * a_xy[0], self.mass * a_xy[1], self.mass * (g + az)], dtype=float)

        # keep roll/pitch near 0, no yaw control here
        roll, pitch, _yaw = base_rpy
        kprp, kdrp = self.gains.kp_rp, self.gains.kd_rp
        tau_rp = np.array(
            [
                -kprp * roll - kdrp * base_omega[0],
                -kprp * pitch - kdrp * base_omega[1],
                0.0,
            ],
            dtype=float,
        )
        return F, tau_rp

    def _solve_force_qp(
        self,
        stance: np.ndarray,
        base_pos: np.ndarray,
        foot_pos_world: np.ndarray,
        F_des: np.ndarray,
        tau_des: np.ndarray,
    ) -> np.ndarray:
        """
        Solve for per-foot forces f_i in world frame. Output shape (4,3).
        """
        idx = np.where(stance)[0].tolist()
        n = len(idx)
        f_out = np.zeros((4, 3), dtype=float)
        if n == 0:
            return f_out

        # decision variable stacked [fx1,fy1,fz1,...]
        f = cp.Variable(3 * n)

        # build centroidal mapping: [sum f; sum r x f] = [F; tau]
        A = np.zeros((6, 3 * n), dtype=float)
        for k, leg_id in enumerate(idx):
            A[0:3, 3 * k : 3 * k + 3] = np.eye(3)
            r = foot_pos_world[leg_id] - base_pos  # world vector from COM to foot
            rx = np.array(
                [
                    [0.0, -r[2], r[1]],
                    [r[2], 0.0, -r[0]],
                    [-r[1], r[0], 0.0],
                ],
                dtype=float,
            )
            A[3:6, 3 * k : 3 * k + 3] = rx

        b = np.hstack([F_des, tau_des])

        # objective: track wrench + small regularization
        # Track net force strongly, and roll/pitch moments softly (helps stability at higher speeds).
        W = np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.05])
        reg = 5e-4
        obj = cp.Minimize(cp.sum_squares(W @ (A @ f - b)) + reg * cp.sum_squares(f))

        constraints = []
        mu = self.gait.mu
        fz_max = self.gait.fz_max
        for k in range(n):
            fx = f[3 * k + 0]
            fy = f[3 * k + 1]
            fz = f[3 * k + 2]
            constraints += [
                fz >= 0.0,
                fz <= fz_max,
                fx <= mu * fz,
                fx >= -mu * fz,
                fy <= mu * fz,
                fy >= -mu * fz,
            ]

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if f.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError("QP failed")
            fv = np.asarray(f.value, dtype=float).reshape(n, 3)
        except Exception:
            # Fallback: simple constrained distribution (always feasible)
            F = np.asarray(F_des, dtype=float).reshape(3)
            fz = max(0.0, float(F[2]) / n)
            for k, leg_id in enumerate(idx):
                fx = float(F[0]) / n
                fy = float(F[1]) / n
                fx = clamp(fx, -mu * fz, mu * fz)
                fy = clamp(fy, -mu * fz, mu * fz)
                f_out[leg_id] = np.array([fx, fy, min(fz, fz_max)], dtype=float)
            return f_out

        for k, leg_id in enumerate(idx):
            f_out[leg_id] = fv[k]
        return f_out

    def step(
        self,
        base_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        base_vel: np.ndarray,
        base_omega: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        foot_pos_world: np.ndarray,
        foot_jacobians: np.ndarray,
        v_cmd_xy: np.ndarray,
        z_cmd: float,
        stance_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        - foot_jacobians: (4,3,12) linear Jacobian for each foot link in world frame.
        Returns (tau_12, debug_dict)
        """
        self._t += self.dt

        base_pos = np.asarray(base_pos, dtype=float).reshape(3)
        base_vel = np.asarray(base_vel, dtype=float).reshape(3)
        base_omega = np.asarray(base_omega, dtype=float).reshape(3)
        q = np.asarray(q, dtype=float).reshape(12)
        qd = np.asarray(qd, dtype=float).reshape(12)
        foot_pos_world = np.asarray(foot_pos_world, dtype=float).reshape(4, 3)
        J = np.asarray(foot_jacobians, dtype=float).reshape(4, 3, 12)
        v_cmd_xy = np.asarray(v_cmd_xy, dtype=float).reshape(2)

        base_rpy = quat_to_rpy(np.asarray(base_quat_xyzw, dtype=float).reshape(4))

        stance_sched = self._stance_mask()
        stance = stance_sched
        if stance_override is not None:
            stance_override = np.asarray(stance_override, dtype=bool).reshape(4)
            # Use contact truth only to gate scheduled stance legs.
            # This preserves swing/stance alternation even if a swing foot is still in contact.
            stance = stance_sched & stance_override
            if not np.any(stance):
                # fallback: if contact info is missing, use schedule
                stance = stance_sched
        self._plan_swing_targets(base_pos, base_rpy, base_vel, base_omega[2], v_cmd_xy, foot_pos_world)

        F_des, tau_des = self._desired_wrench(base_pos, base_rpy, base_vel, base_omega, v_cmd_xy, z_cmd)
        f_world = self._solve_force_qp(stance, base_pos, foot_pos_world, F_des, tau_des)  # (4,3)

        # convert desired foot forces to joint torques: tau = J^T f
        tau = np.zeros(12, dtype=float)
        for i in range(4):
            if stance[i]:
                # Move stance anchor backward to generate forward propulsion.
                # Intuition: if anchor drifts backward while foot is on ground, impedance tries to push ground backward -> body forward.
                self._stance_anchor_world[i, 0:2] += (
                    -self.gains.k_stance_anchor * v_cmd_xy * self.dt
                )
                # stance foot impedance to reduce slip & generate body motion
                p = foot_pos_world[i]
                v = J[i] @ qd
                f_imp = self.gains.kp_stance * (self._stance_anchor_world[i] - p) - self.gains.kd_stance * v
                f_imp = np.clip(f_imp, [-260.0, -260.0, -320.0], [260.0, 260.0, 320.0])
                tau += J[i].T @ (f_world[i] + f_imp)

        # swing legs: task-space PD towards planned target
        kp = self.gains.kp_swing
        kd = self.gains.kd_swing
        for i in range(4):
            if stance[i]:
                continue
            s = self._swing_progress(i)
            ss = self._smoothstep(s)
            p0 = self._swing_start_world[i]
            p1 = self._swing_target_world[i]
            p_des = (1.0 - ss) * p0 + ss * p1
            # add mid-swing height bump
            p_des = p_des.copy()
            p_des[2] += self.gait.swing_height * 4.0 * ss * (1.0 - ss)

            p = foot_pos_world[i]
            v = J[i] @ qd
            f_task = kp * (p_des - p) - kd * v
            # limit swing task force to avoid explosions
            f_task = np.clip(f_task, [-220.0, -220.0, -220.0], [220.0, 220.0, 220.0])
            tau += J[i].T @ f_task

        # joint damping
        tau -= self.gains.kd_joint * qd
        tau = np.clip(tau, -140.0, 140.0)

        debug = {
            "stance": stance.astype(int),
            "F_des": F_des,
            "tau_des": tau_des,
            "f_world": f_world,
            "base_rpy": base_rpy,
        }
        return tau, debug

