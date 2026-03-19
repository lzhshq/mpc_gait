from __future__ import annotations

import numpy as np


def estimate_contact_forces_from_tau(
    tau_12: np.ndarray,
    foot_jacobians: np.ndarray,
    leg_joint_slices: list[slice],
    damping: float = 1e-6,
) -> np.ndarray:
    """
    Per-leg force estimate using tau ≈ J^T f  =>  f ≈ (J^T)^+ tau.

    Inputs
    - tau_12: (12,)
    - foot_jacobians: (4,3,12) linear jacobians
    - leg_joint_slices: 4 slices mapping leg joints in tau vector (3 joints each)

    Output
    - f_est_world: (4,3)
    """
    tau_12 = np.asarray(tau_12, dtype=float).reshape(12)
    J = np.asarray(foot_jacobians, dtype=float).reshape(4, 3, 12)
    f_est = np.zeros((4, 3), dtype=float)
    for i in range(4):
        sl = leg_joint_slices[i]
        tau_leg = tau_12[sl]  # (3,)
        J_leg = J[i][:, sl]   # (3,3)
        # Solve min ||J^T f - tau||^2 + d||f||^2  =>  (J J^T + dI) f = J tau
        A = J_leg @ J_leg.T + damping * np.eye(3)
        b = J_leg @ tau_leg
        try:
            f_est[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            f_est[i] = np.zeros(3)
    return f_est


def estimate_contact_state_from_force(
    f_est_world: np.ndarray,
    fz_threshold: float = 20.0,
    f_norm_threshold: float | None = None,
) -> np.ndarray:
    """
    stance/swing classification based on estimated force magnitude.

    Output: contact (4,) bool
    """
    f = np.asarray(f_est_world, dtype=float).reshape(4, 3)
    if f_norm_threshold is not None:
        return (np.linalg.norm(f, axis=1) >= float(f_norm_threshold))
    return (f[:, 2] >= float(fz_threshold))


def finite_difference_acc(qd_prev: np.ndarray, qd: np.ndarray, dt: float) -> np.ndarray:
    qd_prev = np.asarray(qd_prev, dtype=float)
    qd = np.asarray(qd, dtype=float)
    return (qd - qd_prev) / max(1e-9, float(dt))

