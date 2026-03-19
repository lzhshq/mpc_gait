from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(angle: float) -> float:
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> R (3x3)."""
    x, y, z, w = q_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def quat_to_rpy(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> roll,pitch,yaw."""
    x, y, z, w = q_xyzw
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """roll,pitch,yaw -> quaternion (x,y,z,w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=float)


def fifth_order_poly(p0: np.ndarray, v0: np.ndarray, a0: np.ndarray, p1: np.ndarray, v1: np.ndarray, a1: np.ndarray, t: float, T: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Quintic polynomial interpolation in R^n.
    Returns (p(t), v(t)).
    """
    if T <= 0:
        return p1.copy(), v1.copy()
    s = clamp(t / T, 0.0, 1.0)
    T2, T3, T4, T5 = T * T, T**3, T**4, T**5
    # coefficients for each dimension
    c0 = p0
    c1 = v0
    c2 = a0 / 2.0
    A = (
        20 * (p1 - p0)
        - (8 * v1 + 12 * v0) * T
        - (3 * a0 - a1) * T2
    ) / (2 * T3)
    B = (
        30 * (p0 - p1)
        + (14 * v1 + 16 * v0) * T
        + (3 * a0 - 2 * a1) * T2
    ) / (2 * T4)
    C = (
        12 * (p1 - p0)
        - (6 * v1 + 6 * v0) * T
        - (a0 - a1) * T2
    ) / (2 * T5)
    tt = s * T
    p = c0 + c1 * tt + c2 * tt**2 + A * tt**3 + B * tt**4 + C * tt**5
    v = c1 + 2 * c2 * tt + 3 * A * tt**2 + 4 * B * tt**3 + 5 * C * tt**4
    return p, v


@dataclass(frozen=True)
class LegIndex:
    FL: int = 0
    FR: int = 1
    RL: int = 2
    RR: int = 3

