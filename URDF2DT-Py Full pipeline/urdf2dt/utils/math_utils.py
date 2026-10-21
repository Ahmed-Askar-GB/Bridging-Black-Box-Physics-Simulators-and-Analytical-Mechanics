"""Mathematical and spatial transformation utilities."""

from __future__ import annotations
import numpy as np


def rpy_to_R(rpy: np.ndarray | list[float]) -> np.ndarray:
    """Convert roll-pitch-yaw (XYZ extrinsic / ZYX intrinsic) angles to 3x3 rotation matrix.

    R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr, cr]], dtype=np.float64)

    Ry = np.array([[cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float64)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy, cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    return Rz @ Ry @ Rx


def xyz_rpy_to_T(xyz: np.ndarray | list[float], rpy: np.ndarray | list[float]) -> np.ndarray:
    """Construct 4x4 homogeneous transformation matrix from translation and RPY."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz] using Shepperd's 4-branch method.

    Guarantees qw >= 0 and avoids singularity at 180 degrees.
    """
    assert R.shape == (3, 3), f"Expected 3x3 matrix, got {R.shape}"

    B0 = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
    B1 = 1.0 + R[0, 0] - R[1, 1] - R[2, 2]
    B2 = 1.0 - R[0, 0] + R[1, 1] - R[2, 2]
    B3 = 1.0 - R[0, 0] - R[1, 1] + R[2, 2]

    branch = int(np.argmax([B0, B1, B2, B3]))

    if branch == 0:
        S = 2.0 * np.sqrt(max(0.0, B0))
        qw = S / 4.0
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif branch == 1:
        S = 2.0 * np.sqrt(max(0.0, B1))
        qw = (R[2, 1] - R[1, 2]) / S
        qx = S / 4.0
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif branch == 2:
        S = 2.0 * np.sqrt(max(0.0, B2))
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = S / 4.0
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(max(0.0, B3))
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = S / 4.0

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # Canonicalize qw >= 0
    if q[0] < 0.0:
        q = -q
    norm_q = np.linalg.norm(q)
    if norm_q > 1e-12:
        q = q / norm_q
    return q


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx**2 + qy**2)]
    ], dtype=np.float64)


def quat_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute shortest geodesic angular difference in radians between two unit quaternions.

    theta = 2 * arccos(|q1 . q2|)
    """
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    q2 = np.asarray(q2, dtype=np.float64) / np.linalg.norm(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return float(2.0 * np.arccos(dot))


def skew(v: np.ndarray) -> np.ndarray:
    """Generate 3x3 skew-symmetric matrix for 3D vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)
