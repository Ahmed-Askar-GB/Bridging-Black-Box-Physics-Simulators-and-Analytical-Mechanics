"""Standard Denavit-Hartenberg (DH) parameter extraction via P-Q geometric skew-lines method."""

from __future__ import annotations
import numpy as np
from urdf2dt.containers import Chain, DHTable


def _dh_transform(a: float, alpha: float, theta: float, d: float) -> np.ndarray:
    """Compute standard 4x4 DH elementary transformation matrix.

    A_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0]
    ], dtype=np.float64)


def urdf_extract_dh(chain: Chain, verbose: bool = False) -> DHTable:
    """Extract standard DH parameters using the P-Q common-normal / skew-lines method.

    Returns DHTable with table: [theta_offset, d, a, alpha] and DHpram: [a, alpha, theta_offset, d, sigma].
    """
    n = chain.n
    if n == 0:
        raise ValueError("Cannot extract DH parameters from 0-DOF chain.")

    # Compute world-frame EE node:
    T_ee_world = chain.T_urdf[-1] @ chain.T_ee
    O_ee = T_ee_world[:3, 3]

    # Augmented origins and unit z-axes (N active joints + 1 virtual EE node)
    O = np.column_stack([chain.O, O_ee])  # (3, N+1)
    Z = np.column_stack([chain.Z, chain.Z[:, -1]])  # (3, N+1)

    Q = np.zeros((3, n + 1), dtype=np.float64)
    X = np.zeros((3, n + 1), dtype=np.float64)

    # Base frame seeds
    Q[:, 0] = np.zeros(3)
    X[:, 0] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    DH_table = np.zeros((n, 4), dtype=np.float64)  # [theta_offset, d, a, alpha]
    joint_types: list[str] = []
    sigma = np.ones(n, dtype=np.float64)  # 1=revolute, 2=prismatic

    for i in range(n):
        prev_Z = Z[:, i]
        curr_Z = Z[:, i + 1]
        prev_O = O[:, i]
        curr_O = O[:, i + 1]

        delta_O = curr_O - prev_O
        c = float(np.dot(prev_Z, curr_Z))

        # CASE A: Parallel or anti-parallel consecutive axes (|c| ≈ 1)
        if abs(abs(c) - 1.0) < 1e-5:
            d_val = 0.0
            P_i = Q[:, i].copy()
            Q[:, i + 1] = curr_O + np.dot(P_i - curr_O, curr_Z) * curr_Z

            X_dir = delta_O - np.dot(delta_O, curr_Z) * curr_Z
            norm_X = np.linalg.norm(X_dir)
            if norm_X > 1e-5:
                X[:, i + 1] = X_dir / norm_X
            else:
                X[:, i + 1] = X[:, i].copy()

        # CASE B: Skew or intersecting consecutive axes
        else:
            denom = 1.0 - c**2
            t1 = (np.dot(delta_O, prev_Z) - c * np.dot(delta_O, curr_Z)) / denom
            t2 = (c * np.dot(delta_O, prev_Z) - np.dot(delta_O, curr_Z)) / denom

            P_i = prev_O + t1 * prev_Z
            Q[:, i + 1] = curr_O + t2 * curr_Z

            d_val = float(np.dot(P_i - Q[:, i], prev_Z))
            n_vec = Q[:, i + 1] - P_i
            norm_n = np.linalg.norm(n_vec)

            if norm_n > 1e-5:
                # Sub-case B1: Skew axes
                X[:, i + 1] = n_vec / norm_n
            else:
                # Sub-case B2: Intersecting axes
                X_dir = np.cross(prev_Z, curr_Z)
                norm_cross = np.linalg.norm(X_dir)
                if norm_cross > 1e-12:
                    X_dir = X_dir / norm_cross
                else:
                    X_dir = np.array([1.0, 0.0, 0.0])

                # Choose X_dir direction to preserve continuity with preceding frame X[:, i]
                if np.dot(X_dir, X[:, i]) < -0.1:
                    X_dir = -X_dir
                X[:, i + 1] = X_dir

        # Extract standard DH parameters
        DH_table[i, 1] = d_val  # d
        DH_table[i, 2] = float(np.linalg.norm(Q[:, i + 1] - P_i))  # a

        # alpha: angle from prev_Z to curr_Z about X[:, i+1]
        alpha_val = float(np.arctan2(
            np.dot(np.cross(prev_Z, curr_Z), X[:, i + 1]),
            np.dot(prev_Z, curr_Z)
        ))
        DH_table[i, 3] = alpha_val

        # theta: angle from X[:, i] to X[:, i+1] about prev_Z
        theta_val = float(np.arctan2(
            np.dot(np.cross(X[:, i], X[:, i + 1]), prev_Z),
            np.dot(X[:, i], X[:, i + 1])
        ))
        # Canonicalize to [-pi, pi]
        theta_val = (theta_val + np.pi) % (2.0 * np.pi) - np.pi
        if abs(theta_val) < 1e-4 or abs(abs(theta_val) - np.pi) < 1e-4:
            theta_val = 0.0
        DH_table[i, 0] = theta_val

        j_type = chain.active_joints[i].joint_type
        if j_type == "prismatic":
            joint_types.append("prismatic")
            sigma[i] = 2.0
        else:
            joint_types.append("revolute")
            sigma[i] = 1.0

    # Override last row with physical EE offsets if applicable
    DH_table[-1, 1] = chain.d_ee
    DH_table[-1, 2] = chain.a_ee

    # Build DHpram matrix: [a, alpha, theta_offset, d, sigma]
    DHpram = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        DHpram[i, :] = [
            DH_table[i, 2],  # a
            DH_table[i, 3],  # alpha
            DH_table[i, 0],  # theta_offset
            DH_table[i, 1],  # d
            sigma[i]         # sigma
        ]

    # Compute cumulative standard DH transforms at q=0
    T_dh: list[np.ndarray] = []
    T_accum = np.eye(4, dtype=np.float64)
    for i in range(n):
        A_i = _dh_transform(
            a=DHpram[i, 0],
            alpha=DHpram[i, 1],
            theta=DHpram[i, 2],
            d=DHpram[i, 3]
        )
        T_accum = T_accum @ A_i
        T_dh.append(T_accum.copy())

    if verbose:
        print("\nExtracted Standard DH Parameters Table [theta_off (rad), d (m), a (m), alpha (rad)]:")
        for idx in range(n):
            print(f"  Link {idx+1}: theta={DH_table[idx,0]:8.4f}, d={DH_table[idx,1]:8.4f}, "
                  f"a={DH_table[idx,2]:8.4f}, alpha={DH_table[idx,3]:8.4f} ({joint_types[idx]})")

    return DHTable(
        table=DH_table,
        DHpram=DHpram,
        joint_types=joint_types,
        sigma=sigma,
        T_dh=T_dh,
    )
