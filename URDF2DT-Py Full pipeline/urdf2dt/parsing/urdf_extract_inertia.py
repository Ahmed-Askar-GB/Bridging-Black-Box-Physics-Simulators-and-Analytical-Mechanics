"""Coordinate transformation of URDF link inertial properties into standard DH frames."""

from __future__ import annotations
import warnings
import numpy as np
from urdf2dt.containers import Robot, Chain, DHTable, Inertia


def urdf_extract_inertia(robot: Robot, chain: Chain, dh: DHTable, verbose: bool = False) -> Inertia:
    """Transform URDF link masses, CoM vectors, and inertia tensors into standard DH frames.

    Frame Conventions:
      P_Ci[i]: CoM of link i measured from DH origin {i-1}, expressed in orientation of DH frame {i}.
      I_dh[i]: 3x3 inertia tensor of link i rotated into orientation of DH frame {i}:
               I_dh[i] = R_rel @ I_link[i] @ R_rel.T
               where R_rel = R_dh_i.T @ R_urdf_i.
    """
    n = chain.n
    m = np.zeros(n, dtype=np.float64)
    P_Ci = np.zeros((n, 3), dtype=np.float64)
    I_dh = np.zeros((n, 3, 3), dtype=np.float64)

    for i in range(n):
        link_name = chain.active_links[i]
        link = robot.links.get(link_name)
        if link is None:
            warnings.warn(f"Link '{link_name}' not found in robot. Using zero inertia.")
            continue

        m[i] = link.mass
        if m[i] <= 0.0:
            warnings.warn(f"Link '{link_name}' has non-positive mass: {m[i]} kg.")

        # CoM global position at q=0
        T_urdf_i = chain.T_urdf[i]
        R_urdf_i = T_urdf_i[:3, :3]
        p_urdf_i = T_urdf_i[:3, 3]
        p_com_global = R_urdf_i @ link.com + p_urdf_i

        # Preceding DH frame origin
        if i == 0:
            O_dh_prev = np.zeros(3, dtype=np.float64)
        else:
            O_dh_prev = dh.T_dh[i - 1][:3, 3]

        # Current DH frame rotation
        T_dh_i = dh.T_dh[i]
        R_dh_i = T_dh_i[:3, :3]

        # CoM vector relative to preceding DH origin in current DH frame
        P_Ci[i, :] = R_dh_i.T @ (p_com_global - O_dh_prev)

        # Relative rotation from URDF link frame to DH frame
        R_rel = R_dh_i.T @ R_urdf_i

        # Rotate inertia tensor
        I_dh[i, :, :] = R_rel @ link.inertia @ R_rel.T

        # Check physical realizability via principal moments
        if m[i] > 1e-6:
            eigvals = np.sort(np.linalg.eigvalsh(I_dh[i, :, :]))
            if (eigvals[0] + eigvals[1]) < (eigvals[2] - 1e-7):
                warnings.warn(
                    f"Link '{link_name}' inertia violates triangle inequality: "
                    f"I1+I2={eigvals[0]+eigvals[1]:.4e} < I3={eigvals[2]:.4e}"
                )

    if verbose:
        print("\nTransformed Inertial Parameters in Standard DH Frames:")
        for idx in range(n):
            print(f"  Link {idx+1} ({chain.active_links[idx]}): mass={m[idx]:.3f} kg, "
                  f"P_Ci=[{P_Ci[idx,0]:.4f}, {P_Ci[idx,1]:.4f}, {P_Ci[idx,2]:.4f}] m")

    return Inertia(m=m, P_Ci=P_Ci, I=I_dh)
