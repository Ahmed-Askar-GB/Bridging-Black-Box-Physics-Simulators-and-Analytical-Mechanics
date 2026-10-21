"""Symbolic Forward Kinematics and Shepperd quaternion conversion via CasADi."""

from __future__ import annotations
import casadi as cs
import numpy as np
from urdf2dt.containers import DHTable, Chain


def _ai_casadi(a: float | cs.SX, alpha: float | cs.SX, theta: cs.SX, d: float | cs.SX) -> cs.SX:
    """Compute 4x4 symbolic standard DH elementary transformation matrix."""
    ct = cs.cos(theta)
    st = cs.sin(theta)
    ca = cs.cos(alpha)
    sa = cs.sin(alpha)

    return cs.vertcat(
        cs.horzcat(ct, -st * ca,  st * sa, a * ct),
        cs.horzcat(st,  ct * ca, -ct * sa, a * st),
        cs.horzcat(0.0,      sa,       ca,      d),
        cs.horzcat(0.0,     0.0,      0.0,    1.0)
    )


def fk_casadi(dh: DHTable, chain: Chain, verbose: bool = False) -> dict:
    """Derive symbolic forward kinematics, cumulative link transforms, and EE pose with Shepperd quaternion."""
    n = chain.n
    q = cs.SX.sym("q", n)

    # Build symbolic DH table
    DHp = dh.DHpram  # [a, alpha, theta_offset, d, sigma]
    A_list: list[cs.SX] = []
    T0i_list: list[cs.SX] = []

    T_running = cs.SX.eye(4)

    for i in range(n):
        a_i = float(DHp[i, 0])
        alpha_i = float(DHp[i, 1])
        th_off = float(DHp[i, 2])
        d_off = float(DHp[i, 3])
        sigma_i = float(DHp[i, 4])

        if sigma_i == 1.0:
            theta_i = th_off + q[i]
            d_i = d_off
        else:
            theta_i = th_off
            d_i = d_off + q[i]

        A_i = _ai_casadi(a_i, alpha_i, theta_i, d_i)
        A_list.append(A_i)

        T_running = cs.mtimes(T_running, A_i)
        T0i_list.append(T_running)

    # End-effector transform
    T_ee_const = cs.SX(chain.T_ee)
    T0e = cs.mtimes(T0i_list[-1], T_ee_const)

    p_ee = T0e[:3, 3]
    R_ee = T0e[:3, :3]

    # Shepperd single-branch quaternion adapted for symbolic differentiability
    tr = R_ee[0, 0] + R_ee[1, 1] + R_ee[2, 2]
    # Safeguard sqrt argument against numerical noise
    radicand = cs.fmax(1.0 + tr, 1e-12)
    qw = 0.5 * cs.sqrt(radicand)
    inv4qw = 1.0 / (4.0 * qw)

    qx = (R_ee[2, 1] - R_ee[1, 2]) * inv4qw
    qy = (R_ee[0, 2] - R_ee[2, 0]) * inv4qw
    qz = (R_ee[1, 0] - R_ee[0, 1]) * inv4qw

    quat_ee = cs.vertcat(qw, qx, qy, qz)

    # Compile CasADi Function objects
    fk_fun = cs.Function("fk_fun", [q], [p_ee, R_ee, quat_ee], ["q"], ["p_ee", "R_ee", "quat_ee"])
    fk_ee_fun = cs.Function("fk_ee_fun", [q], [T0e], ["q"], ["T0e"])

    fk_link_funs: list[cs.Function] = []
    for i in range(n):
        f_link = cs.Function(f"fk_link_{i+1}", [q], [T0i_list[i]], ["q"], [f"T0_{i+1}"])
        fk_link_funs.append(f_link)

    return {
        "q": q,
        "n": n,
        "T0i": T0i_list,
        "T0e": T0e,
        "p_ee": p_ee,
        "R_ee": R_ee,
        "quat_ee": quat_ee,
        "fk_fun": fk_fun,
        "fk_ee_fun": fk_ee_fun,
        "fk_link_funs": fk_link_funs,
    }
