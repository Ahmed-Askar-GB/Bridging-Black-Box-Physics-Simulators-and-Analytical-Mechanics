"""Geometric Jacobian and exact auto-diff time-derivative J_dot via CasADi."""

from __future__ import annotations
import casadi as cs
from urdf2dt.containers import DHTable, Chain


def jacobian_casadi(fk: dict, dh: DHTable, chain: Chain, verbose: bool = False) -> dict:
    """Compute the 6xN geometric Jacobian J(q) and exact time-derivative J_dot(q, dq) using CasADi auto-differentiation."""
    n = chain.n
    q = fk["q"]
    dq = cs.SX.sym("dq", n)

    T0i = fk["T0i"]
    p_ee = fk["p_ee"]
    sigma = dh.sigma

    # Base frame actuation axis and origin
    z_list: list[cs.SX] = [cs.vertcat(0.0, 0.0, 1.0)]
    O_list: list[cs.SX] = [cs.vertcat(0.0, 0.0, 0.0)]

    for i in range(1, n):
        z_list.append(T0i[i - 1][:3, 2])
        O_list.append(T0i[i - 1][:3, 3])

    # Construct columns of Jv and Jw
    Jv_cols: list[cs.SX] = []
    Jw_cols: list[cs.SX] = []

    for j in range(n):
        zj = z_list[j]
        Oj = O_list[j]

        if sigma[j] == 1.0:
            # Revolute: Jv = zj x (p_ee - Oj), Jw = zj
            r_arm = p_ee - Oj
            jv_j = cs.cross(zj, r_arm)
            jw_j = zj
        else:
            # Prismatic: Jv = zj, Jw = 0
            jv_j = zj
            jw_j = cs.vertcat(0.0, 0.0, 0.0)

        Jv_cols.append(jv_j)
        Jw_cols.append(jw_j)

    Jv_sym = cs.horzcat(*Jv_cols)
    Jw_sym = cs.horzcat(*Jw_cols)
    J_sym = cs.vertcat(Jv_sym, Jw_sym)

    # Exact time derivative J_dot(q, dq) via algorithmic differentiation
    J_vec = cs.vec(J_sym)
    dJ_dq = cs.jacobian(J_vec, q)
    Jdot_vec = cs.mtimes(dJ_dq, dq)
    Jdot_sym = cs.reshape(Jdot_vec, 6, n)

    # Compile Functions
    J_fun = cs.Function("J_fun", [q], [J_sym], ["q"], ["J"])
    Jv_fun = cs.Function("Jv_fun", [q], [Jv_sym], ["q"], ["Jv"])
    Jw_fun = cs.Function("Jw_fun", [q], [Jw_sym], ["q"], ["Jw"])
    Jdot_fun = cs.Function("Jdot_fun", [q, dq], [Jdot_sym], ["q", "dq"], ["Jdot"])

    return {
        "q": q,
        "dq": dq,
        "J_sym": J_sym,
        "Jv_sym": Jv_sym,
        "Jw_sym": Jw_sym,
        "Jdot_sym": Jdot_sym,
        "J_fun": J_fun,
        "Jv_fun": Jv_fun,
        "Jw_fun": Jw_fun,
        "Jdot_fun": Jdot_fun,
    }
