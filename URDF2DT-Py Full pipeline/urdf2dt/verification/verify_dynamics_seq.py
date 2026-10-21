"""Sequential Link-by-Link Dynamic Isolation and Torque Validation."""

from __future__ import annotations
import numpy as np
from urdf2dt.containers import Chain, DHTable, Inertia, StageResult
from urdf2dt.analytical.dynamics_casadi import dynamics_casadi
from urdf2dt.digital_twin import MuJoCoTwin
from urdf2dt.utils.traj_gen import traj_poly5
from urdf2dt.utils.sign_detection import detect_joint_signs


def verify_dynamics_sequential(
    chain: Chain,
    dh: DHTable,
    inertia: Inertia,
    twin: MuJoCoTwin,
    t_dur: float = 3.0,
    dt: float = 0.005,
    tol_tau: float = 0.1,
    g_vec: tuple[float, float, float] | list[float] | np.ndarray | float | None = None,
    verbose: bool = True
) -> list[StageResult]:
    """Execute sequential link-by-link dynamic isolation:

    For stage i = 1 .. N:
      1. Downstream links k > i have masses and inertias zeroed out.
      2. Non-active joints j != i are frozen at nominal non-zero holding angles.
      3. Joint i is dynamically excited with a 5th-order polynomial trajectory.
      4. Analytical torque tau_i,ana is cross-validated against twin torque tau_i,sim.
    """
    n = chain.n

    # Synchronize customizable gravity between twin and analytical model
    if g_vec is None:
        g_val = twin.gravity.copy()
    else:
        twin.set_gravity(g_vec)
        g_val = twin.gravity.copy()

    # Temporary CasADi model to get link functions for sign detection
    temp_model = dynamics_casadi(dh, inertia, chain, g_vec=g_val)
    joint_signs = detect_joint_signs(chain, temp_model.fk_link_funs)

    q_holding = np.array([0.1, -0.2, 0.2, -0.15, 0.2, -0.1], dtype=np.float64)
    if len(q_holding) < n:
        q_holding = np.concatenate([q_holding, np.zeros(n - len(q_holding))])
    else:
        q_holding = q_holding[:n]

    stage_results: list[StageResult] = []

    if verbose:
        print("\n==================================================================")
        print("     SEQUENTIAL LINK-BY-LINK DYNAMIC PARAMETER VALIDATION         ")
        print("==================================================================")

    for stage_idx in range(n):
        # Stage i (1-based: stage_idx + 1)
        stage_num = stage_idx + 1
        joint_name = chain.active_joints[stage_idx].name
        link_name = chain.active_links[stage_idx]

        # Step A: Downstream Zeroing
        m_stage = inertia.m.copy()
        P_Ci_stage = inertia.P_Ci.copy()
        I_stage = inertia.I.copy()

        # Zero out downstream links k > stage_idx
        for k in range(stage_idx + 1, n):
            m_stage[k] = 0.0
            P_Ci_stage[k, :] = 0.0
            I_stage[k, :, :] = 0.0

        stage_inertia = Inertia(m=m_stage, P_Ci=P_Ci_stage, I=I_stage)
        stage_casadi = dynamics_casadi(dh, stage_inertia, chain, g_vec=g_val)

        # Update MuJoCo twin link inertias
        twin.set_link_inertias(masses=m_stage, inertias=I_stage)

        # Step B: Joint Trajectory Excitation
        q0 = q_holding.copy()
        qf = q_holding.copy()
        # Excite current joint by +0.8 rad
        qf[stage_idx] = q0[stage_idx] + 0.8

        time, q_traj_urdf, dq_traj_urdf, ddq_traj_urdf = traj_poly5(q0, qf, t_dur=t_dur, dt=dt)
        num_steps = len(time)

        tau_twin = np.zeros(num_steps, dtype=np.float64)
        tau_ana = np.zeros(num_steps, dtype=np.float64)

        for step in range(num_steps):
            q_u = q_traj_urdf[step, :]
            dq_u = dq_traj_urdf[step, :]
            ddq_u = ddq_traj_urdf[step, :]

            # MuJoCo inverse dynamics ground truth
            tau_vec_twin = twin.inverse_dynamics(q_u, dq_u, ddq_u)
            tau_twin[step] = tau_vec_twin[stage_idx]

            # Analytical DH evaluation (mapped via joint_signs)
            q_dh = joint_signs * q_u
            dq_dh = joint_signs * dq_u
            ddq_dh = joint_signs * ddq_u

            tau_vec_dh = np.array(stage_casadi.tau_fun(q_dh, dq_dh, ddq_dh)).flatten()
            # Map back to URDF joint direction: tau_urdf = sign * tau_dh
            tau_vec_ana = joint_signs * tau_vec_dh
            tau_ana[step] = tau_vec_ana[stage_idx]

        # Compute error residuals
        err = np.abs(tau_twin - tau_ana)
        max_err = float(np.max(err))
        rms_err = float(np.sqrt(np.mean(err**2)))
        peak_tau = float(np.max(np.abs(tau_twin)))
        passed = bool(rms_err < tol_tau)

        result = StageResult(
            stage=stage_num,
            joint_name=joint_name,
            link_name=link_name,
            passed=passed,
            max_err=max_err,
            rms_err=rms_err,
            peak_torque=peak_tau,
            time=time,
            tau_sim=tau_twin,
            tau_ana=tau_ana,
        )
        stage_results.append(result)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  Stage {stage_num}/{n} [{status}] Joint: {joint_name:20s} | "
                  f"RMS Err: {rms_err:8.4e} N*m | Max Err: {max_err:8.4e} N*m | Peak: {peak_tau:6.2f} N*m")

    # Restore true physical parameters in MuJoCo
    twin.set_link_inertias(masses=inertia.m, inertias=inertia.I)

    return stage_results
