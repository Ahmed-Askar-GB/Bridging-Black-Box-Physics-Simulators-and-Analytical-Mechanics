"""Comprehensive Benchmark: Closed-form CasADi Analytical Simulation vs MuJoCo Physics Engine for UR16e.

Quantifies execution time, throughput, per-step latency, real-time factor (RTF),
and trajectory coincidence between:
  1. Closed-form analytical CasADi forward dynamics (RK4 & Euler integration)
  2. MuJoCo physics engine (mj_step & mj_forward)
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

# Ensure urdf2dt is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from urdf2dt import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
    dynamics_casadi,
    MuJoCoTwin,
    ComputedTorqueController,
)
from urdf2dt.utils import traj_poly5, detect_joint_signs


def find_ur16e_urdf() -> str:
    """Resolve UR16e URDF file path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "End-to-End MATLAB-CasADi Implementation", "test_UR16e", "universalUR16e.urdf"),
        os.path.join(current_dir, "..", "End-to-End MATLAB Implementation", "test_UR16e", "universalUR16e.urdf"),
        os.path.join(current_dir, "tests", "urdf", "universalUR16e.urdf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate universalUR16e.urdf")


def run_casadi_rk4_step(f_dyn, q, dq, tau, dt):
    """4th-order Runge-Kutta numerical integration step for x = [q; dq]."""
    k1_v = dq
    k1_a = np.array(f_dyn(q, dq, tau)).flatten()

    q_m1 = q + 0.5 * dt * k1_v
    v_m1 = dq + 0.5 * dt * k1_a
    k2_v = v_m1
    k2_a = np.array(f_dyn(q_m1, v_m1, tau)).flatten()

    q_m2 = q + 0.5 * dt * k2_v
    v_m2 = dq + 0.5 * dt * k2_a
    k3_v = v_m2
    k3_a = np.array(f_dyn(q_m2, v_m2, tau)).flatten()

    q_end = q + dt * k3_v
    v_end = dq + dt * k3_a
    k4_v = v_end
    k4_a = np.array(f_dyn(q_end, v_end, tau)).flatten()

    q_next = q + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    v_next = dq + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
    return q_next, v_next


def run_casadi_euler_step(f_dyn, q, dq, tau, dt):
    """Semi-implicit Euler integration step."""
    acc = np.array(f_dyn(q, dq, tau)).flatten()
    v_next = dq + dt * acc
    q_next = q + dt * v_next
    return q_next, v_next


def benchmark_ur16e(gravity: tuple[float, float, float] | list[float] | np.ndarray | float = (0.0, 0.0, -9.81)):
    print("\n" + "=" * 80)
    print("      BENCHMARK: CASADI CLOSED-FORM ANALYTICAL DYNAMICS vs. MUJOCO PLANT    ")
    print("                         UNIVERSAL ROBOTS UR16E (6-DOF)                     ")
    print("=" * 80)

    urdf_path = find_ur16e_urdf()
    print(f"  Target URDF: {urdf_path}\n")

    # Ingestion & Model Compilation
    t0 = time.perf_counter()
    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot, verbose=False)
    dh = urdf_extract_dh(chain, verbose=False)
    inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
    model = dynamics_casadi(dh, inertia, chain, g_vec=gravity, verbose=False)
    twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=0.001, gravity=gravity, verbose=False)
    signs = detect_joint_signs(chain, model.fk_link_funs)
    t_setup = time.perf_counter() - t0
    print(f"  [Setup] Models ingested and compiled in {t_setup:.3f} s.")
    print(f"  [Setup] Custom Gravity Vector: {twin.gravity.tolist()} m/s^2")
    print(f"  [Setup] Detected joint alignment signs: {signs.tolist()}\n")

    n = chain.n
    f_dyn = model.forward_dyn_fun

    # Helper to convert between URDF and DH coordinates for CasADi
    def casadi_fwd_acc(q_u, dq_u, tau_u):
        q_dh = signs * q_u
        dq_dh = signs * dq_u
        tau_dh = signs * tau_u
        ddq_dh = np.array(f_dyn(q_dh, dq_dh, tau_dh)).flatten()
        return signs * ddq_dh

    # =========================================================================
    # BENCHMARK 1: Raw Forward Dynamics Single-Step Evaluation (10,000 calls)
    # =========================================================================
    print("-" * 80)
    print(">>> BENCHMARK 1: Raw Forward Dynamics Single Evaluation (N = 10,000 calls)")
    print("-" * 80)
    N_evals = 10000
    rng = np.random.default_rng(42)
    q_test = rng.uniform(-np.pi, np.pi, size=(N_evals, n))
    dq_test = rng.uniform(-2.0, 2.0, size=(N_evals, n))
    tau_test = rng.uniform(-50.0, 50.0, size=(N_evals, n))

    # Warm-up
    for i in range(50):
        _ = casadi_fwd_acc(q_test[i], dq_test[i], tau_test[i])
        twin.data.qpos[:] = q_test[i]
        twin.data.qvel[:] = dq_test[i]
        twin.data.qfrc_applied[:] = tau_test[i]
        import mujoco
        mujoco.mj_forward(twin.model, twin.data)

    # 1A. CasADi Closed-Form Evaluation
    t0 = time.perf_counter()
    for i in range(N_evals):
        _ = casadi_fwd_acc(q_test[i], dq_test[i], tau_test[i])
    t_casadi_raw = time.perf_counter() - t0
    avg_casadi_raw_us = (t_casadi_raw / N_evals) * 1e6
    rate_casadi_raw_khz = (N_evals / t_casadi_raw) / 1e3

    # 1B. MuJoCo mj_forward Evaluation
    t0 = time.perf_counter()
    for i in range(N_evals):
        twin.data.qpos[:] = q_test[i]
        twin.data.qvel[:] = dq_test[i]
        twin.data.qfrc_applied[:] = tau_test[i]
        mujoco.mj_forward(twin.model, twin.data)
        _ = twin.data.qacc.copy()
    t_mujoco_raw = time.perf_counter() - t0
    avg_mujoco_raw_us = (t_mujoco_raw / N_evals) * 1e6
    rate_mujoco_raw_khz = (N_evals / t_mujoco_raw) / 1e3

    print(f"  CasADi Closed-Form : Total = {t_casadi_raw:.4f} s | Avg = {avg_casadi_raw_us:7.2f} us/call | Rate = {rate_casadi_raw_khz:6.2f} kHz")
    print(f"  MuJoCo Engine      : Total = {t_mujoco_raw:.4f} s | Avg = {avg_mujoco_raw_us:7.2f} us/call | Rate = {rate_mujoco_raw_khz:6.2f} kHz")
    print(f"  --> Speedup Ratio  : MuJoCo is {t_casadi_raw / t_mujoco_raw:.2f}x faster for raw forward dynamics evaluation.")
    print()

    # =========================================================================
    # BENCHMARK 2: Raw Inverse Dynamics Single Evaluation (N = 10,000 calls)
    # =========================================================================
    print("-" * 80)
    print(">>> BENCHMARK 2: Raw Inverse Dynamics Single Evaluation (N = 10,000 calls)")
    print("-" * 80)
    ddq_test = rng.uniform(-5.0, 5.0, size=(N_evals, n))

    # 2A. CasADi Closed-Form Inverse Dynamics
    t0 = time.perf_counter()
    for i in range(N_evals):
        q_dh = signs * q_test[i]
        dq_dh = signs * dq_test[i]
        ddq_dh = signs * ddq_test[i]
        _ = signs * np.array(model.tau_fun(q_dh, dq_dh, ddq_dh)).flatten()
    t_casadi_inv = time.perf_counter() - t0
    avg_casadi_inv_us = (t_casadi_inv / N_evals) * 1e6
    rate_casadi_inv_khz = (N_evals / t_casadi_inv) / 1e3

    # 2B. MuJoCo mj_inverse Evaluation
    t0 = time.perf_counter()
    for i in range(N_evals):
        twin.data.qpos[:] = q_test[i]
        twin.data.qvel[:] = dq_test[i]
        twin.data.qacc[:] = ddq_test[i]
        mujoco.mj_inverse(twin.model, twin.data)
        _ = twin.data.qfrc_inverse.copy()
    t_mujoco_inv = time.perf_counter() - t0
    avg_mujoco_inv_us = (t_mujoco_inv / N_evals) * 1e6
    rate_mujoco_inv_khz = (N_evals / t_mujoco_inv) / 1e3

    print(f"  CasADi Closed-Form : Total = {t_casadi_inv:.4f} s | Avg = {avg_casadi_inv_us:7.2f} us/call | Rate = {rate_casadi_inv_khz:6.2f} kHz")
    print(f"  MuJoCo Engine      : Total = {t_mujoco_inv:.4f} s | Avg = {avg_mujoco_inv_us:7.2f} us/call | Rate = {rate_mujoco_inv_khz:6.2f} kHz")
    print(f"  --> Speedup Ratio  : MuJoCo is {t_casadi_inv / t_mujoco_inv:.2f}x faster for raw inverse dynamics evaluation.")
    print()

    # =========================================================================
    # BENCHMARK 3: Continuous Time-Stepping Simulation (T = 5.0 s, dt = 0.001 s -> 5,000 steps)
    # =========================================================================
    print("-" * 80)
    print(">>> BENCHMARK 3: Continuous Passive Simulation under Gravity (5,000 steps)")
    print("-" * 80)
    T_sim = 5.0
    dt = 0.001
    N_steps = int(T_sim / dt)
    time_vec = np.linspace(0.0, T_sim, N_steps)
    q_init = np.array([0.0, -np.pi / 4, np.pi / 4, 0.0, np.pi / 6, 0.0])
    dq_init = np.zeros(n)
    tau_zero = np.zeros(n)

    # 3A. CasADi with RK4 (4 stage evaluations per step)
    q_curr = q_init.copy()
    dq_curr = dq_init.copy()
    q_hist_casadi_rk4 = np.zeros((N_steps, n))
    t0 = time.perf_counter()
    for k in range(N_steps):
        q_hist_casadi_rk4[k, :] = q_curr
        q_curr, dq_curr = run_casadi_rk4_step(casadi_fwd_acc, q_curr, dq_curr, tau_zero, dt)
    t_casadi_rk4 = time.perf_counter() - t0
    rtf_casadi_rk4 = T_sim / t_casadi_rk4
    avg_step_casadi_rk4_us = (t_casadi_rk4 / N_steps) * 1e6

    # 3B. CasADi with Semi-implicit Euler (1 evaluation per step)
    q_curr = q_init.copy()
    dq_curr = dq_init.copy()
    q_hist_casadi_euler = np.zeros((N_steps, n))
    t0 = time.perf_counter()
    for k in range(N_steps):
        q_hist_casadi_euler[k, :] = q_curr
        q_curr, dq_curr = run_casadi_euler_step(casadi_fwd_acc, q_curr, dq_curr, tau_zero, dt)
    t_casadi_euler = time.perf_counter() - t0
    rtf_casadi_euler = T_sim / t_casadi_euler
    avg_step_casadi_euler_us = (t_casadi_euler / N_steps) * 1e6

    # 3C. MuJoCo mj_step (native C integrator)
    twin.data.qpos[:] = q_init
    twin.data.qvel[:] = dq_init
    twin.data.qacc[:] = 0.0
    twin.data.ctrl[:] = 0.0
    twin.data.qfrc_applied[:] = 0.0
    q_hist_mujoco = np.zeros((N_steps, n))
    t0 = time.perf_counter()
    for k in range(N_steps):
        q_hist_mujoco[k, :] = twin.data.qpos.copy()
        mujoco.mj_step(twin.model, twin.data)
    t_mujoco_sim = time.perf_counter() - t0
    rtf_mujoco = T_sim / t_mujoco_sim
    avg_step_mujoco_us = (t_mujoco_sim / N_steps) * 1e6

    # 3D. CasADi Accelerated Batch Simulation (C++ VM mapaccum, 0 Python loop overhead)
    from urdf2dt import create_batch_simulator
    batch_sim_rk4 = create_batch_simulator(model, dt=dt, N_steps=N_steps, method="rk4")
    batch_sim_euler = create_batch_simulator(model, dt=dt, N_steps=N_steps, method="euler")
    x_init_dh = np.concatenate([signs * q_init, signs * dq_init])
    tau_batch = np.zeros((n, N_steps))

    # Warm-up
    _ = batch_sim_rk4(x_init_dh, tau_batch)
    _ = batch_sim_euler(x_init_dh, tau_batch)

    # Benchmark Batch RK4
    t0 = time.perf_counter()
    _ = batch_sim_rk4(x_init_dh, tau_batch)
    t_casadi_batch_rk4 = time.perf_counter() - t0
    rtf_casadi_batch_rk4 = T_sim / t_casadi_batch_rk4
    avg_step_batch_rk4_us = (t_casadi_batch_rk4 / N_steps) * 1e6

    # Benchmark Batch Euler
    t0 = time.perf_counter()
    _ = batch_sim_euler(x_init_dh, tau_batch)
    t_casadi_batch_euler = time.perf_counter() - t0
    rtf_casadi_batch_euler = T_sim / t_casadi_batch_euler
    avg_step_batch_euler_us = (t_casadi_batch_euler / N_steps) * 1e6

    print(f"  CasADi (RK4 Py-Loop)  : Total = {t_casadi_rk4:.4f} s | Step = {avg_step_casadi_rk4_us:7.2f} us | RTF = {rtf_casadi_rk4:7.1f}x real-time")
    print(f"  CasADi (Euler Py-Loop): Total = {t_casadi_euler:.4f} s | Step = {avg_step_casadi_euler_us:7.2f} us | RTF = {rtf_casadi_euler:7.1f}x real-time")
    print(f"  CasADi (Batch RK4)    : Total = {t_casadi_batch_rk4:.4f} s | Step = {avg_step_batch_rk4_us:7.2f} us | RTF = {rtf_casadi_batch_rk4:7.1f}x real-time")
    print(f"  CasADi (Batch Euler)  : Total = {t_casadi_batch_euler:.4f} s | Step = {avg_step_batch_euler_us:7.2f} us | RTF = {rtf_casadi_batch_euler:7.1f}x real-time")
    print(f"  MuJoCo mj_step        : Total = {t_mujoco_sim:.4f} s | Step = {avg_step_mujoco_us:7.2f} us | RTF = {rtf_mujoco:7.1f}x real-time")
    print(f"  --> Batch Acceleration: Batch RK4 is {t_casadi_rk4 / t_casadi_batch_rk4:.2f}x faster than Python RK4; Batch Euler is {t_casadi_euler / t_casadi_batch_euler:.2f}x faster than Python Euler.")
    print(f"  --> Gap vs MuJoCo     : MuJoCo is only {t_casadi_batch_euler / t_mujoco_sim:.2f}x faster than CasADi Batch Euler (down from {t_casadi_euler / t_mujoco_sim:.2f}x).")
    print()

    # =========================================================================
    # BENCHMARK 4: Closed-Loop Computed Torque Control Simulation (5,000 steps)
    # =========================================================================
    print("-" * 80)
    print(">>> BENCHMARK 4: Closed-Loop Computed Torque Trajectory Control (5,000 steps)")
    print("-" * 80)
    q_start = np.zeros(n)
    q_goal = np.array([0.5, -0.8, 0.6, -0.4, 0.5, -0.3])
    time_ref, q_ref, dq_ref, ddq_ref = traj_poly5(q_start, q_goal, T_sim, dt=dt)
    N_steps = len(time_ref)

    ctrl = ComputedTorqueController(model, kp=400.0, kd=40.0, joint_signs=signs)

    # 4A. CasADi Closed-Loop (Controller + RK4 Forward Dynamics)
    q_curr = q_start.copy()
    dq_curr = np.zeros(n)
    q_cl_casadi = np.zeros((N_steps, n))
    t0 = time.perf_counter()
    for k in range(N_steps):
        q_cl_casadi[k, :] = q_curr
        tau_cmd = ctrl.compute_control(q_curr, dq_curr, q_ref[k], dq_ref[k], ddq_ref[k], time_ref[k])
        q_curr, dq_curr = run_casadi_rk4_step(casadi_fwd_acc, q_curr, dq_curr, tau_cmd, dt)
    t_cl_casadi = time.perf_counter() - t0
    rtf_cl_casadi = T_sim / t_cl_casadi
    avg_cl_casadi_us = (t_cl_casadi / N_steps) * 1e6

    # 4B. MuJoCo Closed-Loop (Controller + mj_step)
    twin.data.qpos[:] = q_start
    twin.data.qvel[:] = 0.0
    twin.data.qacc[:] = 0.0
    twin.data.ctrl[:] = 0.0
    twin.data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(twin.model, twin.data)
    q_cl_mujoco = np.zeros((N_steps, n))
    t0 = time.perf_counter()
    for k in range(N_steps):
        q_k = twin.data.qpos.copy()
        dq_k = twin.data.qvel.copy()
        q_cl_mujoco[k, :] = q_k
        tau_cmd = ctrl.compute_control(q_k, dq_k, q_ref[k], dq_ref[k], ddq_ref[k], time_ref[k])
        twin.data.ctrl[:] = tau_cmd
        mujoco.mj_step(twin.model, twin.data)
    t_cl_mujoco = time.perf_counter() - t0
    rtf_cl_mujoco = T_sim / t_cl_mujoco
    avg_cl_mujoco_us = (t_cl_mujoco / N_steps) * 1e6

    # Trajectory coincidence between both engines
    max_discrepancy = np.max(np.abs(q_cl_casadi - q_cl_mujoco))
    rms_discrepancy = np.sqrt(np.mean((q_cl_casadi - q_cl_mujoco)**2))

    print(f"  CasADi Closed-Loop : Total = {t_cl_casadi:.4f} s | Step = {avg_cl_casadi_us:7.2f} us | RTF = {rtf_cl_casadi:7.1f}x real-time")
    print(f"  MuJoCo Closed-Loop : Total = {t_cl_mujoco:.4f} s | Step = {avg_cl_mujoco_us:7.2f} us | RTF = {rtf_cl_mujoco:7.1f}x real-time")
    print(f"  --> Speedup Ratio  : MuJoCo is {t_cl_casadi / t_cl_mujoco:.2f}x faster for closed-loop simulation.")
    print(f"  --> Coincidence    : Max state difference between CasADi & MuJoCo = {max_discrepancy:.4e} rad (RMS = {rms_discrepancy:.4e} rad).")
    print()

    # =========================================================================
    # EXECUTIVE SUMMARY DASHBOARD
    # =========================================================================
    print("=" * 80)
    print("                    EXECUTIVE SUMMARY & SPEEDUP COMPARISON                   ")
    print("=" * 80)
    print(f"{'Task / Simulation Mode':<35} | {'CasADi Time':<12} | {'MuJoCo Time':<12} | {'Speedup':<8} | {'CasADi RTF':<10} | {'MuJoCo RTF':<10}")
    print("-" * 105)
    print(f"{'Raw Forward Dynamics (10k calls)':<35} | {t_casadi_raw:9.4f} s  | {t_mujoco_raw:9.4f} s  | {t_casadi_raw/t_mujoco_raw:6.2f}x  | {'N/A':<10} | {'N/A':<10}")
    print(f"{'Raw Inverse Dynamics (10k calls)':<35} | {t_casadi_inv:9.4f} s  | {t_mujoco_inv:9.4f} s  | {t_casadi_inv/t_mujoco_inv:6.2f}x  | {'N/A':<10} | {'N/A':<10}")
    print(f"{'Passive Sim 5s (Euler Py-Loop)':<35} | {t_casadi_euler:9.4f} s  | {t_mujoco_sim:9.4f} s  | {t_casadi_euler/t_mujoco_sim:6.2f}x  | {rtf_casadi_euler:7.1f}x   | {rtf_mujoco:7.1f}x")
    print(f"{'Passive Sim 5s (RK4 Py-Loop)':<35} | {t_casadi_rk4:9.4f} s  | {t_mujoco_sim:9.4f} s  | {t_casadi_rk4/t_mujoco_sim:6.2f}x  | {rtf_casadi_rk4:7.1f}x   | {rtf_mujoco:7.1f}x")
    print(f"{'Passive Sim 5s (Batch Euler)':<35} | {t_casadi_batch_euler:9.4f} s  | {t_mujoco_sim:9.4f} s  | {t_casadi_batch_euler/t_mujoco_sim:6.2f}x  | {rtf_casadi_batch_euler:7.1f}x   | {rtf_mujoco:7.1f}x")
    print(f"{'Passive Sim 5s (Batch RK4)':<35} | {t_casadi_batch_rk4:9.4f} s  | {t_mujoco_sim:9.4f} s  | {t_casadi_batch_rk4/t_mujoco_sim:6.2f}x  | {rtf_casadi_batch_rk4:7.1f}x   | {rtf_mujoco:7.1f}x")
    print(f"{'Closed-Loop Control 5s (1kHz)':<35} | {t_cl_casadi:9.4f} s  | {t_cl_mujoco:9.4f} s  | {t_cl_casadi/t_cl_mujoco:6.2f}x  | {rtf_cl_casadi:7.1f}x   | {rtf_cl_mujoco:7.1f}x")
    print("=" * 105)

    return {
        "t_casadi_raw": t_casadi_raw,
        "t_mujoco_raw": t_mujoco_raw,
        "t_casadi_inv": t_casadi_inv,
        "t_mujoco_inv": t_mujoco_inv,
        "t_casadi_rk4": t_casadi_rk4,
        "t_casadi_euler": t_casadi_euler,
        "t_casadi_batch_rk4": t_casadi_batch_rk4,
        "t_casadi_batch_euler": t_casadi_batch_euler,
        "t_mujoco_sim": t_mujoco_sim,
        "t_cl_casadi": t_cl_casadi,
        "t_cl_mujoco": t_cl_mujoco,
        "max_discrepancy": max_discrepancy,
        "rms_discrepancy": rms_discrepancy,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UR16e CasADi vs MuJoCo Benchmark")
    parser.add_argument("--gravity", type=float, nargs="+", default=[0.0, 0.0, -9.81],
                        help="Customizable 3D gravity vector [gx gy gz] (default: 0 0 -9.81)")
    args = parser.parse_args()
    g_input = tuple(args.gravity) if len(args.gravity) == 3 else args.gravity[0] if len(args.gravity) == 1 else (0.0, 0.0, -9.81)
    benchmark_ur16e(gravity=g_input)
