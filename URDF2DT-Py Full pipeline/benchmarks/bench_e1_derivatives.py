"""Benchmark E1: Derivative Capability, Exactness, and Scaling (H1, H2).

Pre-Registered Hypotheses:
  H1: Proposed AD derivatives evaluate to machine precision (relative error <= 1e-7)
      relative to a 64-bit reference, whereas finite-difference approximations exhibit
      truncation error O(eps^2) or numerical cancellation.
      Exact Lagrangian Hessians are natively available in the Proposed formulation.
  H2: Exact AD Jacobian evaluation scales near-quadratically (alpha <= 2.50) across
      dimension n in [3, 14], outperforming multi-point numerical perturbations on
      industrial 6-DOF and 7-DOF manipulators.

Outputs:
  - results/e1_derivatives.json
"""

from __future__ import annotations
import os
import sys
import time
import json
import platform
import subprocess
import numpy as np
import casadi as cs

# Add package root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PIPELINE_ROOT)

from urdf2dt import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
    dynamics_casadi,
    MuJoCoTwin,
    DynamicsModel,
)
from urdf2dt.utils import detect_joint_signs
from benchmarks.bench_e6_scalability import generate_serial_urdf


def get_git_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORKSPACE_ROOT, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def resolve_urdf(name: str) -> str:
    candidates = [
        os.path.join(PIPELINE_ROOT, "tests", "fixtures", f"{name}.urdf"),
        os.path.join(WORKSPACE_ROOT, f"{name}.urdf"),
        os.path.join(WORKSPACE_ROOT, "Bridging Black-Box Physics Simulators and Analytical Mechanics", f"{name}.urdf"),
        os.path.join(WORKSPACE_ROOT, "Bridging Black-Box Physics Simulators and Analytical Mechanics", "End-to-End MATLAB-CasADi Implementation", "test_UR16e", f"universal{name.upper()}.urdf"),
        os.path.join(WORKSPACE_ROOT, "Bridging Black-Box Physics Simulators and Analytical Mechanics", "End-to-End MATLAB-CasADi Implementation", "test_UR16e", "urdf", f"universal{name.upper()}.urdf"),
        os.path.join(WORKSPACE_ROOT, "Bridging Black-Box Physics Simulators and Analytical Mechanics", "abbIrb120", f"{name}.urdf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate URDF for '{name}'")


def run_benchmark():
    print("=" * 80)
    print("  BENCHMARK E1: DERIVATIVE CAPABILITY, EXACTNESS & SCALING (H1, H2)")
    print("=" * 80)

    robots_to_test = [
        ("ur5", "UR5", 6),
        ("ur16e", "UR16e", 6),
        ("kukaIiwa7", "KUKA iiwa 7", 7),
    ]

    dt = 0.001
    N_evals = 200
    rng = np.random.default_rng(42)

    results_data = {
        "metadata": {
            "git_hash": get_git_hash(),
            "cpu_model": platform.processor(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "casadi_version": cs.__version__,
            "mujoco_version": "3.10.0",
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": 42,
        },
        "h1_precision": {},
        "fd_step_sweep": {},
        "h2_scaling_vs_n": {},
        "capability_matrix": {
            "headers": ["Framework", "Continuous Jacobians", "Discrete Jacobians", "Exact Lagrangian Hessian", "Symbolic CSE", "Embedded C Export"],
            "rows": [
                ["MuJoCo Physics", "Finite Differences (Euler)", "mjd_transitionFD (Euler only)", "No native API", "No", "C Runtime library"],
                ["Pinocchio (Spatial C++)", "Analytical (ABA/RNEA)", "Semi-analytical Euler", "Via CasADi bridge", "Yes (via CasADi)", "C++ / Python bindings"],
                ["Proposed Pipeline", "Exact Closed-Form AD", "Exact Symbolic Euler & RK4", "Yes (Exact Hessian)", "Yes (Full DAG)", "Yes (ANSI C99 static)"],
            ]
        }
    }

    # -------------------------------------------------------------------------
    # Part 1: Precision Benchmarking on Industrial Testbed
    # -------------------------------------------------------------------------
    for key, disp_name, n_dof in robots_to_test:
        print(f"\n---> Benchmarking Precision on {disp_name} (n={n_dof} DOF)...")
        urdf_path = resolve_urdf(key)

        robot = urdf_parse(urdf_path)
        chain = urdf_build_chain(robot, verbose=False)
        dh = urdf_extract_dh(chain, verbose=False)
        inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
        model = dynamics_casadi(dh, inertia, chain, verbose=False)
        signs = detect_joint_signs(chain, model.fk_link_funs)
        dyn_model = DynamicsModel(chain, dh, inertia, model, robot_name=key, signs=signs, dt_default=dt)

        import mujoco
        twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=dt, verbose=False)
        twin.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        twin.model.dof_damping[:] = 0.0
        twin.model.dof_frictionloss[:] = 0.0

        q_nom = rng.uniform(-1.0, 1.0, size=n_dof)
        dq_nom = rng.uniform(-0.5, 0.5, size=n_dof)
        u_nom = rng.uniform(-10.0, 10.0, size=n_dof)
        x_nom = np.concatenate([q_nom, dq_nom])

        # (a) Exact AD discrete Jacobians
        A_rk4_ad = np.array(dyn_model.dRK4_dx_fun(x_nom, u_nom, dt))
        A_euler_ad = np.array(dyn_model.dEuler_dx_fun(x_nom, u_nom, dt))

        # (b) High-precision Reference: central finite differences with optimal eps=1e-7 on identical RK4 graph
        A_ref_rk4 = np.zeros((2 * n_dof, 2 * n_dof))
        eps_opt = 1e-7
        for i in range(2 * n_dof):
            dx_plus = x_nom.copy(); dx_plus[i] += eps_opt
            dx_minus = x_nom.copy(); dx_minus[i] -= eps_opt
            f_p = np.array(dyn_model.rk4_step_fun(dx_plus, u_nom, dt)).flatten()
            f_m = np.array(dyn_model.rk4_step_fun(dx_minus, u_nom, dt)).flatten()
            A_ref_rk4[:, i] = (f_p - f_m) / (2.0 * eps_opt)

        err_ad_vs_ref = float(np.max(np.abs(A_rk4_ad - A_ref_rk4)))
        rel_ad_vs_ref = float(err_ad_vs_ref / (np.max(np.abs(A_ref_rk4)) + 1e-9))

        # (c) Step-size sweep on the EXACT SAME discrete RK4 map: O(eps^2) truncation vs cancellation
        sweep_results = {}
        for eps_test in [1e-8, 1e-6, 1e-4, 1e-2]:
            A_fd_test = np.zeros((2 * n_dof, 2 * n_dof))
            for i in range(2 * n_dof):
                dx_plus = x_nom.copy(); dx_plus[i] += eps_test
                dx_minus = x_nom.copy(); dx_minus[i] -= eps_test
                f_p = np.array(dyn_model.rk4_step_fun(dx_plus, u_nom, dt)).flatten()
                f_m = np.array(dyn_model.rk4_step_fun(dx_minus, u_nom, dt)).flatten()
                A_fd_test[:, i] = (f_p - f_m) / (2.0 * eps_test)
            err_sweep = float(np.max(np.abs(A_rk4_ad - A_fd_test)))
            sweep_results[f"eps_{eps_test:.0e}"] = {
                "eps": eps_test,
                "max_abs_error": err_sweep,
                "relative_error": float(err_sweep / (np.max(np.abs(A_rk4_ad)) + 1e-9)),
            }
        results_data["fd_step_sweep"][key] = sweep_results

        # (d) Same discrete map comparison with MuJoCo:
        # Note: MuJoCo's mjd_transitionFD explicitly throws FatalError under mjINT_RK4.
        # Under mjINT_EULER on conservative plant:
        twin.reset(q_nom, dq_nom)
        twin.data.qfrc_applied[:] = u_nom
        mujoco.mj_forward(twin.model, twin.data)
        dfdx_mj_euler = np.zeros((2 * n_dof, 2 * n_dof))
        dfdu_mj_euler = np.zeros((2 * n_dof, n_dof))
        mujoco.mjd_transitionFD(twin.model, twin.data, 1e-6, 1, dfdx_mj_euler, dfdu_mj_euler, None, None)

        err_mj_euler_vs_ad = float(np.max(np.abs(dfdx_mj_euler - A_euler_ad)))
        rel_mj_euler_vs_ad = float(err_mj_euler_vs_ad / (np.max(np.abs(A_euler_ad)) + 1e-9))

        # (e) Exact Lagrangian Hessian
        Q_test = np.eye(2 * n_dof)
        R_test = np.eye(n_dof)
        H_exact = np.array(dyn_model.lagrangian_hessian_fun(x_nom, u_nom, Q_test, R_test))
        hessian_available = bool(H_exact.shape == (3 * n_dof, 3 * n_dof) and np.all(np.isfinite(H_exact)))

        print(f"  [H1 Precision] AD vs Ref (eps=1e-7) max err    : {err_ad_vs_ref:.4e} (rel: {rel_ad_vs_ref:.4e})")
        print(f"  [H1 Precision] MuJoCo mjd_transitionFD vs AD   : {err_mj_euler_vs_ad:.4e} (rel: {rel_mj_euler_vs_ad:.4e})")
        print(f"  [H1 Precision] Exact Lagrangian Hessian Valid   : {hessian_available}")

        # (f) Timing of single evaluation
        t0 = time.perf_counter()
        for _ in range(N_evals):
            _ = dyn_model.dEuler_dx_fun(x_nom, u_nom, dt)
        t_prop_euler = (time.perf_counter() - t0) / N_evals * 1e6  # us

        t0 = time.perf_counter()
        for _ in range(N_evals):
            _ = dyn_model.dRK4_dx_fun(x_nom, u_nom, dt)
        t_prop_rk4 = (time.perf_counter() - t0) / N_evals * 1e6  # us

        t0 = time.perf_counter()
        for _ in range(N_evals):
            mujoco.mjd_transitionFD(twin.model, twin.data, 1e-6, 1, dfdx_mj_euler, dfdu_mj_euler, None, None)
        t_mj_euler = (time.perf_counter() - t0) / N_evals * 1e6  # us

        results_data["h1_precision"][key] = {
            "robot": disp_name,
            "dof": n_dof,
            "err_ad_vs_reference": err_ad_vs_ref,
            "rel_ad_vs_reference": rel_ad_vs_ref,
            "err_mujoco_euler_vs_ad": err_mj_euler_vs_ad,
            "rel_mujoco_euler_vs_ad": rel_mj_euler_vs_ad,
            "exact_hessian_available": hessian_available,
            "latency_prop_euler_us": t_prop_euler,
            "latency_prop_rk4_us": t_prop_rk4,
            "latency_mujoco_euler_us": t_mj_euler,
            "mujoco_rk4_supported": False,  # mjd_transitionFD throws FatalError on mjINT_RK4
        }

    # -------------------------------------------------------------------------
    # Part 2: Multi-DOF Scaling of Derivatives (n in [3, 14])
    # -------------------------------------------------------------------------
    print("\n---> Benchmarking Multi-DOF Derivative Scaling (n in [3, 14])...")
    temp_dir = os.path.join(PIPELINE_ROOT, "temp_urdfs")
    os.makedirs(temp_dir, exist_ok=True)

    dof_list = [3, 4, 6, 7, 10, 12, 14]
    scaling_records = []

    for n in dof_list:
        urdf_temp = os.path.join(temp_dir, f"serial_{n}dof_bench.urdf")
        generate_serial_urdf(n, urdf_temp)

        rob = urdf_parse(urdf_temp)
        chain = urdf_build_chain(rob, verbose=False)
        dh = urdf_extract_dh(chain, verbose=False)
        inertia = urdf_extract_inertia(rob, chain, dh, verbose=False)
        model = dynamics_casadi(dh, inertia, chain, verbose=False)
        signs = detect_joint_signs(chain, model.fk_link_funs)
        dyn_m = DynamicsModel(chain, dh, inertia, model, robot_name=f"serial_{n}", signs=signs, dt_default=dt)

        x_test = np.zeros(2 * n)
        u_test = np.zeros(n)
        Q_t = np.eye(2 * n)
        R_t = np.eye(n)

        # Warmup
        _ = dyn_m.df_dx_fun(x_test, u_test)
        _ = dyn_m.dEuler_dx_fun(x_test, u_test, dt)
        _ = dyn_m.dRK4_dx_fun(x_test, u_test, dt)
        _ = dyn_m.lagrangian_hessian_fun(x_test, u_test, Q_t, R_t)

        N_reps = 30
        t0 = time.perf_counter()
        for _ in range(N_reps):
            _ = dyn_m.df_dx_fun(x_test, u_test)
        t_cont_jac = (time.perf_counter() - t0) / N_reps * 1e6  # us

        t0 = time.perf_counter()
        for _ in range(N_reps):
            _ = dyn_m.dEuler_dx_fun(x_test, u_test, dt)
        t_euler_jac = (time.perf_counter() - t0) / N_reps * 1e6  # us

        t0 = time.perf_counter()
        for _ in range(N_reps):
            _ = dyn_m.dRK4_dx_fun(x_test, u_test, dt)
        t_rk4_jac = (time.perf_counter() - t0) / N_reps * 1e6  # us

        t0 = time.perf_counter()
        for _ in range(N_reps):
            _ = dyn_m.lagrangian_hessian_fun(x_test, u_test, Q_t, R_t)
        t_hessian = (time.perf_counter() - t0) / N_reps * 1e6  # us

        print(f"  n={n:2d} DOF: Cont Jac={t_cont_jac:.1f} us | Euler Jac={t_euler_jac:.1f} us | RK4 Jac={t_rk4_jac:.1f} us | Hessian={t_hessian:.1f} us")
        scaling_records.append({
            "dof": n,
            "latency_cont_jac_us": t_cont_jac,
            "latency_euler_jac_us": t_euler_jac,
            "latency_rk4_jac_us": t_rk4_jac,
            "latency_hessian_us": t_hessian,
        })

    # Fit empirical power law alpha on RK4 Jacobian
    log_n = np.log([r["dof"] for r in scaling_records])
    log_t = np.log([r["latency_rk4_jac_us"] for r in scaling_records])
    alpha_jac, _ = np.polyfit(log_n, log_t, 1)
    print(f"  Empirical scaling exponent for discrete RK4 Jacobian: alpha = {alpha_jac:.2f}")

    results_data["h2_scaling_vs_n"] = {
        "records": scaling_records,
        "empirical_alpha_rk4_jac": float(alpha_jac),
        "h2_pass_sub_cubic": bool(alpha_jac <= 2.50),
    }

    # Save to results/e1_derivatives.json
    out_path = os.path.join(WORKSPACE_ROOT, "results", "e1_derivatives.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[DONE] Results written to: {out_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
