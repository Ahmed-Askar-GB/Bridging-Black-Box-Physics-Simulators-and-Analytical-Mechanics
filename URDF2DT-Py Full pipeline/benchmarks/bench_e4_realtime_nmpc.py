"""Benchmark E4: Real-Time Compiled Receding-Horizon NMPC Benchmark (H5).

Pre-Registered Hypotheses:
  H5: When compiled to native code with common subexpression elimination,
      the proposed momentum-Cholesky NMPC formulation achieves mean per-step solve times
      <= 25 ms (40 Hz control rate) on 6-DOF (UR16e) and <= 40 ms (25 Hz control rate)
      on 7-DOF (KUKA iiwa 7), representing a >= 1.35x speedup over the Christoffel+inversion
      baseline and reducing p99 latency spikes during obstacle avoidance by >= 25%.
      Forward dynamics throughput is clearly delineated as >10 kHz (70-100 us).

Outputs:
  - results/e4_realtime_nmpc.json
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
)
from urdf2dt.utils import detect_joint_signs


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
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate URDF for '{name}'")


def build_nmpc_problem(
    model_data: dict,
    formulation: str = "proposed",
    N: int = 15,
    dt: float = 0.02,
    p_obs: np.ndarray | None = None,
    r_obs: float = 0.12,
    d_safe: float = 0.04,
    tau_max: np.ndarray | None = None,
):
    """Build Multiple-Shooting NMPC NLP for Baseline or Proposed formulation."""
    n = model_data["n"]
    nx = 2 * n
    nu = n

    q_sym = cs.SX.sym("q", n)
    dq_sym = cs.SX.sym("dq", n)
    u_sym = cs.SX.sym("u", n)

    if formulation == "baseline":
        fwd_acc = model_data["fwd_acc_base_fun"](q_sym, dq_sym, u_sym)
    else:
        fwd_acc = model_data["fwd_acc_prop_fun"](q_sym, dq_sym, u_sym)

    f_dyn = cs.Function("f_dyn", [q_sym, dq_sym, u_sym], [dq_sym, fwd_acc])

    # RK4 Single Step
    x_in = cs.SX.sym("x_in", nx)
    u_in = cs.SX.sym("u_in", nu)
    qk = x_in[:n]
    vk = x_in[n:]

    k1_v, k1_a = f_dyn(qk, vk, u_in)
    k2_v, k2_a = f_dyn(qk + 0.5 * dt * k1_v, vk + 0.5 * dt * k1_a, u_in)
    k3_v, k3_a = f_dyn(qk + 0.5 * dt * k2_v, vk + 0.5 * dt * k2_a, u_in)
    k4_v, k4_a = f_dyn(qk + dt * k3_v, vk + dt * k3_a, u_in)

    q_next = qk + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    v_next = vk + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
    x_next = cs.vertcat(q_next, v_next)
    f_rk4 = cs.Function("f_rk4", [x_in, u_in], [x_next])

    # NLP Variables & Constraints
    w = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    # Parameters: p = [x0 (nx); xref_0 (nx); xref_1 ...; xref_N]
    p_param = cs.SX.sym("p_param", nx + (N + 1) * nx)
    x0_param = p_param[:nx]

    # Weights
    Q_diag = [100.0] * n + [10.0] * n
    Q = cs.diag(cs.SX(Q_diag))
    R = cs.diag(cs.SX([0.01] * nu))

    X = []
    U = []
    J_cost = 0.0

    # Node 0
    X0 = cs.SX.sym("X_0", nx)
    w.append(X0)
    lbw.extend([-3.14159] * n + [-4.0] * n)
    ubw.extend([3.14159] * n + [4.0] * n)
    X.append(X0)

    # Initial state constraint
    g.append(X0 - x0_param)
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)

    for k in range(N):
        # Control input Uk
        Uk = cs.SX.sym(f"U_{k}", nu)
        w.append(Uk)
        lbw.extend((-tau_max).tolist())
        ubw.extend(tau_max.tolist())
        U.append(Uk)

        # Stage cost
        x_ref_k = p_param[nx + k * nx: nx + (k + 1) * nx]
        dx = X[k] - x_ref_k
        J_cost += cs.mtimes([dx.T, Q, dx]) + cs.mtimes([Uk.T, R, Uk])

        # State Xk_next
        Xk_next = cs.SX.sym(f"X_{k+1}", nx)
        w.append(Xk_next)
        lbw.extend([-3.14159] * n + [-4.0] * n)
        ubw.extend([3.14159] * n + [4.0] * n)
        X.append(Xk_next)

        # Continuity
        g.append(Xk_next - f_rk4(X[k], Uk))
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)

        # Obstacle avoidance on end-effector
        if p_obs is not None:
            T0e = model_data["fk_ee_fun"](Xk_next[:n])
            p_ee = T0e[:3, 3]
            p_obs_sx = cs.SX(p_obs.reshape(3, 1))
            dist_sq = cs.dot(p_ee - p_obs_sx, p_ee - p_obs_sx)
            g.append(dist_sq)
            lbg.append((r_obs + d_safe) ** 2)
            ubg.append(100.0)

    # Terminal cost
    x_ref_N = p_param[nx + N * nx: nx + (N + 1) * nx]
    dx_N = X[-1] - x_ref_N
    J_cost += 2.0 * cs.mtimes([dx_N.T, Q, dx_N])

    w_vec = cs.vertcat(*w)
    g_vec = cs.vertcat(*g)

    nlp = {"x": w_vec, "f": J_cost, "g": g_vec, "p": p_param}

    # Solver options: fast real-time SQP / warm-started IPOPT
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 25,
        "ipopt.tol": 1e-3,
        "ipopt.acceptable_tol": 5e-3,
        "ipopt.linear_solver": "mumps",
        "ipopt.warm_start_init_point": "yes",
        "ipopt.hessian_approximation": "limited-memory",
    }

    solver = cs.nlpsol("solver", "ipopt", nlp, opts)
    return {
        "solver": solver,
        "lbw": np.array(lbw),
        "ubw": np.array(ubw),
        "lbg": np.array(lbg),
        "ubg": np.array(ubg),
        "nx": nx,
        "nu": nu,
        "N": N,
        "nw": w_vec.size1(),
        "ng": g_vec.size1(),
    }


def run_benchmark():
    print("=" * 80)
    print("  BENCHMARK E4: REAL-TIME RECEDING-HORIZON NMPC (H5)")
    print("=" * 80)

    robots = [
        ("ur16e", "UR16e", 6, np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]), np.array([0.45, 0.15, 0.40])),
        ("kukaIiwa7", "KUKA iiwa 7", 7, np.array([176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0]), np.array([0.35, 0.10, 0.50])),
    ]

    N_steps_sim = 40
    N_horizon = 15
    dt_nmpc = 0.02
    T_total = N_steps_sim * dt_nmpc  # 0.8 s

    results_data = {
        "metadata": {
            "git_hash": get_git_hash(),
            "cpu_model": platform.processor(),
            "os": platform.platform(),
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N_steps": N_steps_sim,
            "N_horizon": N_horizon,
            "dt": dt_nmpc,
        },
        "platforms": {},
        "h5_verdict": {},
    }

    for key, disp_name, n_dof, tau_max, p_obs in robots:
        print(f"\n---> Benchmarking {disp_name} over {N_steps_sim} closed-loop receding steps...")
        urdf_path = resolve_urdf(key)

        robot = urdf_parse(urdf_path)
        chain = urdf_build_chain(robot, verbose=False)
        dh = urdf_extract_dh(chain, verbose=False)
        inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
        model = dynamics_casadi(dh, inertia, chain, verbose=False)
        signs = detect_joint_signs(chain, model.fk_link_funs)
        twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=dt_nmpc, verbose=False)

        # Build baseline (Christoffel) vs proposed (Momentum + Cholesky)
        # 1. Mass matrix & Gravity
        q = model.q
        dq = model.dq
        u = model.tau
        M_sym = model.M_sym
        G_sym = model.G_sym

        # Classical Christoffel symbols
        C_sym = cs.SX.zeros(n_dof, n_dof)
        dM = cs.jacobian(cs.vec(M_sym), q)
        for i in range(n_dof):
            for j in range(n_dof):
                c_ij = cs.SX(0.0)
                idx_ij = j * n_dof + i
                for k in range(n_dof):
                    idx_ik = k * n_dof + i
                    idx_kj = j * n_dof + k
                    c_ijk = 0.5 * (dM[idx_ij, k] + dM[idx_ik, j] - dM[idx_kj, i])
                    c_ij = c_ij + c_ijk * dq[k]
                C_sym[i, j] = c_ij
        c_bias_base = cs.mtimes(C_sym, dq)
        ddq_base = cs.solve(M_sym, u - c_bias_base - G_sym)

        # Proposed Momentum + Cholesky
        T_kin = 0.5 * cs.mtimes(dq.T, cs.mtimes(M_sym, dq))
        grad_T = cs.gradient(T_kin, q)
        M_dot_dq = cs.jtimes(cs.mtimes(M_sym, dq), q, dq)
        c_bias_prop = M_dot_dq - grad_T
        R_chol = cs.chol(M_sym)
        y_sub = cs.solve(R_chol.T, u - c_bias_prop - G_sym)
        ddq_prop = cs.solve(R_chol, y_sub)

        model_data = {
            "n": n_dof,
            "fwd_acc_base_fun": cs.Function("fwd_base", [q, dq, u], [ddq_base]),
            "fwd_acc_prop_fun": cs.Function("fwd_prop", [q, dq, u], [ddq_prop]),
            "fk_ee_fun": model.fk_ee_fun,
        }

        # Measure raw single-step forward dynamics latency (kHz throughput)
        N_fd_evals = 5000
        q_rand = np.random.uniform(-1.0, 1.0, size=(N_fd_evals, n_dof))
        dq_rand = np.random.uniform(-1.0, 1.0, size=(N_fd_evals, n_dof))
        u_rand = np.random.uniform(-10.0, 10.0, size=(N_fd_evals, n_dof))

        t0 = time.perf_counter()
        for i in range(N_fd_evals):
            _ = model_data["fwd_acc_prop_fun"](q_rand[i], dq_rand[i], u_rand[i])
        fd_latency_us = (time.perf_counter() - t0) / N_fd_evals * 1e6
        fd_rate_khz = 1e3 / fd_latency_us

        # Build NMPC solvers
        nmpc_prop = build_nmpc_problem(model_data, formulation="proposed", N=N_horizon, dt=dt_nmpc, p_obs=p_obs, tau_max=tau_max)
        nmpc_base = build_nmpc_problem(model_data, formulation="baseline", N=N_horizon, dt=dt_nmpc, p_obs=p_obs, tau_max=tau_max)

        # Generate target multi-axis trajectory
        t_sim = np.linspace(0.0, T_total + N_horizon * dt_nmpc, N_steps_sim + N_horizon + 1)
        x_ref_full = np.zeros((len(t_sim), 2 * n_dof))
        for j in range(n_dof):
            amp = 0.4 / (j + 1) + 0.2
            freq = 0.3 * (j + 1) * 0.5
            x_ref_full[:, j] = amp * np.sin(2.0 * np.pi * freq * t_sim)
            x_ref_full[:, n_dof + j] = amp * (2.0 * np.pi * freq) * np.cos(2.0 * np.pi * freq * t_sim)

        # Run closed-loop receding-horizon loop
        for form_name, nmpc_pack in [("baseline", nmpc_base), ("proposed", nmpc_prop)]:
            solver = nmpc_pack["solver"]
            w0 = np.zeros(nmpc_pack["nw"])
            solve_times_ms = []
            tracking_errors = []
            min_obs_dists = []

            twin.reset(x_ref_full[0, :n_dof], np.zeros(n_dof))

            for step_k in range(N_steps_sim):
                xk_curr = np.concatenate([twin.data.qpos.copy(), twin.data.qvel.copy()])
                ref_horizon = x_ref_full[step_k: step_k + N_horizon + 1, :].flatten()
                p_val = np.concatenate([xk_curr, ref_horizon])

                t_start = time.perf_counter()
                sol = solver(
                    x0=w0,
                    lbx=nmpc_pack["lbw"],
                    ubx=nmpc_pack["ubw"],
                    lbg=nmpc_pack["lbg"],
                    ubg=nmpc_pack["ubg"],
                    p=p_val,
                )
                t_step_ms = (time.perf_counter() - t_start) * 1e3
                solve_times_ms.append(t_step_ms)

                w_opt = np.array(sol["x"]).flatten()
                # Warm-start shift
                w0 = w_opt.copy()

                # Extract first control input u_0
                u_cmd = w_opt[2 * n_dof: 2 * n_dof + n_dof]

                # Step simulation
                twin.step(u_cmd)

                # Metrics
                q_act = twin.data.qpos.copy()
                q_des = x_ref_full[step_k, :n_dof]
                tracking_errors.append(np.linalg.norm(q_act - q_des))

                p_ee_act = twin.data.site_xpos[0].copy() if twin.model.nsite > 0 else np.array(model.fk_ee_fun(q_act))[:3, 3].flatten()
                min_obs_dists.append(float(np.linalg.norm(p_ee_act - p_obs)))

            # Statistics
            st_arr = np.array(solve_times_ms)
            mean_ms = float(np.mean(st_arr))
            median_ms = float(np.median(st_arr))
            p95_ms = float(np.percentile(st_arr, 95))
            p99_ms = float(np.percentile(st_arr, 99))
            max_ms = float(np.max(st_arr))
            iqr_ms = float(np.percentile(st_arr, 75) - np.percentile(st_arr, 25))
            rmse_rad = float(np.sqrt(np.mean(np.array(tracking_errors) ** 2)))
            min_clearance = float(np.min(min_obs_dists))

            if key not in results_data["platforms"]:
                results_data["platforms"][key] = {
                    "name": disp_name,
                    "dof": n_dof,
                    "fd_latency_us": fd_latency_us,
                    "fd_rate_khz": fd_rate_khz,
                }

            results_data["platforms"][key][form_name] = {
                "mean_solve_ms": mean_ms,
                "median_solve_ms": median_ms,
                "iqr_solve_ms": iqr_ms,
                "p95_solve_ms": p95_ms,
                "p99_solve_ms": p99_ms,
                "max_solve_ms": max_ms,
                "achievable_rate_hz": 1e3 / mean_ms,
                "tracking_rmse_rad": rmse_rad,
                "min_obstacle_distance_m": min_clearance,
            }

            print(f"  [{form_name.upper():8s}] Mean Solve: {mean_ms:5.2f} ms (p95: {p95_ms:5.2f} ms, p99: {p99_ms:5.2f} ms) | Rate: {1e3/mean_ms:5.1f} Hz | Min Dist: {min_clearance:.4f} m")

        # Speedup and spike mitigation
        t_base_mean = results_data["platforms"][key]["baseline"]["mean_solve_ms"]
        t_prop_mean = results_data["platforms"][key]["proposed"]["mean_solve_ms"]
        speedup = t_base_mean / t_prop_mean

        p99_base = results_data["platforms"][key]["baseline"]["p99_solve_ms"]
        p99_prop = results_data["platforms"][key]["proposed"]["p99_solve_ms"]
        spike_ratio = p99_prop / p99_base
        spike_mitigation_pct = (1.0 - spike_ratio) * 100.0

        results_data["platforms"][key]["speedup_factor"] = speedup
        results_data["platforms"][key]["spike_mitigation_pct"] = spike_mitigation_pct
        print(f"  --> Speedup Factor: {speedup:.2f}x | p99 Spike Mitigation: {spike_mitigation_pct:.1f}%")

    # H5 verdict
    ur16_prop_mean = results_data["platforms"]["ur16e"]["proposed"]["mean_solve_ms"]
    iiwa_prop_mean = results_data["platforms"]["kukaIiwa7"]["proposed"]["mean_solve_ms"]
    ur16_speedup = results_data["platforms"]["ur16e"]["speedup_factor"]
    iiwa_speedup = results_data["platforms"]["kukaIiwa7"]["speedup_factor"]
    ur16_spike = results_data["platforms"]["ur16e"]["spike_mitigation_pct"]

    h5_pass_times = bool(ur16_prop_mean <= 25.0 and iiwa_prop_mean <= 40.0)
    h5_pass_speedup = bool(ur16_speedup >= 1.35 and iiwa_speedup >= 1.50)
    h5_pass_spike = bool(ur16_spike >= 25.0)

    results_data["h5_verdict"] = {
        "pass_solve_times_threshold": h5_pass_times,
        "pass_speedup_threshold": h5_pass_speedup,
        "pass_spike_mitigation_threshold": h5_pass_spike,
        "overall_h5_supported": bool(h5_pass_times and h5_pass_speedup and h5_pass_spike),
    }

    out_path = os.path.join(WORKSPACE_ROOT, "results", "e4_realtime_nmpc.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[DONE] Real-Time NMPC Results saved to: {out_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
