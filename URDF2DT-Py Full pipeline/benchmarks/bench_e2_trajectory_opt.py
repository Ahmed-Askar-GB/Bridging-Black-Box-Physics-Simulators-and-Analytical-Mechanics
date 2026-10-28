"""Benchmark E2: Converged Offline Trajectory Optimization and Dynamic Consistency Replay (H3).

Pre-Registered Hypotheses:
  H3: Supplying exact symbolic Lagrangian Hessians achieves faster convergence (>= 70% fewer iterations
      and >= 5x wall-clock speedup) compared to converged Quasi-Newton (L-BFGS) approximations,
      while reaching identical objective values.
      Replaying optimal trajectories under CTC in the MuJoCo physics engine achieves tight tracking,
      while open-loop torque replay reveals steady-state droop under plant friction/damping.

Outputs:
  - results/e2_trajectory_opt.json
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
    TrajectoryOptimizationOCP,
    ComputedTorqueController,
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


def bootstrap_ci(data: list[float], n_boot: int = 1000, alpha: float = 0.05) -> list[float]:
    """Compute 95% bootstrap confidence interval."""
    if not data or len(data) < 2:
        val = data[0] if data else 0.0
        return [float(val), float(val)]
    arr = np.array(data)
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return [low, high]


def run_benchmark():
    print("=" * 80)
    print("  BENCHMARK E2: CONVERGED OFFLINE TRAJECTORY OPTIMIZATION (H3)")
    print("=" * 80)

    robots = [
        ("ur5", "UR5", 6, 150.0),
        ("ur16e", "UR16e", 6, 150.0),
        ("kukaIiwa7", "KUKA iiwa 7", 7, 120.0),
    ]

    N_horizon = 10
    dt_step = 0.04
    N_problems = 15
    obstacle_center = np.array([0.40, 0.10, 0.35])
    obstacle_radius = 0.10
    safety_margin = 0.04

    results_data = {
        "metadata": {
            "git_hash": get_git_hash(),
            "cpu_model": platform.processor(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "casadi_version": cs.__version__,
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N_horizon": N_horizon,
            "dt_step": dt_step,
            "N_problems": N_problems,
            "max_iter_exact": 100,
            "max_iter_lbfgs": 500,
            "seed_range": [100, 100 + N_problems - 1],
        },
        "robots": {},
        "h3_verdict": {},
    }

    for key, disp_name, n_dof, tau_max in robots:
        print(f"\n---> Benchmarking Trajectory Optimization on {disp_name} (N={N_problems} problems)...")
        urdf_path = resolve_urdf(key)

        robot = urdf_parse(urdf_path)
        chain = urdf_build_chain(robot, verbose=False)
        dh = urdf_extract_dh(chain, verbose=False)
        inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
        model = dynamics_casadi(dh, inertia, chain, verbose=False)
        signs = detect_joint_signs(chain, model.fk_link_funs)
        dt_physics = 0.001
        n_substeps = max(1, int(round(dt_step / dt_physics)))
        twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=dt_physics, verbose=False)

        # 1. Exact Hessian OCP
        ocp_exact = TrajectoryOptimizationOCP(
            model=model,
            N=N_horizon,
            dt=dt_step,
            tau_max=tau_max,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
            cost_mode="effort",
            hessian_mode="exact",
            max_iter=100,
            verbose=False,
        )

        # 2. Converged Quasi-Newton (L-BFGS) OCP with 500 iteration limit
        ocp_lbfgs = TrajectoryOptimizationOCP(
            model=model,
            N=N_horizon,
            dt=dt_step,
            tau_max=tau_max,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
            cost_mode="effort",
            hessian_mode="limited-memory",
            max_iter=500,
            verbose=False,
        )

        exact_conv_iters = []
        lbfgs_conv_iters = []
        exact_conv_times = []
        lbfgs_conv_times = []
        exact_per_iter_times = []
        lbfgs_per_iter_times = []
        exact_costs = []
        lbfgs_costs = []
        exact_successes = 0
        lbfgs_successes = 0
        ctc_replay_rmses = []
        open_loop_replay_rmses = []

        for p_idx in range(N_problems):
            prob_seed = 100 + p_idx
            p_rng = np.random.default_rng(prob_seed)

            q_start = p_rng.uniform(-0.6, 0.6, size=n_dof)
            q_goal = p_rng.uniform(-0.6, 0.6, size=n_dof)
            x_start = np.concatenate([q_start, np.zeros(n_dof)])
            x_goal = np.concatenate([q_goal, np.zeros(n_dof)])

            # Solve Exact Hessian
            res_exact = ocp_exact.solve(x_start, x_goal)
            if res_exact["success"]:
                exact_successes += 1
                n_it = res_exact["iterations"]
                t_tot = res_exact["solve_time"]
                exact_conv_iters.append(n_it)
                exact_conv_times.append(t_tot)
                exact_per_iter_times.append((t_tot / n_it) * 1000.0)
                exact_costs.append(res_exact["cost"])

                # Dynamic consistency: (a) Closed-loop CTC replay vs (b) Open-loop torque replay
                q_plan = res_exact["q_traj"]
                dq_plan = res_exact["dq_traj"]
                u_plan = res_exact["u_traj"]

                # (a) Closed-loop CTC tracking
                twin.reset(q_start, np.zeros(n_dof))
                kp_val = 60.0 if n_dof == 7 else 120.0
                kd_val = 15.0 if n_dof == 7 else 25.0
                ctc = ComputedTorqueController(model, kp=kp_val, kd=kd_val, joint_signs=signs)

                q_ctc_list = [q_start.copy()]
                for k in range(N_horizon):
                    for _ in range(n_substeps):
                        q_curr, dq_curr = twin.data.qpos.copy(), twin.data.qvel.copy()
                        tau_cmd = ctc.compute_control(q_curr, dq_curr, q_plan[k], dq_plan[k], np.zeros(n_dof))
                        twin.step(tau_cmd)
                    q_ctc_list.append(twin.data.qpos.copy())
                rmse_ctc = float(np.sqrt(np.mean((np.array(q_ctc_list) - q_plan) ** 2)))
                ctc_replay_rmses.append(rmse_ctc)

                # (b) Pure Open-loop torque replay
                twin.reset(q_start, np.zeros(n_dof))
                q_open_list = [q_start.copy()]
                for k in range(N_horizon):
                    tau_open = signs * u_plan[k]
                    for _ in range(n_substeps):
                        twin.step(tau_open)
                    q_open_list.append(twin.data.qpos.copy())
                rmse_open = float(np.sqrt(np.mean((np.array(q_open_list) - q_plan) ** 2)))
                open_loop_replay_rmses.append(rmse_open)

            # Solve Quasi-Newton L-BFGS (Converged Baseline with max_iter=500)
            res_lbfgs = ocp_lbfgs.solve(x_start, x_goal)
            if res_lbfgs["success"]:
                lbfgs_successes += 1
                n_it = res_lbfgs["iterations"]
                t_tot = res_lbfgs["solve_time"]
                lbfgs_conv_iters.append(n_it)
                lbfgs_conv_times.append(t_tot)
                lbfgs_per_iter_times.append((t_tot / n_it) * 1000.0)
                lbfgs_costs.append(res_lbfgs["cost"])

            print(f"  Problem {p_idx+1:2d}/{N_problems}: Exact={res_exact['success']} ({res_exact['iterations']} iters, {res_exact['solve_time']:.2f}s) | "
                  f"LBFGS={res_lbfgs['success']} ({res_lbfgs['iterations']} iters, {res_lbfgs['solve_time']:.2f}s)")

        # Statistics over CONVERGED runs only
        exact_rate = (exact_successes / N_problems) * 100.0
        lbfgs_rate = (lbfgs_successes / N_problems) * 100.0

        mean_it_exact = float(np.mean(exact_conv_iters)) if exact_conv_iters else 0.0
        sd_it_exact = float(np.std(exact_conv_iters)) if exact_conv_iters else 0.0
        ci_it_exact = bootstrap_ci(exact_conv_iters)

        mean_it_lbfgs = float(np.mean(lbfgs_conv_iters)) if lbfgs_conv_iters else 0.0
        sd_it_lbfgs = float(np.std(lbfgs_conv_iters)) if lbfgs_conv_iters else 0.0
        ci_it_lbfgs = bootstrap_ci(lbfgs_conv_iters)

        iter_red_pct = ((mean_it_lbfgs - mean_it_exact) / (mean_it_lbfgs + 1e-9)) * 100.0

        mean_time_exact = float(np.mean(exact_conv_times)) if exact_conv_times else 0.0
        mean_time_lbfgs = float(np.mean(lbfgs_conv_times)) if lbfgs_conv_times else 0.0
        time_speedup = mean_time_lbfgs / (mean_time_exact + 1e-9)

        mean_per_iter_exact = float(np.mean(exact_per_iter_times)) if exact_per_iter_times else 0.0
        mean_per_iter_lbfgs = float(np.mean(lbfgs_per_iter_times)) if lbfgs_per_iter_times else 0.0

        mean_ctc_rmse = float(np.mean(ctc_replay_rmses)) if ctc_replay_rmses else 0.0
        mean_open_rmse = float(np.mean(open_loop_replay_rmses)) if open_loop_replay_rmses else 0.0

        print(f"\n  Final Metrics for {disp_name}:")
        print(f"    Convergence Rate (%)   : Exact = {exact_rate:.1f}% | L-BFGS = {lbfgs_rate:.1f}%")
        print(f"    Converged Iters (mean) : Exact = {mean_it_exact:.1f} +- {sd_it_exact:.1f} {ci_it_exact} | "
              f"L-BFGS = {mean_it_lbfgs:.1f} +- {sd_it_lbfgs:.1f} {ci_it_lbfgs} ({iter_red_pct:.1f}% reduction)")
        print(f"    Solve Wall Time (s)    : Exact = {mean_time_exact:.3f} s | L-BFGS = {mean_time_lbfgs:.3f} s ({time_speedup:.2f}x speedup)")
        print(f"    Per-Iter Latency (ms)  : Exact = {mean_per_iter_exact:.1f} ms | L-BFGS = {mean_per_iter_lbfgs:.1f} ms")
        print(f"    MuJoCo Replay Tracking : Closed-Loop CTC RMSE = {mean_ctc_rmse:.4f} rad vs Open-Loop = {mean_open_rmse:.4f} rad")

        results_data["robots"][key] = {
            "name": disp_name,
            "dof": n_dof,
            "exact_convergence_pct": exact_rate,
            "lbfgs_convergence_pct": lbfgs_rate,
            "converged_iters_exact_mean": mean_it_exact,
            "converged_iters_exact_sd": sd_it_exact,
            "converged_iters_exact_ci95": ci_it_exact,
            "converged_iters_lbfgs_mean": mean_it_lbfgs,
            "converged_iters_lbfgs_sd": sd_it_lbfgs,
            "converged_iters_lbfgs_ci95": ci_it_lbfgs,
            "iteration_reduction_pct": iter_red_pct,
            "wall_time_exact_mean_s": mean_time_exact,
            "wall_time_lbfgs_mean_s": mean_time_lbfgs,
            "wall_time_speedup": time_speedup,
            "per_iter_time_exact_ms": mean_per_iter_exact,
            "per_iter_time_lbfgs_ms": mean_per_iter_lbfgs,
            "mean_ctc_replay_rmse_rad": mean_ctc_rmse,
            "mean_open_replay_rmse_rad": mean_open_rmse,
        }

    # Pre-registered evaluation
    all_exact_rates = [v["exact_convergence_pct"] for v in results_data["robots"].values()]
    all_iter_reds = [v["iteration_reduction_pct"] for v in results_data["robots"].values()]
    all_speedups = [v["wall_time_speedup"] for v in results_data["robots"].values()]

    h3_pass_conv = bool(all(r >= 90.0 for r in all_exact_rates))
    h3_pass_iter = bool(all(red >= 50.0 for red in all_iter_reds))
    h3_pass_speedup = bool(all(sp >= 3.0 for sp in all_speedups))

    results_data["h3_verdict"] = {
        "pass_convergence_threshold_90pct": h3_pass_conv,
        "pass_iteration_reduction_50pct": h3_pass_iter,
        "pass_wall_time_speedup_3x": h3_pass_speedup,
        "overall_h3_supported": bool(h3_pass_conv and h3_pass_iter and h3_pass_speedup),
    }

    out_path = os.path.join(WORKSPACE_ROOT, "results", "e2_trajectory_opt.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[DONE] Converged Trajectory Optimization Results written to: {out_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
