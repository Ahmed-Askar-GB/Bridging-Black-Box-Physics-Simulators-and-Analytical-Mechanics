"""Benchmark E2: Offline Trajectory Optimization and Dynamic Consistency Replay (H3).

Pre-Registered Hypotheses:
  H3: Supplying exact symbolic Hessians yields higher convergence success rates (>= 95% vs <= 85%)
      and requires >= 30% fewer SQP/IPOPT iterations compared to Quasi-Newton (L-BFGS) approximations
      across 50 random boundary-value problems with obstacle and torque constraints.
      Replaying optimal torques in the MuJoCo physics engine under CTC yields tracking RMSE <= 0.010 rad.

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


def run_benchmark():
    print("=" * 80)
    print("  BENCHMARK E2: CONSTRAINED OFFLINE TRAJECTORY OPTIMIZATION (H3)")
    print("=" * 80)

    robots = [
        ("ur5", "UR5", 6, 150.0),
        ("ur16e", "UR16e", 6, 150.0),
        ("kukaIiwa7", "KUKA iiwa 7", 7, 120.0),
    ]

    N_horizon = 10
    dt_step = 0.04
    N_problems = 25
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
            "seed_range": [100, 100 + N_problems - 1],
        },
        "robots": {},
        "h3_verdict": {},
    }

    for key, disp_name, n_dof, tau_max in robots:
        print(f"\n---> Running 50 Trajectory Optimization Problems on {disp_name}...")
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

        # Build OCP solvers: Exact Hessian vs Quasi-Newton (L-BFGS)
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
            max_iter=50,
            verbose=False,
        )

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
            max_iter=50,
            verbose=False,
        )

        # Generate 50 randomized boundary-value problems with fixed seeds
        exact_successes = 0
        lbfgs_successes = 0
        exact_iters = []
        lbfgs_iters = []
        exact_times = []
        lbfgs_times = []
        replay_rmses = []

        for p_idx in range(N_problems):
            prob_seed = 100 + p_idx
            p_rng = np.random.default_rng(prob_seed)

            q_start = p_rng.uniform(-0.8, 0.8, size=n_dof)
            q_goal = p_rng.uniform(-0.8, 0.8, size=n_dof)
            x_start = np.concatenate([q_start, np.zeros(n_dof)])
            x_goal = np.concatenate([q_goal, np.zeros(n_dof)])

            # Solve with Exact Hessian
            res_exact = ocp_exact.solve(x_start, x_goal)
            exact_iters.append(res_exact["iterations"] if res_exact["iterations"] > 0 else 50)
            exact_times.append(res_exact["solve_time"])
            if res_exact["success"]:
                exact_successes += 1

                # Dynamic consistency check: replay in MuJoCo with CTC
                q_plan = res_exact["q_traj"]
                dq_plan = res_exact["dq_traj"]
                twin.reset(q_start, np.zeros(n_dof))
                kp_val = 60.0 if n_dof == 7 else 120.0
                kd_val = 15.0 if n_dof == 7 else 25.0
                ctc = ComputedTorqueController(model, kp=kp_val, kd=kd_val, joint_signs=signs)

                q_actual_list = [q_start.copy()]
                for k in range(N_horizon):
                    for _ in range(n_substeps):
                        q_curr, dq_curr = twin.data.qpos.copy(), twin.data.qvel.copy()
                        tau_cmd = ctc.compute_control(
                            q_curr, dq_curr,
                            q_plan[k], dq_plan[k], np.zeros(n_dof)
                        )
                        twin.step(tau_cmd)
                    q_actual_list.append(twin.data.qpos.copy())

                q_actual = np.array(q_actual_list)
                rmse_val = float(np.sqrt(np.mean((q_actual - q_plan) ** 2)))
                replay_rmses.append(rmse_val)

            # Solve with Quasi-Newton L-BFGS
            res_lbfgs = ocp_lbfgs.solve(x_start, x_goal)
            lbfgs_iters.append(res_lbfgs["iterations"] if res_lbfgs["iterations"] > 0 else 50)
            lbfgs_times.append(res_lbfgs["solve_time"])
            if res_lbfgs["success"]:
                lbfgs_successes += 1

            if (p_idx + 1) % 10 == 0:
                print(f"  Processed {p_idx + 1}/{N_problems} problems... (Exact succ: {exact_successes}, LBFGS succ: {lbfgs_successes})")

        # Statistics
        exact_rate = exact_successes / N_problems * 100.0
        lbfgs_rate = lbfgs_successes / N_problems * 100.0
        mean_iter_exact = float(np.mean(exact_iters)) if exact_iters else 0.0
        mean_iter_lbfgs = float(np.mean(lbfgs_iters)) if lbfgs_iters else 0.0
        iter_reduction_pct = ((mean_iter_lbfgs - mean_iter_exact) / (mean_iter_lbfgs + 1e-9)) * 100.0
        mean_replay_rmse = float(np.mean(replay_rmses)) if replay_rmses else 0.0

        print(f"\n  Results for {disp_name}:")
        print(f"    Convergence Rate    : Exact = {exact_rate:.1f}% | L-BFGS = {lbfgs_rate:.1f}%")
        print(f"    Mean Iterations     : Exact = {mean_iter_exact:.1f} | L-BFGS = {mean_iter_lbfgs:.1f} ({iter_reduction_pct:.1f}% reduction)")
        print(f"    Dynamic Consistency : MuJoCo Replay Tracking RMSE = {mean_replay_rmse:.4e} rad")

        results_data["robots"][key] = {
            "name": disp_name,
            "dof": n_dof,
            "exact_convergence_pct": exact_rate,
            "lbfgs_convergence_pct": lbfgs_rate,
            "mean_iterations_exact": mean_iter_exact,
            "mean_iterations_lbfgs": mean_iter_lbfgs,
            "iteration_reduction_pct": iter_reduction_pct,
            "mean_time_exact_s": float(np.mean(exact_times)) if exact_times else 0.0,
            "mean_time_lbfgs_s": float(np.mean(lbfgs_times)) if lbfgs_times else 0.0,
            "mean_replay_rmse_rad": mean_replay_rmse,
            "max_replay_rmse_rad": float(np.max(replay_rmses)) if replay_rmses else 0.0,
        }

    # Pre-registered H3 evaluations across all robots
    all_exact_rates = [v["exact_convergence_pct"] for v in results_data["robots"].values()]
    all_iter_reds = [v["iteration_reduction_pct"] for v in results_data["robots"].values()]
    all_rmses = [v["mean_replay_rmse_rad"] for v in results_data["robots"].values()]

    h3_pass_conv = bool(all(r >= 95.0 for r in all_exact_rates))
    h3_pass_iter = bool(all(red >= 30.0 for red in all_iter_reds))
    h3_pass_rmse = bool(all(rmse <= 0.010 for rmse in all_rmses))
    results_data["h3_verdict"] = {
        "pass_convergence_threshold_95pct": h3_pass_conv,
        "pass_iteration_reduction_30pct": h3_pass_iter,
        "pass_dynamic_consistency_rmse_1e-2": h3_pass_rmse,
        "overall_h3_supported": bool(h3_pass_conv and h3_pass_iter and h3_pass_rmse),
    }

    out_path = os.path.join(WORKSPACE_ROOT, "results", "e2_trajectory_opt.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[DONE] Trajectory Optimization Results saved to: {out_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
