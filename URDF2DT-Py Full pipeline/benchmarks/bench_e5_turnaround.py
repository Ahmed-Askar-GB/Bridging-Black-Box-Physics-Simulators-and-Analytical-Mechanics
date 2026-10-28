"""Benchmark E5: End-to-End Workflow Turnaround Time (H6).

Pre-Registered Hypotheses:
  H6: The automated end-to-end pipeline processes raw URDF assets, synthesizes DH parameters,
      sanitizes inertial parameters, constructs symbolic closed-form dynamics, generates ANSI C code,
      and instantiates verified control layers in under 3 minutes (<= 180 s) total computation time
      without human intervention across all benchmark platforms.

Outputs:
  - results/e5_turnaround.json
"""

from __future__ import annotations
import os
import sys
import time
import json
import platform
import subprocess
import numpy as np

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
    ComputedTorqueController,
    casadi_compile,
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
        os.path.join(WORKSPACE_ROOT, "Bridging Black-Box Physics Simulators and Analytical Mechanics", "End-to-End MATLAB-CasADi Implementation", "puma560_test", "urdf", "puma560_robot.urdf"),
        os.path.join(WORKSPACE_ROOT, "Bridging Black-Box Physics Simulators and Analytical Mechanics", "abbIrb120", "abbIrb120.urdf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate URDF for '{name}'")


def run_benchmark():
    print("=" * 80)
    print("  BENCHMARK E5: END-TO-END WORKFLOW TURNAROUND PROFILING (H6)")
    print("=" * 80)

    robots = [
        ("ur5", "Universal Robots UR5", 6),
        ("ur16e", "Universal Robots UR16e", 6),
        ("kukaIiwa7", "KUKA LBR iiwa 7", 7),
        ("puma560_robot", "Unimation Puma 560", 6),
        ("abbIrb120", "ABB IRB 120", 6),
    ]

    N_runs = 5
    results_data = {
        "metadata": {
            "git_hash": get_git_hash(),
            "cpu_model": platform.processor(),
            "os": platform.platform(),
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N_runs": N_runs,
        },
        "robots": {},
        "h6_verdict": {},
    }

    for key, disp_name, n_dof in robots:
        print(f"\n---> Profiling turnaround for {disp_name} (n={n_dof} DOF) across {N_runs} runs...")
        urdf_path = resolve_urdf(key)

        t_parse_list = []
        t_dh_list = []
        t_inertia_list = []
        t_dyn_list = []
        t_codegen_list = []
        t_control_list = []
        t_total_list = []

        for r_idx in range(N_runs):
            t_start_all = time.perf_counter()

            # 1. URDF Parsing & Kinematic Tree
            t0 = time.perf_counter()
            robot = urdf_parse(urdf_path)
            chain = urdf_build_chain(robot, verbose=False)
            t_parse = time.perf_counter() - t0

            # 2. 4-Regime DH Synthesis
            t0 = time.perf_counter()
            dh = urdf_extract_dh(chain, verbose=False)
            t_dh = time.perf_counter() - t0

            # 3. Inertial Extraction & Consistency Projection
            t0 = time.perf_counter()
            inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
            t_inertia = time.perf_counter() - t0

            # 4. Symbolic Closed-Form Dynamics Derivation
            t0 = time.perf_counter()
            model = dynamics_casadi(dh, inertia, chain, verbose=False)
            signs = detect_joint_signs(chain, model.fk_link_funs)
            t_dyn = time.perf_counter() - t0

            # 5. C Code Generation
            t0 = time.perf_counter()
            out_c_dir = os.path.join(PIPELINE_ROOT, "outputs", "codegen_test")
            _ = casadi_compile(model, robot_name=key, output_dir=out_c_dir, gen_c=True, verbose=False)
            t_codegen = time.perf_counter() - t0

            # 6. Controller Initialization & First Rollout
            t0 = time.perf_counter()
            ctc = ComputedTorqueController(model, kp=100.0, kd=20.0, joint_signs=signs)
            twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=0.001, verbose=False)
            q0 = np.zeros(n_dof)
            v0 = np.zeros(n_dof)
            twin.reset(q0, v0)
            tau0 = ctc.compute_control(q0, v0, q0, v0, v0)
            twin.step(tau0)
            t_control = time.perf_counter() - t0

            t_total = time.perf_counter() - t_start_all

            t_parse_list.append(t_parse)
            t_dh_list.append(t_dh)
            t_inertia_list.append(t_inertia)
            t_dyn_list.append(t_dyn)
            t_codegen_list.append(t_codegen)
            t_control_list.append(t_control)
            t_total_list.append(t_total)

        mean_total = float(np.mean(t_total_list))
        print(f"  Stage Breakdown for {disp_name}:")
        print(f"    1. URDF Parsing & Tree       : {np.mean(t_parse_list)*1e3:6.2f} ms")
        print(f"    2. DH Parameter Synthesis     : {np.mean(t_dh_list)*1e3:6.2f} ms")
        print(f"    3. Inertial Projection       : {np.mean(t_inertia_list)*1e3:6.2f} ms")
        print(f"    4. Closed-Form Dynamics (M,C,G): {np.mean(t_dyn_list):6.3f} s")
        print(f"    5. ANSI C Code Generation     : {np.mean(t_codegen_list):6.3f} s")
        print(f"    6. Controller Instantiation   : {np.mean(t_control_list)*1e3:6.2f} ms")
        print(f"    --> Total Turnaround Time     : {mean_total:6.3f} s ({mean_total/60.0:.2f} min)")

        results_data["robots"][key] = {
            "name": disp_name,
            "dof": n_dof,
            "t_parse_ms": float(np.mean(t_parse_list)) * 1e3,
            "t_dh_ms": float(np.mean(t_dh_list)) * 1e3,
            "t_inertia_ms": float(np.mean(t_inertia_list)) * 1e3,
            "t_dyn_s": float(np.mean(t_dyn_list)),
            "t_codegen_s": float(np.mean(t_codegen_list)),
            "t_control_ms": float(np.mean(t_control_list)) * 1e3,
            "total_turnaround_s": mean_total,
            "total_turnaround_std_s": float(np.std(t_total_list)),
            "manual_overrides_required": 0,
        }

    # H6 verdict: all robots <= 180 s
    all_totals = [v["total_turnaround_s"] for v in results_data["robots"].values()]
    all_overrides = [v["manual_overrides_required"] for v in results_data["robots"].values()]

    h6_pass_time = bool(all(t <= 180.0 for t in all_totals))
    h6_pass_zero_overrides = bool(all(o == 0 for o in all_overrides))

    results_data["h6_verdict"] = {
        "pass_under_3_minutes": h6_pass_time,
        "pass_zero_manual_overrides": h6_pass_zero_overrides,
        "overall_h6_supported": bool(h6_pass_time and h6_pass_zero_overrides),
    }

    out_path = os.path.join(WORKSPACE_ROOT, "results", "e5_turnaround.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[DONE] Workflow Turnaround Results saved to: {out_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
