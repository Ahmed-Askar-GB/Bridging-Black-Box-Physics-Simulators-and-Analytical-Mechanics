"""Master One-Click CLI Orchestrator for URDF2DT-Py."""

from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np

# Ensure urdf2dt is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from urdf2dt import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
    dynamics_casadi,
    casadi_compile,
    MuJoCoTwin,
    ComputedTorqueController,
    verify_invariants,
    verify_kinematics,
    verify_dynamics_sequential,
    print_verification_dashboard,
    export_verification_report,
    VerificationReport,
)
from urdf2dt.utils import traj_poly5, detect_joint_signs


def run_pipeline(
    urdf_path: str,
    out_dir: str = "outputs",
    gen_c: bool = True,
    t_dur: float = 2.0,
    verbose: bool = True
) -> VerificationReport:
    """Execute the master 6-phase URDF to analytical CasADi and Digital Twin pipeline."""
    t_start = time.time()

    print("\n" + "=" * 80)
    print("       URDF2DT-PY: ROBOT DYNAMICS, DIGITAL TWIN & CONTROL PIPELINE       ")
    print("=" * 80)
    print(f"  URDF File : {urdf_path}")
    print(f"  Output Dir: {out_dir}")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # STAGE 1: Parsing Layer
    # -------------------------------------------------------------------------
    print(">>> [STAGE 1/6] PARSING LAYER <<<")
    t0 = time.time()
    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot, verbose=verbose)
    dh = urdf_extract_dh(chain, verbose=verbose)
    inertia = urdf_extract_inertia(robot, chain, dh, verbose=verbose)
    print(f"  [OK] Parsed {len(robot.links)} links, {len(robot.joints)} joints.")
    print(f"  [OK] Extracted {chain.n}-DOF active kinematic chain.")
    print(f"  [OK] Standard DH parameters table generated via geometric P-Q method.")
    print(f"  [OK] Inertial properties transformed into standard DH frames.")
    print(f"  Stage 1 completed in {time.time() - t0:.3f} s\n")

    # -------------------------------------------------------------------------
    # STAGE 2: Analytical CasADi Kinematics & Lagrangian Dynamics
    # -------------------------------------------------------------------------
    print(">>> [STAGE 2/6] ANALYTICAL CASADI ENGINE <<<")
    t0 = time.time()
    model = dynamics_casadi(dh, inertia, chain, verbose=verbose)
    print("  [OK] Forward Kinematics compiled (T0i, T0e, Shepperd quaternion).")
    print("  [OK] Geometric Jacobian and exact auto-diff J_dot compiled.")
    print("  [OK] Lagrangian M(q)*ddq + C(q,dq)*dq + G(q) = tau compiled.")
    print(f"  Stage 2 completed in {time.time() - t0:.3f} s\n")

    # -------------------------------------------------------------------------
    # STAGE 3: Model Compilation & Embedded C Code Generation
    # -------------------------------------------------------------------------
    print(">>> [STAGE 3/6] MODEL COMPILATION & EMBEDDED C CODE GENERATION <<<")
    t0 = time.time()
    comp_dir = os.path.join(out_dir, "compilation")
    comp_res = casadi_compile(model, robot_name=robot.name, output_dir=comp_dir, gen_c=gen_c, verbose=verbose)
    if gen_c:
        print(f"  [OK] Generated embedded C code: {comp_res['c_filepath']}")
        print(f"  [OK] Generated C header: {comp_res['h_filepath']}")
    print(f"  Stage 3 completed in {time.time() - t0:.3f} s\n")

    # -------------------------------------------------------------------------
    # STAGE 4: Headless Digital Twin Initialization
    # -------------------------------------------------------------------------
    print(">>> [STAGE 4/6] DIGITAL TWIN (MUJOCO ENGINE) <<<")
    t0 = time.time()
    twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=0.001, verbose=verbose)
    print(f"  [OK] MuJoCo Twin plant initialized (nq={twin.nq}, nv={twin.nv}, nu={twin.nu}).")
    print("  [OK] Anchored coincident End-Effector sensor site on terminal body.")
    print(f"  Stage 4 completed in {time.time() - t0:.3f} s\n")

    # -------------------------------------------------------------------------
    # STAGE 5: Invariant & Kinematic Ground-Truth Cross-Validation
    # -------------------------------------------------------------------------
    print(">>> [STAGE 5/6] INVARIANT & KINEMATIC CROSS-VALIDATION <<<")
    t0 = time.time()
    inv_res = verify_invariants(robot, dh, inertia, model, num_samples=10)
    for k, v in inv_res.items():
        status = "PASSED" if v else "FAILED"
        print(f"  - Invariant check '{k}': {status}")

    kin_res = verify_kinematics(model, twin, chain, num_samples=20, verbose=verbose)
    print(f"  Stage 5 completed in {time.time() - t0:.3f} s\n")

    # -------------------------------------------------------------------------
    # STAGE 6: Sequential Dynamic Isolation & Control Verification
    # -------------------------------------------------------------------------
    print(">>> [STAGE 6/6] SEQUENTIAL DYNAMIC ISOLATION & CONTROL <<<")
    t0 = time.time()
    stage_results = verify_dynamics_sequential(
        chain=chain,
        dh=dh,
        inertia=inertia,
        twin=twin,
        t_dur=t_dur,
        dt=0.005,
        tol_tau=0.1,
        verbose=verbose,
    )

    # Test Computed Torque Control tracking in closed loop
    joint_signs = detect_joint_signs(chain, model.fk_link_funs)
    controller = ComputedTorqueController(model=model, kp=120.0, kd=25.0, joint_signs=joint_signs)
    q0 = np.zeros(chain.n)
    qf = np.array([0.5, -0.4, 0.6, -0.5, 0.3, -0.2])[:chain.n]
    sim_time, q_ref, dq_ref, ddq_ref = traj_poly5(q0, qf, t_dur=t_dur, dt=0.001)

    sim_res = twin.simulate_trajectory(sim_time, q_ref, dq_ref, ddq_ref, controller=controller)
    tracking_err = np.linalg.norm(sim_res["q_sim"] - q_ref, axis=1)
    max_tracking_err = float(np.max(tracking_err))
    print(f"\n  [OK] Closed-loop Computed Torque simulation completed (Max error: {max_tracking_err:.4e} rad).")
    print(f"  Stage 6 completed in {time.time() - t0:.3f} s\n")

    # Assemble Verification Report
    all_stages_passed = all(st.passed for st in stage_results)
    all_passed = bool(all_stages_passed and (kin_res["passed"] == 1.0) and all(inv_res.values()))
    max_dyn_err = max(st.max_err for st in stage_results) if stage_results else 0.0

    report = VerificationReport(
        all_passed=all_passed,
        max_fk_pos_err=kin_res["max_pos_err"],
        max_fk_quat_err=kin_res["max_quat_err"],
        max_jac_err=kin_res["max_jac_err"],
        max_dyn_err=max_dyn_err,
        stage_results=stage_results,
        summary=f"Pipeline executed in {time.time() - t_start:.2f} s. All passed: {all_passed}.",
    )

    # Print Dashboard & Export
    print_verification_dashboard(report)
    export_files = export_verification_report(report, output_dir=out_dir)
    print(f"Report exported to: {export_files['json']} and {export_files['csv']}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URDF2DT-Py Master Pipeline Orchestrator")
    default_urdf = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "ur5.urdf")
    parser.add_argument("--urdf", type=str, default=default_urdf, help="Path to URDF file")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--t-dur", type=float, default=2.0, help="Excitation trajectory duration")
    parser.add_argument("--no-c", action="store_true", help="Disable C code generation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose messages")

    args = parser.parse_args()

    report = run_pipeline(
        urdf_path=args.urdf,
        out_dir=args.out_dir,
        gen_c=not args.no_c,
        t_dur=args.t_dur,
        verbose=not args.quiet,
    )

    sys.exit(0 if report.all_passed else 1)
