"""UR16e Animated Pipeline Test with Live 3D MuJoCo Viewer and Matplotlib Dashboards.

Validates the full URDF2DT-Py pipeline on the Universal Robots UR16e model,
launches a real-time 3D MuJoCo animation with an analytical CasADi EE overlay marker,
and generates multi-panel kinematic, dynamic torque, and closed-loop tracking plots.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer  # Explicit submodule import required on Windows

# Ensure package is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from urdf2dt import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
    dynamics_casadi,
    MuJoCoTwin,
    ComputedTorqueController,
    verify_invariants,
    verify_kinematics,
    verify_dynamics_sequential,
    print_verification_dashboard,
    export_verification_report,
    VerificationReport,
)
from urdf2dt.utils import traj_poly5, multi_sine_excitation, detect_joint_signs, quat_error, R_to_quat

# Default asset directories for UR16e
PRIMARY_UR16E_DIR = os.path.abspath(
    os.path.join(
        PKG_ROOT, "..", "End-to-End MATLAB-CasADi Implementation", "test_UR16e"
    )
)
FALLBACK_FIXTURE_PATH = os.path.join(PKG_ROOT, "tests", "fixtures", "ur16e.urdf")


def discover_ur16e_assets(target_dir: str = PRIMARY_UR16E_DIR) -> str:
    """Scan directory for UR16e URDF file and resolve mesh references."""
    if os.path.exists(target_dir):
        for root, _, files in os.walk(target_dir):
            for f in files:
                if f.lower().endswith(".urdf") and "16e" in f.lower():
                    cand = os.path.join(root, f)
                    if os.path.exists(cand):
                        return os.path.abspath(cand)

    if os.path.exists(FALLBACK_FIXTURE_PATH):
        return os.path.abspath(FALLBACK_FIXTURE_PATH)

    raise FileNotFoundError(f"Could not locate UR16e URDF in {target_dir} or {FALLBACK_FIXTURE_PATH}")


def prepare_ur16e_mesh_xml(urdf_path: str) -> str:
    """Preprocess URDF XML by resolving package://ur_description to absolute paths on disk."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    base_dir = os.path.dirname(urdf_path)

    ur_desc_dir = None
    search_dirs = [
        base_dir,
        os.path.join(base_dir, "ur_description"),
        os.path.abspath(os.path.join(base_dir, "..", "test_UR16e", "ur_description")),
        os.path.abspath(os.path.join(base_dir, "..", "test_UR16e")),
    ]
    for d in search_dirs:
        if os.path.exists(os.path.join(d, "meshes")):
            ur_desc_dir = d
            break
        elif os.path.exists(os.path.join(d, "ur_description", "meshes")):
            ur_desc_dir = os.path.join(d, "ur_description")
            break

    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "")
        if fn.startswith("package://ur_description/"):
            rel_sub = fn[len("package://ur_description/"):]
            if ur_desc_dir is not None:
                new_fn = os.path.join(ur_desc_dir, rel_sub).replace("\\", "/")
                if os.path.exists(new_fn):
                    mesh.attrib["filename"] = new_fn

    return ET.tostring(root, encoding="unicode")


def run_ur16e_verification_and_animation(
    urdf_path: str | None = None,
    t_dur: float = 6.0,
    dt: float = 0.002,
    headless: bool = False,
    show_plots: bool = True,
    save_dir: str | None = None,
    verbose: bool = True
) -> dict:
    """Run full pipeline, 3D animated simulation, and diagnostic dashboard for UR16e."""
    if urdf_path is None:
        urdf_path = discover_ur16e_assets()

    if verbose:
        print("\n" + "=" * 80)
        print("    UNIVERSAL ROBOTS UR16E: ANIMATED PIPELINE VERIFICATION & DIGITAL TWIN     ")
        print("=" * 80)
        print(f"  Target URDF : {urdf_path}")
        print(f"  Simulation  : Duration={t_dur:.1f} s, Timestep={dt*1000:.1f} ms ({int(1.0/dt)} Hz)")
        print(f"  Viewer Mode : {'Headless (No GUI Window)' if headless else 'Interactive 3D Window (MuJoCo Viewer)'}")
        print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # STAGE 1: Analytical Model Generation (CasADi)
    # -------------------------------------------------------------------------
    if verbose:
        print(">>> [STAGE 1/4] ANALYTICAL CASADI MODEL DERIVATION <<<")
    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot, verbose=False)
    dh = urdf_extract_dh(chain, verbose=False)
    inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
    model = dynamics_casadi(dh, inertia, chain, verbose=False)

    joint_signs = detect_joint_signs(chain, model.fk_link_funs)

    if verbose:
        print(f"  [OK] Parsed robot '{robot.name}' with {chain.n} active DOF.")
        print(f"  [OK] Standard DH Parameters table generated (Total Moving Mass: {np.sum(inertia.m):.2f} kg).")
        print(f"  [OK] Autonomous Joint Sign Alignment Map: {joint_signs}")

    # -------------------------------------------------------------------------
    # STAGE 2: Persistent Trajectory Generation
    # -------------------------------------------------------------------------
    if verbose:
        print("\n>>> [STAGE 2/4] PERSISTENT EXCITATION TRAJECTORY GENERATION <<<")
    q0 = np.array([0.0, -np.pi / 4, np.pi / 3, -np.pi / 2, -np.pi / 3, 0.0])
    qf = np.array([0.8, -np.pi / 2, 0.2, -0.2, 0.8, -0.6])
    time_vec, q_ref, dq_ref, ddq_ref = traj_poly5(q0, qf, t_dur=t_dur, dt=dt)
    num_steps = len(time_vec)
    if verbose:
        print(f"  [OK] Generated 6-DOF smooth minimum-jerk trajectory with {num_steps} steps.")

    # -------------------------------------------------------------------------
    # STAGE 3: MuJoCo Twin & Live 3D Animation Overlay
    # -------------------------------------------------------------------------
    if verbose:
        print("\n>>> [STAGE 3/4] MUJOCO DIGITAL TWIN REPLAY & CASADI 3D OVERLAY <<<")
    twin = MuJoCoTwin(urdf_path, chain=chain, dh=dh, dt=dt, verbose=False)
    controller = ComputedTorqueController(model, kp=350.0, kd=45.0, joint_signs=joint_signs)

    pos_mj = np.zeros((num_steps, 3))
    pos_casadi = np.zeros((num_steps, 3))
    quat_mj = np.zeros((num_steps, 4))
    quat_casadi = np.zeros((num_steps, 4))
    vel_lin_mj = np.zeros((num_steps, 3))
    vel_lin_casadi = np.zeros((num_steps, 3))
    vel_ang_mj = np.zeros((num_steps, 3))
    vel_ang_casadi = np.zeros((num_steps, 3))
    tau_inv_mj = np.zeros((num_steps, chain.n))
    tau_inv_casadi = np.zeros((num_steps, chain.n))
    q_cl_sim = np.zeros((num_steps, chain.n))

    site_id = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    viewer = None
    if not headless:
        try:
            viewer = mujoco.viewer.launch_passive(twin.model, twin.data)
            # Preset optimal camera viewpoint framing the robot
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.45], dtype=np.float64)
            viewer.cam.distance = 2.25
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            if verbose:
                print("  [OK] 3D MuJoCo viewer window opened successfully with optimal camera framing.")
                print("       Rendering robot motion with live CasADi analytical EE overlay (green sphere).")
        except Exception as e:
            print(f"  [!] Note: Viewer launch exception ({e}). Continuing in headless mode.")
            viewer = None

    t_anim_start = time.time()
    trail_positions = []

    # Mode A: Kinematic & Inverse Dynamics Replay
    for k in range(num_steps):
        q_k = q_ref[k, :]
        dq_k = dq_ref[k, :]
        ddq_k = ddq_ref[k, :]

        # Update MuJoCo state
        twin.data.qpos[:] = q_k
        twin.data.qvel[:] = dq_k
        twin.data.qacc[:] = ddq_k

        # Inverse dynamics torque ground truth
        mujoco.mj_inverse(twin.model, twin.data)
        tau_inv_mj[k, :] = twin.data.qfrc_inverse.copy()

        # Forward kinematics for sensors
        mujoco.mj_forward(twin.model, twin.data)
        if site_id >= 0:
            pos_mj[k, :] = twin.data.site_xpos[site_id].copy()
            quat_mj[k, :] = R_to_quat(twin.data.site_xmat[site_id].reshape(3, 3))
            jac_p = np.zeros((3, twin.model.nv))
            jac_r = np.zeros((3, twin.model.nv))
            mujoco.mj_jacSite(twin.model, twin.data, jac_p, jac_r, site_id)
            vel_lin_mj[k, :] = jac_p @ dq_k
            vel_ang_mj[k, :] = jac_r @ dq_k

        # Evaluate Analytical CasADi Model
        q_dh = joint_signs * q_k
        dq_dh = joint_signs * dq_k
        ddq_dh = joint_signs * ddq_k

        p_ana, R_ana, q_ana = model.fk_fun(q_dh)
        pos_casadi[k, :] = np.array(p_ana).flatten()
        quat_casadi[k, :] = np.array(q_ana).flatten()

        J_ana = np.array(model.J_fun(q_dh))
        J_aligned = J_ana * joint_signs[None, :]
        v_spatial = J_aligned @ dq_k
        vel_lin_casadi[k, :] = v_spatial[:3]
        vel_ang_casadi[k, :] = v_spatial[3:]

        tau_dh = np.array(model.tau_fun(q_dh, dq_dh, ddq_dh)).flatten()
        tau_inv_casadi[k, :] = joint_signs * tau_dh

        # Live 3D Overlay in Viewer Window
        if viewer is not None and viewer.is_running():
            if k % 12 == 0:
                trail_positions.append(pos_casadi[k, :].copy())
                if len(trail_positions) > 60:
                    trail_positions.pop(0)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                # Green semi-transparent sphere marker at CasADi analytical EE pose
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.028, 0.0, 0.0], dtype=np.float64),
                        pos=pos_casadi[k, :],
                        mat=np.array(R_ana).flatten(),
                        rgba=np.array([0.0, 1.0, 0.2, 0.75], dtype=np.float32),
                    )
                    viewer.user_scn.ngeom += 1

                # Trajectory trail spheres
                for pt in trail_positions:
                    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.008, 0.0, 0.0], dtype=np.float64),
                            pos=pt,
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0.2, 0.6, 1.0, 0.45], dtype=np.float32),
                        )
                        viewer.user_scn.ngeom += 1

            viewer.sync()

            # Smooth real-time playback pacing
            t_expected = t_anim_start + (k * dt)
            t_diff = t_expected - time.time()
            if t_diff > 0:
                time.sleep(t_diff)

    if verbose:
        print("  [OK] Kinematic & Inverse Dynamics replay phase complete.")

    # Mode B: Dynamic Forward Simulation under Computed Torque Control
    if verbose:
        print("  Running Dynamic Forward Simulation under Computed Torque Control...")

    twin.data.qpos[:] = q_ref[0, :]
    twin.data.qvel[:] = dq_ref[0, :]
    mujoco.mj_forward(twin.model, twin.data)
    t_cl_start = time.time()

    for k in range(num_steps):
        q_curr = twin.data.qpos.copy()
        dq_curr = twin.data.qvel.copy()

        q_cl_sim[k, :] = q_curr

        tau_cmd = controller.compute_control(
            q=q_curr,
            dq=dq_curr,
            q_d=q_ref[k, :],
            dq_d=dq_ref[k, :],
            ddq_d=ddq_ref[k, :],
            t=time_vec[k]
        )

        if twin.model.nu > 0:
            twin.data.ctrl[:] = tau_cmd
        mujoco.mj_step(twin.model, twin.data)

        # Sync forward simulation in viewer if open
        if viewer is not None and viewer.is_running() and (k % 4 == 0):
            viewer.sync()
            t_expected = t_cl_start + (k * dt)
            t_diff = t_expected - time.time()
            if t_diff > 0:
                time.sleep(t_diff)

    if viewer is not None and viewer.is_running():
        if verbose:
            print("  [OK] Trajectory animation complete. MuJoCo viewer remains active for inspection.")

    tracking_err = np.linalg.norm(q_cl_sim - q_ref, axis=1)

    # -------------------------------------------------------------------------
    # STAGE 4: Quantitative Verification & Dashboard
    # -------------------------------------------------------------------------
    if verbose:
        print("\n>>> [STAGE 4/4] QUANTITATIVE ACCURACY AUDIT <<<")

    pos_residuals = np.linalg.norm(pos_mj - pos_casadi, axis=1)
    max_pos_err = float(np.max(pos_residuals))
    rms_pos_err = float(np.sqrt(np.mean(pos_residuals**2)))

    quat_residuals = np.array([quat_error(quat_mj[i, :], quat_casadi[i, :]) for i in range(num_steps)])
    max_quat_err = float(np.max(quat_residuals))
    rms_quat_err = float(np.sqrt(np.mean(quat_residuals**2)))

    vel_lin_err = np.linalg.norm(vel_lin_mj - vel_lin_casadi, axis=1)
    max_vel_lin_err = float(np.max(vel_lin_err))
    rms_vel_lin_err = float(np.sqrt(np.mean(vel_lin_err**2)))

    vel_ang_err = np.linalg.norm(vel_ang_mj - vel_ang_casadi, axis=1)
    max_vel_ang_err = float(np.max(vel_ang_err))
    rms_vel_ang_err = float(np.sqrt(np.mean(vel_ang_err**2)))

    tau_residuals = np.abs(tau_inv_mj - tau_inv_casadi)
    per_joint_max_tau = np.max(tau_residuals, axis=0)
    per_joint_rms_tau = np.sqrt(np.mean(tau_residuals**2, axis=0))

    max_track_err = float(np.max(tracking_err))
    rms_track_err = float(np.sqrt(np.mean(tracking_err**2)))

    # Scale-aware tolerances for heavy-payload UR16e arm (27 kg moving mass, ~300 N*m torques)
    tol_pos = 1e-4
    tol_quat = 1e-3
    tol_tau = 2.0  # 2.0 N*m scale-aware tolerance for 300 N*m industrial joints (<0.7% rel error)
    tol_track = 0.05

    passed = bool(
        max_pos_err < tol_pos and
        max_quat_err < tol_quat and
        np.all(per_joint_rms_tau < tol_tau) and
        max_track_err < tol_track
    )

    if verbose:
        print("\n" + "=" * 80)
        print("          UNIVERSAL ROBOTS UR16E VALIDATION REPORT (MUJOCO vs CASADI)         ")
        print("=" * 80)
        print(f"  Overall Validation Status    : {'>>> PASSED <<<' if passed else '>>> FAILED <<<'}")
        print(f"  EE Position Error (RMS / Max): {rms_pos_err:10.4e} m  /  {max_pos_err:10.4e} m")
        print(f"  EE Quat Error     (RMS / Max): {rms_quat_err:10.4e} rad / {max_quat_err:10.4e} rad")
        print(f"  EE Linear Vel Err (RMS / Max): {rms_vel_lin_err:10.4e} m/s / {max_vel_lin_err:10.4e} m/s")
        print(f"  EE Angular Vel Err(RMS / Max): {rms_vel_ang_err:10.4e} r/s / {max_vel_ang_err:10.4e} r/s")
        print(f"  Closed-Loop Tracking (Max)   : {max_track_err:10.4e} rad")
        print("-" * 80)
        print(f"{'Joint':<6} {'Joint Name':<24} {'RMS Torque Err (N*m)':<24} {'Max Torque Err (N*m)':<24}")
        print("-" * 80)
        for j_idx in range(chain.n):
            j_name = chain.active_joints[j_idx].name
            print(f"{j_idx+1:<6} {j_name:<24} {per_joint_rms_tau[j_idx]:<24.4e} {per_joint_max_tau[j_idx]:<24.4e}")
        print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # STAGE 5: Multi-Panel Visual Comparison Plots (Matplotlib)
    # -------------------------------------------------------------------------
    def _safe_path(p: str) -> str:
        if os.name == "nt" and len(os.path.abspath(p)) >= 240 and not p.startswith("\\\\?\\"):
            return "\\\\?\\" + os.path.abspath(p)
        return p

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(urdf_path), "results")
    os.makedirs(_safe_path(save_dir), exist_ok=True)

    pkg_outputs_dir = os.path.join(PKG_ROOT, "outputs_ur16e")
    os.makedirs(pkg_outputs_dir, exist_ok=True)

    if headless:
        matplotlib.use("Agg")

    # Figure 1: Kinematics Comparison
    fig1 = plt.figure(figsize=(14, 10))
    fig1.suptitle("Universal Robots UR16e: Kinematic Verification (MuJoCo vs. CasADi Analytical)", fontsize=14, fontweight="bold")

    ax1 = fig1.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(pos_mj[:, 0], pos_mj[:, 1], pos_mj[:, 2], "b-", linewidth=2.0, label="MuJoCo Sensor")
    ax1.plot(pos_casadi[:, 0], pos_casadi[:, 1], pos_casadi[:, 2], "r--", linewidth=1.5, label="CasADi Analytical")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D End-Effector Workspace Path")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.plot(time_vec, pos_mj[:, 0], "b-", label="p_x (MuJoCo)")
    ax2.plot(time_vec, pos_mj[:, 1], "g-", label="p_y (MuJoCo)")
    ax2.plot(time_vec, pos_mj[:, 2], "k-", label="p_z (MuJoCo)")
    ax2.plot(time_vec, pos_casadi[:, 0], "r--", label="p_x (CasADi)")
    ax2.plot(time_vec, pos_casadi[:, 1], "y--", label="p_y (CasADi)")
    ax2.plot(time_vec, pos_casadi[:, 2], "m--", label="p_z (CasADi)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title(f"Position Components & Error (Max: {max_pos_err:.2e} m)")
    ax2.legend(loc="upper right", ncol=2)
    ax2.grid(True)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.plot(time_vec, quat_mj[:, 0], "b-", label="qw (MuJoCo)")
    ax3.plot(time_vec, quat_mj[:, 1], "g-", label="qx")
    ax3.plot(time_vec, quat_mj[:, 2], "r-", label="qy")
    ax3.plot(time_vec, quat_mj[:, 3], "k-", label="qz")
    ax3.plot(time_vec, quat_casadi[:, 0], "c--", label="qw (CasADi)")
    ax3.plot(time_vec, quat_casadi[:, 1], "y--", label="qx")
    ax3.plot(time_vec, quat_casadi[:, 2], "m--", label="qy")
    ax3.plot(time_vec, quat_casadi[:, 3], "orange", linestyle="--", label="qz")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Quaternion")
    ax3.set_title(f"End-Effector Orientation (Max Err: {max_quat_err:.2e} rad)")
    ax3.legend(loc="upper right", ncol=2)
    ax3.grid(True)

    ax4 = fig1.add_subplot(2, 2, 4)
    v_norm_mj = np.linalg.norm(vel_lin_mj, axis=1)
    v_norm_ana = np.linalg.norm(vel_lin_casadi, axis=1)
    w_norm_mj = np.linalg.norm(vel_ang_mj, axis=1)
    w_norm_ana = np.linalg.norm(vel_ang_casadi, axis=1)
    ax4.plot(time_vec, v_norm_mj, "b-", label="||v|| MuJoCo")
    ax4.plot(time_vec, v_norm_ana, "r--", label="||v|| CasADi J(q)*dq")
    ax4.plot(time_vec, w_norm_mj, "g-", label="||w|| MuJoCo")
    ax4.plot(time_vec, w_norm_ana, "m--", label="||w|| CasADi J(q)*dq")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Speed (m/s, rad/s)")
    ax4.set_title(f"Spatial Velocity Comparison (Max Lin Err: {max_vel_lin_err:.2e} m/s)")
    ax4.legend(loc="upper right")
    ax4.grid(True)

    plt.tight_layout()
    fig1_path = _safe_path(os.path.join(save_dir, "ur16e_kinematics_comparison.png"))
    fig1.savefig(fig1_path, dpi=200)
    fig1.savefig(_safe_path(os.path.join(pkg_outputs_dir, "ur16e_kinematics_comparison.png")), dpi=200)

    # Figure 2: Dynamics & Inverse Torque Comparison
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    fig2.suptitle("Universal Robots UR16e: Inverse Dynamics Torque (MuJoCo RNEA vs. CasADi Lagrangian)", fontsize=14, fontweight="bold")

    for i in range(chain.n):
        row = i // 2
        col = i % 2
        ax = axes2[row, col]
        ax.plot(time_vec, tau_inv_mj[:, i], "b-", linewidth=2.0, label="MuJoCo Ground Truth")
        ax.plot(time_vec, tau_inv_casadi[:, i], "r--", linewidth=1.5, label="CasADi Analytical")
        ax.set_ylabel(f"Tau_{i+1} (N*m)")
        ax.set_title(f"Joint {i+1}: {chain.active_joints[i].name} (RMSE: {per_joint_rms_tau[i]:.2e} N*m)")
        ax.grid(True)
        if row == 2:
            ax.set_xlabel("Time (s)")
        if i == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    fig2_path = _safe_path(os.path.join(save_dir, "ur16e_dynamics_torque_comparison.png"))
    fig2.savefig(fig2_path, dpi=200)
    fig2.savefig(_safe_path(os.path.join(pkg_outputs_dir, "ur16e_dynamics_torque_comparison.png")), dpi=200)

    # Figure 3: Closed-Loop Tracking Performance
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig3.suptitle("Universal Robots UR16e: Closed-Loop Tracking under CasADi Computed Torque Control", fontsize=14, fontweight="bold")

    for i in range(chain.n):
        ax3a.plot(time_vec, q_cl_sim[:, i], "-", label=f"Joint {i+1} Actual")
        ax3a.plot(time_vec, q_ref[:, i], "--", alpha=0.7)
    ax3a.set_ylabel("Joint Positions (rad)")
    ax3a.set_title("Joint Angles: Reference (Dashed) vs. Forward MuJoCo Closed-Loop (Solid)")
    ax3a.grid(True)
    ax3a.legend(loc="upper right", ncol=3)

    for i in range(chain.n):
        err_i = q_cl_sim[:, i] - q_ref[:, i]
        ax3b.plot(time_vec, err_i, "-", label=f"e_{i+1}")
    ax3b.set_xlabel("Time (s)")
    ax3b.set_ylabel("Tracking Error (rad)")
    ax3b.set_title(f"Joint Coordinate Errors (Max Absolute Error: {max_track_err:.3e} rad)")
    ax3b.grid(True)
    ax3b.legend(loc="upper right", ncol=6)

    plt.tight_layout()
    fig3_path = _safe_path(os.path.join(save_dir, "ur16e_closed_loop_tracking.png"))
    fig3.savefig(fig3_path, dpi=200)
    fig3.savefig(_safe_path(os.path.join(pkg_outputs_dir, "ur16e_closed_loop_tracking.png")), dpi=200)

    if verbose:
        print(f"  [OK] Saved comparison figures:")
        print(f"       - {fig1_path}")
        print(f"       - {fig2_path}")
        print(f"       - {fig3_path}")

    if show_plots and not headless:
        plt.show()
    else:
        plt.close("all")

    return {
        "passed": passed,
        "max_pos_err": max_pos_err,
        "rms_pos_err": rms_pos_err,
        "max_quat_err": max_quat_err,
        "rms_quat_err": rms_quat_err,
        "max_vel_lin_err": max_vel_lin_err,
        "per_joint_rms_tau": per_joint_rms_tau,
        "max_track_err": max_track_err,
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
    }


def test_ur16e_animated_pipeline():
    """PyTest entrypoint. Runs headless if HEADLESS=1 or CI environment, otherwise shows brief animation."""
    headless = os.environ.get("HEADLESS", "0") == "1"
    res = run_ur16e_verification_and_animation(
        t_dur=2.0,
        dt=0.005,
        headless=headless,
        show_plots=False,
        verbose=True,
    )
    assert res["passed"] is True
    assert res["max_pos_err"] < 1e-4
    assert res["max_quat_err"] < 1e-3
    assert np.all(res["per_joint_rms_tau"] < 2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR16e 3D Animation & Verification Script")
    parser.add_argument("--urdf", type=str, default=None, help="Path to UR16e URDF")
    parser.add_argument("--duration", type=float, default=6.0, help="Trajectory duration (s)")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep (s)")
    parser.add_argument("--headless", action="store_true", help="Run without opening 3D viewer or plot GUI")
    parser.add_argument("--no-plots", action="store_true", help="Do not display interactive plots at the end")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save output plots")

    args = parser.parse_args()

    results = run_ur16e_verification_and_animation(
        urdf_path=args.urdf,
        t_dur=args.duration,
        dt=args.dt,
        headless=args.headless,
        show_plots=not args.no_plots,
        save_dir=args.save_dir,
        verbose=True,
    )

    sys.exit(0 if results["passed"] else 1)
