"""Universal Interactive 3D MuJoCo Animation and Real-Time Verification Dashboard.

Supports ANY serial robot manipulator (e.g. UR16e, UR5, ABB IRB 120, Puma 560, SCARA, KUKA):
  - Autonomous ROS package URI and mesh resolution (STL/OBJ/DAE fallback).
  - Autonomous procedural visual link capsules if meshes are absent (zero crash guarantee).
  - Reach-based auto-framing camera calibrated to manipulator dimension R_reach.
  - Limit-aware N-DOF smooth trajectory synthesis (5th-order polynomial).
  - Real-time analytical CasADi EE overlay (target sphere, triad, continuous trajectory trail).
  - Dynamic telemetry HUD table formatting any number of joints.
  - Keyboard Hotkeys:
      [Space] : Pause / Resume
      [R]     : Re-animate / Replay from t = 0
      [L]     : Toggle continuous loop replay
      [M]     : Toggle Mode (Kinematic Replay <-> Computed Torque Control)
      [+]     : Speed up playback (up to 4.0x)
      [-]     : Slow down playback (down to 0.25x)
      [C]     : Reset Camera to optimal framing
  - Window Persistence: Holds final pose and stays interactive without closing.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Ensure urdf2dt is discoverable on path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from urdf2dt.analytical.dynamics_casadi import dynamics_casadi
from urdf2dt.analytical.fk_casadi import fk_casadi
from urdf2dt.analytical.jacobian_casadi import jacobian_casadi
from urdf2dt.containers import Chain, DHTable
from urdf2dt.control.computed_torque import ComputedTorqueController
from urdf2dt.parsing.urdf_build_chain import urdf_build_chain
from urdf2dt.parsing.urdf_extract_dh import urdf_extract_dh
from urdf2dt.parsing.urdf_extract_inertia import urdf_extract_inertia
from urdf2dt.parsing.urdf_parse import urdf_parse
from urdf2dt.utils.math_utils import R_to_quat, quat_error
from urdf2dt.utils.traj_gen import traj_poly5


def find_default_urdf() -> str:
    """Discover a default URDF in the workspace if none is explicitly passed."""
    candidates = [
        current_dir.parent / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "universalUR16e.urdf",
        current_dir / "tests" / "fixtures" / "ur16e.urdf",
        current_dir.parent / "universalUR5.urdf",
        current_dir.parent / "abbIrb120" / "abbIrb120.urdf",
        current_dir / "tests" / "fixtures" / "ur5.urdf",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())

    # Fallback to any .urdf in parent directories
    matches = glob.glob(str(current_dir.parent / "**/*.urdf"), recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not find any .urdf file in workspace. Please provide --urdf <path>.")


def generate_ndof_trajectory(
    chain: Chain,
    t_dur: float = 6.0,
    dt: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize smooth 5th-order polynomial trajectory tailored to the active joints and limits."""
    n = chain.n
    q_start = np.zeros(n)
    q_target = np.zeros(n)

    for i, joint in enumerate(chain.active_joints):
        q_min = getattr(joint, "lower", None)
        q_max = getattr(joint, "upper", None)
        is_bounded = (q_min is not None) and (q_max is not None) and (q_max > q_min)

        alt_sign = 1.0 if (i % 2 == 0) else -1.0

        if is_bounded and abs(q_max - q_min) < 4.0 * np.pi:
            q_mid = 0.5 * (q_min + q_max)
            span = 0.5 * (q_max - q_min)
            # Oscillate smoothly within middle 60% of joint range
            amp = min(span * 0.35, 1.0)
            q_start[i] = q_mid - amp * alt_sign
            q_target[i] = q_mid + amp * alt_sign
        else:
            # Continuous or unbounded joint
            amp = 0.8
            q_start[i] = -0.4 * amp * alt_sign
            q_target[i] = 0.6 * amp * alt_sign

    time_vec, q_ref, dq_ref, ddq_ref = traj_poly5(q_start, q_target, t_dur, dt)
    return q_ref, dq_ref, ddq_ref, time_vec


def compute_reach_and_camera(dh: DHTable) -> tuple[float, np.ndarray, float]:
    """Calculate manipulator reach and optimal camera framing parameters from Standard DH table."""
    # dh.table is (N, 4) with columns: [theta_offset, d, a, alpha]
    d_vals = dh.table[:, 1]
    a_vals = dh.table[:, 2]
    reach = float(np.sum(np.abs(a_vals) + np.abs(d_vals)))
    reach = max(0.35, reach)

    lookat = np.array([0.0, 0.0, 0.40 * reach], dtype=np.float64)
    distance = max(1.1, 2.2 * reach)
    return reach, lookat, distance


def build_universal_mjcf(
    urdf_path: str,
    chain: Chain,
    dh: DHTable,
    mesh_dir: str | None = None,
    dt: float = 0.002,
) -> str:
    """Compile any arbitrary URDF into an enhanced MJCF with 3D meshes/capsules, EE site, floor, and lighting."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    search_dirs = []
    if mesh_dir and os.path.isdir(mesh_dir):
        search_dirs.append(os.path.abspath(mesh_dir))
    search_dirs.extend([
        urdf_dir,
        os.path.join(urdf_dir, "meshes"),
        os.path.join(urdf_dir, "..", "meshes"),
        os.path.join(urdf_dir, "..", "test_UR16e"),
        os.path.join(urdf_dir, "..", "STL_Files"),
        os.path.abspath("End-to-End MATLAB-CasADi Implementation/test_UR16e"),
        os.path.abspath("STL_Files"),
    ])

    # 1. Resolve mesh URIs across arbitrary package:// prefixes
    for m in root.iter("mesh"):
        fn = m.attrib.get("filename", "")
        if not fn:
            continue

        rel_path = fn
        if fn.startswith("package://"):
            parts = fn.split("package://")[-1].split("/", 1)
            rel_path = parts[1] if len(parts) > 1 else parts[0]

        resolved = None
        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            candidates = [
                os.path.join(sdir, rel_path),
                os.path.join(sdir, os.path.basename(rel_path)),
                os.path.join(sdir, "visual", os.path.basename(rel_path)),
                os.path.join(sdir, "collision", os.path.basename(rel_path)),
            ]
            for cand in candidates:
                if os.path.exists(cand) and cand.lower().endswith((".stl", ".obj")):
                    resolved = cand
                    break
                if cand.lower().endswith(".dae"):
                    stl_cand = cand[:-4] + ".stl"
                    if os.path.exists(stl_cand):
                        resolved = stl_cand
                        break
            if resolved:
                break

        if resolved and os.path.exists(resolved):
            m.set("filename", resolved.replace("\\", "/"))

    # 2. Safeguard non-zero inertias for moving bodies
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", attrib={"value": "0.1"})
            ET.SubElement(inertial, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(
                inertial,
                "inertia",
                attrib={"ixx": "1e-4", "iyy": "1e-4", "izz": "1e-4", "ixy": "0", "ixz": "0", "iyz": "0"},
            )
        else:
            m_elem = inertial.find("mass")
            if m_elem is not None:
                try:
                    if float(m_elem.attrib.get("value", 0.0)) <= 1e-6:
                        m_elem.attrib["value"] = "0.1"
                except ValueError:
                    m_elem.attrib["value"] = "0.1"

    # Convert to initial MJCF
    s_urdf = ET.tostring(root, encoding="unicode")
    try:
        m_tmp = mujoco.MjModel.from_xml_string(s_urdf)
    except Exception:
        # Fall back gracefully to procedural geoms if meshes fail
        for link in root.findall("link"):
            for v in list(link.findall("visual")):
                link.remove(v)
            for c in list(link.findall("collision")):
                link.remove(c)
        s_clean = ET.tostring(root, encoding="unicode")
        m_tmp = mujoco.MjModel.from_xml_string(s_clean)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        tmp_path = f.name
    f.close()

    try:
        mujoco.mj_saveLastXML(tmp_path, m_tmp)
        mjcf_tree = ET.parse(tmp_path)
        mjcf_root = mjcf_tree.getroot()

        # Set contype=0 and conaffinity=0 on all geoms so physics is free of unwanted contact friction
        for g in mjcf_root.iter("geom"):
            g.attrib["contype"] = "0"
            g.attrib["conaffinity"] = "0"
            if "rgba" not in g.attrib and "material" not in g.attrib:
                g.attrib["rgba"] = "0.74 0.78 0.84 1.0"

        # Ensure every robot body has at least one visual geom
        for b in mjcf_root.iter("body"):
            if len(b.findall("geom")) == 0:
                ET.SubElement(
                    b,
                    "geom",
                    attrib={
                        "type": "capsule",
                        "size": "0.038 0.075",
                        "rgba": "0.68 0.72 0.78 1.0",
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "1",
                    },
                )

        # Inject Visual Environment: Skybox, Floor, Lighting
        asset = mjcf_root.find("asset")
        if asset is None:
            asset = ET.SubElement(mjcf_root, "asset")

        ET.SubElement(
            asset,
            "texture",
            attrib={
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.25 0.35 0.45",
                "rgb2": "0.05 0.07 0.10",
                "width": "512",
                "height": "512",
            },
        )
        ET.SubElement(
            asset,
            "texture",
            attrib={
                "name": "grid_tex",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.18 0.22 0.26",
                "rgb2": "0.24 0.28 0.32",
                "width": "512",
                "height": "512",
                "mark": "edge",
                "markrgb": "0.35 0.45 0.55",
            },
        )
        ET.SubElement(
            asset,
            "material",
            attrib={
                "name": "floor_mat",
                "texture": "grid_tex",
                "texrepeat": "3 3",
                "texuniform": "true",
                "reflectance": "0.15",
            },
        )

        wb = mjcf_root.find("worldbody")
        if wb is not None:
            reach = float(np.sum(np.abs(dh.table[:, 2]) + np.abs(dh.table[:, 1])))
            floor_size = max(3.0, reach * 2.5)
            ET.SubElement(
                wb,
                "geom",
                attrib={
                    "name": "ground_plane",
                    "type": "plane",
                    "size": f"{floor_size} {floor_size} 0.05",
                    "material": "floor_mat",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
            ET.SubElement(
                wb,
                "light",
                attrib={
                    "name": "sun",
                    "directional": "true",
                    "pos": f"0 0 {max(3.0, reach * 3.0):.2f}",
                    "dir": "0 0 -1",
                    "castshadow": "true",
                    "diffuse": "0.85 0.85 0.85",
                },
            )
            ET.SubElement(
                wb,
                "light",
                attrib={
                    "name": "fill",
                    "directional": "false",
                    "pos": f"{reach * 1.5:.2f} {reach * 1.8:.2f} {reach * 2.0:.2f}",
                    "dir": "-0.6 -0.7 -0.5",
                    "diffuse": "0.45 0.45 0.50",
                },
            )

        # Inject Coincident End-Effector Site matching Analytical Frame
        if chain.active_links:
            tip_name = chain.active_links[-1]
            tip_body = None
            for b in mjcf_root.iter("body"):
                if b.attrib.get("name") == tip_name:
                    tip_body = b
                    break

            if tip_body is not None:
                site = ET.SubElement(tip_body, "site")
                site.attrib["name"] = "ee_site"
                T_u0 = chain.T_urdf[-1]
                T_dh0 = dh.T_dh[-1] @ chain.T_ee
                T_target = np.linalg.inv(T_u0) @ T_dh0
                pos = T_target[:3, 3]
                quat = R_to_quat(T_target[:3, :3])

                site.attrib["pos"] = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
                site.attrib["quat"] = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
                site.attrib["size"] = "0.012"
                site.attrib["rgba"] = "1.0 0.2 0.2 0.8"

        # Inject Actuators
        act_container = mjcf_root.find("actuator")
        if act_container is None:
            act_container = ET.SubElement(mjcf_root, "actuator")

        for j in chain.active_joints:
            motor = ET.SubElement(act_container, "motor")
            motor.attrib["name"] = f"motor_{j.name}"
            motor.attrib["joint"] = j.name
            motor.attrib["gear"] = "1"
            motor.attrib["ctrllimited"] = "false"

        return ET.tostring(mjcf_root, encoding="unicode")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def animate_robot(
    urdf_path: str | None = None,
    mesh_dir: str | None = None,
    t_dur: float = 6.0,
    dt: float = 0.002,
    initial_mode: str = "kinematic",
    start_paused: bool = False,
    loop: bool = False,
    speed: float = 1.0,
) -> None:
    """Universal interactive 3D MuJoCo animation and real-time verification viewer for any robot."""
    if urdf_path is None:
        urdf_path = find_default_urdf()
    urdf_path = str(Path(urdf_path).resolve())

    print("\n======================================================================")
    print("      UNIVERSAL ROBOT INTERACTIVE 3D ANIMATION & DIGITAL TWIN         ")
    print("======================================================================")
    print(f"  [+] URDF Model: {urdf_path}")
    if mesh_dir:
        print(f"  [+] Mesh Directory: {mesh_dir}")

    # -------------------------------------------------------------------------
    # STAGE 1: Analytical Model Synthesis
    # -------------------------------------------------------------------------
    print("\n>>> [1/4] Synthesizing Analytical CasADi Model...")
    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertias = urdf_extract_inertia(chain, dh)

    cas_fk = fk_casadi(chain, dh)
    cas_jac = jacobian_casadi(chain, dh, cas_fk)
    cas_dyn = dynamics_casadi(chain, dh, inertias, cas_fk, cas_jac)

    # Autonomous Joint Axis Alignment
    joint_signs = np.ones(chain.n)
    for i in range(chain.n):
        z_u = chain.active_joints[i].axis
        z_d = dh.z_axes[i]
        dot_val = float(np.dot(z_u, z_d))
        joint_signs[i] = 1.0 if dot_val >= 0.0 else -1.0

    reach, cam_lookat, cam_distance = compute_reach_and_camera(dh)

    print(f"  [OK] Extracted {chain.n}-DOF Standard DH Parameters.")
    print(f"  [OK] Manipulator Reach: {reach:.3f} m")
    print(f"  [OK] Autonomous Joint Sign Map: {joint_signs.tolist()}")

    controller = ComputedTorqueController(
        model=cas_dyn,
        joint_signs=joint_signs,
        kp=160.0,
        kd=28.0,
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Limit-Aware N-DOF Reference Trajectory
    # -------------------------------------------------------------------------
    print(f"\n>>> [2/4] Generating Limit-Aware {chain.n}-DOF Trajectory (T = {t_dur:.1f}s)...")
    q_ref, dq_ref, ddq_ref, time_vec = generate_ndof_trajectory(chain, t_dur=t_dur, dt=dt)
    num_steps = len(time_vec)

    # Pre-evaluate Analytical CasADi Model for All Steps
    print("  Pre-computing analytical forward kinematics & inverse dynamics...")
    pos_casadi = np.zeros((num_steps, 3))
    rot_casadi = np.zeros((num_steps, 3, 3))
    quat_casadi = np.zeros((num_steps, 4))
    tau_casadi = np.zeros((num_steps, chain.n))

    for k in range(num_steps):
        q_dh = joint_signs * q_ref[k, :]
        dq_dh = joint_signs * dq_ref[k, :]
        ddq_dh = joint_signs * ddq_ref[k, :]

        p_k, R_k, quat_k = cas_dyn.fk_fun(q_dh)
        pos_casadi[k, :] = np.array(p_k).flatten()
        rot_casadi[k, :, :] = np.array(R_k).reshape(3, 3)
        quat_casadi[k, :] = np.array(quat_k).flatten()

        tau_dh = np.array(cas_dyn.tau_fun(q_dh, dq_dh, ddq_dh)).flatten()
        tau_casadi[k, :] = joint_signs * tau_dh

    # -------------------------------------------------------------------------
    # STAGE 3: Build Universal MuJoCo Model
    # -------------------------------------------------------------------------
    print("\n>>> [3/4] Compiling Enhanced MuJoCo Digital Twin Scene...")
    mjcf_xml = build_universal_mjcf(urdf_path, chain, dh, mesh_dir=mesh_dir, dt=dt)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    model.opt.timestep = dt
    model.dof_damping[:] = 0.0
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    print(f"  [OK] MuJoCo Model compiled. Geoms: {model.ngeom}, Meshes: {model.nmesh}, Site ID: {site_id}")

    # -------------------------------------------------------------------------
    # STAGE 4: Interactive Simulation Loop
    # -------------------------------------------------------------------------
    print("\n>>> [4/4] Launching Universal Interactive Viewer...")
    print("----------------------------------------------------------------------")
    print("  KEYBOARD SHORTCUTS:")
    print("    [Space] : Pause / Resume animation")
    print("    [R]     : Re-animate / Restart trajectory from t = 0")
    print("    [L]     : Toggle continuous loop mode")
    print("    [M]     : Toggle Mode (Kinematic Replay <-> Computed Torque Control)")
    print("    [+] / = : Increase playback speed")
    print("    [-]     : Decrease playback speed")
    print("    [C]     : Reset Camera to optimal framing")
    print("    Close the viewer window [X] to exit.")
    print("----------------------------------------------------------------------\n")

    is_paused = start_paused
    restart_requested = False
    is_looping = loop
    sim_mode = initial_mode.lower()
    playback_speed = max(0.1, min(speed, 5.0))
    reset_camera_flag = True

    def key_callback(keycode: int) -> None:
        nonlocal is_paused, restart_requested, is_looping, sim_mode, playback_speed, reset_camera_flag

        if keycode == 32:  # Spacebar
            is_paused = not is_paused
            status = "PAUSED" if is_paused else "PLAYING"
            print(f"\n[KEY EVENT] Space pressed -> Simulation {status}")
        elif keycode in (82, 114):  # R or r
            restart_requested = True
            print("\n[KEY EVENT] 'R' pressed -> Restarting trajectory from t = 0.0s...")
        elif keycode in (76, 108):  # L or l
            is_looping = not is_looping
            print(f"\n[KEY EVENT] 'L' pressed -> Continuous Loop: {'ENABLED' if is_looping else 'DISABLED'}")
        elif keycode in (77, 109):  # M or m
            sim_mode = "control" if sim_mode == "kinematic" else "kinematic"
            restart_requested = True
            print(f"\n[KEY EVENT] 'M' pressed -> Switched Mode to: {sim_mode.upper()} (Restarting)")
        elif keycode in (61, 43):  # + or =
            playback_speed = min(4.0, playback_speed * 1.5)
            print(f"\n[KEY EVENT] '+' pressed -> Playback Speed: {playback_speed:.2f}x")
        elif keycode == 45:  # -
            playback_speed = max(0.25, playback_speed / 1.5)
            print(f"\n[KEY EVENT] '-' pressed -> Playback Speed: {playback_speed:.2f}x")
        elif keycode in (67, 99):  # C or c
            reset_camera_flag = True
            print("\n[KEY EVENT] 'C' pressed -> Resetting camera to optimal framing.")

    def set_camera(v: mujoco.viewer.Handle) -> None:
        v.cam.lookat[:] = cam_lookat
        v.cam.distance = cam_distance
        v.cam.azimuth = 135.0
        v.cam.elevation = -20.0

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    k = 0
    trail_positions: list[np.ndarray] = []
    last_hud_print_time = 0.0
    frame_count = 0

    data.qpos[:] = q_ref[0, :]
    data.qvel[:] = dq_ref[0, :]
    data.qacc[:] = ddq_ref[0, :]
    mujoco.mj_forward(model, data)

    t_wall_last = time.time()
    t_sim_accum = 0.0

    print("Viewer running. Displaying live measurements...\n")

    while viewer.is_running():
        # Lock optimal camera viewpoint on initial frames
        if frame_count < 10 or reset_camera_flag:
            set_camera(viewer)
            reset_camera_flag = False
        frame_count += 1

        # Re-animation handler
        if restart_requested:
            k = 0
            t_sim_accum = 0.0
            trail_positions.clear()
            data.qpos[:] = q_ref[0, :]
            data.qvel[:] = dq_ref[0, :]
            data.qacc[:] = ddq_ref[0, :]
            if model.nu > 0:
                data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            restart_requested = False
            t_wall_last = time.time()
            print(f"\n>>> RE-ANIMATING [{sim_mode.upper()} MODE] <<<")

        t_wall_now = time.time()
        dt_wall = t_wall_now - t_wall_last
        t_wall_last = t_wall_now

        trajectory_active = k < num_steps

        if trajectory_active and not is_paused:
            t_sim_accum += dt_wall * playback_speed
            steps_to_advance = int(t_sim_accum / dt)
            if steps_to_advance > 0:
                t_sim_accum -= steps_to_advance * dt
                steps_to_advance = min(steps_to_advance, 10)
            else:
                steps_to_advance = 1

            for _ in range(steps_to_advance):
                if k >= num_steps:
                    break

                q_des = q_ref[k, :]
                dq_des = dq_ref[k, :]
                ddq_des = ddq_ref[k, :]

                if sim_mode == "kinematic":
                    data.qpos[:] = q_des
                    data.qvel[:] = dq_des
                    data.qacc[:] = ddq_des
                    mujoco.mj_inverse(model, data)
                    mujoco.mj_forward(model, data)
                else:
                    q_curr = data.qpos.copy()
                    dq_curr = data.qvel.copy()
                    tau_cmd = controller.compute_control(
                        q=q_curr,
                        dq=dq_curr,
                        q_d=q_des,
                        dq_d=dq_des,
                        ddq_d=ddq_des,
                        t=time_vec[k],
                    )
                    if model.nu > 0:
                        data.ctrl[:] = tau_cmd
                    mujoco.mj_step(model, data)

                k += 1

            if k >= num_steps:
                if is_looping:
                    k = 0
                    trail_positions.clear()
                    print("\n[LOOP] Trajectory reached end -> Replaying from start...")
                else:
                    print("\n======================================================================")
                    print(f"  [TRAJECTORY FINISHED] End of {t_dur:.1f}s trajectory reached.")
                    print("  Viewer is ACTIVE and holding final pose.")
                    print("  --> Press [R] to Re-animate / Replay.")
                    print("  --> Press [Space] to Pause/Play, [L] to Toggle Continuous Loop.")
                    print("  --> Press [M] to Toggle Mode between Kinematic and Closed-Loop Control.")
                    print("  --> Use mouse to orbit, pan, and zoom.")
                    print("======================================================================\n")

        # Query Sensors & Diagnostics
        current_step = min(k, num_steps - 1)
        if site_id >= 0:
            pos_mj = data.site_xpos[site_id].copy()
            rot_mj = data.site_xmat[site_id].reshape(3, 3).copy()
        else:
            pos_mj = data.xpos[model.nbody - 1].copy()
            rot_mj = data.xmat[model.nbody - 1].reshape(3, 3).copy()

        quat_mj = R_to_quat(rot_mj)
        p_cas = pos_casadi[current_step, :]
        R_cas = rot_casadi[current_step, :, :]
        q_cas = quat_casadi[current_step, :]

        e_pos = np.linalg.norm(pos_mj - p_cas)
        e_rot = quat_error(quat_mj, q_cas)
        tau_err = 0.0
        if sim_mode == "kinematic":
            tau_mj = data.qfrc_inverse.copy()
            tau_err = float(np.max(np.abs(tau_casadi[current_step, :] - tau_mj)))

        # Update 3D Overlay Visuals
        if trajectory_active:
            if len(trail_positions) == 0 or np.linalg.norm(pos_mj - trail_positions[-1]) > (0.015 * reach):
                trail_positions.append(pos_mj.copy())
                if len(trail_positions) > 75:
                    trail_positions.pop(0)

        with viewer.lock():
            viewer.user_scn.ngeom = 0

            # 1. CasADi analytical target sphere
            sphere_rad = max(0.012, 0.024 * reach)
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([sphere_rad, 0.0, 0.0], dtype=np.float64),
                    pos=p_cas,
                    mat=R_cas.flatten(),
                    rgba=np.array([0.0, 1.0, 0.2, 0.75], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            # 2. Analytical Coordinate Triad (X red, Y green, Z blue)
            axis_len = max(0.035, 0.065 * reach)
            axis_rad = max(0.002, 0.0035 * reach)
            triad_axes = [
                (R_cas[:, 0], np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32)),
                (R_cas[:, 1], np.array([0.1, 1.0, 0.1, 0.9], dtype=np.float32)),
                (R_cas[:, 2], np.array([0.1, 0.4, 1.0, 0.9], dtype=np.float32)),
            ]
            for axis_vec, col in triad_axes:
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    p_mid = p_cas + 0.5 * axis_len * axis_vec
                    R_cyl = np.eye(3)
                    z_cyl = axis_vec / np.linalg.norm(axis_vec)
                    v_tmp = np.array([0.0, 0.0, 1.0])
                    if abs(np.dot(z_cyl, v_tmp)) > 0.9:
                        v_tmp = np.array([1.0, 0.0, 0.0])
                    x_cyl = np.cross(v_tmp, z_cyl)
                    x_cyl /= np.linalg.norm(x_cyl)
                    y_cyl = np.cross(z_cyl, x_cyl)
                    R_cyl = np.column_stack([x_cyl, y_cyl, z_cyl])

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=np.array([axis_rad, axis_len * 0.5, 0.0], dtype=np.float64),
                        pos=p_mid,
                        mat=R_cyl.flatten(),
                        rgba=col,
                    )
                    viewer.user_scn.ngeom += 1

            # 3. Trajectory Trail
            trail_rad = max(0.004, 0.007 * reach)
            for pt in trail_positions:
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([trail_rad, 0.0, 0.0], dtype=np.float64),
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.15, 0.75, 1.0, 0.45], dtype=np.float32),
                    )
                    viewer.user_scn.ngeom += 1

        viewer.sync()

        # Dynamic N-DOF Terminal HUD Table
        if (t_wall_now - last_hud_print_time) > 0.15:
            last_hud_print_time = t_wall_now
            t_curr = time_vec[current_step]
            pct = (current_step / max(1, num_steps - 1)) * 100.0
            state_str = "PAUSED" if is_paused else ("PLAYING" if trajectory_active else "HOLDING (REPLAY READY)")
            q_deg = np.rad2deg(data.qpos[:chain.n])
            q_str = ",".join(f"{v:3.0f}" for v in q_deg[:min(6, chain.n)])
            if chain.n > 6:
                q_str += ",..."

            print(
                f"\r[{state_str:^15}] t={t_curr:4.2f}/{t_dur:.1f}s ({pct:4.1f}%) | "
                f"Mode={sim_mode.upper()[:3]} | Spd={playback_speed:3.1f}x | "
                f"EE-Pos Err: {e_pos * 1000.0:8.4e} mm | "
                f"Rot Err: {e_rot:8.4e} rad | "
                f"Tau Diff: {tau_err:8.4e} Nm | "
                f"q=[{q_str}]",
                end="",
                flush=True,
            )

        time.sleep(0.002)

    print("\n\nViewer closed by user. Universal session finished.")


def main() -> None:
    """CLI Entry point for universal robot interactive animation viewer."""
    parser = argparse.ArgumentParser(
        description="Universal Robot Interactive 3D MuJoCo Animation with Live Analytical CasADi Verification"
    )
    parser.add_argument("--urdf", type=str, default=None, help="Path to any robot .urdf file")
    parser.add_argument("--mesh-dir", type=str, default=None, help="Optional explicit mesh/asset directory")
    parser.add_argument("--duration", "-T", type=float, default=6.0, help="Trajectory duration in seconds (default: 6.0)")
    parser.add_argument("--dt", type=float, default=0.002, help="Timestep in seconds (default: 0.002)")
    parser.add_argument(
        "--mode",
        type=str,
        default="kinematic",
        choices=["kinematic", "control"],
        help="Initial mode: 'kinematic' (Inverse Dynamics Replay) or 'control' (Computed Torque Control)",
    )
    parser.add_argument("--paused", action="store_true", help="Start simulation in paused state")
    parser.add_argument("--loop", action="store_true", help="Enable continuous trajectory replay loop")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")

    args = parser.parse_args()

    animate_robot(
        urdf_path=args.urdf,
        mesh_dir=args.mesh_dir,
        t_dur=args.duration,
        dt=args.dt,
        initial_mode=args.mode,
        start_paused=args.paused,
        loop=args.loop,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
