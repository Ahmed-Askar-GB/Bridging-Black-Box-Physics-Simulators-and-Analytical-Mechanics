"""Standalone Interactive 3D MuJoCo Animation and Real-Time Verification Dashboard for UR16e.

Features:
  1. Complete 3D mesh rendering of the Universal Robots UR16e (binary STLs with proper materials).
  2. Grounded visual floor (checkerboard grid), gradient skybox, and shadow-casting directional lighting.
  3. Pre-framed, optimal camera viewpoint (elevation, azimuth, distance, and lookat centering).
  4. Live 3D CasADi analytical End-Effector overlay (sphere marker, orientation triad, and trajectory trail).
  5. Live real-time telemetry HUD displaying:
     - Joint angles q_1..q_6
     - CasADi Analytical EE pose vs. MuJoCo Sensor EE pose
     - Position error ||e_pos|| (machine precision verification)
     - Orientation error ||e_rot|| (quaternion geodesic)
     - Commanded vs. Ground-Truth Inverse Dynamics Torques tau_1..tau_6
     - Closed-loop tracking error norm ||q - q_d||
  6. Interactive Keyboard Controls:
     - [Spacebar] : Pause / Resume animation
     - [R] / [r]  : Re-animate / Restart simulation from t = 0
     - [L] / [l]  : Toggle continuous loop mode on/off
     - [M] / [m]  : Toggle Mode (Mode A: Kinematic/InvDyn Replay <-> Mode B: Computed Torque Forward Sim)
     - [+] / [=]  : Increase playback speed (1.5x, 2.0x, 4.0x)
     - [-]        : Decrease playback speed (0.75x, 0.5x, 0.25x)
     - [C] / [c]  : Reset camera to default optimal framing
  7. Window Persistence: The viewer NEVER closes automatically at the end of the trajectory;
     it holds the robot and remains interactive until explicitly closed by the user.
"""

from __future__ import annotations

import argparse
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


def find_ur16e_urdf() -> tuple[str, str]:
    """Locate the UR16e URDF file and base mesh directory."""
    candidate_paths = [
        current_dir.parent / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "universalUR16e.urdf",
        current_dir / ".." / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "universalUR16e.urdf",
        Path(r"C:\Users\User\OneDrive - Egypt Japan University Of Science and Technology (E-JUST)\1- Research\Data-Driven\URDF Dynamic\End-to-End MATLAB-CasADi Implementation\test_UR16e\universalUR16e.urdf"),
    ]

    for p in candidate_paths:
        if p.exists():
            urdf_path = str(p.resolve())
            mesh_base_dir = str(p.parent.resolve()).replace("\\", "/")
            return urdf_path, mesh_base_dir

    raise FileNotFoundError(
        "Could not find universalUR16e.urdf in candidate paths. "
        "Please provide the path via the --urdf argument."
    )


def build_enhanced_mjcf(
    urdf_path: str,
    mesh_base_dir: str,
    chain: Chain,
    dh: DHTable,
    dt: float = 0.002,
) -> str:
    """Compile URDF into an enhanced MJCF with 3D STL meshes, EE site, floor, skybox, and lights."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Rewrite mesh filenames: convert package:// paths to local STL files
    for mesh_elem in root.iter("mesh"):
        fname = mesh_elem.attrib.get("filename", "")
        if fname:
            # Replace package://ur_description with local directory
            fname = fname.replace("package://ur_description", mesh_base_dir)
            fname = fname.replace("package://ur_e_description", mesh_base_dir)
            # Prefer STL meshes over DAE for MuJoCo native support
            if fname.endswith(".dae"):
                fname_stl = fname[:-4] + ".stl"
                if os.path.exists(fname_stl):
                    fname = fname_stl
            mesh_elem.set("filename", fname.replace("\\", "/"))

    # Convert to initial MJCF via MuJoCo compiler
    urdf_string = ET.tostring(root, encoding="unicode")
    m_tmp = mujoco.MjModel.from_xml_string(urdf_string)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        tmp_path = f.name
    # Close immediately to release Windows file lock
    try:
        mujoco.mj_saveLastXML(tmp_path, m_tmp)
        mjcf_tree = ET.parse(tmp_path)
        mjcf_root = mjcf_tree.getroot()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Set non-colliding flags and clean metallic visual colors on all geoms
    for g in mjcf_root.iter("geom"):
        g.attrib["contype"] = "0"
        g.attrib["conaffinity"] = "0"
        if g.attrib.get("name") != "ground_plane":
            g.attrib["rgba"] = "0.78 0.82 0.88 1.0"

    # 1. Enhance Visual Assets (Skybox, Grid Floor Texture, Materials)
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
            "roughness": "0.4",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "robot_metal",
            "rgba": "0.75 0.78 0.82 1.0",
            "reflectance": "0.3",
            "specular": "0.5",
            "shininess": "0.8",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "robot_dark",
            "rgba": "0.22 0.25 0.28 1.0",
            "reflectance": "0.2",
            "specular": "0.4",
            "shininess": "0.6",
        },
    )

    # 2. Add Ground Plane & Lights to Worldbody
    wb = mjcf_root.find("worldbody")
    if wb is not None:
        # Checkered Ground Plane
        ET.SubElement(
            wb,
            "geom",
            attrib={
                "name": "ground_plane",
                "type": "plane",
                "size": "4 4 0.05",
                "material": "floor_mat",
                "condim": "3",
            },
        )
        # Overhead directional sun with shadows
        ET.SubElement(
            wb,
            "light",
            attrib={
                "name": "sun",
                "directional": "true",
                "pos": "0 0 4.0",
                "dir": "0 0 -1",
                "castshadow": "true",
                "diffuse": "0.85 0.85 0.85",
                "specular": "0.25 0.25 0.25",
            },
        )
        # Front-right fill light
        ET.SubElement(
            wb,
            "light",
            attrib={
                "name": "fill_light",
                "directional": "false",
                "pos": "2.0 2.5 3.0",
                "dir": "-0.6 -0.7 -0.5",
                "castshadow": "false",
                "diffuse": "0.45 0.45 0.50",
            },
        )

    # 3. Inject Coincident End-Effector Site matching Analytical Frame
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

    # 4. Inject Actuators for Closed-Loop Torque Control
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


def set_optimal_camera(viewer: mujoco.viewer.Handle) -> None:
    """Set optimal camera perspective framing the entire UR16e robot upright."""
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.45], dtype=np.float64)
    viewer.cam.distance = 2.25
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -22.0


def animate_ur16e(
    urdf_path: str | None = None,
    t_dur: float = 6.0,
    dt: float = 0.002,
    initial_mode: str = "kinematic",
    start_paused: bool = False,
    loop: bool = False,
    speed: float = 1.0,
) -> None:
    """Launch interactive 3D MuJoCo animation for UR16e (delegates to universal engine)."""
    if urdf_path is None:
        urdf_path, mesh_base_dir = find_ur16e_urdf()
    else:
        mesh_base_dir = str(Path(urdf_path).parent.resolve())

    from animate_robot_interactive import animate_robot

    animate_robot(
        urdf_path=urdf_path,
        mesh_dir=mesh_base_dir,
        t_dur=t_dur,
        dt=dt,
        initial_mode=initial_mode,
        start_paused=start_paused,
        loop=loop,
        speed=speed,
    )


def _legacy_animate_ur16e(
    urdf_path: str | None = None,
    t_dur: float = 6.0,
    dt: float = 0.002,
    initial_mode: str = "kinematic",
    start_paused: bool = False,
    loop: bool = False,
    speed: float = 1.0,
) -> None:
    """Launch interactive 3D MuJoCo animation for UR16e with live analytical CasADi overlay and telemetry."""
    if urdf_path is None:
        urdf_path, mesh_base_dir = find_ur16e_urdf()
    else:
        urdf_path = str(Path(urdf_path).resolve())
        mesh_base_dir = str(Path(urdf_path).parent.resolve()).replace("\\", "/")

    print("\n======================================================================")
    print("      UR16e INTERACTIVE 3D MUJOCO ANIMATION & VERIFICATION VIEWER     ")
    print("======================================================================")
    print(f"  [+] URDF Model: {urdf_path}")
    print(f"  [+] Mesh Directory: {mesh_base_dir}")

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

    # Compute sign alignment: sign_map[i] = sign(round(z_urdf . z_dh))
    joint_signs = np.ones(chain.n)
    for i in range(chain.n):
        z_u = chain.active_joints[i].axis
        z_d = dh.z_axes[i]
        dot_val = float(np.dot(z_u, z_d))
        joint_signs[i] = 1.0 if dot_val >= 0.0 else -1.0

    print(f"  [OK] Extracted {chain.n}-DOF Standard DH Parameters.")
    print(f"  [OK] Autonomous Joint Sign Map: {joint_signs.tolist()}")

    controller = ComputedTorqueController(
        model=cas_dyn,
        joint_signs=joint_signs,
        kp=160.0,
        kd=28.0,
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Persistent Excitation Reference Trajectory
    # -------------------------------------------------------------------------
    print("\n>>> [2/4] Generating 500 Hz Smooth 6-DOF Trajectory (T = {:.1f}s)...".format(t_dur))
    time_vec = np.arange(0.0, t_dur, dt)
    num_steps = len(time_vec)

    q_start = np.array([0.0, -np.pi / 2, np.pi / 3, -np.pi / 3, np.pi / 2, 0.0])
    q_target = np.array([np.pi / 3, -np.pi / 4, np.pi / 2, -np.pi / 2, -np.pi / 3, np.pi / 2])

    q_ref, dq_ref, ddq_ref = traj_poly5(q_start, q_target, t_dur, dt)
    num_steps = min(num_steps, len(q_ref))
    q_ref = q_ref[:num_steps]
    dq_ref = dq_ref[:num_steps]
    ddq_ref = ddq_ref[:num_steps]
    time_vec = time_vec[:num_steps]

    # Pre-evaluate Analytical CasADi Model for All Steps
    print("  Pre-computing analytical CasADi forward kinematics and inverse dynamics...")
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
    # STAGE 3: Build Enhanced MuJoCo Model
    # -------------------------------------------------------------------------
    print("\n>>> [3/4] Compiling Enhanced MuJoCo Scene with STLs and Lighting...")
    mjcf_xml = build_enhanced_mjcf(urdf_path, mesh_base_dir, chain, dh, dt=dt)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    model.opt.timestep = dt
    model.dof_damping[:] = 0.0
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    print(f"  [OK] MuJoCo Model compiled. Geoms: {model.ngeom}, Lights: {model.nlight}, Site ID: {site_id}")

    # -------------------------------------------------------------------------
    # STAGE 4: Interactive Simulation Loop
    # -------------------------------------------------------------------------
    print("\n>>> [4/4] Launching Interactive MuJoCo Viewer Window...")
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

    # State variables controlled by keyboard callback
    is_paused = start_paused
    restart_requested = False
    is_looping = loop
    sim_mode = initial_mode.lower()  # "kinematic" or "control"
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
            print("\n[KEY EVENT] 'C' pressed -> Resetting camera to optimal view.")

    # Launch passive viewer
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # Initial state setup
    k = 0
    trail_positions: list[np.ndarray] = []
    last_hud_print_time = 0.0

    # Reset MuJoCo state to trajectory start
    data.qpos[:] = q_ref[0, :]
    data.qvel[:] = dq_ref[0, :]
    data.qacc[:] = ddq_ref[0, :]
    mujoco.mj_forward(model, data)

    t_wall_last = time.time()
    t_sim_accum = 0.0
    frame_count = 0

    print("Viewer running. Displaying live measurements...\n")

    while viewer.is_running():
        # Lock optimal camera viewpoint on initial startup frames
        if frame_count < 10 or reset_camera_flag:
            set_optimal_camera(viewer)
            reset_camera_flag = False
        frame_count += 1

        # Handle Re-animation / Restart Request
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
            print(f"\n>>> RE-ANIMATING SIMULATION [{sim_mode.upper()} MODE] <<<")

        # Time pacing
        t_wall_now = time.time()
        dt_wall = t_wall_now - t_wall_last
        t_wall_last = t_wall_now

        trajectory_active = k < num_steps

        if trajectory_active and not is_paused:
            # Advance simulation time based on speed
            t_sim_accum += dt_wall * playback_speed
            steps_to_advance = int(t_sim_accum / dt)
            if steps_to_advance > 0:
                t_sim_accum -= steps_to_advance * dt
                # Step limit to avoid freezing
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
                    # Mode A: Kinematic & Inverse Dynamics Replay
                    data.qpos[:] = q_des
                    data.qvel[:] = dq_des
                    data.qacc[:] = ddq_des
                    mujoco.mj_inverse(model, data)
                    mujoco.mj_forward(model, data)
                else:
                    # Mode B: Dynamic Forward Simulation with Computed Torque Control
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

            # End of trajectory handling
            if k >= num_steps:
                if is_looping:
                    k = 0
                    trail_positions.clear()
                    print("\n[LOOP] Trajectory reached end -> Replaying from start...")
                else:
                    print("\n======================================================================")
                    print("  [TRAJECTORY FINISHED] End of 6.0s trajectory reached.")
                    print("  Viewer is ACTIVE and holding final pose.")
                    print("  --> Press [R] to Re-animate / Replay.")
                    print("  --> Press [Space] to Pause/Play, [L] to Toggle Continuous Loop.")
                    print("  --> Press [M] to Toggle Mode between Kinematic and Closed-Loop Control.")
                    print("  --> Use mouse to orbit, pan, and zoom.")
                    print("======================================================================\n")

        # Query Current Live State & Sensors
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

        # Update 3D Overlay Visuals in Viewer
        if trajectory_active:
            if len(trail_positions) == 0 or np.linalg.norm(pos_mj - trail_positions[-1]) > 0.02:
                trail_positions.append(pos_mj.copy())
                if len(trail_positions) > 75:
                    trail_positions.pop(0)

        with viewer.lock():
            viewer.user_scn.ngeom = 0

            # 1. Semi-transparent bright green sphere at CasADi analytical EE target
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.024, 0.0, 0.0], dtype=np.float64),
                    pos=p_cas,
                    mat=R_cas.flatten(),
                    rgba=np.array([0.0, 1.0, 0.2, 0.75], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

            # 2. Analytical Coordinate Triad (X=Red, Y=Green, Z=Blue)
            axis_len = 0.065
            axis_rad = 0.0035
            triad_axes = [
                (R_cas[:, 0], np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32)),  # X
                (R_cas[:, 1], np.array([0.1, 1.0, 0.1, 0.9], dtype=np.float32)),  # Y
                (R_cas[:, 2], np.array([0.1, 0.4, 1.0, 0.9], dtype=np.float32)),  # Z
            ]
            for axis_vec, col in triad_axes:
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    p_mid = p_cas + 0.5 * axis_len * axis_vec
                    R_cyl = np.eye(3)
                    z_cyl = axis_vec / np.linalg.norm(axis_vec)
                    # Simple orientation for cylinder
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

            # 3. Continuous 3D Trajectory Trail (Cyan spheres)
            for pt in trail_positions:
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.007, 0.0, 0.0], dtype=np.float64),
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.15, 0.75, 1.0, 0.45], dtype=np.float32),
                    )
                    viewer.user_scn.ngeom += 1

        viewer.sync()

        # Live Terminal HUD Table (printed every ~0.15s)
        if (t_wall_now - last_hud_print_time) > 0.15:
            last_hud_print_time = t_wall_now
            t_curr = time_vec[current_step]
            pct = (current_step / (num_steps - 1)) * 100.0
            state_str = "PAUSED" if is_paused else ("PLAYING" if trajectory_active else "HOLDING (REPLAY READY)")
            q_deg = np.rad2deg(data.qpos[:6])

            print(
                f"\r[{state_str:^15}] t={t_curr:4.2f}/{t_dur:.1f}s ({pct:4.1f}%) | "
                f"Mode={sim_mode.upper()[:3]} | Spd={playback_speed:3.1f}x | "
                f"EE-Pos Err: {e_pos * 1000.0:8.4e} mm | "
                f"Rot Err: {e_rot:8.4e} rad | "
                f"Tau Diff: {tau_err:8.4e} Nm | "
                f"q=[{q_deg[0]:4.0f},{q_deg[1]:4.0f},{q_deg[2]:4.0f},{q_deg[3]:4.0f},{q_deg[4]:4.0f},{q_deg[5]:4.0f}]",
                end="",
                flush=True,
            )

        # Gentle sleep to prevent 100% CPU usage
        time.sleep(0.002)

    print("\n\nViewer closed by user. Interactive session finished.")


def main() -> None:
    """CLI Entry point for UR16e interactive animation viewer."""
    parser = argparse.ArgumentParser(
        description="UR16e Interactive 3D MuJoCo Animation with Live Analytical CasADi Verification"
    )
    parser.add_argument("--urdf", type=str, default=None, help="Path to UR16e URDF file")
    parser.add_argument("--duration", "-T", type=float, default=6.0, help="Trajectory duration in seconds (default: 6.0)")
    parser.add_argument("--dt", type=float, default=0.002, help="Timestep in seconds (default: 0.002)")
    parser.add_argument(
        "--mode",
        type=str,
        default="kinematic",
        choices=["kinematic", "control"],
        help="Initial playback mode: 'kinematic' (Inverse Dynamics Replay) or 'control' (Computed Torque Forward Dynamics)",
    )
    parser.add_argument("--paused", action="store_true", help="Start simulation in paused state")
    parser.add_argument("--loop", action="store_true", help="Enable continuous trajectory looping")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 1.0, 1.5, 2.0)")

    args = parser.parse_args()

    animate_ur16e(
        urdf_path=args.urdf,
        t_dur=args.duration,
        dt=args.dt,
        initial_mode=args.mode,
        start_paused=args.paused,
        loop=args.loop,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
