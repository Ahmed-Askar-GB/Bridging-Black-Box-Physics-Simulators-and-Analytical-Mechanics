"""Headless MuJoCo Digital Twin plant for forward time-stepping and inverse dynamics ground truth."""

from __future__ import annotations
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from urdf2dt.containers import Chain, DHTable
from urdf2dt.utils.math_utils import R_to_quat


class MuJoCoTwin:
    """Headless MuJoCo plant wrapping MjModel and MjData with automated site injection, inverse dynamics, and time-stepping."""

    def __init__(
        self,
        urdf_path: str,
        chain: Chain | None = None,
        dh: DHTable | None = None,
        dt: float = 0.001,
        mesh_dir: str | None = None,
        gravity: tuple[float, float, float] | list[float] | np.ndarray | float = (0.0, 0.0, -9.81),
        verbose: bool = False
    ):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

        self.urdf_path = urdf_path
        self.chain = chain
        self.dh = dh
        self.dt = dt
        self.mesh_dir = mesh_dir
        self.verbose = verbose

        # Process customizable gravity
        if np.isscalar(gravity):
            self.gravity = np.array([0.0, 0.0, -abs(float(gravity))], dtype=np.float64)
        else:
            self.gravity = np.asarray(gravity, dtype=np.float64).flatten()
            if len(self.gravity) != 3:
                raise ValueError(f"Gravity vector must have 3 components [gx, gy, gz], got {len(self.gravity)}")

        # Compile URDF to MJCF with site and actuators
        self.xml_string = self._compile_mjcf(urdf_path, chain, dh, mesh_dir=mesh_dir)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.model.opt.timestep = dt
        self.model.opt.gravity[:] = self.gravity
        self.model.dof_damping[:] = 0.0
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        if verbose:
            print(f"MuJoCo Twin initialized: nq={self.nq}, nv={self.nv}, nu={self.nu}, gravity={self.gravity.tolist()}")

    def set_gravity(self, gravity: tuple[float, float, float] | list[float] | np.ndarray | float) -> None:
        """Dynamically update physics engine gravity vector."""
        if np.isscalar(gravity):
            self.gravity = np.array([0.0, 0.0, -abs(float(gravity))], dtype=np.float64)
        else:
            self.gravity = np.asarray(gravity, dtype=np.float64).flatten()
            if len(self.gravity) != 3:
                raise ValueError(f"Gravity vector must have 3 components [gx, gy, gz], got {len(self.gravity)}")
        self.model.opt.gravity[:] = self.gravity

    def _compile_mjcf(
        self,
        urdf_path: str,
        chain: Chain | None,
        dh: DHTable | None,
        mesh_dir: str | None = None,
    ) -> str:
        """Parse URDF, convert to native MJCF with 3D meshes/procedural visuals, EE site, and actuators."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        mesh_dirs = []
        if mesh_dir and os.path.isdir(mesh_dir):
            mesh_dirs.append(os.path.abspath(mesh_dir))
        mesh_dirs.extend([
            urdf_dir,
            os.path.join(urdf_dir, "meshes"),
            os.path.join(urdf_dir, "..", "meshes"),
            os.path.join(urdf_dir, "..", "test_UR16e"),
            os.path.join(urdf_dir, "..", "STL_Files"),
            os.path.abspath("End-to-End MATLAB-CasADi Implementation/test_UR16e"),
            os.path.abspath("STL_Files"),
        ])
        mesh_base_dir = mesh_dirs[0]

        # Adapt mesh filenames to local binary STLs where available across arbitrary ROS packages
        for m in root.iter("mesh"):
            fn = m.attrib.get("filename", "")
            if not fn:
                continue

            # Strip generic package://<package_name>/ prefix
            rel_path = fn
            if fn.startswith("package://"):
                parts = fn.split("package://")[-1].split("/", 1)
                rel_path = parts[1] if len(parts) > 1 else parts[0]

            resolved = None
            for sdir in mesh_dirs:
                if not os.path.isdir(sdir):
                    continue
                trials = [
                    os.path.join(sdir, rel_path),
                    os.path.join(sdir, os.path.basename(rel_path)),
                    os.path.join(sdir, "visual", os.path.basename(rel_path)),
                    os.path.join(sdir, "collision", os.path.basename(rel_path)),
                ]
                for cand in trials:
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

        # Safeguard: ensure every link has non-zero inertial properties for MuJoCo
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

        s_urdf = ET.tostring(root, encoding="unicode")
        try:
            m_tmp = mujoco.MjModel.from_xml_string(s_urdf)
        except Exception:
            # Fall back to headless procedural visuals if meshes are missing on disk
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

            # Set contype="0" and conaffinity="0" on all geoms so contacts don't disturb unconstrained physics
            for g in mjcf_root.iter("geom"):
                g.attrib["contype"] = "0"
                g.attrib["conaffinity"] = "0"
                if "rgba" not in g.attrib and "material" not in g.attrib:
                    g.attrib["rgba"] = "0.72 0.75 0.80 1.0"

            # Ensure every robot body has at least one visual geom
            for b in mjcf_root.iter("body"):
                if len(b.findall("geom")) == 0:
                    ET.SubElement(
                        b,
                        "geom",
                        attrib={
                            "type": "capsule",
                            "size": "0.04 0.08",
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
                ET.SubElement(
                    wb,
                    "geom",
                    attrib={
                        "name": "ground_plane",
                        "type": "plane",
                        "size": "4 4 0.05",
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
                        "pos": "0 0 4.0",
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
                        "pos": "2.0 2.5 3.0",
                        "dir": "-0.6 -0.7 -0.5",
                        "diffuse": "0.45 0.45 0.50",
                    },
                )

            # Active joints to actuate
            active_joint_names: list[str] = []
            if chain is not None:
                active_joint_names = [j.name for j in chain.active_joints]
            else:
                for j in root.findall("joint"):
                    if j.attrib.get("type") in ("revolute", "continuous", "prismatic"):
                        active_joint_names.append(j.attrib["name"])

            # Inject EE site into last body if chain is provided
            if chain is not None and chain.active_links:
                tip_name = chain.active_links[-1]
                tip_body = None
                for b in mjcf_root.iter("body"):
                    if b.attrib.get("name") == tip_name:
                        tip_body = b
                        break

                if tip_body is not None:
                    site = ET.SubElement(tip_body, "site")
                    site.attrib["name"] = "ee_site"

                    # Compute relative transform from terminal link to DH EE frame
                    if dh is not None:
                        T_u0 = chain.T_urdf[-1]
                        T_dh0 = dh.T_dh[-1] @ chain.T_ee
                        T_target = np.linalg.inv(T_u0) @ T_dh0
                        pos = T_target[:3, 3]
                        quat = R_to_quat(T_target[:3, :3])
                    else:
                        pos = chain.T_ee[:3, 3]
                        quat = R_to_quat(chain.T_ee[:3, :3])

                    site.attrib["pos"] = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
                    site.attrib["quat"] = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"

            # Inject actuators
            act_container = mjcf_root.find("actuator")
            if act_container is None:
                act_container = ET.SubElement(mjcf_root, "actuator")

            for j_name in active_joint_names:
                motor = ET.SubElement(act_container, "motor")
                motor.attrib["name"] = f"motor_{j_name}"
                motor.attrib["joint"] = j_name
                motor.attrib["gear"] = "1"
                motor.attrib["ctrllimited"] = "false"

            return ET.tostring(mjcf_root, encoding="unicode")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute end-effector position and unit quaternion at joint configuration q."""
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if site_id >= 0:
            pos = self.data.site_xpos[site_id].copy()
            mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        else:
            body_id = self.model.nbody - 1
            pos = self.data.xpos[body_id].copy()
            mat = self.data.xmat[body_id].reshape(3, 3).copy()

        quat = R_to_quat(mat)
        return pos, quat

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute 6xN geometric Jacobian at joint configuration q."""
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if site_id >= 0:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
        else:
            body_id = self.model.nbody - 1
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)

        return np.vstack([jac_pos, jac_rot])

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute symmetric positive-definite mass matrix M(q) via MuJoCo's CRBA engine."""
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, self.data, M)
        return M

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Evaluate exact inverse dynamics tau = M(q)*a + C(q, v)*v + G(q) via MuJoCo RNEA."""
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)
        self.data.qvel[:] = np.asarray(v, dtype=np.float64)
        self.data.qacc[:] = np.asarray(a, dtype=np.float64)
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def set_link_inertias(
        self,
        masses: np.ndarray | list[float],
        coms: np.ndarray | list[list[float]] | None = None,
        inertias: np.ndarray | None = None
    ) -> None:
        """Programmatically override link masses, CoMs, and inertia tensors (e.g. for sequential isolation)."""
        masses = np.asarray(masses, dtype=np.float64)
        num_bodies = min(len(masses), self.model.nbody - 1)
        for i in range(num_bodies):
            body_id = i + 1
            self.model.body_mass[body_id] = float(masses[i])
            if coms is not None:
                self.model.body_ipos[body_id] = np.asarray(coms[i], dtype=np.float64)
            if inertias is not None:
                I_mat = inertias[i]
                if I_mat.shape == (3, 3):
                    eigvals = np.linalg.eigvalsh(I_mat)
                    self.model.body_inertia[body_id] = np.maximum(eigvals, 1e-9)
                else:
                    self.model.body_inertia[body_id] = np.maximum(I_mat, 1e-9)

    def simulate_trajectory(
        self,
        time: np.ndarray,
        q_traj: np.ndarray,
        dq_traj: np.ndarray,
        ddq_traj: np.ndarray,
        controller=None
    ) -> dict[str, np.ndarray]:
        """Simulate forward in time along a trajectory, logging state and actuation torques."""
        num_steps = len(time)
        q_sim = np.zeros_like(q_traj)
        dq_sim = np.zeros_like(dq_traj)
        tau_sim = np.zeros_like(q_traj)
        ee_pos = np.zeros((num_steps, 3), dtype=np.float64)
        ee_quat = np.zeros((num_steps, 4), dtype=np.float64)

        self.data.qpos[:] = q_traj[0, :]
        self.data.qvel[:] = dq_traj[0, :]
        mujoco.mj_forward(self.model, self.data)

        for k in range(num_steps):
            q_k = self.data.qpos.copy()
            dq_k = self.data.qvel.copy()

            q_sim[k, :] = q_k
            dq_sim[k, :] = dq_k

            pos, quat = self.forward_kinematics(q_k)
            ee_pos[k, :] = pos
            ee_quat[k, :] = quat

            if controller is not None:
                tau_cmd = controller.compute_control(
                    q=q_k,
                    dq=dq_k,
                    q_d=q_traj[k, :],
                    dq_d=dq_traj[k, :],
                    ddq_d=ddq_traj[k, :],
                    t=time[k]
                )
            else:
                tau_cmd = self.inverse_dynamics(q_k, dq_k, ddq_traj[k, :])

            tau_sim[k, :] = tau_cmd

            if self.model.nu > 0:
                self.data.ctrl[:] = tau_cmd
            mujoco.mj_step(self.model, self.data)

        return {
            "time": time,
            "q_sim": q_sim,
            "dq_sim": dq_sim,
            "tau_sim": tau_sim,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
        }
