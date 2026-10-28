"""Unified DynamicsModel artifact export API.

Exports CasADi functions, exact higher-order analytical derivatives,
serialized binary artifacts, generated ANSI C code, and JSON metadata.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import time
import subprocess
import numpy as np
import casadi as cs

from urdf2dt.containers import DHTable, Inertia, Chain, CasadiModel
from urdf2dt.analytical.dynamics_casadi import dynamics_casadi


class DynamicsModel:
    """Unified container and export interface for robot kinematics and dynamics."""

    def __init__(
        self,
        chain: Chain,
        dh: DHTable,
        inertia: Inertia,
        model: CasadiModel,
        robot_name: str = "robot",
        signs: np.ndarray | None = None,
        dt_default: float = 0.001,
        sanitization_report: dict | None = None,
    ):
        self.chain = chain
        self.dh = dh
        self.inertia = inertia
        self.model = model
        self.robot_name = robot_name
        self.n = chain.n
        self.signs = signs if signs is not None else np.ones(self.n)
        self.dt_default = dt_default
        self.sanitization_report = sanitization_report or {}

        # Primary symbolic variables
        self.q = model.q
        self.dq = model.dq
        self.ddq = model.ddq
        self.tau = model.tau
        self.x = cs.vertcat(self.q, self.dq)

        # Build exact derivative functions
        self._build_derivative_functions()

    def _build_derivative_functions(self):
        """Construct exact first- and second-order derivatives via Algorithmic Differentiation."""
        n = self.n
        q = self.q
        dq = self.dq
        tau = self.tau
        x = self.x

        # 1. Continuous-time forward dynamics state derivative:
        # dx/dt = f_cont(x, u) = [dq; ddq(q, dq, tau)]
        ddq_expr = self.model.forward_dyn_fun(q, dq, tau)
        f_cont = cs.vertcat(dq, ddq_expr)
        self.f_cont_fun = cs.Function("f_cont", [x, tau], [f_cont], ["x", "u"], ["x_dot"])

        # Continuous Jacobians: df/dx (2n x 2n), df/du (2n x n)
        df_dx = cs.jacobian(f_cont, x)
        df_du = cs.jacobian(f_cont, tau)
        self.df_dx_fun = cs.Function("df_dx", [x, tau], [df_dx], ["x", "u"], ["df_dx"])
        self.df_du_fun = cs.Function("df_du", [x, tau], [df_du], ["x", "u"], ["df_du"])

        # 2. Discrete-time Semi-Implicit Euler integrator & exact discrete Jacobians
        dt_sym = cs.SX.sym("dt")
        v_next_euler = dq + dt_sym * ddq_expr
        q_next_euler = q + dt_sym * v_next_euler
        x_next_euler = cs.vertcat(q_next_euler, v_next_euler)
        self.euler_step_fun = cs.Function("euler_step", [x, tau, dt_sym], [x_next_euler], ["x", "u", "dt"], ["x_next"])
        self.dEuler_dx_fun = cs.Function("dEuler_dx", [x, tau, dt_sym], [cs.jacobian(x_next_euler, x)], ["x", "u", "dt"], ["A"])
        self.dEuler_du_fun = cs.Function("dEuler_du", [x, tau, dt_sym], [cs.jacobian(x_next_euler, tau)], ["x", "u", "dt"], ["B"])

        # 3. Discrete-time RK4 integrator & exact discrete Jacobians
        k1_v = dq
        k1_a = ddq_expr

        q_m1 = q + 0.5 * dt_sym * k1_v
        v_m1 = dq + 0.5 * dt_sym * k1_a
        k2_v = v_m1
        k2_a = self.model.forward_dyn_fun(q_m1, v_m1, tau)

        q_m2 = q + 0.5 * dt_sym * k2_v
        v_m2 = dq + 0.5 * dt_sym * k2_a
        k3_v = v_m2
        k3_a = self.model.forward_dyn_fun(q_m2, v_m2, tau)

        q_end = q + dt_sym * k3_v
        v_end = dq + dt_sym * k3_a
        k4_v = v_end
        k4_a = self.model.forward_dyn_fun(q_end, v_end, tau)

        q_next = q + (dt_sym / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        v_next = dq + (dt_sym / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
        x_next_rk4 = cs.vertcat(q_next, v_next)

        self.rk4_step_fun = cs.Function("rk4_step", [x, tau, dt_sym], [x_next_rk4], ["x", "u", "dt"], ["x_next"])

        # RK4 State transition Jacobian A = dx_next/dx, Control Jacobian B = dx_next/du
        A_rk4 = cs.jacobian(x_next_rk4, x)
        B_rk4 = cs.jacobian(x_next_rk4, tau)
        self.dRK4_dx_fun = cs.Function("dRK4_dx", [x, tau, dt_sym], [A_rk4], ["x", "u", "dt"], ["A"])
        self.dRK4_du_fun = cs.Function("dRK4_du", [x, tau, dt_sym], [B_rk4], ["x", "u", "dt"], ["B"])

        # 3. Exact Lagrangian Objective Hessian:
        # L(x, u) = 0.5 * x^T Q x + 0.5 * u^T R u
        Q_sym = cs.SX.sym("Q", 2 * n, 2 * n)
        R_sym = cs.SX.sym("R", n, n)
        loss = 0.5 * cs.mtimes([x.T, Q_sym, x]) + 0.5 * cs.mtimes([tau.T, R_sym, tau])
        xu = cs.vertcat(x, tau)
        hess_loss = cs.hessian(loss, xu)[0]
        self.lagrangian_hessian_fun = cs.Function(
            "lagrangian_hessian",
            [x, tau, Q_sym, R_sym],
            [hess_loss],
            ["x", "u", "Q", "R"],
            ["H"]
        )

    # -------------------------------------------------------------------------
    # Serialization & Export
    # -------------------------------------------------------------------------
    def export_all(self, output_dir: str = "exported_model", compile_c: bool = False) -> dict:
        """Export full model bundle: .casadi files, ANSI C code, and JSON metadata."""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 1. Save metadata JSON
        meta_path = os.path.join(output_dir, f"{self.robot_name}_metadata.json")
        meta = {
            "robot_name": self.robot_name,
            "dof": self.n,
            "joint_names": [j.name for j in self.chain.active_joints],
            "link_names": self.chain.active_links,
            "joint_signs": self.signs.tolist(),
            "dh_parameters": {
                "a": self.dh.DHpram[:, 0].tolist(),
                "alpha": self.dh.DHpram[:, 1].tolist(),
                "theta_offset": self.dh.DHpram[:, 2].tolist(),
                "d": self.dh.DHpram[:, 3].tolist(),
                "sigma": self.dh.DHpram[:, 4].tolist(),
            },
            "inertia_parameters": {
                "masses": self.inertia.m.tolist(),
                "com_offsets": self.inertia.P_Ci.tolist(),
                "inertia_tensors": self.inertia.I.tolist(),
            },
            "gravity": self.model.g_vec.tolist(),
            "sanitization_report": self.sanitization_report,
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        results["metadata_path"] = meta_path

        # 2. Generate ANSI C code
        c_filename = f"{self.robot_name}_dynamics.c"
        c_filepath = os.path.join(output_dir, c_filename)
        h_filepath = os.path.join(output_dir, f"{self.robot_name}_dynamics.h")

        cg = cs.CodeGenerator(c_filename)
        cg.add(self.model.fk_fun)
        cg.add(self.model.fk_ee_fun)
        cg.add(self.model.J_fun)
        cg.add(self.model.M_fun)
        cg.add(self.model.C_fun)
        cg.add(self.model.G_fun)
        cg.add(self.model.tau_fun)
        cg.add(self.model.forward_dyn_fun)
        cg.add(self.f_cont_fun)
        cg.add(self.df_dx_fun)
        cg.add(self.df_du_fun)
        cg.add(self.rk4_step_fun)
        cg.add(self.dRK4_dx_fun)
        cg.add(self.dRK4_du_fun)

        curr_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            cg.generate()
        finally:
            os.chdir(curr_dir)

        results["c_filepath"] = c_filepath
        results["h_filepath"] = h_filepath

        # 3. Optional compilation to shared object (.dll / .so)
        if compile_c:
            dll_path = self.compile_shared_library(output_dir, c_filename)
            results["dll_filepath"] = dll_path

        return results

    def compile_shared_library(self, output_dir: str, c_filename: str) -> str | None:
        """Compile generated C file into native dynamic library using GCC if available."""
        c_path = os.path.join(output_dir, c_filename)
        stem = os.path.splitext(c_filename)[0]
        ext = ".dll" if sys.platform == "win32" else ".so"
        out_dll = os.path.join(output_dir, f"{stem}{ext}")

        # Search for gcc in PATH or local tools/w64devkit
        tools_gcc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "w64devkit", "bin", "gcc.exe"))
        gcc_cmd = tools_gcc if os.path.exists(tools_gcc) else "gcc"

        cmd = [
            gcc_cmd,
            "-O3",
            "-fPIC",
            "-shared",
            "-march=native",
            c_path,
            "-o",
            out_dll,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and os.path.exists(out_dll):
                return out_dll
            else:
                print(f"[Warning] C compilation failed with returncode {res.returncode}: {res.stderr}")
                return None
        except Exception as e:
            print(f"[Warning] C compiler invocation failed: {e}")
            return None


def export_dynamics_model(
    urdf_path: str,
    output_dir: str = "exported_model",
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    compile_c: bool = False,
) -> DynamicsModel:
    """End-to-end convenience loader: URDF -> Sanitization -> DynamicsModel -> Artifact Export."""
    from urdf2dt.parsing import urdf_parse, urdf_build_chain, urdf_extract_dh, urdf_extract_inertia
    from urdf2dt.utils import detect_joint_signs

    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot, verbose=False)
    dh = urdf_extract_dh(chain, verbose=False)
    inertia = urdf_extract_inertia(robot, chain, dh, verbose=False)
    model = dynamics_casadi(dh, inertia, chain, g_vec=gravity, verbose=False)
    signs = detect_joint_signs(chain, model.fk_link_funs)

    dyn_model = DynamicsModel(
        chain=chain,
        dh=dh,
        inertia=inertia,
        model=model,
        robot_name=robot.name,
        signs=signs,
    )
    dyn_model.export_all(output_dir=output_dir, compile_c=compile_c)
    return dyn_model
