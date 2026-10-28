"""Unified Trajectory Optimization and Optimal Control Problem (OCP) Builder.

Constructs Multiple-Shooting RK4 formulations for both:
  1. Offline Trajectory Optimization (boundary value problems with exact Hessians)
  2. Receding-Horizon NMPC (fast parametric solves with warm-starting)

Features:
  - Exact CasADi symbolic Hessian vs Quasi-Newton (L-BFGS)
  - Joint position, velocity, and torque limit constraints
  - Workspace spherical obstacle avoidance constraints
  - Export to compiled C code for millisecond-level execution
"""

from __future__ import annotations
import os
import time
import numpy as np
import casadi as cs

from urdf2dt.containers import CasadiModel


class TrajectoryOptimizationOCP:
    """Unified Multiple-Shooting RK4 OCP builder."""

    def __init__(
        self,
        model: CasadiModel,
        N: int = 20,
        dt: float = 0.02,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
        dq_max: np.ndarray | None = None,
        tau_max: np.ndarray | float = 150.0,
        obstacle_center: np.ndarray | None = None,
        obstacle_radius: float = 0.12,
        safety_margin: float = 0.04,
        cost_mode: str = "effort",  # "effort", "tracking", "terminal"
        hessian_mode: str = "exact",  # "exact" or "limited-memory"
        integrator: str = "euler",  # "euler" (symplectic) or "rk4"
        ipopt_tol: float = 1e-4,
        max_iter: int = 100,
        verbose: bool = False,
    ):
        self.model = model
        self.n = model.n
        self.nx = 2 * self.n
        self.nu = self.n
        self.N = N
        self.dt = dt
        self.integrator = integrator

        # Limits
        self.q_min = np.asarray(q_min) if q_min is not None else -np.pi * np.ones(self.n)
        self.q_max = np.asarray(q_max) if q_max is not None else np.pi * np.ones(self.n)
        self.dq_max = np.asarray(dq_max) if dq_max is not None else 5.0 * np.ones(self.n)

        if np.isscalar(tau_max):
            self.tau_max = float(tau_max) * np.ones(self.n)
        else:
            self.tau_max = np.asarray(tau_max, dtype=np.float64)

        # Obstacle
        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius
        self.safety_margin = safety_margin
        self.r_safe = obstacle_radius + safety_margin

        self.cost_mode = cost_mode
        self.hessian_mode = hessian_mode
        self.ipopt_tol = ipopt_tol
        self.max_iter = max_iter
        self.verbose = verbose

        self._build_nlp()

    def _build_nlp(self) -> None:
        """Construct raw CasADi NLP symbolic representation."""
        n = self.n
        nx = self.nx
        nu = self.nu
        N = self.N
        dt = self.dt

        # Continuous forward dynamics function
        q_sym = self.model.q
        dq_sym = self.model.dq
        tau_sym = self.model.tau
        fwd_acc = self.model.forward_dyn_fun(q_sym, dq_sym, tau_sym)
        f_dyn = cs.Function("f_dyn", [q_sym, dq_sym, tau_sym], [dq_sym, fwd_acc])

        # Single-step integration function: (x_k, u_k) -> x_{k+1}
        x_in = cs.SX.sym("x_in", nx)
        u_in = cs.SX.sym("u_in", nu)
        qk = x_in[:n]
        vk = x_in[n:]

        if self.integrator == "euler":
            # Symplectic / Semi-implicit Euler: energy-conserving single stage
            v_dot, a_dot = f_dyn(qk, vk, u_in)
            v_next = vk + dt * a_dot
            q_next = qk + dt * v_next
            x_next = cs.vertcat(q_next, v_next)
        else:
            # RK4 Integration
            k1_v, k1_a = f_dyn(qk, vk, u_in)
            k2_v, k2_a = f_dyn(qk + 0.5 * dt * k1_v, vk + 0.5 * dt * k1_a, u_in)
            k3_v, k3_a = f_dyn(qk + 0.5 * dt * k2_v, vk + 0.5 * dt * k2_a, u_in)
            k4_v, k4_a = f_dyn(qk + dt * k3_v, vk + dt * k3_a, u_in)

            q_next = qk + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            v_next = vk + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
            x_next = cs.vertcat(q_next, v_next)

        self.f_step = cs.Function("f_step", [x_in, u_in], [x_next])
        self.f_rk4 = self.f_step  # Backward compatibility alias

        # Decision variables: w = [x_0; u_0; x_1; u_1; ...; x_N]
        # Parameters: p = [x_0_param; x_goal_param; Q_pos; Q_vel; R_tau]
        w = []
        w0 = []
        lbw = []
        ubw = []

        g = []
        lbg = []
        ubg = []

        # Parameters: p = [x_init (nx); x_target (nx)]
        p_init = cs.SX.sym("p_init", nx)
        p_target = cs.SX.sym("p_target", nx)
        p = cs.vertcat(p_init, p_target)

        # State and control symbols along horizon
        X = []
        U = []

        # x_0
        X0 = cs.SX.sym("X_0", nx)
        w.append(X0)
        lbw.extend(list(np.concatenate([self.q_min, -self.dq_max])))
        ubw.extend(list(np.concatenate([self.q_max, self.dq_max])))
        w0.extend([0.0] * nx)
        X.append(X0)

        # Initial state constraint: x_0 - p_init == 0
        g.append(X0 - p_init)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)

        J_cost = 0.0

        for k in range(N):
            # Control u_k
            Uk = cs.SX.sym(f"U_{k}", nu)
            w.append(Uk)
            lbw.extend(list(-self.tau_max))
            ubw.extend(list(self.tau_max))
            w0.extend([0.0] * nu)
            U.append(Uk)

            # State x_{k+1}
            Xk_next = cs.SX.sym(f"X_{k+1}", nx)
            w.append(Xk_next)
            lbw.extend(list(np.concatenate([self.q_min, -self.dq_max])))
            ubw.extend(list(np.concatenate([self.q_max, self.dq_max])))
            w0.extend([0.0] * nx)
            X.append(Xk_next)

            # Multiple shooting continuity: x_{k+1} - f_rk4(x_k, u_k) == 0
            x_step = self.f_rk4(X[k], Uk)
            g.append(Xk_next - x_step)
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)

            # Stage Cost
            if self.cost_mode == "effort":
                J_cost += 0.01 * cs.dot(Uk, Uk)
                if k > 0:
                    J_cost += 0.005 * cs.dot(Uk - U[k - 1], Uk - U[k - 1])
                # Small state regularization
                J_cost += 0.1 * cs.dot(Xk_next[:n] - p_target[:n], Xk_next[:n] - p_target[:n])
            elif self.cost_mode == "tracking":
                e_q = X[k][:n] - p_target[:n]
                e_v = X[k][n:] - p_target[n:]
                J_cost += 10.0 * cs.dot(e_q, e_q) + 1.0 * cs.dot(e_v, e_v) + 0.01 * cs.dot(Uk, Uk)

            # Workspace obstacle avoidance: ||p_ee(q_k) - p_obs||^2 >= r_safe^2
            if self.obstacle_center is not None:
                p_obs = cs.SX(np.asarray(self.obstacle_center).reshape(3, 1))
                T0e = self.model.fk_ee_fun(Xk_next[:n])
                p_ee = T0e[:3, 3]
                dist_sq = cs.dot(p_ee - p_obs, p_ee - p_obs)
                g.append(dist_sq)
                lbg.append(self.r_safe ** 2)
                ubg.append(100.0)  # Large upper bound

        # Terminal cost
        e_q_term = X[-1][:n] - p_target[:n]
        e_v_term = X[-1][n:] - p_target[n:]
        J_cost += 100.0 * cs.dot(e_q_term, e_q_term) + 10.0 * cs.dot(e_v_term, e_v_term)

        # Assemble NLP structures
        self.w_sym = cs.vertcat(*w)
        self.g_sym = cs.vertcat(*g)
        self.p_sym = p
        self.J_sym = J_cost

        self.lbw = np.array(lbw, dtype=np.float64)
        self.ubw = np.array(ubw, dtype=np.float64)
        self.lbg = np.array(lbg, dtype=np.float64)
        self.ubg = np.array(ubg, dtype=np.float64)
        self.w0_default = np.array(w0, dtype=np.float64)

        nlp = {"x": self.w_sym, "f": self.J_sym, "g": self.g_sym, "p": self.p_sym}

        ipopt_opts = {
            "print_level": 5 if self.verbose else 0,
            "max_iter": self.max_iter,
            "tol": self.ipopt_tol,
            "hessian_approximation": self.hessian_mode,
            "linear_solver": "mumps",
        }
        solver_opts = {
            "ipopt": ipopt_opts,
            "print_time": 1 if self.verbose else 0,
        }

        self.solver = cs.nlpsol("solver", "ipopt", nlp, solver_opts)
        self.nlp_dict = nlp

    def solve(
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        w0_guess: np.ndarray | None = None,
    ) -> dict:
        """Execute trajectory optimization or NMPC solve."""
        x_init = np.asarray(x_init, dtype=np.float64).flatten()
        x_target = np.asarray(x_target, dtype=np.float64).flatten()
        p_val = np.concatenate([x_init, x_target])

        # Interpolated guess if none provided
        if w0_guess is None:
            w0 = self._interpolate_guess(x_init, x_target)
        else:
            w0 = np.asarray(w0_guess, dtype=np.float64).flatten()

        t0 = time.perf_counter()
        sol = self.solver(
            x0=w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p_val,
        )
        solve_time = time.perf_counter() - t0

        stats = self.solver.stats()
        ret_status = stats.get("return_status", "")
        success = bool(stats.get("success", False) or ret_status in ["Solve_Succeeded", "Solved_To_Acceptable_Level"])
        iter_count = stats.get("iter_count", -1)
        w_opt = np.array(sol["x"]).flatten()
        cost_val = float(sol["f"])

        # Unpack trajectories
        q_traj, dq_traj, u_traj = self._unpack_solution(w_opt)

        return {
            "success": success,
            "solve_time": solve_time,
            "iterations": iter_count,
            "cost": cost_val,
            "q_traj": q_traj,
            "dq_traj": dq_traj,
            "u_traj": u_traj,
            "w_opt": w_opt,
            "stats": stats,
        }

    def _interpolate_guess(self, x_init: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """Linear interpolation state guess with zero control."""
        w_guess = []
        for k in range(self.N + 1):
            alpha = k / float(self.N)
            xk = (1.0 - alpha) * x_init + alpha * x_target
            w_guess.extend(list(xk))
            if k < self.N:
                w_guess.extend([0.0] * self.nu)
        return np.array(w_guess, dtype=np.float64)

    def _unpack_solution(self, w_opt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract state and control matrices from serialized solution vector."""
        n = self.n
        nx = self.nx
        nu = self.nu
        N = self.N

        q_list = []
        dq_list = []
        u_list = []

        idx = 0
        # x_0
        q_list.append(w_opt[idx: idx + n])
        dq_list.append(w_opt[idx + n: idx + nx])
        idx += nx

        for k in range(N):
            # u_k
            u_list.append(w_opt[idx: idx + nu])
            idx += nu
            # x_{k+1}
            q_list.append(w_opt[idx: idx + n])
            dq_list.append(w_opt[idx + n: idx + nx])
            idx += nx

        return np.array(q_list), np.array(dq_list), np.array(u_list)
