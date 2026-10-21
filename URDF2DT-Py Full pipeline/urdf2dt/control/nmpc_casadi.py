"""Nonlinear Model Predictive Control (NMPC) for trajectory tracking via CasADi Opti."""

from __future__ import annotations
import numpy as np
import casadi as cs
from urdf2dt.containers import CasadiModel


class CasadiNMPC:
    """Multiple-shooting Nonlinear MPC for robotic manipulators using CasADi Opti and symbolic forward dynamics."""

    def __init__(
        self,
        model: CasadiModel,
        N_horizon: int = 15,
        dt: float = 0.02,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
        tau_max: float = 150.0,
        Q_pos: float = 100.0,
        Q_vel: float = 10.0,
        R_tau: float = 0.01,
        joint_signs: np.ndarray | None = None,
        verbose: bool = False
    ):
        self.model = model
        self.n = model.n
        self.N = N_horizon
        self.dt = dt
        self.signs = np.ones(self.n) if joint_signs is None else np.asarray(joint_signs)
        self.verbose = verbose

        # Weights
        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.R_tau = R_tau

        # Constraints
        self.q_min = -2.0 * np.pi * np.ones(self.n) if q_min is None else np.asarray(q_min)
        self.q_max = 2.0 * np.pi * np.ones(self.n) if q_max is None else np.asarray(q_max)
        self.tau_max = tau_max

        self._build_opti_problem()

    def _build_opti_problem(self) -> None:
        """Construct the parametric CasADi Opti multiple-shooting NLP."""
        self.opti = cs.Opti()

        nx = 2 * self.n
        nu = self.n

        # Decision variables
        self.X = self.opti.variable(nx, self.N + 1)  # State trajectory: [q; v]
        self.U = self.opti.variable(nu, self.N)      # Control trajectory: tau

        # Parameters
        self.X0_param = self.opti.parameter(nx)
        self.X_ref_param = self.opti.parameter(nx, self.N + 1)

        # Objective function
        obj = 0
        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_ref_k = self.X_ref_param[:, k]

            e_q = x_k[:self.n] - x_ref_k[:self.n]
            e_v = x_k[self.n:] - x_ref_k[self.n:]

            obj += self.Q_pos * cs.dot(e_q, e_q)
            obj += self.Q_vel * cs.dot(e_v, e_v)
            obj += self.R_tau * cs.dot(u_k, u_k)

        # Terminal cost
        e_q_N = self.X[:self.n, self.N] - self.X_ref_param[:self.n, self.N]
        e_v_N = self.X[self.n:, self.N] - self.X_ref_param[self.n:, self.N]
        obj += (2.0 * self.Q_pos) * cs.dot(e_q_N, e_q_N)
        obj += (2.0 * self.Q_vel) * cs.dot(e_v_N, e_v_N)

        self.opti.minimize(obj)

        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.X0_param)

        # Multiple shooting dynamic constraints via RK4 integration
        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]

            # Forward dynamics vector field
            def f_dyn(x, u):
                qk = x[:self.n]
                vk = x[self.n:]
                # Forward dynamics in CasADi model
                ddqk = self.model.forward_dyn_fun(qk, vk, u)
                return cs.vertcat(vk, ddqk)

            k1 = f_dyn(x_k, u_k)
            k2 = f_dyn(x_k + 0.5 * self.dt * k1, u_k)
            k3 = f_dyn(x_k + 0.5 * self.dt * k2, u_k)
            k4 = f_dyn(x_k + self.dt * k3, u_k)
            x_next = x_k + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            self.opti.subject_to(self.X[:, k + 1] == x_next)

            # Box constraints
            self.opti.subject_to(self.opti.bounded(self.q_min, self.X[:self.n, k], self.q_max))
            self.opti.subject_to(self.opti.bounded(-self.tau_max, u_k, self.tau_max))

        self.opti.subject_to(self.opti.bounded(self.q_min, self.X[:self.n, self.N], self.q_max))

        # Solver options
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 50,
            "tol": 1e-4,
            "print_level": 0 if not self.verbose else 5,
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def solve(self, x0: np.ndarray, x_ref_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the NMPC problem given current state x0=[q; v] and reference horizon x_ref_traj."""
        self.opti.set_value(self.X0_param, np.asarray(x0, dtype=np.float64))
        self.opti.set_value(self.X_ref_param, np.asarray(x_ref_traj, dtype=np.float64))

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            # First control action
            tau_0 = u_opt[:, 0] if self.N > 1 else u_opt
            return np.asarray(tau_0, dtype=np.float64), np.asarray(x_opt, dtype=np.float64)
        except Exception:
            # Fallback to zero torque if solver fails
            u_fallback = np.asarray(self.opti.debug.value(self.U), dtype=np.float64)
            tau_0 = u_fallback[:, 0] if self.N > 1 else u_fallback
            return tau_0, np.zeros((2 * self.n, self.N + 1))
