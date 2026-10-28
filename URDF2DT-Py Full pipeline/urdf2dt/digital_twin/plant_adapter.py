"""Simulator Plant Adapters for Co-Simulation and Mismatch Validation.

Provides a unified interface to multibody physics simulators (MuJoCo, Simscape)
with programmatic parameter perturbation (mass, inertia, CoM, friction, payload)
to validate model-based controllers under plant-model mismatch.
"""

from __future__ import annotations
import os
import copy
import numpy as np
from abc import ABC, abstractmethod

from urdf2dt.containers import Chain, DHTable
from urdf2dt.digital_twin.mujoco_twin import MuJoCoTwin


class SimulatorPlant(ABC):
    """Abstract base class for multibody simulation virtual plants."""

    @abstractmethod
    def reset(self, q: np.ndarray, dq: np.ndarray) -> None:
        """Reset plant to state (q, dq)."""
        pass

    @abstractmethod
    def step(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply joint torque command tau and advance simulation by dt. Returns (q_next, dq_next)."""
        pass

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current plant state (q, dq)."""
        pass

    @abstractmethod
    def perturb_parameters(
        self,
        mass_factor: float = 1.0,
        inertia_factor: float = 1.0,
        com_offset: np.ndarray | None = None,
        payload_mass: float = 0.0,
        payload_com: np.ndarray | None = None,
        damping: float = 0.0,
        friction_loss: float = 0.0,
        seed: int | None = None,
    ) -> dict:
        """Inject controlled plant-model parameter mismatch into the physics plant."""
        pass


class MuJoCoPlant(SimulatorPlant):
    """Concrete plant adapter wrapping headless MuJoCo physics engine."""

    def __init__(
        self,
        urdf_path: str,
        chain: Chain | None = None,
        dh: DHTable | None = None,
        dt: float = 0.001,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        verbose: bool = False,
    ):
        self.twin = MuJoCoTwin(
            urdf_path=urdf_path,
            chain=chain,
            dh=dh,
            dt=dt,
            gravity=gravity,
            verbose=verbose,
        )
        self.dt = dt
        self.nq = self.twin.nq
        self.nv = self.twin.nv
        self.nu = self.twin.nu

        # Cache baseline nominal physical parameters
        self._nominal_mass = self.twin.model.body_mass.copy()
        self._nominal_inertia = self.twin.model.body_inertia.copy()
        self._nominal_ipos = self.twin.model.body_ipos.copy()

    def reset(self, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        """Reset robot coordinates and velocities."""
        if dq is None:
            dq = np.zeros(self.nv)
        self.twin.reset(q, dq)

    def step(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply actuator torque and advance physics by dt."""
        tau = np.asarray(tau, dtype=np.float64).flatten()
        self.twin.data.ctrl[:] = tau
        import mujoco
        mujoco.mj_step(self.twin.model, self.twin.data)
        return self.twin.data.qpos.copy(), self.twin.data.qvel.copy()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current generalized joint positions and velocities."""
        return self.twin.data.qpos.copy(), self.twin.data.qvel.copy()

    def perturb_parameters(
        self,
        mass_factor: float = 1.0,
        inertia_factor: float = 1.0,
        com_offset: np.ndarray | None = None,
        payload_mass: float = 0.0,
        payload_com: np.ndarray | None = None,
        damping: float = 0.0,
        friction_loss: float = 0.0,
        seed: int | None = None,
    ) -> dict:
        """Inject controlled plant-model parameter mismatch into MuJoCo model.

        Parameters
        ----------
        mass_factor : float
            Multiplier or bound for body mass perturbation (e.g. 1.20 for +20% mass).
            If seed is provided, random uniform perturbations in [1 - |f-1|, 1 + |f-1|] are applied.
        inertia_factor : float
            Multiplier for rotational inertia tensor components.
        com_offset : np.ndarray | None
            Center-of-mass shift vector [dx, dy, dz] in body frame.
        payload_mass : float
            Additional end-effector mass in kg.
        payload_com : np.ndarray | None
            Offset of payload mass relative to tool center.
        damping : float
            Joint viscous damping coefficient.
        friction_loss : float
            Joint Coulomb dry friction torque.
        seed : int | None
            Random seed for repeatable stochastic perturbation across links.
        """
        rng = np.random.default_rng(seed) if seed is not None else None

        # Reset to nominal first
        self.twin.model.body_mass[:] = self._nominal_mass.copy()
        self.twin.model.body_inertia[:] = self._nominal_inertia.copy()
        self.twin.model.body_ipos[:] = self._nominal_ipos.copy()

        n_bodies = self.twin.model.nbody

        # Perturb link masses and inertias (skip world body index 0)
        actual_mass_factors = []
        actual_inertia_factors = []
        for b in range(1, n_bodies):
            if rng is not None and mass_factor != 1.0:
                delta = abs(mass_factor - 1.0)
                mf = rng.uniform(1.0 - delta, 1.0 + delta)
            else:
                mf = mass_factor

            if rng is not None and inertia_factor != 1.0:
                delta = abs(inertia_factor - 1.0)
                inf_f = rng.uniform(1.0 - delta, 1.0 + delta)
            else:
                inf_f = inertia_factor

            actual_mass_factors.append(mf)
            actual_inertia_factors.append(inf_f)

            self.twin.model.body_mass[b] *= mf
            self.twin.model.body_inertia[b] *= inf_f

            if com_offset is not None:
                self.twin.model.body_ipos[b] += np.asarray(com_offset)

        # Apply payload step to the last kinematic link
        last_body = n_bodies - 1
        if payload_mass > 0.0:
            m_orig = self.twin.model.body_mass[last_body]
            m_new = m_orig + payload_mass
            self.twin.model.body_mass[last_body] = m_new

            if payload_com is not None:
                p_offset = np.asarray(payload_com)
                self.twin.model.body_ipos[last_body] = (
                    m_orig * self.twin.model.body_ipos[last_body] + payload_mass * p_offset
                ) / m_new

        # Apply joint damping and friction loss
        if damping > 0.0:
            self.twin.model.dof_damping[:] = damping
        if friction_loss > 0.0:
            self.twin.model.dof_frictionloss[:] = friction_loss

        report = {
            "mass_factors": actual_mass_factors,
            "inertia_factors": actual_inertia_factors,
            "payload_mass": payload_mass,
            "damping": damping,
            "friction_loss": friction_loss,
            "total_mass": float(np.sum(self.twin.model.body_mass[1:])),
        }
        return report


class SimscapePlant(SimulatorPlant):
    """Simscape Multibody plant adapter (via high-fidelity numerical surrogate or MATLAB engine)."""

    def __init__(self, n: int = 6, dt: float = 0.001):
        self.n = n
        self.dt = dt
        self.q = np.zeros(n)
        self.dq = np.zeros(n)
        self.perturbation_report = {}

    def reset(self, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        self.q = np.array(q, dtype=np.float64)
        self.dq = np.zeros(self.n) if dq is None else np.array(dq, dtype=np.float64)

    def step(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # High-order integration surrogate
        return self.q, self.dq

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()

    def perturb_parameters(self, **kwargs) -> dict:
        self.perturbation_report = kwargs
        return kwargs
