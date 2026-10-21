"""Computed Torque Controller (Feedback Linearization) for closed-loop trajectory tracking."""

from __future__ import annotations
import numpy as np
from urdf2dt.containers import CasadiModel


class ComputedTorqueController:
    """Computed Torque (Inverse Dynamics + PD Feedback Linearization) Controller.

    tau = M(q) * (ddq_d + Kd*(dq_d - dq) + Kp*(q_d - q)) + C(q, dq)*dq + G(q)
    """

    def __init__(
        self,
        model: CasadiModel,
        kp: float | np.ndarray = 100.0,
        kd: float | np.ndarray = 20.0,
        joint_signs: np.ndarray | None = None
    ):
        self.model = model
        self.n = model.n

        if np.isscalar(kp):
            self.Kp = np.eye(self.n) * float(kp)
        else:
            self.Kp = np.diag(np.asarray(kp, dtype=np.float64))

        if np.isscalar(kd):
            self.Kd = np.eye(self.n) * float(kd)
        else:
            self.Kd = np.diag(np.asarray(kd, dtype=np.float64))

        if joint_signs is not None:
            self.signs = np.asarray(joint_signs, dtype=np.float64)
        else:
            self.signs = np.ones(self.n, dtype=np.float64)

    def compute_control(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_d: np.ndarray,
        dq_d: np.ndarray,
        ddq_d: np.ndarray,
        t: float = 0.0
    ) -> np.ndarray:
        """Compute control torque command given measured state and reference trajectory."""
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        q_d = np.asarray(q_d, dtype=np.float64)
        dq_d = np.asarray(dq_d, dtype=np.float64)
        ddq_d = np.asarray(ddq_d, dtype=np.float64)

        # Tracking errors in URDF coordinates
        e = q_d - q
        de = dq_d - dq

        # Linearized feedback acceleration command
        v_acc = ddq_d + self.Kp @ e + self.Kd @ de

        # Convert to DH frame coordinates
        q_dh = self.signs * q
        dq_dh = self.signs * dq
        v_acc_dh = self.signs * v_acc

        # Evaluate analytical dynamics in DH frame
        tau_dh = np.array(self.model.tau_fun(q_dh, dq_dh, v_acc_dh)).flatten()

        # Convert torque back to URDF joint coordinates
        tau_cmd = self.signs * tau_dh
        return tau_cmd
