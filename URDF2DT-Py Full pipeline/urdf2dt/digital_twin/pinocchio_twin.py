"""Pinocchio ground-truth digital twin wrapper for multibody spatial algebra."""

from __future__ import annotations
import os
import numpy as np

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    try:
        import pin
        PINOCCHIO_AVAILABLE = True
    except ImportError:
        PINOCCHIO_AVAILABLE = False


class PinocchioTwin:
    """Wrapper around Pinocchio rigid body dynamics library (RNEA, CRBA, ABA, Jacobians)."""

    def __init__(self, urdf_path: str):
        if not PINOCCHIO_AVAILABLE:
            raise ImportError(
                "Pinocchio is not installed in the current Python environment. "
                "On Windows, Pinocchio is typically installed via conda-forge: 'conda install -c conda-forge pinocchio'. "
                "Alternatively, use MuJoCoTwin for native headless physics and ground-truth validation."
            )

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute end-effector position and rotation matrix at joint angles q."""
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # Use last active operational frame
        frame_id = self.model.nframes - 1
        placement = self.data.oMf[frame_id]
        return placement.translation.copy(), placement.rotation.copy()

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute generalized inertia matrix M(q) via CRBA (Composite Rigid Body Algorithm)."""
        q = np.asarray(q, dtype=np.float64)
        return pin.crba(self.model, self.data, q).copy()

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Compute generalized joint forces/torques via RNEA (Recursive Newton-Euler Algorithm)."""
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        return pin.rnea(self.model, self.data, q, v, a).copy()

    def forward_dynamics(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Compute joint accelerations ddq via ABA (Articulated Body Algorithm)."""
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        return pin.aba(self.model, self.data, q, v, tau).copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute 6xN geometric Jacobian in world-aligned frame."""
        q = np.asarray(q, dtype=np.float64)
        pin.computeJointJacobians(self.model, self.data, q)
        frame_id = self.model.nframes - 1
        return pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
