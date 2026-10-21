"""Model-based control design package."""

from urdf2dt.control.computed_torque import ComputedTorqueController
from urdf2dt.control.nmpc_casadi import CasadiNMPC

__all__ = [
    "ComputedTorqueController",
    "CasadiNMPC",
]
