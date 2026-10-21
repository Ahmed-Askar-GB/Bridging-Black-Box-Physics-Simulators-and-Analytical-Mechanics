"""URDF2DT-Py: URDF to Analytical CasADi Model, Digital Twin & Control Framework."""

__version__ = "0.1.0"

from urdf2dt.containers import (
    Robot,
    Link,
    Joint,
    Chain,
    DHTable,
    Inertia,
    CasadiModel,
    StageResult,
    VerificationReport,
)
from urdf2dt.parsing import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
)
from urdf2dt.analytical import (
    fk_casadi,
    jacobian_casadi,
    dynamics_casadi,
    create_batch_simulator,
    casadi_compile,
)
from urdf2dt.digital_twin import MuJoCoTwin, PinocchioTwin, PINOCCHIO_AVAILABLE
from urdf2dt.control import ComputedTorqueController, CasadiNMPC
from urdf2dt.verification import (
    verify_invariants,
    verify_kinematics,
    verify_dynamics_sequential,
    print_verification_dashboard,
    export_verification_report,
)

__all__ = [
    "Robot",
    "Link",
    "Joint",
    "Chain",
    "DHTable",
    "Inertia",
    "CasadiModel",
    "StageResult",
    "VerificationReport",
    "urdf_parse",
    "urdf_build_chain",
    "urdf_extract_dh",
    "urdf_extract_inertia",
    "fk_casadi",
    "jacobian_casadi",
    "dynamics_casadi",
    "create_batch_simulator",
    "casadi_compile",
    "MuJoCoTwin",
    "PinocchioTwin",
    "PINOCCHIO_AVAILABLE",
    "ComputedTorqueController",
    "CasadiNMPC",
    "verify_invariants",
    "verify_kinematics",
    "verify_dynamics_sequential",
    "print_verification_dashboard",
    "export_verification_report",
]
