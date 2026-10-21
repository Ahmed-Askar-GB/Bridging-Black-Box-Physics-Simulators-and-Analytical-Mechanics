"""Cross-validation between analytical CasADi kinematics/Jacobians and digital twin engines."""

from __future__ import annotations
import numpy as np
from urdf2dt.containers import CasadiModel, Chain
from urdf2dt.digital_twin import MuJoCoTwin
from urdf2dt.utils.math_utils import quat_error
from urdf2dt.utils.sign_detection import detect_joint_signs


def verify_kinematics(
    model: CasadiModel,
    twin: MuJoCoTwin,
    chain: Chain,
    num_samples: int = 20,
    pos_tol: float = 1e-4,
    quat_tol: float = 1e-3,
    jac_tol: float = 1e-3,
    verbose: bool = False
) -> dict[str, float]:
    """Cross-validate Forward Kinematics (position and orientation) and geometric Jacobians

    between the analytical DH CasADi model and the Digital Twin across random configurations.
    """
    n = chain.n
    joint_signs = detect_joint_signs(chain, model.fk_link_funs)

    max_pos_err = 0.0
    max_quat_err = 0.0
    max_jac_err = 0.0

    rng = np.random.default_rng(123)

    for idx in range(num_samples):
        # Generate random URDF joint configuration
        q_urdf = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
        # DH angles mapped via joint signs
        q_dh = joint_signs * q_urdf

        # Analytical evaluation
        p_ana, _, quat_ana = model.fk_fun(q_dh)
        p_ana = np.array(p_ana).flatten()
        quat_ana = np.array(quat_ana).flatten()
        J_ana = np.array(model.J_fun(q_dh))

        # Twin evaluation
        p_twin, quat_twin = twin.forward_kinematics(q_urdf)
        J_twin = twin.jacobian(q_urdf)

        pos_err = float(np.linalg.norm(p_ana - p_twin))
        q_err = float(quat_error(quat_ana, quat_twin))

        # Re-scale Jacobian columns by joint signs for directional alignment
        J_ana_aligned = J_ana * joint_signs[None, :]
        jac_err = float(np.linalg.norm(J_ana_aligned - J_twin))

        if pos_err > max_pos_err:
            max_pos_err = pos_err
        if q_err > max_quat_err:
            max_quat_err = q_err
        if jac_err > max_jac_err:
            max_jac_err = jac_err

    passed = (max_pos_err <= pos_tol) and (max_quat_err <= quat_tol) and (max_jac_err <= jac_tol)

    if verbose:
        print("\n--- Kinematic Verification Results ---")
        print(f"  Max Position Error   : {max_pos_err:.4e} m (tol: {pos_tol:.1e})")
        print(f"  Max Quaternion Error : {max_quat_err:.4e} rad (tol: {quat_tol:.1e})")
        print(f"  Max Jacobian Error   : {max_jac_err:.4e} (tol: {jac_tol:.1e})")
        print(f"  Overall Status       : {'PASSED' if passed else 'FAILED'}")

    return {
        "max_pos_err": max_pos_err,
        "max_quat_err": max_quat_err,
        "max_jac_err": max_jac_err,
        "passed": float(passed),
    }
