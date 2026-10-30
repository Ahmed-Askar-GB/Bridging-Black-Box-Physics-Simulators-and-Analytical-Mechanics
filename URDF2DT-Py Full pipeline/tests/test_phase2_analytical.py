"""Tests for Phase 2: Analytical CasADi Engine."""

import os
import numpy as np
import pytest
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
    casadi_compile,
)

FIXTURE_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur5.urdf")


@pytest.fixture(scope="module")
def parsed_models():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)
    model = dynamics_casadi(dh, inertia, chain)
    return robot, chain, dh, inertia, model


def test_fk_casadi(parsed_models):
    _, chain, dh, _, model = parsed_models
    q_test = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    p_ee, R_ee, quat_ee = model.fk_fun(q_test)
    p_ee = np.array(p_ee).flatten()
    R_ee = np.array(R_ee)
    quat_ee = np.array(quat_ee).flatten()

    # Position must have finite norms
    assert np.all(np.isfinite(p_ee))

    # Rotation matrix SO(3) checks
    assert np.allclose(R_ee @ R_ee.T, np.eye(3), atol=1e-8)
    assert abs(np.linalg.det(R_ee) - 1.0) < 1e-8

    # Quaternion unit norm
    assert abs(np.linalg.norm(quat_ee) - 1.0) < 1e-8


def test_jacobian_and_jdot(parsed_models):
    _, chain, dh, _, model = parsed_models
    q = np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.7])
    dq = np.array([0.05, -0.1, 0.15, -0.2, 0.25, -0.3])

    J = np.array(model.J_fun(q))
    Jdot = np.array(model.Jdot_fun(q, dq))

    assert J.shape == (6, 6)
    assert Jdot.shape == (6, 6)

    # Finite difference check for Jdot: Jdot(q, dq) ≈ (J(q + eps*dq) - J(q - eps*dq)) / (2*eps)
    eps = 1e-7
    J_plus = np.array(model.J_fun(q + eps * dq))
    J_minus = np.array(model.J_fun(q - eps * dq))
    Jdot_fd = (J_plus - J_minus) / (2.0 * eps)

    assert np.allclose(Jdot, Jdot_fd, atol=1e-5)


def test_dynamics_properties(parsed_models):
    _, chain, dh, inertia, model = parsed_models
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    dq = np.array([0.05, -0.1, 0.15, -0.2, 0.25, -0.3])
    ddq = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])

    M = np.array(model.M_fun(q))
    C = np.array(model.C_fun(q, dq))
    G = np.array(model.G_fun(q)).flatten()
    tau = np.array(model.tau_fun(q, dq, ddq)).flatten()

    # 1. Mass matrix symmetry & SPD
    assert np.allclose(M, M.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)

    # 2. Static gravity check: at dq=0, ddq=0 -> tau == G(q)
    tau_static = np.array(model.tau_fun(q, np.zeros(6), np.zeros(6))).flatten()
    assert np.allclose(tau_static, G, atol=1e-8)

    # 3. Forward dynamics round-trip: ddq_sol = M^-1 * (tau - C*dq - G) == ddq
    ddq_calc = np.array(model.forward_dyn_fun(q, dq, tau)).flatten()
    assert np.allclose(ddq_calc, ddq, atol=1e-8)

    # 4. Skew-symmetry of M_dot - 2*C:
    # dM/dt along (q, dq) via central differences
    eps = 1e-7
    M_plus = np.array(model.M_fun(q + eps * dq))
    M_minus = np.array(model.M_fun(q - eps * dq))
    M_dot = (M_plus - M_minus) / (2.0 * eps)

    N_mat = M_dot - 2.0 * C
    # Skew-symmetric: N + N.T ≈ 0
    assert np.allclose(N_mat + N_mat.T, np.zeros((6, 6)), atol=1e-4)


def test_casadi_compile(parsed_models, tmp_path):
    robot, _, _, _, model = parsed_models
    out_dir = str(tmp_path / "codegen")
    res = casadi_compile(model, robot_name=robot.name, output_dir=out_dir, gen_c=True)

    assert os.path.exists(res["c_filepath"])
    assert os.path.exists(res["h_filepath"])
    assert os.path.getsize(res["c_filepath"]) > 1000
    assert os.path.getsize(res["h_filepath"]) > 100


def test_matrix_free_hvp(parsed_models):
    """Test matrix-free Hessian-vector products based on Pearlmutter (1994) and Margossian (2019)."""
    _, _, _, _, model = parsed_models
    assert model.hvp_G_fun is not None

    q = np.array([0.15, -0.25, 0.35, -0.45, 0.55, -0.65])
    v = np.array([0.02, -0.04, 0.06, -0.08, 0.10, -0.12])

    # 1. Evaluate HVP via forward-over-reverse AD
    hvp_val = np.array(model.hvp_G_fun(q, v)).flatten()

    # 2. Compare against central difference finite differences of G(q)
    # Theoretical noise floor is O(sqrt(eps_mach)) ~ 1.49e-8
    eps = 1e-7
    G_plus = np.array(model.G_fun(q + eps * v)).flatten()
    G_minus = np.array(model.G_fun(q - eps * v)).flatten()
    hvp_fd = (G_plus - G_minus) / (2.0 * eps)

    assert np.allclose(hvp_val, hvp_fd, atol=1e-5)
    assert np.linalg.norm(hvp_val - hvp_fd) < 1e-5
