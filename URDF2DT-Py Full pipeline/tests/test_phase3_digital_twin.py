"""Tests for Phase 3: Digital Twin Engines (MuJoCo and Pinocchio)."""

import os
import numpy as np
import pytest
from urdf2dt.parsing import urdf_parse, urdf_build_chain
from urdf2dt.digital_twin import MuJoCoTwin, PinocchioTwin, PINOCCHIO_AVAILABLE

FIXTURE_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur5.urdf")


@pytest.fixture(scope="module")
def chain():
    robot = urdf_parse(FIXTURE_URDF)
    return urdf_build_chain(robot)


def test_mujoco_twin_initialization(chain):
    twin = MuJoCoTwin(FIXTURE_URDF, chain=chain, dt=0.001)
    assert twin.nq == 6
    assert twin.nv == 6
    assert twin.nu == 6


def test_mujoco_twin_kinematics_and_jacobian(chain):
    twin = MuJoCoTwin(FIXTURE_URDF, chain=chain, dt=0.001)
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    pos, quat = twin.forward_kinematics(q)
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-6

    J = twin.jacobian(q)
    assert J.shape == (6, 6)
    assert np.all(np.isfinite(J))


def test_mujoco_twin_dynamics(chain):
    twin = MuJoCoTwin(FIXTURE_URDF, chain=chain, dt=0.001)
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    v = np.array([0.05, -0.1, 0.15, -0.2, 0.25, -0.3])
    a = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])

    # Mass matrix SPD
    M = twin.mass_matrix(q)
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(M) > 0.0)

    # Inverse dynamics
    tau = twin.inverse_dynamics(q, v, a)
    assert tau.shape == (6,)
    assert np.all(np.isfinite(tau))


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not available in environment")
def test_pinocchio_twin():
    p_twin = PinocchioTwin(FIXTURE_URDF)
    q = np.zeros(6)
    pos, rot = p_twin.forward_kinematics(q)
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)

    M = p_twin.mass_matrix(q)
    assert M.shape == (6, 6)
