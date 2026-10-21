"""Tests for Phase 4: Verification Suite (Invariants and Kinematics Cross-Validation)."""

import os
import numpy as np
import pytest
from urdf2dt.parsing import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
)
from urdf2dt.analytical import dynamics_casadi
from urdf2dt.digital_twin import MuJoCoTwin
from urdf2dt.verification import verify_invariants, verify_kinematics

FIXTURE_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur5.urdf")


@pytest.fixture(scope="module")
def setup_models():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)
    model = dynamics_casadi(dh, inertia, chain)
    twin = MuJoCoTwin(FIXTURE_URDF, chain=chain, dh=dh, dt=0.001)
    return robot, chain, dh, inertia, model, twin


def test_verify_invariants(setup_models):
    robot, _, dh, inertia, model, _ = setup_models
    res = verify_invariants(robot, dh, inertia, model, num_samples=10)
    assert res["mass_positivity"] is True
    assert res["triangle_inequality"] is True
    assert res["fk_so3_orthogonality"] is True
    assert res["mass_matrix_spd"] is True


def test_verify_kinematics_cross_validation(setup_models):
    _, chain, _, _, model, twin = setup_models
    res = verify_kinematics(
        model=model,
        twin=twin,
        chain=chain,
        num_samples=15,
        pos_tol=1e-3,
        quat_tol=1e-3,
        jac_tol=1e-2,
        verbose=False,
    )
    assert res["passed"] == 1.0
