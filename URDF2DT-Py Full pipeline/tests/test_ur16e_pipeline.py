"""End-to-End Pipeline Verification for Universal Robots UR16e (universalUR16e.urdf)."""

import os
import numpy as np
import pytest
from urdf2dt import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
    dynamics_casadi,
    MuJoCoTwin,
    ComputedTorqueController,
    verify_invariants,
    verify_kinematics,
    verify_dynamics_sequential,
)
from urdf2dt.utils import traj_poly5, detect_joint_signs

UR16E_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur16e.urdf")


@pytest.fixture(scope="module")
def ur16e_models():
    robot = urdf_parse(UR16E_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)
    model = dynamics_casadi(dh, inertia, chain)
    twin = MuJoCoTwin(UR16E_URDF, chain=chain, dh=dh, dt=0.001)
    return robot, chain, dh, inertia, model, twin


def test_ur16e_parsing_and_dh(ur16e_models):
    robot, chain, dh, inertia, _, _ = ur16e_models

    assert robot.name == "ur16e_robot"
    assert chain.n == 6
    assert len(chain.active_joints) == 6
    assert len(chain.active_links) == 6

    # Verify UR16e physical link lengths:
    # d1 = 0.1807 m, a2 = 0.4784 m, a3 = 0.3600 m, d4 = 0.1741 m, d5 = 0.1198 m
    assert abs(dh.table[0, 1] - 0.1807) < 1e-3
    assert abs(dh.table[1, 2] - 0.4784) < 1e-3
    assert abs(dh.table[2, 2] - 0.3600) < 1e-3
    assert abs(dh.table[3, 1] - 0.1741) < 1e-3
    assert abs(dh.table[4, 1] - 0.1198) < 1e-3

    # Total moving mass of UR16e active links is 27.26 kg
    total_mass = float(np.sum(inertia.m))
    assert abs(total_mass - 27.26) < 0.1


def test_ur16e_invariants(ur16e_models):
    robot, _, dh, inertia, model, _ = ur16e_models
    res = verify_invariants(robot, dh, inertia, model, num_samples=10)
    assert res["mass_positivity"] is True
    assert res["triangle_inequality"] is True
    assert res["fk_so3_orthogonality"] is True
    assert res["mass_matrix_spd"] is True


def test_ur16e_kinematics_cross_validation(ur16e_models):
    _, chain, _, _, model, twin = ur16e_models
    res = verify_kinematics(
        model=model,
        twin=twin,
        chain=chain,
        num_samples=15,
        pos_tol=1e-4,
        quat_tol=1e-3,
        jac_tol=1e-3,
        verbose=False,
    )
    assert res["passed"] == 1.0
    assert res["max_pos_err"] < 1e-6
    assert res["max_quat_err"] < 1e-6
    assert res["max_jac_err"] < 1e-6


def test_ur16e_sequential_dynamics_isolation(ur16e_models):
    _, chain, dh, inertia, _, twin = ur16e_models
    stage_results = verify_dynamics_sequential(
        chain=chain,
        dh=dh,
        inertia=inertia,
        twin=twin,
        t_dur=2.0,
        dt=0.01,
        tol_tau=0.05,
        verbose=True,
    )

    assert len(stage_results) == 6
    for st in stage_results:
        assert st.passed is True, f"Stage {st.stage} ({st.joint_name}) failed: RMS {st.rms_err:.4e} N*m"
        assert st.rms_err < 0.05


def test_ur16e_computed_torque_tracking(ur16e_models):
    _, chain, _, _, model, twin = ur16e_models
    signs = detect_joint_signs(chain, model.fk_link_funs)
    controller = ComputedTorqueController(model, kp=300.0, kd=40.0, joint_signs=signs)

    q0 = np.zeros(6)
    qf = np.array([0.3, -0.4, 0.3, -0.2, 0.3, -0.2])
    time, q_ref, dq_ref, ddq_ref = traj_poly5(q0, qf, t_dur=2.0, dt=0.001)

    res = twin.simulate_trajectory(time, q_ref, dq_ref, ddq_ref, controller=controller)
    pos_err = np.linalg.norm(res["q_sim"] - q_ref, axis=1)

    assert np.max(pos_err) < 0.01
