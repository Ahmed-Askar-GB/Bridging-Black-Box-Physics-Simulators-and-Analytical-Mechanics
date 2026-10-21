"""Tests for Phase 6: Model-Based Control (Computed Torque & CasADi NMPC)."""

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
from urdf2dt.control import ComputedTorqueController, CasadiNMPC
from urdf2dt.utils import traj_poly5, detect_joint_signs

FIXTURE_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur5.urdf")


@pytest.fixture(scope="module")
def control_setup():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)
    model = dynamics_casadi(dh, inertia, chain)
    twin = MuJoCoTwin(FIXTURE_URDF, chain=chain, dh=dh, dt=0.001)
    signs = detect_joint_signs(chain, model.fk_link_funs)
    return model, twin, chain, signs


def test_computed_torque_controller(control_setup):
    model, twin, chain, signs = control_setup
    controller = ComputedTorqueController(model, kp=300.0, kd=40.0, joint_signs=signs)

    # Trajectory tracking test
    q0 = np.zeros(6)
    qf = np.array([0.2, -0.3, 0.2, -0.1, 0.2, -0.1])
    time, q_ref, dq_ref, ddq_ref = traj_poly5(q0, qf, t_dur=2.0, dt=0.001)

    res = twin.simulate_trajectory(time, q_ref, dq_ref, ddq_ref, controller=controller)
    pos_err = np.linalg.norm(res["q_sim"] - q_ref, axis=1)

    # Ensure tracking error converges and remains tiny (< 0.01 rad)
    assert np.max(pos_err) < 0.01


def test_casadi_nmpc_step(control_setup):
    model, _, chain, signs = control_setup
    nmpc = CasadiNMPC(model, N_horizon=5, dt=0.05, joint_signs=signs, verbose=False)

    x0 = np.zeros(12)  # [q; v]
    # Reference trajectory over horizon
    x_ref = np.zeros((12, 6))
    for k in range(6):
        x_ref[:6, k] = np.array([0.1, -0.1, 0.1, 0.0, 0.0, 0.0])

    tau_cmd, x_opt = nmpc.solve(x0, x_ref)

    assert tau_cmd.shape == (6,)
    assert np.all(np.isfinite(tau_cmd))
    assert x_opt.shape == (12, 6)
