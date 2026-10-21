"""Tests for Phase 5: Sequential Link-by-Link Dynamic Isolation."""

import os
import numpy as np
import pytest
from urdf2dt.parsing import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
)
from urdf2dt.digital_twin import MuJoCoTwin
from urdf2dt.verification import verify_dynamics_sequential

FIXTURE_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur5.urdf")


def test_sequential_dynamics_isolation():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)
    twin = MuJoCoTwin(FIXTURE_URDF, chain=chain, dh=dh, dt=0.005)

    stage_results = verify_dynamics_sequential(
        chain=chain,
        dh=dh,
        inertia=inertia,
        twin=twin,
        t_dur=2.0,
        dt=0.01,
        tol_tau=0.1,
        verbose=True,
    )

    assert len(stage_results) == 6
    for st in stage_results:
        assert st.passed is True, f"Stage {st.stage} ({st.joint_name}) failed with RMS error {st.rms_err:.4e} N*m"
        assert st.rms_err < 0.1
        assert st.max_err < 0.2
