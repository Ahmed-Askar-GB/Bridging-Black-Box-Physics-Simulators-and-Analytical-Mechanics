"""Tests for Phase 1: Parsing and Parameter Extraction."""

import os
import numpy as np
import pytest
from urdf2dt.parsing import (
    urdf_parse,
    urdf_build_chain,
    urdf_extract_dh,
    urdf_extract_inertia,
)

FIXTURE_URDF = os.path.join(os.path.dirname(__file__), "fixtures", "ur5.urdf")


def test_urdf_parse():
    robot = urdf_parse(FIXTURE_URDF)
    assert robot.name == "ur5"
    assert len(robot.links) > 0
    assert len(robot.joints) > 0
    assert robot.root_link in ("world", "base_link")


def test_urdf_build_chain():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    assert chain.n == 6
    assert len(chain.active_joints) == 6
    assert len(chain.active_links) == 6
    assert chain.O.shape == (3, 6)
    assert chain.Z.shape == (3, 6)


def test_urdf_extract_dh():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)

    assert dh.table.shape == (6, 4)
    assert dh.DHpram.shape == (6, 5)
    assert len(dh.T_dh) == 6

    # Verify standard UR5 DH parameters:
    # a values: approximately [0, -0.425, -0.39225, 0, 0, 0] or positive magnitudes
    # link offsets d: shoulder ~0.089, wrist1 ~0.109, wrist2 ~0.09465, wrist3 ~0.0823
    assert abs(dh.table[1, 2] - 0.425) < 1e-3 or abs(dh.table[1, 2] + 0.425) < 1e-3
    assert abs(dh.table[2, 2] - 0.39225) < 1e-3 or abs(dh.table[2, 2] + 0.39225) < 1e-3
    assert abs(dh.table[0, 1] - 0.089159) < 1e-3


def test_urdf_extract_inertia():
    robot = urdf_parse(FIXTURE_URDF)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)

    assert inertia.m.shape == (6,)
    assert inertia.P_Ci.shape == (6, 3)
    assert inertia.I.shape == (6, 3, 3)

    # Masses must be positive
    assert np.all(inertia.m > 0.0)

    # Inertia tensors must be symmetric
    for i in range(6):
        I_i = inertia.I[i]
        assert np.allclose(I_i, I_i.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(I_i)
        assert np.all(eigvals > 0.0)
