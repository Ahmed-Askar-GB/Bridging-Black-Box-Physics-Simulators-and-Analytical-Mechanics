"""Autonomous robot-agnostic joint axis sign detection between URDF and standard DH."""

from __future__ import annotations
import numpy as np
from typing import List, Callable, Any
from urdf2dt.containers import Chain


def detect_joint_signs(chain: Chain, fk_link_funs: List[Callable[..., Any]]) -> np.ndarray:
    """Automatically determine the sign alignment (+1.0 or -1.0) between URDF joint actuation axes and DH z-axes.

    At q=0:
      z_urdf[i] is the world-frame actuation axis of joint i.
      z_dh[i-1] is the world-frame z-axis of DH frame {i-1}.
      Because both define the same physical axis line of rotation:
        sign_map[i] = sign(round(dot(z_urdf[i], z_dh[i-1])))
    """
    n = chain.n
    joint_signs = np.ones(n, dtype=np.float64)
    q_zero = np.zeros(n, dtype=np.float64)

    for i in range(n):
        z_urdf = chain.Z[:, i]
        if i == 0:
            # First DH frame actuation axis is world [0, 0, 1]
            z_dh = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            # Frame {i-1} cumulative transform
            T_prev = np.array(fk_link_funs[i - 1](q_zero), dtype=np.float64)
            z_dh = T_prev[:3, 2]

        d = float(np.dot(z_urdf, z_dh))
        s = round(d)
        if abs(s) >= 0.5:
            joint_signs[i] = np.sign(s)
        else:
            joint_signs[i] = 1.0

    return joint_signs
