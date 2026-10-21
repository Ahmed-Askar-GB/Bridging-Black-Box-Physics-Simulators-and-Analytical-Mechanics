"""Trajectory generation utilities for excitation and verification."""

from __future__ import annotations
import numpy as np
from typing import Tuple


def traj_poly5(
    q0: np.ndarray | list[float],
    qf: np.ndarray | list[float],
    t_dur: float,
    dt: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth 5th-order polynomial trajectory (minimum jerk) with zero boundary velocities and accelerations.

    q(t) = q0 + (qf - q0) * (10 tau^3 - 15 tau^4 + 6 tau^5)
    dq(t) = (qf - q0) * (30 tau^2 - 60 tau^3 + 30 tau^4) / t_dur
    ddq(t) = (qf - q0) * (60 tau - 180 tau^2 + 120 tau^3) / t_dur^2

    Returns:
        time: (M,) time array
        q: (M, N) joint positions
        dq: (M, N) joint velocities
        ddq: (M, N) joint accelerations
    """
    q0 = np.asarray(q0, dtype=np.float64)
    qf = np.asarray(qf, dtype=np.float64)
    assert q0.shape == qf.shape, "q0 and qf must have matching shape"

    num_steps = max(2, int(np.round(t_dur / dt)) + 1)
    time = np.linspace(0.0, t_dur, num_steps)
    tau = time / t_dur

    # Polynomial blend functions
    s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
    ds = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / t_dur
    dds = (60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)) / (t_dur**2)

    delta_q = qf - q0
    # Outer product for trajectory shapes (M, N)
    q = q0[None, :] + np.outer(s, delta_q)
    dq = np.outer(ds, delta_q)
    ddq = np.outer(dds, delta_q)

    return time, q, dq, ddq


def multi_sine_excitation(
    q0: np.ndarray | list[float],
    amplitudes: np.ndarray | list[float],
    frequencies: np.ndarray | list[float],
    t_dur: float,
    dt: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate persistent multi-harmonic sinusoidal excitation for parameter identification and stress testing."""
    q0 = np.asarray(q0, dtype=np.float64)
    amp = np.asarray(amplitudes, dtype=np.float64)
    freq = np.asarray(frequencies, dtype=np.float64)

    num_steps = max(2, int(np.round(t_dur / dt)) + 1)
    time = np.linspace(0.0, t_dur, num_steps)
    n = len(q0)

    q = np.zeros((num_steps, n), dtype=np.float64)
    dq = np.zeros((num_steps, n), dtype=np.float64)
    ddq = np.zeros((num_steps, n), dtype=np.float64)

    for i in range(n):
        w = 2.0 * np.pi * freq[i]
        a = amp[i]
        # 1 - cos(w t) ensures q(0) = q0, dq(0) = 0
        q[:, i] = q0[i] + a * (1.0 - np.cos(w * time))
        dq[:, i] = a * w * np.sin(w * time)
        ddq[:, i] = a * (w**2) * np.cos(w * time)

    return time, q, dq, ddq
