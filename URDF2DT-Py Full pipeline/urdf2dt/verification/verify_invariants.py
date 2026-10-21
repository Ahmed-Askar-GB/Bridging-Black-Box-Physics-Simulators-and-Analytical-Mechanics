"""Mathematical invariant and physical realizability verification."""

from __future__ import annotations
import numpy as np
from urdf2dt.containers import Robot, DHTable, Inertia, CasadiModel


def verify_invariants(
    robot: Robot,
    dh: DHTable,
    inertia: Inertia,
    model: CasadiModel,
    num_samples: int = 10,
    tol: float = 1e-7
) -> dict[str, bool]:
    """Verify physical and mathematical invariants:

    1. Link masses strictly positive.
    2. Inertia tensors satisfy triangle inequality (principal moments).
    3. Forward kinematics rotation matrices lie in SO(3).
    4. Generalized Mass Matrix M(q) is strictly symmetric positive-definite.
    """
    n = dh.table.shape[0]
    results = {}

    # 1. Mass positivity
    results["mass_positivity"] = bool(np.all(inertia.m > 0.0))

    # 2. Triangle inequality for link inertias
    triangle_ok = True
    for i in range(n):
        I_i = inertia.I[i, :, :]
        eigvals = np.sort(np.linalg.eigvalsh(I_i))
        if (eigvals[0] + eigvals[1]) < (eigvals[2] - tol):
            triangle_ok = False
            break
    results["triangle_inequality"] = triangle_ok

    # 3. SO(3) checks on FK
    so3_ok = True
    rng = np.random.default_rng(42)
    for _ in range(num_samples):
        q_rand = rng.uniform(-np.pi, np.pi, size=n)
        _, R_ee, quat_ee = model.fk_fun(q_rand)
        R = np.array(R_ee)
        quat = np.array(quat_ee).flatten()

        ortho_err = np.linalg.norm(R @ R.T - np.eye(3))
        det_err = abs(np.linalg.det(R) - 1.0)
        quat_err = abs(np.linalg.norm(quat) - 1.0)

        if ortho_err > tol or det_err > tol or quat_err > tol:
            so3_ok = False
            break
    results["fk_so3_orthogonality"] = so3_ok

    # 4. M(q) symmetry and SPD
    spd_ok = True
    for _ in range(num_samples):
        q_rand = rng.uniform(-np.pi, np.pi, size=n)
        M = np.array(model.M_fun(q_rand))
        sym_err = np.linalg.norm(M - M.T)
        eigvals = np.linalg.eigvalsh(M)

        if sym_err > tol or np.any(eigvals <= 0.0):
            spd_ok = False
            break
    results["mass_matrix_spd"] = spd_ok

    return results
