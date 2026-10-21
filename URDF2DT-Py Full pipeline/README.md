# URDF2DT-Py

**URDF to Analytical CasADi Model, Digital Twin & Control Framework**

An end-to-end, modular, headless robotics pipeline in Python that ingests any arbitrary serial manipulator `.urdf` file, extracts standard Denavit–Hartenberg (DH) parameters via the geometric P-Q skew-lines method, projects link inertial parameters into DH frames, derives exact symbolic analytical kinematics and Lagrangian dynamics using CasADi (`SX`), cross-validates against Pinocchio and MuJoCo ground-truth engines at machine precision, and provides model-based control synthesis (Computed Torque Control & CasADi NMPC).

---

## Key Features
- **Phase 1: Parsing & Geometric DH Extraction (`urdf2dt.parsing`)**:
  - Full XML ingestion of links, joints, and rotated inertia tensors.
  - Breadth-First Search (BFS) active serial chain resolution.
  - Skew-lines (P-Q foot-points) standard DH parameter derivation (handles parallel, skew, intersecting axes).
  - Projection of link CoM and inertia tensors into standard DH frames.
- **Phase 2: Analytical CasADi Engine (`urdf2dt.analytical`)**:
  - Symbolic Forward Kinematics $T_0^i(q)$ and singularity-free Shepperd quaternion conversion.
  - Geometric Jacobian $J(q)$ and exact auto-diff time-derivative $\dot{J}(q, \dot{q})$.
  - Lagrangian Mass Matrix $M(q)$, Coriolis Matrix $C(q, \dot{q})$ via symbolic Christoffel symbols, and Gravity $G(q)$.
  - C code generation (`.c`/`.h`) and standalone serialization.
- **Phase 3: Dual-Engine Digital Twin (`urdf2dt.digital_twin`)**:
  - `PinocchioTwin`: Complete wrapper for RNEA, CRBA, ABA, Jacobians, and `pinocchio.casadi` bridge.
  - `MuJoCoTwin`: Native headless physics plant, EE site injection, `mj_inverse`, `mj_fullM`, `mj_jac`, and forward integration.
- **Phase 4 & 5: Autonomous Verification (`urdf2dt.verification`)**:
  - Autonomous robot-agnostic joint sign detection between URDF axes and DH $z$-axes.
  - Physical invariant checks ($SO(3)$, triangle inequality, mass matrix SPD).
  - Link-by-link sequential dynamic isolation with 5th-order polynomial excitation.
- **Phase 6: Model-Based Control (`urdf2dt.control`)**:
  - Computed Torque Controller (inverse dynamics feedforward + PD).
  - Multiple-shooting Trajectory Tracking NMPC in CasADi `Opti`.

---

## Installation

```bash
pip install -e .
```

To run tests:
```bash
pytest tests/ -v
```

To execute master pipeline:
```bash
python run_pipeline.py --urdf tests/fixtures/ur5.urdf
```
