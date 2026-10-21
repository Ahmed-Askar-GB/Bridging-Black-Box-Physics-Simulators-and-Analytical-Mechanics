# Bridging Black-Box Physics Simulators and Analytical Mechanics
### A Tri-Paradigm Framework for Interpretable Robot Modeling, Dynamic Verification, and Real-Time Nonlinear Control

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.6%2B-orange.svg)](https://web.casadi.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0%2B-red.svg)](https://mujoco.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)

---

## Overview

This repository hosts the official implementation of the **Tri-Paradigm Robot Digital Twin Framework**, accompanying the scientific paper:
> **"Bridging Black-Box Physics Simulators and White-Box Analytical Mechanics: A Tri-Paradigm Framework for Interpretable Robot Modeling, Verification, and Nonlinear Control"**

The framework automates the end-to-end derivation of analytical rigid-body mechanics from raw CAD-derived Unified Robot Description Format (URDF) models into three complementary software architectures:

```
Bridging Black-Box Physics Simulators and Analytical Mechanics/
├── End-to-End MATLAB Implementation/        # Tier 1: Pure MATLAB Symbolic (Glass-box inspection)
├── End-to-End MATLAB-CasADi Implementation/ # Tier 2: MATLAB/CasADi + Simscape Multibody (MBD)
├── URDF2DT-Py Full pipeline/                # Tier 3: Python/CasADi + MuJoCo (Real-time NMPC/CTC)
├── STL_Files/                               # 3D visual and collision CAD meshes (.stl, .dae)
├── abbIrb120/                               # Benchmark robot: ABB IRB 120 URDF and models
├── universalUR5.urdf                        # Benchmark robot: Universal Robots UR5
├── DH_dyn.m                                 # Baseline numeric dynamics (cross-validation)
├── DH_dyn_casadi.m                          # Baseline CasADi dynamics (cross-validation)
├── DH_fk.m                                  # Baseline forward kinematics (cross-validation)
└── DH_jacobian.m                            # Baseline Jacobian calculation (cross-validation)
```

---

## Architectural Tiers

| Architecture Tier | Language & Ecosystem | Operational Focus | Primary Performance |
| :--- | :--- | :--- | :--- |
| **Tier 1: `MATLAB-Sym`** | MATLAB Symbolic Math | Pedagogical derivation, glass-box transparency, Lyapunov proofs | Analytical cosine/sine polynomial extraction; $195.0~\mu\text{s}$ vectorized ID |
| **Tier 2: `MATLAB-CasADi`** | MATLAB + CasADi (SX) | Industrial Model-Based Design (MBD), Simscape twin synthesis | Rapid graph compilation ($0.22\text{ s}$); programmatic Simscape `.slx` build |
| **Tier 3: `Python-MuJoCo`** | Python 3.10+ + CasADi + MuJoCo | Open-source, real-time control ($>1\text{ kHz}$), embedded C-code | **$78.7~\mu\text{s}$ forward dynamics (Cholesky)**; $12.4\times$ real-time batch ODE |

---

## Key Algorithmic Innovations

1. **Euler–Lagrange Momentum Derivative for Coriolis Bias**:
   Replaces the bloated Christoffel symbol tensor $\sum_{j,k} c_{ijk}\dot{q}_j\dot{q}_k$ ($\mathcal{O}(n^4)$ complexity, 216 partial derivatives on 6-DOF arms) with a direct forward-mode directional derivative:
   $$\mathbf{c}_{\text{bias}}(\bm{q}, \dot{\bm{q}}) = \operatorname{jtimes}(\bm{M}(\bm{q})\dot{\bm{q}},\, \bm{q},\, \dot{\bm{q}}) - \frac{1}{2}\nabla_{\bm{q}}\left(\dot{\bm{q}}^T \bm{M}(\bm{q})\dot{\bm{q}}\right)$$
   Reduces symbolic graph operations by **63.2%** while maintaining machine-precision parity ($4.16 \times 10^{-17}\text{ N}\cdot\text{m}$).

2. **Inversion-Free Symbolic Cholesky Factorization**:
   Proves $\bm{M}(\bm{q}) \succ 0$ (SPD) and decomposes $\bm{M} = \bm{R}^T \bm{R}$. Forward dynamics $\ddot{\bm{q}}$ evaluates via two triangular substitutions without symbolic matrix inversion, accelerating single-step simulation by **$1.92\times$** ($78.7~\mu\text{s}$ vs $151.5~\mu\text{s}$).

3. **Fully Parameterized 3D Gravity Vector ($\bm{g} \in \mathbb{R}^3$)**:
   Enables seamless digital twin simulation for terrestrial, inclined gantry, microgravity ($\bm{g}=\bm{0}$), lunar ($1.62\text{ m/s}^2$), and Martian ($3.71\text{ m/s}^2$) environments.

4. **Five-Stage Autonomous Fail-Fast Verification Protocol**:
   Automates physical invariance validation, Monte Carlo forward kinematics certification ($N=1,000$), geometric Jacobian consistency, link-by-link sequential dynamic isolation, and closed-loop Computed Torque Control tracking.

---

## Quickstart Guide

### Tier 3: Python/CasADi + MuJoCo

#### 1. Setup Environment
```bash
cd "URDF2DT-Py Full pipeline"
python -m pip install -e .
```

#### 2. Run Autonomous Verification Suite
```bash
python -m pytest -v
```
All 22 unit and integration tests validate parsing, kinematics, dynamics, MuJoCo cross-validation, and closed-loop tracking.

#### 3. Run Universal Robots UR16e Benchmark
```bash
python benchmark_ur16e_casadi_vs_mujoco.py
```

#### 4. Launch 3D Interactive Telemetry & Animation
```bash
python animate_ur16e_interactive.py
```

---

### Tiers 1 & 2: MATLAB & Simscape Digital Twin

In MATLAB (R2023b or later):

```matlab
% For Tier 1 (Pure MATLAB Symbolic):
cd('End-to-End MATLAB Implementation');
report = run_pipeline();

% For Tier 2 (MATLAB-CasADi + Simscape Digital Twin):
cd('End-to-End MATLAB-CasADi Implementation');
report = run_pipeline();
```

---

## Citation

If you use this codebase or digital twin methodology in your research, please cite:

```bibtex
@article{ihab2026triparadigm,
  title={Bridging Black-Box Physics Simulators and White-Box Analytical Mechanics: A Tri-Paradigm Framework for Interpretable Robot Modeling, Verification, and Nonlinear Control},
  author={Ihab, Abdelrahman and Askar, Ahmed and El-Hussieny, Haitham},
  journal={Nature Scientific Reports},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
