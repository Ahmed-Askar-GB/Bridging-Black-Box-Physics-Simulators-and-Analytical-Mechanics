# Bridging Black-Box Physics Simulators and Analytical Mechanics
### A Tri-Paradigm Framework for Interpretable Robot Modeling, Dynamic Verification, and Real-Time Nonlinear Control

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.6%2B-orange.svg)](https://web.casadi.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0%2B-red.svg)](https://mujoco.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)

---

## Overview

This repository provides the official implementation of the **CADENCE** (*Computer-Aided Derivation of Exact Nonlinear Continuum-Equated Dynamics for Robotics*) tri-paradigm framework, accompanying the paper:

> **"Bridging Black-Box Physics Simulators and Analytical Mechanics: A Tri-Paradigm Framework for Interpretable Robot Modeling, Dynamic Verification, and Real-Time Nonlinear Control"**

CADENCE automates the end-to-end transformation of raw, CAD-derived Unified Robot Description Format (URDF) descriptions into minimal canonical Denavit--Hartenberg (DH) parameters, maps inertial tensors with spectral realizability sanitization, and compiles closed-form equations of motion across three interoperable execution paradigms:

```
Bridging Black-Box Physics Simulators and Analytical Mechanics/
├── End-to-End MATLAB Implementation/        # Tier 1: Pure MATLAB Symbolic (Glass-box inspection)
│   ├── parsing/                             # URDF parser, kinematic chain builder, DH extraction
│   ├── kinematics/                          # Symbolic forward kinematics & geometric Jacobians
│   ├── dynamics/                            # Symbolic Euler-Lagrange dynamics (Momentum & Christoffel)
│   ├── control/                             # Computed torque control & fmincon trajectory optimization
│   ├── verification/                        # Algebraic skew-symmetry & static balance validators
│   └── utils/                               # Standardized robot path resolver & math utilities
│
├── End-to-End MATLAB-CasADi Implementation/ # Tier 2: MATLAB/CasADi + Simscape Multibody (MBD)
│   ├── parsing/                             # Graph extraction and DH parameter classification
│   ├── kinematics/                          # CasADi SX kinematics & Lie-group SO(3) mappings
│   ├── dynamics/                            # Algorithmic differentiation dynamics (Cholesky + HVP)
│   ├── control/                             # CTC feedback linearization & CasADi Opti NMPC
│   ├── benchmarks/                          # 5-stage verification, 15-BVP traj opt, mismatch, NMPC
│   ├── digital_twin/                        # Programmatic Simscape Multibody (.slx) twin builder
│   └── utils/                               # Joint sign detection & scale-aware tolerances
│
├── URDF2DT-Py Full pipeline/                # Tier 3: Python 3.10+ / CasADi + MuJoCo (Real-Time Control)
│   ├── urdf2dt/                             # Core library (parsing, analytical, digital twin, control)
│   ├── tests/                               # 24-test pytest suite (invariants, kinematics, dynamics, CTC)
│   └── benchmarks/                          # Scaling, trajectory optimization, mismatch, NMPC
│
├── abbIrb120/                               # Benchmark robot: ABB IRB 120 URDF and mesh files
├── STL_Files/                               # 3D visual and collision CAD meshes (.stl, .dae)
├── universalUR5.urdf                        # Benchmark robot: Universal Robots UR5
└── kukaIiwa7.urdf                           # Benchmark robot: KUKA LBR iiwa 7
```

---

## Architectural Paradigms

| Architecture Tier | Language & Ecosystem | Operational Focus | Key Performance Metrics |
| :--- | :--- | :--- | :--- |
| **Tier 1: `MATLAB-Sym`** | MATLAB Symbolic Math | Pedagogical derivation, glass-box transparency, Lyapunov proofs | Exact closed-form algebraic expressions; analytical skew-symmetry proof; zero external dependencies |
| **Tier 2: `MATLAB-CasADi`** | MATLAB + CasADi (SX) | Industrial Model-Based Design (MBD), Simscape twin synthesis | Rapid graph compilation ($<0.08\text{ s}$); programmatic Simscape `.slx` build; multiple-shooting NMPC |
| **Tier 3: `Python-MuJoCo`** | Python 3.10+ + CasADi + MuJoCo | Open-source, zero-licensing, high-rate control ($>10\text{ kHz}$), C-code | **$78.7~\mu\text{s}$ forward dynamics (Cholesky)**; $1.57\times$--$1.63\times$ NMPC solve speedup |

---

## Core Algorithmic & Matrix Calculus Foundations

### 1. Directional Momentum Coriolis Formulation ($O(n^2)$ Complexity)
Rather than evaluating the classical $n \times n \times n$ rank-3 Christoffel symbol tensor:
$$c_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{kj}}{\partial q_i}\right)$$
which requires unrolling $n^3$ partial derivatives (e.g., 216 expressions for $n=6$; 343 for $n=7$), CADENCE derives the Coriolis bias vector $\mathbf{c}_{\text{bias}}(\mathbf{q}, \dot{\mathbf{q}}) = \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$ as the exact directional Lie derivative of generalized momentum $\mathbf{p} = \mathbf{M}(\mathbf{q})\dot{\mathbf{q}}$:
$$\mathbf{c}_{\text{bias}}(\mathbf{q}, \dot{\mathbf{q}}) = \frac{d}{dt}\left(\mathbf{M}(\mathbf{q})\dot{\mathbf{q}}\right) - \frac{1}{2}\nabla_{\mathbf{q}}\left(\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q})\dot{\mathbf{q}}\right)$$
This operator-theoretic approach slashes symbolic graph operation counts by **44.3%** at $n=6$, **52.1%** at $n=7$, and **76.8%** at $n=14$, while preserving machine-precision dynamic consistency ($\le 4.44 \times 10^{-15}\text{ N}\cdot\text{m}$).

### 2. Inversion-Free Symbolic Cholesky Factorization
Because the generalized mass matrix is strictly Symmetric Positive-Definite ($\mathbf{M}(\mathbf{q}) \succ 0$), CADENCE factors $\mathbf{M}(\mathbf{q}) = \mathbf{R}^T \mathbf{R}$ via symbolic Cholesky decomposition. Forward dynamics $\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{c}_{\text{bias}} - \mathbf{G})$ evaluates via two triangular substitution passes:
$$\mathbf{R}^T \mathbf{y} = \boldsymbol{\tau} - \mathbf{c}_{\text{bias}}(\mathbf{q}, \dot{\mathbf{q}}) - \mathbf{G}(\mathbf{q}), \quad \mathbf{R} \ddot{\mathbf{q}} = \mathbf{y}$$
This eliminates pivoting overhead and cuts forward dynamics operations by over **50%** relative to general LU inversion. On high-DOF chains ($n=14$), Cholesky evaluation executes in **$374.2~\mu\text{s}$** ($>2.6\text{ kHz}$ throughput), delivering a **$4.48\times$ speedup** over explicit inversion ($1,676.0~\mu\text{s}$).

### 3. Matrix-Free Hessian-Vector Products (HVP)
Grounded in modern operator-theoretic matrix calculus (*Edelman, Johnson, & Bright, MIT 18.S096 / 18.063, 2025*), CADENCE computes directional curvature:
$$\nabla_{\mathbf{q}}^2 V(\mathbf{q}) \cdot \mathbf{v} = \frac{\partial \mathbf{G}(\mathbf{q})}{\partial \mathbf{q}} \mathbf{v} = \left.\frac{d}{d\alpha} \mathbf{G}(\mathbf{q} + \alpha \mathbf{v})\right|_{\alpha=0}$$
using forward-over-reverse algorithmic differentiation in $\mathcal{O}(\operatorname{cost}(\mathbf{G}))$ operations, bypassing dense $n \times n$ Hessian matrix storage during SQP/Newton iterations.

### 4. Inertial Realizability & Spectral Projection
Mesh approximations in raw URDF files frequently generate invalid inertia tensors violating the rigid-body triangle inequality ($I_1 + I_2 \ge I_3$). CADENCE applies spectral projection onto the manifold of physically realizable pseudo-inertia matrices $\boldsymbol{\mathcal{J}} \succ 0$, restoring positive-definiteness without perturbing principal coordinate axes.

---

## Supported Benchmark Manipulators

CADENCE includes comprehensive, verified model configurations for five diverse manipulator architectures:

| Platform | DOFs | Morphology & Wrist Type | Nominal Mass | Ingestion Status | Verification Pass Rate |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Universal Robots UR5** | 6 | Non-spherical wrist, cantilever | $18.4\text{ kg}$ | Verified | **100% (5/5 Stages)** |
| **Universal Robots UR16e** | 6 | Non-spherical wrist, heavy payload | $33.1\text{ kg}$ | Verified | **100% (5/5 Stages)** |
| **ABB IRB 120** | 6 | Inline spherical wrist, compact | $25.0\text{ kg}$ | Verified | **100% (5/5 Stages)** |
| **Unimation Puma 560** | 6 | Offset spherical wrist, classic benchmark | $54.2\text{ kg}$ | Verified | **100% (5/5 Stages)** |
| **KUKA LBR iiwa 7** | 7 | Kinematically redundant arm ($R \bot R \dots$) | $23.9\text{ kg}$ | Verified | **100% (5/5 Stages)** |
| **Synthetic Chains** | $3 \le n \le 14$ | Serial kinematic chains for scaling studies | Scalable | Verified | **100% ($n \in [3,14]$)** |

---

## Quickstart Guide

### 1. Tier 3: Python 3.10+ / CasADi + MuJoCo

#### Setup Environment
```bash
cd "URDF2DT-Py Full pipeline"
pip install -e .
```

#### Run the PyTest Verification Suite (24 Tests)
```bash
python -m pytest tests/ -v
```
All unit tests validate URDF parsing, DH extraction, CasADi kinematics, momentum Coriolis dynamics, matrix-free HVPs, MuJoCo cross-validation, sequential link isolation, and closed-loop Computed Torque Control.

#### Run Master Tri-Platform Parity Benchmark
```bash
python benchmarks/run_tri_platform_master_benchmark.py
```
Ingests all 5 benchmark manipulators, benchmarks derivation throughput, checks static balance $\boldsymbol{\tau}(\mathbf{q}, \mathbf{0}, \mathbf{0}) == \mathbf{G}(\mathbf{q})$, and confirms forward dynamics roundtrip errors to machine precision ($\le 9.77 \times 10^{-15}\text{ rad/s}^2$).

#### Run Constrained Offline Trajectory Optimization
```bash
python benchmarks/bench_e2_trajectory_opt.py
```
Solves 15 boundary-value problems per robot, comparing Exact Symbolic Lagrangian Hessians vs Quasi-Newton (L-BFGS).

#### Run Receding-Horizon NMPC with Obstacle Avoidance
```bash
python benchmarks/bench_e4_realtime_nmpc.py
```

---

### 2. Tier 2: MATLAB-CasADi + Simscape Multibody

In MATLAB (R2023b or later):

```matlab
% Add Tier 2 directories to path
cd('End-to-End MATLAB-CasADi Implementation');
addpath(pwd);
addpath('parsing', 'kinematics', 'dynamics', 'control', 'verification', 'benchmarks', 'utils');

% 1. Run 5-Stage Multi-Robot Verification Suite across all 5 manipulators
run_multirobot_5stage_verification();

% 2. Benchmark Offline Trajectory Optimization across 15 BVPs (Table 4 Parity)
benchmark_trajectory_opt_15bvp({'UR5', 'UR16e', 'KUKA_iiwa'}, 15);

% 3. Benchmark Closed-Loop Control under 4-Level Parametric Mismatch (Table 5 Parity)
benchmark_mismatch_control(10);

% 4. Benchmark Receding-Horizon NMPC with Obstacle Avoidance (Table 6 Parity)
benchmark_nmpc_ur16e_kuka(40);

% 5. Benchmark Multi-DOF Scalability from n=3 to n=14 (Table 3 Parity)
benchmark_multi_dof_scalability([3, 4, 6, 7, 10, 12, 14]);
```

---

### 3. Tier 1: Pure MATLAB Symbolic Math (`MATLAB-Sym`)

For glass-box algebraic inspection, textbook derivations, and Lyapunov stability proofs with zero external toolboxes:

```matlab
% Add Tier 1 directories to path
cd('End-to-End MATLAB Implementation');
addpath(pwd);
addpath('parsing', 'kinematics', 'dynamics', 'control', 'verification', 'utils');

% 1. Derive symbolic equations and inspect algebraic character lengths
results = run_multirobot_sym_verification();

% 2. Inspect individual robot (e.g., UR5)
res = run_track1_sym('UR5');
disp(res.dyn.M_sym(1, 1)); % Glass-box mass matrix element
disp(res.dyn.c_bias_sym(1)); % Momentum Coriolis bias
```

---

## Experimental Validation Summary

The benchmarks included in this repository reproduce the empirical results published in the paper:

### 1. Multi-Robot 5-Stage Verification (Table 2 Parity)
- **Forward Kinematics Position Error:** $< 8.8 \times 10^{-10}\text{ m}$ across all platforms ($5.8 \times 10^{-16}\text{ m}$ on UR5).
- **Unit Quaternion Orientation Error:** $< 4.2 \times 10^{-8}$ across $1,000$ Monte Carlo configurations.
- **Physical Realizability:** Triangle inequality restored via spectral projection; $\lambda_{\min}(\mathbf{M}) > 0$ strictly maintained.
- **Verification Pass Rate:** **100%** across all stages.

### 2. Multi-DOF Scalability ($n \in [3, 14]$) (Table 3 Parity)
- **Empirical Complexity Exponent:** Momentum Coriolis scales with power-law exponent $\alpha = 2.29$ vs $\alpha = 3.25$ for classical Christoffel symbols.
- **Instruction Graph Compression:** $-44.3\%$ at $n=6$, $-52.1\%$ at $n=7$, and **$-76.8\%$** at $n=14$.
- **Real-Time Determinism:** Symbolic Cholesky executes in **$374.2~\mu\text{s}$** at $n=14$ ($2.67\text{ kHz}$ update rate), delivering a **$4.48\times$ speedup** over explicit inversion ($1,676.0~\mu\text{s}$, $596\text{ Hz}$).

### 3. Trajectory Optimization with Exact Hessians (Table 4 Parity)
- **Convergence Rate:** Exact Lagrangian Hessians achieve **100.0% convergence** across all 45 boundary-value problems (UR5, UR16e, KUKA iiwa).
- **Quasi-Newton Failure:** L-BFGS times out ($0\%$ convergence) on the heavy-payload UR16e and achieves only $6.7\%$ convergence on the 7-DOF KUKA iiwa within 500 iterations.
- **Iteration Reduction:** Exact curvature cuts solver iteration count by **96.8%** and accelerates solve times by up to **$14.39\times$**.

### 4. Closed-Loop Control & Parametric Mismatch (Table 5 Parity)
- **Tracking Error Reduction:** Full White-Box Computed Torque Control (CTC) slashes RMS tracking error by **98.0%** over model-free decentralized PID under nominal continuous maneuvers ($0.0155\text{ rad}$ vs $0.7841\text{ rad}$) while eliminating high-frequency torque chattering.
- **Parametric Uncertainty:** CTC preserves superior tracking accuracy under $\pm 10\%$ (Level 1) and $\pm 20\%$ (Level 2) mass/inertia mismatch.

### 5. Receding-Horizon NMPC with Obstacle Avoidance (Table 6 Parity)
- **KKT Dynamic Jacobian Compression:** Instruction count reduced by **$-40.6\%$** on UR16e and **$-48.2\%$** on KUKA iiwa.
- **Solve Acceleration:** Mean solve time accelerated by **$1.57\times$** on UR16e and **$1.63\times$** on KUKA iiwa.
- **Latency Spike Mitigation:** 99th-percentile (p99) solve latency spike cut by **$43.5\%$** on UR16e and **$25.9\%$** on KUKA iiwa during active obstacle activation.

---

## Citation

If you use this framework or codebase in your academic research, please cite:

```bibtex
@article{ihab2026cadence,
  title={Bridging Black-Box Physics Simulators and Analytical Mechanics: A Tri-Paradigm Framework for Interpretable Robot Modeling, Dynamic Verification, and Real-Time Nonlinear Control},
  author={Ihab, Abdelrahman and Askar, Ahmed and El-Hussieny, Haitham},
  journal={Nature Scientific Reports},
  year={2026}
}
```

---

## License

This project is open-source and released under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
