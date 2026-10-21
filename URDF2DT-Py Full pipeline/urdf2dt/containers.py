"""Containers and structured dataclasses for URDF2DT-Py."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import casadi as cs


@dataclass
class Link:
    """URDF Link representation."""
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Joint:
    """URDF Joint representation."""
    name: str
    parent: str
    child: str
    joint_type: str  # 'revolute', 'continuous', 'prismatic', 'fixed'
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    lower: float = -np.pi
    upper: float = np.pi
    effort: float = 100.0
    velocity: float = 10.0
    T_parent_child: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_effective: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class Robot:
    """Parsed URDF robot model."""
    name: str
    links: Dict[str, Link] = field(default_factory=dict)
    joints: Dict[str, Joint] = field(default_factory=dict)
    root_link: str = ""


@dataclass
class Chain:
    """Ordered active serial kinematic chain extracted from URDF."""
    n: int
    active_joints: List[Joint]
    active_links: List[str]
    active_start: str
    O: np.ndarray  # (3, N) world frame joint origins at q=0
    Z: np.ndarray  # (3, N) world frame joint z-axes at q=0
    T_urdf: List[np.ndarray]  # N cumulative transforms at q=0
    T_root_to_j1: np.ndarray  # (4, 4) fixed transform prior to joint 1
    T_ee: np.ndarray  # (4, 4) accumulated fixed transform from link N to EE
    d_ee: float = 0.0
    a_ee: float = 0.0


@dataclass
class DHTable:
    """Standard Denavit-Hartenberg parameters table and transforms."""
    table: np.ndarray  # (N, 4): [theta_offset, d, a, alpha]
    DHpram: np.ndarray  # (N, 5): [a, alpha, theta_offset, d, sigma]
    joint_types: List[str]
    sigma: np.ndarray  # (N,): 1 for revolute, 2 for prismatic
    T_dh: List[np.ndarray]  # N cumulative transforms at q=0


@dataclass
class Inertia:
    """Inertial parameters transformed into standard DH frames."""
    m: np.ndarray  # (N,) mass in kg
    P_Ci: np.ndarray  # (N, 3) CoM relative to preceding DH origin in current DH frame
    I: np.ndarray  # (N, 3, 3) rotational inertia tensors in DH frame orientations


@dataclass
class CasadiModel:
    """Symbolic and compiled CasADi kinematics and Lagrangian dynamics."""
    n: int
    q: cs.SX
    dq: cs.SX
    ddq: cs.SX
    tau: cs.SX
    M_sym: cs.SX
    C_sym: cs.SX
    G_sym: cs.SX
    tau_sym: cs.SX
    fk_fun: cs.Function
    fk_ee_fun: cs.Function
    fk_link_funs: List[cs.Function]
    J_fun: cs.Function
    Jv_fun: cs.Function
    Jw_fun: cs.Function
    Jdot_fun: cs.Function
    M_fun: cs.Function
    C_fun: cs.Function
    G_fun: cs.Function
    tau_fun: cs.Function
    forward_dyn_fun: cs.Function
    step_fun: Optional[cs.Function] = None
    batch_sim_fun: Optional[Any] = None
    g_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))


@dataclass
class StageResult:
    """Sequential dynamic verification result for a single link/joint."""
    stage: int
    joint_name: str
    link_name: str
    passed: bool
    max_err: float
    rms_err: float
    peak_torque: float
    time: np.ndarray
    tau_sim: np.ndarray
    tau_ana: np.ndarray


@dataclass
class VerificationReport:
    """Overall cross-engine verification summary."""
    all_passed: bool
    max_fk_pos_err: float = 0.0
    max_fk_quat_err: float = 0.0
    max_jac_err: float = 0.0
    max_dyn_err: float = 0.0
    stage_results: List[StageResult] = field(default_factory=list)
    summary: str = ""
