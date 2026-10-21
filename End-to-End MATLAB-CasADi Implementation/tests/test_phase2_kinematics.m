%% test_phase2_kinematics.m
% =========================================================================
% INTEGRATION TEST — Phase 2: CasADi Kinematics & Dynamics
% =========================================================================
% Validates that the CasADi symbolic pipeline produces correct results
% by cross-checking against:
%   (A) The existing numeric DH_fk.m    (FK comparison)
%   (B) The existing numeric DH_jacobian.m (Jacobian comparison)
%   (C) Finite-difference Jacobian (numerical ground truth for J and J-dot)
%   (D) The existing DH_dyn_casadi.m    (Dynamics M, C, G comparison)
%   (E) Structural/physical checks      (M SPD, tau=G at rest, etc.)
%   (F) traj_poly5 boundary conditions  (utility function check)
%
% KEY NOTES ON CONVENTIONS:
%   - DH_fk.m     : sigma 1 = revolute (same as our convention)
%   - DH_jacobian : sigma 1 = revolute (same as our convention)
%   - DH_dyn_casadi: sigma 1 = revolute (same as our convention)
%   - fk_ee_fun   : returns T_0^N * T_ee.  For UR5, T_ee = eye(4), so
%                   fk_ee_fun output equals DH_fk output (no trailing offset).
%   - DH_dyn_casadi signature: (DHpram, P_Ci, q, dq, I, m, g [, ddq])
%     P_Ci is [Nx3] — the function transposes it internally.
%
% =========================================================================
% HOW TO RUN:
%   cd to the "End-to-End MATLAB-CasADi Implementation" folder, then:
%     run tests/test_phase2_kinematics.m
%
% PASS criterion: "ALL TESTS PASSED" printed, assert passes.
% =========================================================================

clear; clc;

% -------------------------------------------------------------------------
% 0. Setup paths
% -------------------------------------------------------------------------
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
root_dir   = fileparts(parent_dir);

addpath(fullfile(parent_dir, 'parsing'));
addpath(fullfile(parent_dir, 'kinematics'));
addpath(fullfile(parent_dir, 'dynamics'));
addpath(fullfile(parent_dir, 'utils'));
addpath(root_dir);   % for DH_fk.m, DH_jacobian.m, DH_dyn_casadi.m

urdf_file = fullfile(root_dir, 'universalUR5.urdf');

fprintf('============================================================\n');
fprintf('  Phase 2 Integration Test — CasADi Kinematics & Dynamics\n');
fprintf('  Robot: UR5 (%s)\n', urdf_file);
fprintf('============================================================\n\n');

% -------------------------------------------------------------------------
% 1. Build full pipeline (Phase 1 extraction)
% -------------------------------------------------------------------------
fprintf('--- Building Phase 1 extraction pipeline ---\n');
robot   = urdf_parse(urdf_file);
chain   = urdf_build_chain(robot, false);
dh      = urdf_extract_dh(chain, false);
inertia = urdf_extract_inertia(robot, chain, dh, false);
N       = chain.n;
fprintf('  Robot: %d-DOF serial chain extracted.\n\n', N);

% -------------------------------------------------------------------------
% 2. Build Phase 2 CasADi models
% -------------------------------------------------------------------------
fprintf('--- Building Phase 2 CasADi models ---\n');
fprintf('  [1/3] Building FK ...\n');
fk  = fk_casadi(dh, chain);

fprintf('  [2/3] Building Jacobian ...\n');
jac = jacobian_casadi(fk, dh, chain);

fprintf('  [3/3] Building Dynamics ...\n');
opts_dyn.verbose = true;
dyn = dynamics_casadi(dh, inertia, chain, opts_dyn);
fprintf('\n');

% -------------------------------------------------------------------------
% 3. Define test configurations
% -------------------------------------------------------------------------
% Use several configurations to test robustness (not just q=0).
% Each column of Q_test is one joint configuration [N x 1].
Q_test = [ ...
    zeros(N, 1), ...                                  % config 1: zero config
    [0.1; -0.5;  0.3; -0.2;  0.4; 0.1], ...          % config 2: generic A
    [pi/4; -pi/3; pi/6; -pi/4; pi/3; -pi/6], ...     % config 3: notable angles
    [-0.8; 1.2; -0.6; 0.9; -1.1; 0.5] ...            % config 4: large-angle
];
% Q_test is [N x 4] — each column is one test configuration
n_configs = size(Q_test, 2);

TOL_FK  = 1e-10;   % FK position/rotation tolerance (near-machine precision)
TOL_JAC = 1e-6;    % Jacobian tolerance (CasADi vs DH_jacobian.m)
TOL_FD  = 1e-5;    % Finite-difference Jacobian tolerance
TOL_DYN = 1e-6;    % Dynamics M, C, G tolerance vs DH_dyn_casadi.m

% =========================================================================
% TEST A — FK comparison: fk_casadi vs DH_fk.m
% =========================================================================
% fk_ee_fun returns T_0^N * chain.T_ee.
% DH_fk with P_point=[] returns T_0^N (no tool offset, just last-link frame).
% Since chain.T_ee = eye(4) for UR5 (no trailing fixed joints), both should match.
% =========================================================================
fprintf('--- Test A: FK comparison (fk_casadi vs DH_fk.m) ---\n');
test_A_pass = true;
max_pos_err = 0;
max_rot_err = 0;

for c = 1:n_configs
    % Extract the c-th column and ensure it is a column vector [N x 1]
    q_num = Q_test(:, c);   % [N x 1] double
    q_num = q_num(:);       % explicit column guarantee (safety)

    % --- New pipeline: fk_casadi ---
    % fk_ee_fun is a casadi.Function with one input 'q' [Nx1] and one output 'T0e' [4x4].
    % Calling it with positional syntax returns a casadi.DM object.
    % full() converts DM to a plain MATLAB double matrix.
    T0e_new = full(fk.fk_ee_fun(q_num));   % [4x4] double: T_0^EE
    p_new   = T0e_new(1:3, 4);             % [3x1] EE position
    R_new   = T0e_new(1:3, 1:3);           % [3x3] EE rotation

    % --- Legacy: DH_fk.m ---
    % Signature: [pose, T0e, Ro] = DH_fk(DHpram, q, joint_types, P_point)
    % DHpram has 5 columns [a, alpha, theta_off, d, sigma], so joint_types
    % is read from DHpram column 5 automatically (no need to pass joint_types).
    % P_point = [] → defaults to [0;0;0] (no tool offset, stop at last DH frame).
    [~, T0e_leg] = DH_fk(dh.DHpram, q_num, [], []);
    p_leg = T0e_leg(1:3, 4);   % [3x1]
    R_leg = T0e_leg(1:3, 1:3); % [3x3]

    % Compute position error (Euclidean norm of position vector difference)
    pos_err = norm(p_new - p_leg);
    max_pos_err = max(max_pos_err, pos_err);

    % Rotation error: Frobenius norm of rotation matrix difference
    % ||R_new - R_leg||_F should be ~0 if the FK is identical
    rot_err = norm(R_new - R_leg, 'fro');
    max_rot_err = max(max_rot_err, rot_err);

    if pos_err > TOL_FK || rot_err > TOL_FK
        fprintf('  [FAIL] Config %d: pos_err=%.2e m, rot_err=%.2e (Frob)\n', ...
                c, pos_err, rot_err);
        test_A_pass = false;
    end
end

if test_A_pass
    fprintf('  [PASS] FK matches DH_fk.m over %d configs.\n', n_configs);
    fprintf('         Max position error : %.2e m\n', max_pos_err);
    fprintf('         Max rotation error : %.2e (Frobenius norm)\n\n', max_rot_err);
else
    fprintf('  [FAIL] FK mismatch vs DH_fk.m.\n\n');
end

% =========================================================================
% TEST B — Jacobian comparison: jacobian_casadi vs DH_jacobian.m
% =========================================================================
% DH_jacobian.m uses the same DH convention (1=revolute) and builds the
% geometric Jacobian at the EE origin (P_point=[]).
% Since T_ee=eye(4) for UR5, the Jacobians should be identical.
% =========================================================================
fprintf('--- Test B: Jacobian comparison (jacobian_casadi vs DH_jacobian.m) ---\n');
test_B_pass = true;
max_J_err   = 0;

for c = 1:n_configs
    q_num = Q_test(:, c);
    q_num = q_num(:);   % ensure column vector

    % New: jacobian_casadi — J_fun maps q [Nx1] → J [6xN]
    J_new = full(jac.J_fun(q_num));   % [6xN] double

    % Legacy: DH_jacobian.m
    % Signature: [J, Jv, Jw, T0e, Ro] = DH_jacobian(DHpram, q, joint_types, P_point)
    % DHpram has 5 cols → joint_types from col 5; P_point=[] → origin of last frame.
    [J_leg, ~, ~] = DH_jacobian(dh.DHpram, q_num, [], []);   % [6xN]

    % Element-wise maximum absolute difference
    J_err = max(max(abs(J_new - J_leg)));
    max_J_err = max(max_J_err, J_err);

    if J_err > TOL_JAC
        fprintf('  [FAIL] Config %d: max Jacobian element error = %.4e  (tol=%.0e)\n', ...
                c, J_err, TOL_JAC);
        test_B_pass = false;
    end
end

if test_B_pass
    fprintf('  [PASS] Jacobian matches DH_jacobian.m over %d configs.\n', n_configs);
    fprintf('         Max element-wise error: %.2e\n\n', max_J_err);
else
    fprintf('  [FAIL] Jacobian mismatch vs DH_jacobian.m.\n\n');
end

% =========================================================================
% TEST C — Finite-difference Jacobian validation
% =========================================================================
% Numerically validates the analytical Jacobian and J-dot independently
% of any other reference implementation.
%
% Linear Jacobian column j:
%   Jv_fd(:,j) ≈ [p_ee(q + eps·e_j) - p_ee(q - eps·e_j)] / (2·eps)
%
% Angular Jacobian column j (via rotation matrix central differences):
%   dR ≈ [R(q+eps·e_j) - R(q-eps·e_j)] / (2·eps)
%   ω× = dR · R0'   (where R0 = R(q))
%   Extract: ω = [ω×(3,2); ω×(1,3); ω×(2,1)]
%
%   Note on sign convention for skew-symmetric matrix extraction:
%   If ω× = [  0  -ωz  ωy ]
%            [ ωz    0 -ωx ]
%            [-ωy   ωx   0 ]
%   Then: ωx = ω×(3,2), ωy = ω×(1,3), ωz = ω×(2,1)
%   (This is the vee-map extraction: ω = vee(ω×))
% =========================================================================
fprintf('--- Test C: Finite-difference Jacobian validation ---\n');
test_C_pass = true;
FD_EPS      = 1e-6;   % finite-difference step (1 µrad / µm)
max_fd_err  = 0;

% Test at config 2 (away from q=0 to avoid accidental symmetry)
q_test_fd = Q_test(:, 2);
q_test_fd = q_test_fd(:);

J_fd_pos = zeros(3, N);   % finite-difference linear Jacobian  [3xN]
J_fd_ang = zeros(3, N);   % finite-difference angular Jacobian [3xN]

% Compute base rotation at q_test_fd (used for angular FD)
T0_base = full(fk.fk_ee_fun(q_test_fd));
R0_base = T0_base(1:3, 1:3);   % [3x3] rotation matrix at test point

for j = 1:N
    % Unit perturbation vector along joint j
    e_j     = zeros(N, 1);
    e_j(j)  = 1;

    % Forward and backward perturbed EE transforms
    T_fwd = full(fk.fk_ee_fun(q_test_fd + FD_EPS * e_j));   % [4x4]
    T_bwd = full(fk.fk_ee_fun(q_test_fd - FD_EPS * e_j));   % [4x4]

    % --- Linear Jacobian column j ---
    % Central difference of EE position → linear velocity w.r.t. q_j perturbation
    J_fd_pos(:, j) = (T_fwd(1:3, 4) - T_bwd(1:3, 4)) / (2*FD_EPS);

    % --- Angular Jacobian column j ---
    % From kinematics: dR/dq_j = [ω_j]_× · R  → ω_j = vee([dR/dq_j · R'])
    % Central difference of rotation matrix w.r.t. q_j perturbation:
    dR_dqj = (T_fwd(1:3, 1:3) - T_bwd(1:3, 1:3)) / (2*FD_EPS);  % [3x3]

    % The skew-symmetric angular velocity matrix:
    %   omega_cross = dR/dq_j · R0'
    % Its entries satisfy: omega_cross = -omega_cross' (skew-symmetric)
    omega_cross = dR_dqj * R0_base';   % [3x3] skew-symmetric matrix

    % Vee-map extraction: map skew-symmetric matrix → 3-vector
    % For ω× = [  0  -ωz  ωy ]
    %          [ ωz    0 -ωx ]
    %          [-ωy   ωx   0 ]
    % ωx = ω×(3,2),  ωy = ω×(1,3),  ωz = ω×(2,1)
    J_fd_ang(:, j) = [omega_cross(3,2); omega_cross(1,3); omega_cross(2,1)];
end

% Stack FD Jacobian: [linear; angular] — same ordering as geometric Jacobian
J_fd_full = [J_fd_pos; J_fd_ang];   % [6xN] finite-difference Jacobian

% Compare analytical vs finite-difference Jacobian
J_an   = full(jac.J_fun(q_test_fd));           % [6xN] analytical
fd_err = max(max(abs(J_an - J_fd_full)));       % max element-wise error

if fd_err > TOL_FD
    fprintf('  [FAIL] Jacobian vs FD: max error = %.4e  (tol = %.0e)\n', fd_err, TOL_FD);
    test_C_pass = false;
else
    fprintf('  [PASS] Analytical Jacobian matches finite-diff (max err = %.2e, tol = %.0e).\n', ...
            fd_err, TOL_FD);
end

% J-dot finite-difference validation
% Computes J-dot by differencing J along the trajectory direction dq.
% Method: J_dot ≈ [J(q + dt*dq) - J(q - dt*dq)] / (2*dt)
% This uses the directional derivative: dJ/dt = sum_k (∂J/∂q_k) * dq_k
fprintf('  Validating J-dot via finite-difference ...\n');
dq_test  = [0.1; 0.2; -0.1; 0.05; 0.15; -0.05];   % arbitrary test velocity [N x 1]

% Analytical J-dot from jacobian_casadi
Jdot_an  = full(jac.Jdot_fun(q_test_fd, dq_test));  % [6xN] analytical J-dot

% Numerical J-dot: central-difference J along trajectory q(t) = q + t*dq
dt_fd    = 1e-5;                                                    % time step
J_fwd_t  = full(jac.J_fun(q_test_fd + dt_fd * dq_test));   % J(q + dt*dq) [6xN]
J_bwd_t  = full(jac.J_fun(q_test_fd - dt_fd * dq_test));   % J(q - dt*dq) [6xN]
Jdot_fd  = (J_fwd_t - J_bwd_t) / (2*dt_fd);                % [6xN] numerical J-dot

jdot_err = max(max(abs(Jdot_an - Jdot_fd)));
% J-dot FD is less accurate (double differentiation + truncation), so use looser tol
if jdot_err > TOL_FD * 50
    fprintf('  [FAIL] J-dot vs FD: max error = %.4e  (tol = %.0e)\n', jdot_err, TOL_FD*50);
    test_C_pass = false;
else
    fprintf('  [PASS] J-dot matches finite-diff (max err = %.2e, tol = %.0e).\n', ...
            jdot_err, TOL_FD*50);
end

if test_C_pass
    fprintf('  [PASS] Test C complete.\n\n');
else
    fprintf('  [FAIL] Test C complete.\n\n');
end

% =========================================================================
% TEST D — Dynamics comparison: dynamics_casadi vs DH_dyn_casadi.m
% =========================================================================
% Cross-validates M(q), C(q,dq), G(q) against the legacy DH_dyn_casadi.m.
%
% DH_dyn_casadi.m signature:
%   [M, C, G, Ro, fns] = DH_dyn_casadi(DHpram, P_Ci, q, dq, I, m, g [, ddq])
%
%   - DHpram: [Nx5] with sigma in column 5 (1=revolute)
%   - P_Ci:   [Nx3] CoM positions in DH frame (function transposes internally)
%   - q, dq:  symbolic or numeric joint state
%   - I:      [3x3xN] inertia tensors
%   - m:      [Nx1] link masses
%   - g:      [3x1] gravity vector
%
% M, C, G returned as CasADi SX expressions (evaluated at the given q/dq).
% Use full() to convert DM to double after evaluating at numeric values.
% =========================================================================
fprintf('--- Test D: Dynamics comparison (dynamics_casadi vs DH_dyn_casadi.m) ---\n');
test_D_pass = true;

% Choose a non-trivial configuration for dynamics testing
q_dyn  = Q_test(:, 2);   % generic config A [N x 1]
dq_dyn = [0.1; -0.2; 0.15; -0.1; 0.2; -0.05];   % non-zero velocities [N x 1]

% --- New pipeline: dynamics_casadi ---
% dyn.M_fun, C_fun, G_fun are casadi.Function objects compiled from symbolic exprs.
M_new = full(dyn.M_fun(q_dyn));          % [NxN] double
C_new = full(dyn.C_fun(q_dyn, dq_dyn)); % [NxN] double
G_new = full(dyn.G_fun(q_dyn));          % [Nx1] double

% --- Legacy: DH_dyn_casadi.m ---
% Pass empty q and dq so DH_dyn_casadi creates purely symbolic variables internally
% and returns compiled Function objects in fns_leg.
[~, ~, ~, ~, fns_leg] = DH_dyn_casadi( ...
    dh.DHpram, ...       % [Nx5]: DH params with sigma in col 5
    inertia.P_Ci, ...    % [Nx3]: CoM in DH frame (legacy transposes internally)
    [], ...              % empty -> creates symbolic SX q internally
    [], ...              % empty -> creates symbolic SX dq internally
    inertia.I, ...       % [3x3xN]: inertia tensors
    inertia.m, ...       % [Nx1]: link masses
    [0; 0; -9.81] ...    % [3x1]: gravity vector
);

% Evaluate compiled legacy functions at numeric test state (q_dyn, dq_dyn)
M_leg = full(fns_leg.M_fun(q_dyn));           % [NxN] double
C_leg = full(fns_leg.C_fun(q_dyn, dq_dyn));   % [NxN] double
G_leg = full(fns_leg.G_fun(q_dyn));           % [Nx1] double

% Compare element-wise
M_err = max(max(abs(M_new - M_leg)));
C_err = max(max(abs(C_new - C_leg)));
G_err = max(abs(G_new  - G_leg));

if M_err > TOL_DYN
    fprintf('  [FAIL] M mismatch vs DH_dyn_casadi: max err = %.4e  (tol=%.0e)\n', M_err, TOL_DYN);
    test_D_pass = false;
else
    fprintf('  [PASS] M(q) matches DH_dyn_casadi (max err = %.2e).\n', M_err);
end
if C_err > TOL_DYN
    fprintf('  [FAIL] C mismatch vs DH_dyn_casadi: max err = %.4e  (tol=%.0e)\n', C_err, TOL_DYN);
    test_D_pass = false;
else
    fprintf('  [PASS] C(q,dq) matches DH_dyn_casadi (max err = %.2e).\n', C_err);
end
if G_err > TOL_DYN
    fprintf('  [FAIL] G mismatch vs DH_dyn_casadi: max err = %.4e  (tol=%.0e)\n', G_err, TOL_DYN);
    test_D_pass = false;
else
    fprintf('  [PASS] G(q) matches DH_dyn_casadi (max err = %.2e).\n', G_err);
end

if test_D_pass
    fprintf('  [PASS] Test D complete.\n\n');
else
    fprintf('  [FAIL] Test D complete.\n\n');
end

% =========================================================================
% TEST E — Physical sanity checks on dynamics
% =========================================================================
% Physical constraints that M(q) must satisfy for any physically-consistent
% robot model:
%   1. Symmetry:           M = M'  (follows from Lagrangian formulation)
%   2. Positive definite:  all eigenvalues of M > 0 (kinetic energy > 0 for dq≠0)
%   3. Static torque:      tau(q, 0, 0) = G(q)  (no motion → only gravity)
% =========================================================================
fprintf('--- Test E: Physical sanity checks ---\n');
test_E_pass = true;

for c = 1:n_configs
    q_num = Q_test(:, c);
    q_num = q_num(:);

    % --- Check 1: M must be symmetric ---
    M_num   = full(dyn.M_fun(q_num));   % [NxN] double
    sym_err = max(max(abs(M_num - M_num')));
    if sym_err > 1e-10
        fprintf('  [FAIL] Config %d: M not symmetric (sym_err = %.2e)\n', c, sym_err);
        test_E_pass = false;
    end

    % --- Check 2: M must be positive definite ---
    eigs_M = eig(M_num);                % eigenvalues of M
    min_eig = min(eigs_M);
    if min_eig <= 0
        fprintf('  [FAIL] Config %d: M not PD (min eigenvalue = %.4e)\n', c, min_eig);
        test_E_pass = false;
    end

    % --- Check 3: tau(q, 0, 0) must equal G(q) ---
    % At zero velocity and acceleration, the EOM reduces to:
    %   tau = M(q)·0 + C(q,0)·0 + G(q) = G(q)
    G_num   = full(dyn.G_fun(q_num));
    tau_num = full(dyn.tau_fun(q_num, zeros(N,1), zeros(N,1)));
    tau_G_err = max(abs(tau_num - G_num));
    if tau_G_err > 1e-10
        fprintf('  [FAIL] Config %d: tau(q,0,0) ≠ G(q) (max err = %.2e)\n', c, tau_G_err);
        test_E_pass = false;
    end
end

if test_E_pass
    fprintf('  [PASS] M symmetric PD and tau(q,0,0)=G(q) at all %d configs.\n\n', n_configs);
else
    fprintf('  [FAIL] Physical sanity check failed.\n\n');
end

% =========================================================================
% TEST F — traj_poly5 boundary conditions and shape check
% =========================================================================
% A 5th-order polynomial trajectory must satisfy:
%   q(0)    = q0,     q(T)    = q1    (position boundary conditions)
%   dq(0)   = 0,      dq(T)   = 0     (zero-velocity boundary conditions)
%   ddq(0)  = 0,      ddq(T)  = 0     (zero-acceleration boundary conditions)
%
% Also validates that the numerical derivative of q matches dq.
% =========================================================================
fprintf('--- Test F: traj_poly5 utility function ---\n');
test_F_pass = true;

q0_f = zeros(N, 1);
q1_f = [0.5; -0.3; 0.4; -0.2; 0.6; -0.1];
T_f  = 3.0;
t_f  = linspace(0, T_f, 10000);

[q_f, dq_f, ddq_f] = traj_poly5(q0_f, q1_f, T_f, t_f);

% -- Boundary check: q(t=0) = q0, q(t=T) = q1 ---
if norm(q_f(:,1)   - q0_f) > 1e-12
    fprintf('  [FAIL] q(0) ≠ q0: err = %.2e\n', norm(q_f(:,1) - q0_f));
    test_F_pass = false;
end
if norm(q_f(:,end) - q1_f) > 1e-12
    fprintf('  [FAIL] q(T) ≠ q1: err = %.2e\n', norm(q_f(:,end) - q1_f));
    test_F_pass = false;
end

% -- Zero velocity at boundaries: dq(0) = 0, dq(T) = 0 ---
if norm(dq_f(:,1)  ) > 1e-12
    fprintf('  [FAIL] dq(0) ≠ 0: err = %.2e\n', norm(dq_f(:,1)));
    test_F_pass = false;
end
if norm(dq_f(:,end)) > 1e-10   % slightly looser at t=T due to polynomial rounding
    fprintf('  [FAIL] dq(T) ≠ 0: err = %.2e\n', norm(dq_f(:,end)));
    test_F_pass = false;
end

% -- Zero acceleration at boundaries: ddq(0) = 0, ddq(T) = 0 ---
if norm(ddq_f(:,1)  ) > 1e-12
    fprintf('  [FAIL] ddq(0) ≠ 0: err = %.2e\n', norm(ddq_f(:,1)));
    test_F_pass = false;
end
if norm(ddq_f(:,end)) > 1e-10
    fprintf('  [FAIL] ddq(T) ≠ 0: err = %.2e\n', norm(ddq_f(:,end)));
    test_F_pass = false;
end

% -- Finite-difference consistency: dq ≈ d(q)/dt in the mid-range ---
% Use central differences on q values at mid-range indices (away from endpoints).
% dq_fd(:,k) ≈ (q(:,k+1) - q(:,k-1)) / (2*dt)
t_mid = 100:5000;   % mid-range indices
dt    = t_f(2) - t_f(1);                              % time step [s]
% Forward diff (simpler, less accurate, but good for shape check):
dq_fd     = diff(q_f(:, t_mid), 1, 2) / dt;           % [N x (len(t_mid)-1)]
dq_an_mid = dq_f(:, t_mid(1:end-1));                  % analytical dq at same indices
fd_traj_err = max(max(abs(dq_fd - dq_an_mid)));
if fd_traj_err > 1e-4
    fprintf('  [FAIL] traj dq vs FD: max err = %.4e  (tol = 1e-4)\n', fd_traj_err);
    test_F_pass = false;
end

if test_F_pass
    fprintf('  [PASS] traj_poly5: boundary conditions and shape verified.\n\n');
else
    fprintf('  [FAIL] traj_poly5 check failed.\n\n');
end

% =========================================================================
% SUMMARY
% =========================================================================
all_passed = test_A_pass && test_B_pass && test_C_pass && ...
             test_D_pass && test_E_pass && test_F_pass;

fprintf('============================================================\n');
fprintf('  Results Summary:\n');
fprintf('  A (FK vs DH_fk.m)           : %s\n', pass_str(test_A_pass));
fprintf('  B (Jacobian vs DH_jacobian)  : %s\n', pass_str(test_B_pass));
fprintf('  C (FD Jacobian + Jdot)       : %s\n', pass_str(test_C_pass));
fprintf('  D (Dynamics vs DH_dyn_casadi): %s\n', pass_str(test_D_pass));
fprintf('  E (Physical sanity — M,G)    : %s\n', pass_str(test_E_pass));
fprintf('  F (traj_poly5 boundaries)    : %s\n', pass_str(test_F_pass));
fprintf('------------------------------------------------------------\n');
if all_passed
    fprintf('  >>> ALL TESTS PASSED <<<\n');
else
    fprintf('  >>> SOME TESTS FAILED — see details above <<<\n');
end
fprintf('============================================================\n');

assert(all_passed, 'Phase 2 integration tests failed.');

% =========================================================================
%  LOCAL HELPER
% =========================================================================
function s = pass_str(flag)
    % Returns 'PASS' or 'FAIL' string for the summary table.
    if flag, s = 'PASS'; else, s = 'FAIL'; end
end
