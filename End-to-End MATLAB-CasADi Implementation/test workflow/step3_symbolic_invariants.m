function [fk, jac, dyn, joint_signs] = step3_symbolic_invariants(dh, chain, inertia, cfg)
% STEP3_SYMBOLIC_INVARIANTS  Compile CasADi kinematics/dynamics & verify mathematical invariants.
%
% Performs:
%   1. Symbolic compilation of Forward Kinematics, Jacobians, and Lagrangian Dynamics.
%   2. Autonomous detection of joint actuation signs.
%   3. Invariant check: Mass matrix M(q) strict symmetry (|M - M^T| < tol.sym).
%   4. Invariant check: Mass matrix M(q) strict positive definiteness (min eig > 0).
%   5. Invariant check: Static gravitational balance (|tau(q,0,0) - G(q)| < tol.gravity).
%
% =========================================================================
% INPUTS:
%   dh          (struct) DH parameters from step2_extract_parameters.
%   chain       (struct) Kinematic chain from step1_parse_chain.
%   inertia     (struct) Inertia parameters from step2_extract_parameters.
%   cfg         (struct) Configuration struct with active tolerances.
%
% OUTPUTS:
%   fk          (struct) CasADi Forward Kinematics function objects.
%   jac         (struct) CasADi Jacobian function objects.
%   dyn         (struct) CasADi Lagrangian dynamic function objects.
%   joint_signs [Nx1]    Actuation sign vector (+1 or -1 per joint).
% =========================================================================

    fprintf('\n>>> STEP 3: Symbolic Kinematics & Lagrangian Dynamic Invariants ...\n');
    n = chain.n;

    fk  = fk_casadi(dh, chain, struct('verbose', false));
    jac = jacobian_casadi(fk, dh, chain, struct('verbose', false));
    dyn = dynamics_casadi(dh, inertia, chain, struct('verbose', false));

    joint_signs = detect_joint_signs(chain, fk);
    fprintf('  Auto-detected joint signs: %s\n', mat2str(joint_signs));

    % Comprehensive Invariants & Physical Realizability Audit
    inv_opts = struct('num_samples', 10, 'tol', cfg.tol.sym, 'verbose', true);
    [inv_results, all_passed] = verify_invariants(dh, inertia, fk, dyn, inv_opts);
    assert(all_passed, 'step3_symbolic_invariants:InvariantCheckFailed', ...
           'One or more physical/mathematical invariants failed verification.');

end
