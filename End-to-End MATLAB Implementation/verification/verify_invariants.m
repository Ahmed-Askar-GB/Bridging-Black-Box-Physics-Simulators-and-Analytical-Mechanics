function [results, all_passed] = verify_invariants(dh, inertia, fk, dyn, opts)
% VERIFY_INVARIANTS  Validate physical realizability and mathematical invariants.
%
% Tests:
%   1. Mass positivity: m_i > 0 for all links.
%   2. Physical realizability: Rotational inertia tensors satisfy the triangle
%      inequality on principal moments: lambda_1 + lambda_2 >= lambda_3.
%   3. Lie-Group SO(3) & Quaternion: Rotations satisfy R*R' = I, det(R) = +1,
%      and unit quaternion normalization ||q|| = 1.
%   4. Generalized Mass Matrix M(q): Strictly symmetric positive-definite (SPD)
%      with min(eig(M)) > 0 across multiple random configurations.
%   5. Static balance invariant: tau(q, 0, 0) == G(q).
%
% =========================================================================
% INPUTS:
%   dh      (struct) Output from urdf_extract_dh().
%   inertia (struct) Output from urdf_extract_inertia().
%   fk      (struct) Output from fk_casadi().
%   dyn     (struct) Output from dynamics_casadi().
%   opts    (struct) [optional] Options:
%                      .num_samples (int)     Number of random poses (default: 20)
%                      .tol         (double)  General numerical tolerance (default: 1e-7)
%                      .verbose     (logical) Print summary to console (default: true)
%
% OUTPUTS:
%   results    (struct)  Detailed boolean flags and residuals for each invariant.
%   all_passed (logical) True if all tests passed without violation.
% =========================================================================

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    num_samples = 20;   if isfield(opts, 'num_samples'), num_samples = opts.num_samples; end
    tol         = 1e-7; if isfield(opts, 'tol'),         tol         = opts.tol; end
    verbose     = true; if isfield(opts, 'verbose'),     verbose     = opts.verbose; end

    n = size(dh.table, 1);
    results = struct();

    if verbose
        fprintf('\n====================================================================\n');
        fprintf('       MATHEMATICAL INVARIANTS & PHYSICAL REALIZABILITY AUDIT        \n');
        fprintf('====================================================================\n');
    end

    % -------------------------------------------------------------------------
    % 1. Link Mass Positivity
    % -------------------------------------------------------------------------
    m_vec = inertia.m(:);
    results.mass_positivity = all(m_vec > 0);
    if verbose
        if results.mass_positivity
            fprintf('  [PASS] Mass Positivity: All %d links have positive mass (min: %.4f kg).\n', ...
                    n, min(m_vec));
        else
            fprintf('  [FAIL] Mass Positivity: Non-positive mass detected!\n');
        end
    end

    % -------------------------------------------------------------------------
    % 2. Inertia Triangle Inequality (Physical Realizability)
    % -------------------------------------------------------------------------
    % For any real physical rigid body, principal moments must satisfy:
    %   I1 + I2 >= I3,  I1 + I3 >= I2,  I2 + I3 >= I1
    triangle_ok = true;
    max_triangle_violation = 0;

    for i = 1:n
        if iscell(inertia.I)
            I_i = inertia.I{i};
        elseif ndims(inertia.I) == 3 && size(inertia.I, 3) == n
            I_i = inertia.I(:, :, i);
        elseif ndims(inertia.I) == 3 && size(inertia.I, 1) == n
            I_i = squeeze(inertia.I(i, :, :));
        else
            I_i = inertia.I(:, :, i);
        end

        eigs = sort(eig(I_i)); % eigs(1) <= eigs(2) <= eigs(3)
        violation = eigs(3) - (eigs(1) + eigs(2));
        if violation > max_triangle_violation
            max_triangle_violation = violation;
        end
        if violation > tol
            triangle_ok = false;
        end
    end
    results.triangle_inequality = triangle_ok;
    results.max_triangle_violation = max_triangle_violation;

    if verbose
        if triangle_ok
            fprintf('  [PASS] Inertia Triangle Inequality: Verified on all links (max residual: %.2e).\n', ...
                    max_triangle_violation);
        else
            fprintf('  [FAIL] Inertia Triangle Inequality: Physical violation detected (residual: %.2e)!\n', ...
                    max_triangle_violation);
        end
    end

    % -------------------------------------------------------------------------
    % 3. Lie-Group SO(3) & Quaternion Normalization
    % -------------------------------------------------------------------------
    so3_ok = true;
    max_ortho_err = 0;
    max_det_err   = 0;
    max_quat_err  = 0;

    rng(42); % Deterministic seed
    for k = 1:num_samples
        q_rand = -pi + 2*pi * rand(n, 1);
        [~, R_ee, quat_ee] = fk.fk_fun(q_rand);
        R_val    = full(R_ee);
        quat_val = full(quat_ee);

        ortho_err = norm(R_val * R_val' - eye(3), 'fro');
        det_err   = abs(det(R_val) - 1.0);
        quat_err  = abs(norm(quat_val) - 1.0);

        if ortho_err > max_ortho_err, max_ortho_err = ortho_err; end
        if det_err > max_det_err,     max_det_err   = det_err; end
        if quat_err > max_quat_err,   max_quat_err  = quat_err; end

        if ortho_err > tol || det_err > tol || quat_err > tol
            so3_ok = false;
        end
    end
    results.fk_so3_orthogonality = so3_ok;
    results.max_ortho_err = max_ortho_err;
    results.max_det_err   = max_det_err;
    results.max_quat_err  = max_quat_err;

    if verbose
        if so3_ok
            fprintf('  [PASS] Lie-Group SO(3) & Quaternions: R*R''=I, det(R)=1, ||q||=1 (ortho: %.2e, det: %.2e, quat: %.2e).\n', ...
                    max_ortho_err, max_det_err, max_quat_err);
        else
            fprintf('  [FAIL] Lie-Group SO(3) violation detected!\n');
        end
    end

    % -------------------------------------------------------------------------
    % 4. Generalized Mass Matrix M(q) Symmetry & Positive-Definiteness (SPD)
    % -------------------------------------------------------------------------
    spd_ok = true;
    max_sym_err = 0;
    min_eig_val = inf;

    for k = 1:num_samples
        q_rand = -pi + 2*pi * rand(n, 1);
        M_val   = full(dyn.M_fun(q_rand));
        sym_err = norm(M_val - M_val', 'fro');
        eigs_M  = eig(M_val);
        cur_min = min(eigs_M);

        if sym_err > max_sym_err, max_sym_err = sym_err; end
        if cur_min < min_eig_val, min_eig_val = cur_min; end

        if sym_err > tol || cur_min <= 0
            spd_ok = false;
        end
    end
    results.mass_matrix_spd = spd_ok;
    results.max_sym_err     = max_sym_err;
    results.min_eigenvalue  = min_eig_val;

    if verbose
        if spd_ok
            fprintf('  [PASS] Mass Matrix SPD: M(q) is strictly symmetric positive-definite (sym: %.2e, min eig: %.4e).\n', ...
                    max_sym_err, min_eig_val);
        else
            fprintf('  [FAIL] Mass Matrix is not strictly SPD!\n');
        end
    end

    % -------------------------------------------------------------------------
    % 5. Static Gravitational Balance: tau(q, 0, 0) == G(q)
    % -------------------------------------------------------------------------
    static_ok = true;
    max_grav_err = 0;

    for k = 1:num_samples
        q_rand = -pi + 2*pi * rand(n, 1);
        G_val   = full(dyn.G_fun(q_rand));
        tau_val = full(dyn.tau_fun(q_rand, zeros(n, 1), zeros(n, 1)));
        err_g   = norm(tau_val - G_val, inf);

        if err_g > max_grav_err, max_grav_err = err_g; end
        if err_g > tol
            static_ok = false;
        end
    end
    results.static_balance   = static_ok;
    results.max_gravity_err  = max_grav_err;

    if verbose
        if static_ok
            fprintf('  [PASS] Static Gravitational Balance: tau(q,0,0) == G(q) (max error: %.2e N.m).\n', ...
                    max_grav_err);
        else
            fprintf('  [FAIL] Static balance violation detected (error: %.2e N.m)!\n', max_grav_err);
        end
        fprintf('====================================================================\n\n');
    end

    all_passed = results.mass_positivity && ...
                 results.triangle_inequality && ...
                 results.fk_so3_orthogonality && ...
                 results.mass_matrix_spd && ...
                 results.static_balance;

end
