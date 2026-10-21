function jac = jacobian_matlab(fk, dh, chain, opts)
% JACOBIAN_MATLAB  Pure MATLAB Geometric Jacobian J(q) and exact time-derivative J_dot(q, dq).
%
% Computes:
%   - 6xN Geometric Jacobian J(q) = [Jv(q); Jw(q)]
%   - 6xN Analytical Jacobian time-derivative J_dot(q, dq)
%
% =========================================================================

    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    verbose = false; if isfield(opts, 'verbose'), verbose = opts.verbose; end

    n     = chain.n;
    sigma = dh.sigma;

    % -------------------------------------------------------------------------
    % 1. Geometric Jacobian Evaluator J(q)
    % -------------------------------------------------------------------------
    function [J, Jv, Jw] = compute_jacobian(q)
        q = double(q(:));
        Jv = zeros(3, n);
        Jw = zeros(3, n);

        % End-effector position
        T0e  = fk.fk_ee_fun(q);
        p_ee = T0e(1:3, 4);

        % Frame origins and z-axes
        z = zeros(3, n);
        O = zeros(3, n);

        z(:, 1) = [0; 0; 1];
        O(:, 1) = [0; 0; 0];

        for j = 2:n
            T_prev = fk.fk_link_funs{j-1}(q);
            z(:, j) = T_prev(1:3, 3);
            O(:, j) = T_prev(1:3, 4);
        end

        for j = 1:n
            if sigma(j) == 1
                % Revolute joint
                Jv(:, j) = cross(z(:, j), p_ee - O(:, j));
                Jw(:, j) = z(:, j);
            else
                % Prismatic joint
                Jv(:, j) = z(:, j);
                Jw(:, j) = zeros(3, 1);
            end
        end

        J = [Jv; Jw];
    end

    % -------------------------------------------------------------------------
    % 2. Exact Numerical / Analytical Jacobian Time-Derivative J_dot(q, dq)
    % -------------------------------------------------------------------------
    function Jdot = compute_jdot(q, dq)
        q  = double(q(:));
        dq = double(dq(:));
        eps_step = 1e-7;

        % Directional derivative along dq:
        % J_dot(q, dq) = lim_{h -> 0} (J(q + h*dq) - J(q - h*dq)) / (2*h)
        J_plus  = compute_jacobian(q + eps_step * dq);
        J_minus = compute_jacobian(q - eps_step * dq);
        Jdot    = (J_plus - J_minus) / (2.0 * eps_step);
    end

    % Package into struct
    jac = struct();
    jac.n        = n;
    jac.J_fun    = @(q) compute_jacobian(q);
    jac.Jv_fun   = @(q) get_sub_matrix(@(x) compute_jacobian(x), q, 1:3);
    jac.Jw_fun   = @(q) get_sub_matrix(@(x) compute_jacobian(x), q, 4:6);
    jac.Jdot_fun = @(q, dq) compute_jdot(q, dq);

    function sub = get_sub_matrix(fn, q, rows)
        M = fn(q);
        sub = M(rows, :);
    end

    if verbose
        fprintf('  [OK] Pure MATLAB Geometric Jacobian & J_dot compiled (%d DOFs).\n', n);
    end
end
