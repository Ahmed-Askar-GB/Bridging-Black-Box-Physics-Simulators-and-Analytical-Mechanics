function dyn = dynamics_matlab(dh, inertia, chain, opts)
% DYNAMICS_MATLAB  Pure MATLAB Lagrangian and Newton-Euler robot dynamics.
%
% Computes:
%   - Generalized Mass Matrix M(q)
%   - Coriolis and Centrifugal Matrix C(q, dq) and vector C(q, dq)*dq
%   - Gravity Vector G(q)
%   - Inverse Dynamics tau(q, dq, ddq) = M(q)*ddq + C(q, dq)*dq + G(q)
%   - Forward Dynamics ddq = forward_dyn_fun(q, dq, tau) = M(q)^{-1} * (tau - C*dq - G)
%
% Uses 100% pure MATLAB without external toolboxes or CasADi dependencies.
%
% =========================================================================

    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    g_vec   = [0; 0; -9.81]; if isfield(opts, 'g_vec'),   g_vec   = opts.g_vec(:); end
    verbose = false;         if isfield(opts, 'verbose'), verbose = opts.verbose; end

    n     = chain.n;
    DHp   = dh.DHpram;   % [Nx5]: [a, alpha, theta_off, d, sigma]
    sigma = dh.sigma;
    m_vec = inertia.m(:);
    P_Ci  = inertia.P_Ci;

    % Normalize inertia tensors to [3x3xN]
    I_tensors = zeros(3, 3, n);
    for i = 1:n
        if iscell(inertia.I)
            I_tensors(:, :, i) = inertia.I{i};
        elseif ndims(inertia.I) == 3 && size(inertia.I, 3) == n
            I_tensors(:, :, i) = inertia.I(:, :, i);
        elseif ndims(inertia.I) == 3 && size(inertia.I, 1) == n
            I_tensors(:, :, i) = squeeze(inertia.I(i, :, :));
        else
            I_tensors(:, :, i) = inertia.I(:, :, i);
        end
    end

    % -------------------------------------------------------------------------
    % Helper: Cumulative transforms and frame origins/axes for state q
    % -------------------------------------------------------------------------
    function [T0i, z_axes, O_all, p_Ci_all] = compute_kinematic_frames(q)
        q = double(q(:));
        T0i      = cell(n, 1);
        z_axes   = zeros(3, n);
        O_all    = zeros(3, n);
        p_Ci_all = zeros(3, n);

        % Base frame seed
        z_axes(:, 1) = [0; 0; 1];
        O_all(:, 1)  = [0; 0; 0];

        T_running = eye(4);
        for i = 1:n
            a_i     = DHp(i, 1);
            alpha_i = DHp(i, 2);
            th_off  = DHp(i, 3);
            d_off   = DHp(i, 4);
            sigma_i = DHp(i, 5);

            if sigma_i == 1
                theta_i = th_off + q(i);
                d_i     = d_off;
            else
                theta_i = th_off;
                d_i     = d_off + q(i);
            end

            ct = cos(theta_i); st = sin(theta_i);
            ca = cos(alpha_i); sa = sin(alpha_i);

            A_i = [ct,  -st*ca,   st*sa,   a_i*ct;
                   st,   ct*ca,  -ct*sa,   a_i*st;
                    0,      sa,      ca,      d_i;
                    0,       0,       0,       1 ];

            T_running = T_running * A_i;
            T0i{i}    = T_running;

            if i >= 2
                z_axes(:, i) = T0i{i-1}(1:3, 3);
                O_all(:, i)  = T0i{i-1}(1:3, 4);
            end

            % Global CoM position of link i
            R_i = T_running(1:3, 1:3);
            if i == 1
                p_Ci_all(:, 1) = R_i * P_Ci(1, :)';
            else
                p_Ci_all(:, i) = O_all(:, i) + R_i * P_Ci(i, :)';
            end
        end
    end

    % -------------------------------------------------------------------------
    % 1. Per-Link CoM Jacobians
    % -------------------------------------------------------------------------
    function [Jv_all, Jw_all] = compute_link_jacobians(q)
        [~, z_axes, O_all, p_Ci_all] = compute_kinematic_frames(q);
        Jv_all = cell(n, 1);
        Jw_all = cell(n, 1);

        for i = 1:n
            Jv_i = zeros(3, n);
            Jw_i = zeros(3, n);
            p_Ci = p_Ci_all(:, i);

            for j = 1:i
                z_j_prev = z_axes(:, j);
                O_j_prev = O_all(:, j);

                if sigma(j) == 1
                    Jv_i(:, j) = cross(z_j_prev, p_Ci - O_j_prev);
                    Jw_i(:, j) = z_j_prev;
                else
                    Jv_i(:, j) = z_j_prev;
                    Jw_i(:, j) = zeros(3, 1);
                end
            end
            Jv_all{i} = Jv_i;
            Jw_all{i} = Jw_i;
        end
    end

    % -------------------------------------------------------------------------
    % 2. Generalized Mass Matrix M(q)
    % -------------------------------------------------------------------------
    function M = compute_mass_matrix(q)
        q = double(q(:));
        [T0i, ~, ~, ~] = compute_kinematic_frames(q);
        [Jv_all, Jw_all] = compute_link_jacobians(q);

        M = zeros(n, n);
        for i = 1:n
            m_i   = m_vec(i);
            R_i   = T0i{i}(1:3, 1:3);
            I_i_0 = R_i * I_tensors(:, :, i) * R_i';  % rotated to base frame

            Jv_i = Jv_all{i};
            Jw_i = Jw_all{i};

            M = M + m_i * (Jv_i' * Jv_i) + (Jw_i' * I_i_0 * Jw_i);
        end
        % Ensure exact numerical symmetry
        M = 0.5 * (M + M');
    end

    % -------------------------------------------------------------------------
    % 3. Gravity Vector G(q)
    % -------------------------------------------------------------------------
    function G = compute_gravity_vector(q)
        q = double(q(:));
        [Jv_all, ~] = compute_link_jacobians(q);

        G = zeros(n, 1);
        for i = 1:n
            G = G - m_vec(i) * (Jv_all{i}' * g_vec);
        end
    end

    % -------------------------------------------------------------------------
    % 4. Coriolis & Centrifugal Matrix C(q, dq) via numerical Christoffel
    % -------------------------------------------------------------------------
    function C = compute_coriolis_matrix(q, dq)
        q  = double(q(:));
        dq = double(dq(:));
        eps_step = 1e-7;

        % Compute partial derivatives dM/dq_k using central differences
        dM_dq = zeros(n, n, n);
        for k = 1:n
            q_plus  = q; q_plus(k)  = q_plus(k) + eps_step;
            q_minus = q; q_minus(k) = q_minus(k) - eps_step;

            M_plus  = compute_mass_matrix(q_plus);
            M_minus = compute_mass_matrix(q_minus);

            dM_dq(:, :, k) = (M_plus - M_minus) / (2.0 * eps_step);
        end

        % Build C matrix via Christoffel symbols:
        % c_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{kj}/dq_i)
        % C_{ij}  = sum_k c_{ijk} * dq_k
        C = zeros(n, n);
        for i = 1:n
            for j = 1:n
                c_ij = 0;
                for k = 1:n
                    c_ijk = 0.5 * (dM_dq(i, j, k) + dM_dq(i, k, j) - dM_dq(k, j, i));
                    c_ij  = c_ij + c_ijk * dq(k);
                end
                C(i, j) = c_ij;
            end
        end
    end

    % -------------------------------------------------------------------------
    % 5. Momentum Coriolis Bias: c_bias = d(M*dq)/dt - 0.5*grad_q(dq'*M*dq)
    % -------------------------------------------------------------------------
    function c_bias = compute_c_bias(q, dq)
        q  = double(q(:));
        dq = double(dq(:));
        eps_step = 1e-7;

        % Total time derivative d(M*dq)/dt via finite difference along dq
        M_curr = compute_mass_matrix(q);
        p_curr = M_curr * dq;

        q_next_step = q + eps_step * dq;
        M_next_step = compute_mass_matrix(q_next_step);
        p_next_step = M_next_step * dq;
        dp_dt = (p_next_step - p_curr) / eps_step;

        % Gradient of kinetic energy w.r.t. q
        grad_T = zeros(n, 1);
        for k = 1:n
            qk_p = q; qk_p(k) = qk_p(k) + eps_step;
            qk_m = q; qk_m(k) = qk_m(k) - eps_step;
            T_p = 0.5 * (dq' * compute_mass_matrix(qk_p) * dq);
            T_m = 0.5 * (dq' * compute_mass_matrix(qk_m) * dq);
            grad_T(k) = (T_p - T_m) / (2.0 * eps_step);
        end

        c_bias = dp_dt - grad_T;
    end

    % -------------------------------------------------------------------------
    % 6. Inverse Dynamics: tau = M(q)*ddq + c_bias(q, dq) + G(q)
    % -------------------------------------------------------------------------
    function tau = compute_inverse_dynamics(q, dq, ddq)
        q   = double(q(:));
        dq  = double(dq(:));
        ddq = double(ddq(:));

        M = compute_mass_matrix(q);
        c_bias = compute_c_bias(q, dq);
        G = compute_gravity_vector(q);

        tau = M * ddq + c_bias + G;
    end

    % -------------------------------------------------------------------------
    % 7. Forward Dynamics: ddq = R \ (R' \ (tau - c_bias - G)) via Cholesky
    % -------------------------------------------------------------------------
    function ddq = compute_forward_dynamics(q, dq, tau)
        q   = double(q(:));
        dq  = double(dq(:));
        tau = double(tau(:));

        M = compute_mass_matrix(q);
        c_bias = compute_c_bias(q, dq);
        G = compute_gravity_vector(q);
        bias_f = tau - c_bias - G;

        try
            R = chol(M);
            ddq = R \ (R' \ bias_f);
        catch
            ddq = M \ bias_f;
        end
    end

    % Package into output struct
    dyn = struct();
    dyn.n               = n;
    dyn.M_fun           = @(q) compute_mass_matrix(q);
    dyn.C_fun           = @(q, dq) compute_coriolis_matrix(q, dq);
    dyn.c_bias_fun      = @(q, dq) compute_c_bias(q, dq);
    dyn.G_fun           = @(q) compute_gravity_vector(q);
    dyn.tau_fun         = @(q, dq, ddq) compute_inverse_dynamics(q, dq, ddq);
    dyn.forward_dyn_fun = @(q, dq, tau) compute_forward_dynamics(q, dq, tau);

    if verbose
        fprintf('  [OK] Pure MATLAB Dynamics compiled (%d DOFs).\n', n);
    end
end
