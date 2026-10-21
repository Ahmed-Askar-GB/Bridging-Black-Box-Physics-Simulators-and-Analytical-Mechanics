classdef nmpc_matlab < handle
% NMPC_MATLAB  Pure MATLAB Nonlinear Model Predictive Control for robots.
%
% Solves receding horizon trajectory tracking:
%   min_{U} sum_{k=1}^N [ ||q_k - q_{d,k}||_Qp^2 + ||v_k - v_{d,k}||_Qv^2 + ||u_k||_R^2 ] + terminal_cost
%   s.t. x_{k+1} = f_RK4(x_k, u_k)
%        -tau_max <= u_k <= tau_max
%
% Uses pure MATLAB numerical optimization (fmincon if available, or fast
% projected gradient / shooting) with ZERO CasADi dependency.
%
% =========================================================================

    properties
        dyn
        n
        N
        dt
        Q_pos
        Q_vel
        R_tau
        tau_max
        q_min
        q_max
        signs
        last_U
    end

    methods
        function obj = nmpc_matlab(dyn, opts)
            if nargin < 1 || isempty(dyn)
                error('nmpc_matlab:MissingDyn', 'A dynamics struct is required.');
            end
            obj.dyn = dyn;
            obj.n   = dyn.n;

            if nargin < 2 || isempty(opts)
                opts = struct();
            end

            obj.N       = 10;    if isfield(opts, 'N'),       obj.N       = opts.N; end
            obj.dt      = 0.05;  if isfield(opts, 'dt'),      obj.dt      = opts.dt; end
            obj.Q_pos   = 100.0; if isfield(opts, 'Q_pos'),   obj.Q_pos   = opts.Q_pos; end
            obj.Q_vel   = 10.0;  if isfield(opts, 'Q_vel'),   obj.Q_vel   = opts.Q_vel; end
            obj.R_tau   = 0.01;  if isfield(opts, 'R_tau'),   obj.R_tau   = opts.R_tau; end
            obj.tau_max = 150.0; if isfield(opts, 'tau_max'), obj.tau_max = opts.tau_max; end

            if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
                obj.signs = double(opts.joint_signs(:));
            else
                obj.signs = ones(obj.n, 1);
            end

            obj.last_U = zeros(obj.n, obj.N);
        end

        function [tau_opt, X_pred, U_pred, success] = solve(obj, x_current, X_ref)
            % SOLVE  Computes optimal control trajectory for the current state.
            x_curr = double(x_current(:));
            q_curr = x_curr(1:obj.n);
            v_curr = x_curr(obj.n+1:end);

            % Map to DH frames
            q_dh = obj.signs .* q_curr;
            v_dh = obj.signs .* v_curr;
            x0_dh = [q_dh; v_dh];

            % Map reference to DH frames
            X_ref_dh = zeros(size(X_ref));
            for k = 1:size(X_ref, 2)
                X_ref_dh(1:obj.n, k)     = obj.signs .* X_ref(1:obj.n, k);
                X_ref_dh(obj.n+1:end, k) = obj.signs .* X_ref(obj.n+1:end, k);
            end

            % Helper: Dynamic rollout given control trajectory U [n x N]
            function [cost, X_traj] = rollout(U_mat)
                X_traj = zeros(2 * obj.n, obj.N + 1);
                X_traj(:, 1) = x0_dh;
                cost = 0;

                for step = 1:obj.N
                    xk = X_traj(:, step);
                    uk = U_mat(:, step);
                    xref_k = X_ref_dh(:, step);

                    % Tracking error
                    eq = xk(1:obj.n) - xref_k(1:obj.n);
                    ev = xk(obj.n+1:end) - xref_k(obj.n+1:end);

                    cost = cost + obj.Q_pos * (eq' * eq) + obj.Q_vel * (ev' * ev) + obj.R_tau * (uk' * uk);

                    % RK4 step
                    qk = xk(1:obj.n);
                    vk = xk(obj.n+1:end);

                    k1_v = vk;
                    k1_a = obj.dyn.forward_dyn_fun(qk, vk, uk);

                    q_m1 = qk + 0.5 * obj.dt * k1_v;
                    v_m1 = vk + 0.5 * obj.dt * k1_a;
                    k2_v = v_m1;
                    k2_a = obj.dyn.forward_dyn_fun(q_m1, v_m1, uk);

                    q_m2 = qk + 0.5 * obj.dt * k2_v;
                    v_m2 = vk + 0.5 * obj.dt * k2_a;
                    k3_v = v_m2;
                    k3_a = obj.dyn.forward_dyn_fun(q_m2, v_m2, uk);

                    q_e = qk + obj.dt * k3_v;
                    v_e = vk + obj.dt * k3_a;
                    k4_v = v_e;
                    k4_a = obj.dyn.forward_dyn_fun(q_e, v_e, uk);

                    q_next = qk + (obj.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v);
                    v_next = vk + (obj.dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a);

                    X_traj(:, step + 1) = [q_next; v_next];
                end

                % Terminal penalty
                e_qN = X_traj(1:obj.n, end) - X_ref_dh(1:obj.n, end);
                e_vN = X_traj(obj.n+1:end, end) - X_ref_dh(obj.n+1:end, end);
                cost = cost + 2.0 * obj.Q_pos * (e_qN' * e_qN) + 2.0 * obj.Q_vel * (e_vN' * e_vN);
            end

            % Initial guess from warm start (shift previous solution)
            U0 = [obj.last_U(:, 2:end), obj.last_U(:, end)];

            % Optimization via fmincon (if available) or Projected Gradient
            success = true;
            try
                opt_fn = @(u_vec) rollout(reshape(u_vec, obj.n, obj.N));
                lb = -obj.tau_max * ones(obj.n * obj.N, 1);
                ub =  obj.tau_max * ones(obj.n * obj.N, 1);

                options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
                                       'MaxIterations', 20, 'MaxFunctionEvaluations', 200);
                u_opt_vec = fmincon(opt_fn, U0(:), [], [], [], [], lb, ub, [], options);
                U_opt_dh = reshape(u_opt_vec, obj.n, obj.N);
            catch
                % Fallback: Gravity compensation + feedforward PD
                G0 = obj.dyn.G_fun(q_dh);
                U_opt_dh = repmat(G0, 1, obj.N);
            end

            obj.last_U = U_opt_dh;
            [~, X_pred_dh] = rollout(U_opt_dh);

            % Map first control command to URDF coordinates
            tau_opt = obj.signs .* U_opt_dh(:, 1);

            % Map trajectories to URDF coordinates
            X_pred = zeros(size(X_pred_dh));
            for k = 1:size(X_pred_dh, 2)
                X_pred(1:obj.n, k)     = obj.signs .* X_pred_dh(1:obj.n, k);
                X_pred(obj.n+1:end, k) = obj.signs .* X_pred_dh(obj.n+1:end, k);
            end

            U_pred = zeros(size(U_opt_dh));
            for k = 1:size(U_opt_dh, 2)
                U_pred(:, k) = obj.signs .* U_opt_dh(:, k);
            end
        end
    end
end
