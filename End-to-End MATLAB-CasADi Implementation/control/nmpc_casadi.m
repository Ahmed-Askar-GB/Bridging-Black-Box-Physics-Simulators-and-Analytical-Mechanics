classdef nmpc_casadi < handle
% NMPC_CASADI  Nonlinear Model Predictive Control for robots using CasADi Opti.
%
% Implements a multiple-shooting receding horizon NMPC tracking controller:
%   min  sum_{k=1}^N [ ||q_k - q_ref_k||_Qp^2 + ||v_k - v_ref_k||_Qv^2 + ||tau_k||_R^2 ] + terminal_cost
%   s.t. x_{k+1} = f_RK4(x_k, tau_k)
%        x_0 = x_current
%        q_min <= q_k <= q_max
%        -tau_max <= tau_k <= tau_max
%
% =========================================================================
% USAGE:
%   nmpc = nmpc_casadi(dyn, opts);
%   [tau_opt, X_pred, U_pred] = nmpc.solve(x_current, X_ref);
% =========================================================================

    properties
        dyn
        n
        N
        dt
        opti
        X
        U
        X0_param
        X_ref_param
        signs
        solver_opts
        last_sol
    end

    methods
        function obj = nmpc_casadi(dyn, opts)
            % CONSTRUCTOR
            import casadi.*

            if nargin < 1 || isempty(dyn)
                error('nmpc_casadi:MissingDyn', 'A compiled CasADi dynamics struct with forward_dyn_fun is required.');
            end
            obj.dyn = dyn;

            if isfield(dyn, 'n')
                obj.n = dyn.n;
            else
                obj.n = dyn.forward_dyn_fun.size1_in(0);
            end

            if nargin < 2 || isempty(opts)
                opts = struct();
            end

            % Extract options with intelligent defaults
            obj.N  = 15;        if isfield(opts, 'N'),  obj.N  = opts.N; end
            obj.dt = 0.02;      if isfield(opts, 'dt'), obj.dt = opts.dt; end

            Q_pos   = 100.0;    if isfield(opts, 'Q_pos'),   Q_pos   = opts.Q_pos; end
            Q_vel   = 10.0;     if isfield(opts, 'Q_vel'),   Q_vel   = opts.Q_vel; end
            R_tau   = 0.01;     if isfield(opts, 'R_tau'),   R_tau   = opts.R_tau; end
            tau_max = 150.0;    if isfield(opts, 'tau_max'), tau_max = opts.tau_max; end

            q_min   = -2*pi*ones(obj.n, 1);
            if isfield(opts, 'q_min') && ~isempty(opts.q_min)
                q_min = double(opts.q_min(:));
            end

            q_max   = 2*pi*ones(obj.n, 1);
            if isfield(opts, 'q_max') && ~isempty(opts.q_max)
                q_max = double(opts.q_max(:));
            end

            if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
                obj.signs = double(opts.joint_signs(:));
            else
                obj.signs = ones(obj.n, 1);
            end

            verbose = false;
            if isfield(opts, 'verbose'), verbose = opts.verbose; end

            % Build Opti problem
            obj.opti = Opti();
            nx = 2 * obj.n;
            nu = obj.n;

            % Variables
            obj.X = obj.opti.variable(nx, obj.N + 1);  % [q; dq]
            obj.U = obj.opti.variable(nu, obj.N);      % tau

            % Parameters
            obj.X0_param    = obj.opti.parameter(nx, 1);
            obj.X_ref_param = obj.opti.parameter(nx, obj.N + 1);

            % Objective
            cost = 0;
            for k = 1:obj.N
                x_k     = obj.X(:, k);
                u_k     = obj.U(:, k);
                x_ref_k = obj.X_ref_param(:, k);

                e_q = x_k(1:obj.n) - x_ref_k(1:obj.n);
                e_v = x_k(obj.n+1:end) - x_ref_k(obj.n+1:end);

                cost = cost + Q_pos * dot(e_q, e_q) + Q_vel * dot(e_v, e_v) + R_tau * dot(u_k, u_k);
            end

            % Terminal cost
            e_q_N = obj.X(1:obj.n, obj.N+1) - obj.X_ref_param(1:obj.n, obj.N+1);
            e_v_N = obj.X(obj.n+1:end, obj.N+1) - obj.X_ref_param(obj.n+1:end, obj.N+1);
            cost  = cost + 2.0 * Q_pos * dot(e_q_N, e_q_N) + 2.0 * Q_vel * dot(e_v_N, e_v_N);

            obj.opti.minimize(cost);

            % Initial state constraint
            obj.opti.subject_to(obj.X(:, 1) == obj.X0_param);

            % Multiple-shooting dynamic constraints via RK4
            dt_sym = obj.dt;
            for k = 1:obj.N
                x_k = obj.X(:, k);
                u_k = obj.U(:, k);

                q_k  = x_k(1:obj.n);
                v_k  = x_k(obj.n+1:end);

                % RK4 evaluation using forward_dyn_fun
                k1_v = v_k;
                k1_a = obj.dyn.forward_dyn_fun(q_k, v_k, u_k);

                k2_q = q_k + 0.5 * dt_sym * k1_v;
                k2_v = v_k + 0.5 * dt_sym * k1_a;
                k2_a = obj.dyn.forward_dyn_fun(k2_q, k2_v, u_k);

                k3_q = q_k + 0.5 * dt_sym * k2_v;
                k3_v = v_k + 0.5 * dt_sym * k2_a;
                k3_a = obj.dyn.forward_dyn_fun(k3_q, k3_v, u_k);

                k4_q = q_k + dt_sym * k3_v;
                k4_v = v_k + dt_sym * k3_a;
                k4_a = obj.dyn.forward_dyn_fun(k4_q, k4_v, u_k);

                q_next = q_k + (dt_sym / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v);
                v_next = v_k + (dt_sym / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a);

                obj.opti.subject_to(obj.X(:, k+1) == [q_next; v_next]);

                % Torque limit constraints
                obj.opti.subject_to(-tau_max <= u_k <= tau_max);

                % Joint limits
                obj.opti.subject_to(q_min <= x_k(1:obj.n) <= q_max);
            end

            % Terminal joint limit
            obj.opti.subject_to(q_min <= obj.X(1:obj.n, obj.N+1) <= q_max);

            % Configure solver
            ipopt_opts = struct();
            if ~verbose
                ipopt_opts.print_level = 0;
                obj.opti.solver('ipopt', struct('print_time', false), ipopt_opts);
            else
                ipopt_opts.print_level = 3;
                obj.opti.solver('ipopt', struct('print_time', true), ipopt_opts);
            end

            obj.last_sol = [];
        end

        function [tau_opt, X_pred, U_pred, success] = solve(obj, x_current, X_ref)
            % SOLVE  Solves the optimal control NLP for the next control action.
            %
            % INPUTS:
            %   x_current: [2n x 1] current state [q; v] in URDF frames
            %   X_ref:     [2n x (N+1)] reference trajectory horizon in URDF frames
            %
            % OUTPUTS:
            %   tau_opt:   [n x 1] optimal control torque for step 1 in URDF frames
            %   X_pred:    [2n x (N+1)] predicted state trajectory
            %   U_pred:    [n x N] predicted control trajectory
            %   success:   (logical) solver convergence flag

            x_curr = double(x_current(:));
            q_curr = x_curr(1:obj.n);
            v_curr = x_curr(obj.n+1:end);

            % Map to analytical DH frames
            q_dh = obj.signs .* q_curr;
            v_dh = obj.signs .* v_curr;
            x0_dh = [q_dh; v_dh];

            % Map reference trajectory to DH frames
            X_ref_dh = zeros(size(X_ref));
            for k = 1:size(X_ref, 2)
                X_ref_dh(1:obj.n, k)     = obj.signs .* X_ref(1:obj.n, k);
                X_ref_dh(obj.n+1:end, k) = obj.signs .* X_ref(obj.n+1:end, k);
            end

            % Set parameters
            obj.opti.set_value(obj.X0_param, x0_dh);
            obj.opti.set_value(obj.X_ref_param, X_ref_dh);

            % Warm start if previous solution available
            if ~isempty(obj.last_sol)
                try
                    obj.opti.set_initial(obj.opti.lam_g, obj.last_sol.value(obj.opti.lam_g));
                    obj.opti.set_initial(obj.X, obj.last_sol.value(obj.X));
                    obj.opti.set_initial(obj.U, obj.last_sol.value(obj.U));
                catch
                    % Fallback to reference initialization
                    obj.opti.set_initial(obj.X, X_ref_dh);
                end
            else
                obj.opti.set_initial(obj.X, X_ref_dh);
            end

            success = true;
            try
                sol = obj.opti.solve();
                obj.last_sol = sol;

                X_pred_dh = full(sol.value(obj.X));
                U_pred_dh = full(sol.value(obj.U));

                % Optimal first control action in DH
                u0_dh = U_pred_dh(:, 1);

                % Map back to URDF frames
                tau_opt = obj.signs .* u0_dh;

                % Map predicted states and controls back to URDF
                X_pred = zeros(size(X_pred_dh));
                for k = 1:size(X_pred_dh, 2)
                    X_pred(1:obj.n, k)     = obj.signs .* X_pred_dh(1:obj.n, k);
                    X_pred(obj.n+1:end, k) = obj.signs .* X_pred_dh(obj.n+1:end, k);
                end

                U_pred = zeros(size(U_pred_dh));
                for k = 1:size(U_pred_dh, 2)
                    U_pred(:, k) = obj.signs .* U_pred_dh(:, k);
                end

            catch ME
                success = false;
                warning('nmpc_casadi:SolveFailed', 'Opti solve failed: %s. Using gravity compensation.', ME.message);
                % Fallback: Gravity compensation
                G_val = full(obj.dyn.G_fun(q_dh));
                tau_opt = obj.signs .* G_val;
                X_pred = X_ref;
                U_pred = repmat(tau_opt, 1, obj.N);
            end
        end
    end
end
