classdef trajectory_optimization_casadi < handle
% TRAJECTORY_OPTIMIZATION_CASADI  Multiple-Shooting Trajectory Optimization OCP.
%
% Implements direct multiple shooting for robot boundary-value problems (BVPs):
%   min_{w} sum_{k=0}^{N-1} [ 0.01*||u_k||^2 + 0.005*||u_k - u_{k-1}||^2 + 0.1*||q_k - q_target||^2 ] + terminal_cost
%   s.t.  x_0 = x_init
%         x_{k+1} = f_step(x_k, u_k)   (Euler or RK4)
%         q_min <= q_k <= q_max
%         -dq_max <= v_k <= dq_max
%         -tau_max <= u_k <= tau_max
%         ||p_ee(q_k) - p_obs||^2 >= (r_obs + d_safe)^2
%
% Supports:
%   - Exact analytical Lagrangian Hessian ('exact')
%   - Quasi-Newton limited memory approximation ('limited-memory' / L-BFGS)
%
% =========================================================================

    properties
        dyn
        fk
        n
        nx
        nu
        N
        dt
        integrator
        hessian_mode
        q_min
        q_max
        dq_max
        tau_max
        obstacle_center
        obstacle_radius
        safety_margin
        r_safe
        opti
        X
        U
        P_init
        P_target
        verbose
    end

    methods
        function obj = trajectory_optimization_casadi(dyn, fk, opts)
            import casadi.*

            if nargin < 1 || isempty(dyn)
                error('trajectory_optimization_casadi:MissingDyn', 'A compiled dynamics struct is required.');
            end
            obj.dyn = dyn;
            obj.n   = dyn.n;
            obj.nx  = 2 * obj.n;
            obj.nu  = obj.n;

            if nargin < 2 || isempty(fk)
                error('trajectory_optimization_casadi:MissingFK', 'Forward kinematics struct is required.');
            end
            obj.fk = fk;

            if nargin < 3 || isempty(opts)
                opts = struct();
            end

            obj.N            = 10;        if isfield(opts, 'N'),            obj.N            = opts.N; end
            obj.dt           = 0.04;      if isfield(opts, 'dt'),           obj.dt           = opts.dt; end
            obj.integrator   = 'rk4';     if isfield(opts, 'integrator'),   obj.integrator   = opts.integrator; end
            obj.hessian_mode = 'exact';   if isfield(opts, 'hessian_mode'), obj.hessian_mode = opts.hessian_mode; end
            obj.verbose      = false;     if isfield(opts, 'verbose'),      obj.verbose      = opts.verbose; end

            obj.q_min  = -pi * ones(obj.n, 1); if isfield(opts, 'q_min'),  obj.q_min  = opts.q_min(:); end
            obj.q_max  =  pi * ones(obj.n, 1); if isfield(opts, 'q_max'),  obj.q_max  = opts.q_max(:); end
            obj.dq_max = 5.0 * ones(obj.n, 1); if isfield(opts, 'dq_max'), obj.dq_max = opts.dq_max(:); end

            if isfield(opts, 'tau_max') && ~isempty(opts.tau_max)
                if isscalar(opts.tau_max)
                    obj.tau_max = opts.tau_max * ones(obj.n, 1);
                else
                    obj.tau_max = opts.tau_max(:);
                end
            else
                obj.tau_max = 150.0 * ones(obj.n, 1);
            end

            % Obstacle avoidance settings
            obj.obstacle_center = [];
            if isfield(opts, 'obstacle_center') && ~isempty(opts.obstacle_center)
                obj.obstacle_center = opts.obstacle_center(:);
            end
            obj.obstacle_radius = 0.12; if isfield(opts, 'obstacle_radius'), obj.obstacle_radius = opts.obstacle_radius; end
            obj.safety_margin   = 0.04; if isfield(opts, 'safety_margin'),   obj.safety_margin   = opts.safety_margin; end
            obj.r_safe          = obj.obstacle_radius + obj.safety_margin;

            obj.build_ocp();
        end

        function build_ocp(obj)
            import casadi.*

            obj.opti = Opti();

            % Decision Variables
            obj.X = obj.opti.variable(obj.nx, obj.N + 1);  % [q; v]
            obj.U = obj.opti.variable(obj.nu, obj.N);      % tau

            % Parameters
            obj.P_init   = obj.opti.parameter(obj.nx, 1);
            obj.P_target = obj.opti.parameter(obj.nx, 1);

            % Boundary condition at k = 0
            obj.opti.subject_to(obj.X(:, 1) == obj.P_init);

            cost = 0;
            q_target = obj.P_target(1:obj.n);
            v_target = obj.P_target(obj.n+1:end);

            for k = 1:obj.N
                x_k = obj.X(:, k);
                u_k = obj.U(:, k);
                q_k = x_k(1:obj.n);
                v_k = x_k(obj.n+1:end);

                % State bounds
                obj.opti.subject_to(obj.q_min <= q_k <= obj.q_max);
                obj.opti.subject_to(-obj.dq_max <= v_k <= obj.dq_max);

                % Control torque bounds
                obj.opti.subject_to(-obj.tau_max <= u_k <= obj.tau_max);

                % Step integration constraint
                if strcmpi(obj.integrator, 'euler')
                    % Semi-implicit Euler
                    a_k = obj.dyn.forward_dyn_fun(q_k, v_k, u_k);
                    v_next = v_k + obj.dt * a_k;
                    q_next = q_k + obj.dt * v_next;
                else
                    % RK4
                    k1_v = v_k;
                    k1_a = obj.dyn.forward_dyn_fun(q_k, v_k, u_k);

                    k2_q = q_k + 0.5 * obj.dt * k1_v;
                    k2_v = v_k + 0.5 * obj.dt * k1_a;
                    k2_a = obj.dyn.forward_dyn_fun(k2_q, k2_v, u_k);

                    k3_q = q_k + 0.5 * obj.dt * k2_v;
                    k3_v = v_k + 0.5 * obj.dt * k2_a;
                    k3_a = obj.dyn.forward_dyn_fun(k3_q, k3_v, u_k);

                    k4_q = q_k + obj.dt * k3_v;
                    k4_v = v_k + obj.dt * k3_a;
                    k4_a = obj.dyn.forward_dyn_fun(k4_q, k4_v, u_k);

                    q_next = q_k + (obj.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v);
                    v_next = v_k + (obj.dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a);
                end

                obj.opti.subject_to(obj.X(:, k+1) == [q_next; v_next]);

                % Obstacle avoidance constraint
                if ~isempty(obj.obstacle_center)
                    T_ee = obj.fk.fk_ee_fun(q_k);
                    p_ee = T_ee(1:3, 4);
                    dist_sq = dot(p_ee - obj.obstacle_center, p_ee - obj.obstacle_center);
                    obj.opti.subject_to(dist_sq >= obj.r_safe^2);
                end

                % Cost terms
                cost = cost + 0.01 * dot(u_k, u_k);
                if k > 1
                    u_prev = obj.U(:, k-1);
                    cost = cost + 0.005 * dot(u_k - u_prev, u_k - u_prev);
                end
                cost = cost + 0.1 * dot(q_k - q_target, q_k - q_target);
            end

            % Terminal bounds & cost
            x_N = obj.X(:, obj.N + 1);
            q_N = x_N(1:obj.n);
            v_N = x_N(obj.n+1:end);
            obj.opti.subject_to(obj.q_min <= q_N <= obj.q_max);
            obj.opti.subject_to(-obj.dq_max <= v_N <= obj.dq_max);

            cost = cost + 50.0 * dot(q_N - q_target, q_N - q_target) + 5.0 * dot(v_N - v_target, v_N - v_target);
            obj.opti.minimize(cost);

            % Solver configuration
            p_opts = struct('print_time', false);
            s_opts = struct('print_level', 0, 'tol', 1e-4);

            if strcmpi(obj.hessian_mode, 'limited-memory')
                s_opts.hessian_approximation = 'limited-memory';
                s_opts.max_iter = 500;
            else
                s_opts.hessian_approximation = 'exact';
                s_opts.max_iter = 100;
            end

            if obj.verbose
                p_opts.print_time = true;
                s_opts.print_level = 3;
            end

            obj.opti.solver('ipopt', p_opts, s_opts);
        end

        function sol_data = solve_bvp(obj, x_init, x_target)
            % SOLVE_BVP  Solves the boundary-value problem for given endpoints.
            x_init   = double(x_init(:));
            x_target = double(x_target(:));

            obj.opti.set_value(obj.P_init, x_init);
            obj.opti.set_value(obj.P_target, x_target);

            % Initial guess: linear interpolation in state, zero in control
            X0 = zeros(obj.nx, obj.N + 1);
            for k = 1:obj.N+1
                alpha = (k - 1) / obj.N;
                X0(:, k) = (1 - alpha) * x_init + alpha * x_target;
            end
            obj.opti.set_initial(obj.X, X0);
            obj.opti.set_initial(obj.U, zeros(obj.nu, obj.N));

            t_start = tic;
            success = false;
            iter_count = 0;

            try
                sol = obj.opti.solve();
                t_solve = toc(t_start);
                success = true;
                stats = obj.opti.stats();
                iter_count = stats.iter_count;

                X_opt = sol.value(obj.X);
                U_opt = sol.value(obj.U);
                J_cost = sol.value(obj.opti.f);
            catch ME
                t_solve = toc(t_start);
                try
                    stats = obj.opti.stats();
                    iter_count = stats.iter_count;
                catch
                    iter_count = 500;
                end
                X_opt = [];
                U_opt = [];
                J_cost = inf;
            end

            sol_data = struct();
            sol_data.success = success;
            sol_data.solve_time_s = t_solve;
            sol_data.iterations = iter_count;
            sol_data.cost = J_cost;
            sol_data.X_opt = X_opt;
            sol_data.U_opt = U_opt;
            sol_data.hessian_mode = obj.hessian_mode;
        end
    end
end
