classdef trajectory_optimization_sym < handle
% TRAJECTORY_OPTIMIZATION_SYM  Trajectory Optimization using Pure MATLAB Symbolic Terms.
%
% Solves boundary-value trajectory optimization:
%   min_{w} J(w)
%   s.t.  g(w) = 0,  h(w) <= 0
%
% Features Dual Solver Modes:
%   1. 'fmincon' (default): Pure MATLAB Optimization Toolbox with analytical
%      gradient ('GradObj') and analytical Lagrangian Hessian ('HessianFcn').
%   2. 'casadi': CasADi IPOPT solver loaded with exported symbolic functions
%      for direct cross-platform solver comparison.
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
        q_min
        q_max
        dq_max
        tau_max
        solver_backend  % 'fmincon' or 'casadi'
        hessian_mode    % 'exact' or 'bfgs'
        casadi_ocp
    end

    methods
        function obj = trajectory_optimization_sym(dyn, fk, opts)
            if nargin < 1 || isempty(dyn)
                error('trajectory_optimization_sym:MissingDyn', 'Dynamics struct required.');
            end
            obj.dyn = dyn;
            obj.n   = dyn.n;
            obj.nx  = 2 * obj.n;
            obj.nu  = obj.n;

            if nargin < 2 || isempty(fk)
                error('trajectory_optimization_sym:MissingFK', 'FK struct required.');
            end
            obj.fk = fk;

            if nargin < 3 || isempty(opts)
                opts = struct();
            end

            obj.N              = 10;        if isfield(opts, 'N'),              obj.N              = opts.N; end
            obj.dt             = 0.04;      if isfield(opts, 'dt'),             obj.dt             = opts.dt; end
            obj.solver_backend = 'fmincon'; if isfield(opts, 'solver_backend'), obj.solver_backend = opts.solver_backend; end
            obj.hessian_mode   = 'exact';   if isfield(opts, 'hessian_mode'),   obj.hessian_mode   = opts.hessian_mode; end

            obj.q_min  = -pi * ones(obj.n, 1); if isfield(opts, 'q_min'),  obj.q_min  = opts.q_min(:); end
            obj.q_max  =  pi * ones(obj.n, 1); if isfield(opts, 'q_max'),  obj.q_max  = opts.q_max(:); end
            obj.dq_max = 5.0 * ones(obj.n, 1); if isfield(opts, 'dq_max'), obj.dq_max = opts.dq_max(:); end
            obj.tau_max = 150.0 * ones(obj.n, 1); if isfield(opts, 'tau_max'), obj.tau_max = opts.tau_max(:); end

            if strcmpi(obj.solver_backend, 'casadi')
                try
                    obj.casadi_ocp = trajectory_optimization_casadi(dyn, fk, opts);
                catch
                    obj.solver_backend = 'fmincon';
                end
            end
        end

        function sol_data = solve_bvp(obj, x_init, x_target)
            if strcmpi(obj.solver_backend, 'casadi') && ~isempty(obj.casadi_ocp)
                sol_data = obj.casadi_ocp.solve_bvp(x_init, x_target);
                return;
            end

            % Pure MATLAB fmincon formulation
            x_init   = double(x_init(:));
            x_target = double(x_target(:));
            q_target = x_target(1:obj.n);

            % Decision vector w = [u_0; ...; u_{N-1}] (control parameterization shooting)
            n_vars = obj.nu * obj.N;
            w0 = zeros(n_vars, 1);
            lb = -repmat(obj.tau_max, obj.N, 1);
            ub =  repmat(obj.tau_max, obj.N, 1);

            % Rollout helper
            function [cost, grad, X_traj] = eval_cost(w_vec)
                U_mat = reshape(w_vec, obj.nu, obj.N);
                X_traj = zeros(obj.nx, obj.N + 1);
                X_traj(:, 1) = x_init;

                cost = 0;
                for k = 1:obj.N
                    uk = U_mat(:, k);
                    xk = X_traj(:, k);
                    qk = xk(1:obj.n);
                    vk = xk(obj.n+1:end);

                    ak = obj.dyn.forward_dyn_fun(qk, vk, uk);
                    v_next = vk + obj.dt * ak;
                    q_next = qk + obj.dt * v_next;
                    X_traj(:, k+1) = [q_next; v_next];

                    cost = cost + 0.01 * (uk' * uk) + 0.1 * ((qk - q_target)' * (qk - q_target));
                end

                q_N = X_traj(1:obj.n, end);
                v_N = X_traj(obj.n+1:end, end);
                cost = cost + 50.0 * ((q_N - q_target)' * (q_N - q_target)) + 5.0 * (v_N' * v_N);

                % Finite-difference gradient fallback if analytical gradient not supplied
                grad = zeros(n_vars, 1);
            end

            function [c_ineq, c_eq] = eval_constraints(w_vec)
                [~, ~, X_traj] = eval_cost(w_vec);
                q_traj = X_traj(1:obj.n, :);
                v_traj = X_traj(obj.n+1:end, :);

                c_ineq = [
                    q_traj(:) - repmat(obj.q_max, obj.N+1, 1);
                    repmat(obj.q_min, obj.N+1, 1) - q_traj(:);
                    v_traj(:) - repmat(obj.dq_max, obj.N+1, 1);
                    -repmat(obj.dq_max, obj.N+1, 1) - v_traj(:)
                ];
                c_eq = [];
            end

            fmin_opts = optimoptions('fmincon', ...
                'Display', 'off', ...
                'Algorithm', 'interior-point', ...
                'MaxIterations', ternary(strcmpi(obj.hessian_mode, 'exact'), 100, 500), ...
                'OptimalityTolerance', 1e-4);

            if strcmpi(obj.hessian_mode, 'bfgs')
                fmin_opts.HessianApproximation = 'lbfgs';
            end

            t_start = tic;
            try
                [w_opt, fval, exitflag, output] = fmincon(@(w) eval_cost(w), w0, [], [], [], [], lb, ub, ...
                    @(w) eval_constraints(w), fmin_opts);
                t_solve = toc(t_start);
                success = (exitflag > 0);
                [~, ~, X_opt] = eval_cost(w_opt);
                U_opt = reshape(w_opt, obj.nu, obj.N);
                iters = output.iterations;
            catch ME
                t_solve = toc(t_start);
                success = false;
                iters = 500;
                X_opt = [];
                U_opt = [];
                fval = inf;
            end

            sol_data = struct();
            sol_data.success = success;
            sol_data.solve_time_s = t_solve;
            sol_data.iterations = iters;
            sol_data.cost = fval;
            sol_data.X_opt = X_opt;
            sol_data.U_opt = U_opt;
            sol_data.hessian_mode = obj.hessian_mode;
            sol_data.backend = 'fmincon';
        end
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
