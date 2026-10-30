function results = benchmark_nmpc_ur16e_kuka(num_steps)
% BENCHMARK_NMPC_UR16E_KUKA  Replicates Table 6 of CADENCE manuscript.
%
% Compares Baseline (Christoffel + Inversion) versus Proposed (Momentum + Cholesky)
% across 40 closed-loop receding-horizon NMPC steps with obstacle avoidance:
%   - UR16e (6-DOF)
%   - KUKA LBR iiwa 7 (7-DOF redundant chain)
%
% Metrics:
%   1. Mean and Median solve latency (ms)
%   2. 95th (p95) and 99th (p99) percentile latency spike mitigation
%   3. Speedup factor
%   4. Cumulative NLP cost and tracking RMSE
%
% =========================================================================

    if nargin < 1 || isempty(num_steps)
        num_steps = 40;
    end

    this_dir = fileparts(mfilename('fullpath'));
    base_dir = fileparts(this_dir);

    addpath(base_dir);
    addpath(fullfile(base_dir, 'parsing'));
    addpath(fullfile(base_dir, 'kinematics'));
    addpath(fullfile(base_dir, 'dynamics'));
    addpath(fullfile(base_dir, 'control'));
    addpath(fullfile(base_dir, 'utils'));

    fprintf('\n====================================================================\n');
    fprintf('  CADENCE CLOSED-LOOP NMPC BENCHMARK WITH OBSTACLES (TABLE 6 PARITY)\n');
    fprintf('  UR16e & KUKA iiwa across %d Steps (Horizon N=15, dt=0.02s)        \n', num_steps);
    fprintf('====================================================================\n\n');

    robots = {'UR16e', 'KUKA_iiwa'};
    results = struct();

    for r = 1:length(robots)
        name = robots{r};
        fprintf('\n>>> Benchmarking NMPC for [%s] <<<\n', name);

        try
            urdf_path = get_robot_urdf_path(name);
        catch ME
            fprintf('    [SKIP] %s: %s\n', name, ME.message);
            continue;
        end

        robot   = urdf_parse(urdf_path);
        chain   = urdf_build_chain(robot, false);
        dh      = urdf_extract_dh(chain, false);
        inertia = urdf_extract_inertia(robot, chain, dh, false);
        n       = chain.n;

        fk = fk_casadi(dh, chain, struct('verbose', false));
        signs = detect_joint_signs(chain, fk.fk_link_funs);

        % Obstacle parameters
        obs_center = [0.35; 0.10; 0.50];
        obs_radius = 0.12;
        safe_margin = 0.04;

        % Torque saturation bounds
        if strcmpi(name, 'UR16e')
            tau_max = 150.0;
        else
            tau_max = [176; 176; 110; 110; 110; 40; 40];
        end

        % Build 1. Baseline Model (Christoffel + Inversion)
        fprintf('  Building Baseline Model (Christoffel + Inversion)...\n');
        dyn_base = dynamics_casadi(dh, inertia, chain, ...
            struct('use_momentum_coriolis', false, 'use_cholesky', false, 'verbose', false));

        % Build 2. Proposed Model (Momentum + Cholesky)
        fprintf('  Building Proposed Model (Momentum + Cholesky)...\n');
        dyn_prop = dynamics_casadi(dh, inertia, chain, ...
            struct('use_momentum_coriolis', true, 'use_cholesky', true, 'verbose', false));

        nmpc_opts = struct('N', 15, 'dt', 0.02, 'fk', fk, ...
                           'obstacle_center', obs_center, 'obstacle_radius', obs_radius, ...
                           'safety_margin', safe_margin, 'tau_max', tau_max, ...
                           'joint_signs', signs, 'verbose', false);

        nmpc_base = nmpc_casadi(dyn_base, nmpc_opts);
        nmpc_prop = nmpc_casadi(dyn_prop, nmpc_opts);

        % Multi-harmonic reference trajectory
        t_vec = (0:num_steps + 15) * 0.02;
        X_ref_full = zeros(2 * n, length(t_vec));
        for k = 1:length(t_vec)
            t = t_vec(k);
            q_ref_k = 0.3 * sin(1.5 * t) * ones(n, 1);
            v_ref_k = 0.45 * cos(1.5 * t) * ones(n, 1);
            X_ref_full(:, k) = [q_ref_k; v_ref_k];
        end

        % Evaluate Baseline
        fprintf('  Simulating Baseline Closed-Loop NMPC (%d steps)...\n', num_steps);
        x_curr_base = X_ref_full(:, 1);
        times_base = zeros(num_steps, 1);
        costs_base = zeros(num_steps, 1);
        errors_base = zeros(num_steps, 1);

        for step = 1:num_steps
            X_horizon = X_ref_full(:, step:step+15);
            [tau_opt, X_pred, ~, succ, info] = nmpc_base.solve(x_curr_base, X_horizon);
            times_base(step) = info.solve_time_ms;
            costs_base(step) = info.cost;

            err = x_curr_base(1:n) - X_horizon(1:n, 1);
            errors_base(step) = norm(err);

            % Step forward in simulator
            acc = full(dyn_prop.forward_dyn_fun(signs.*x_curr_base(1:n), signs.*x_curr_base(n+1:end), signs.*tau_opt));
            acc = signs .* acc;
            v_next = x_curr_base(n+1:end) + 0.02 * acc;
            q_next = x_curr_base(1:n) + 0.02 * v_next;
            x_curr_base = [q_next; v_next];
        end

        % Evaluate Proposed
        fprintf('  Simulating Proposed Closed-Loop NMPC (%d steps)...\n', num_steps);
        x_curr_prop = X_ref_full(:, 1);
        times_prop = zeros(num_steps, 1);
        costs_prop = zeros(num_steps, 1);
        errors_prop = zeros(num_steps, 1);

        for step = 1:num_steps
            X_horizon = X_ref_full(:, step:step+15);
            [tau_opt, X_pred, ~, succ, info] = nmpc_prop.solve(x_curr_prop, X_horizon);
            times_prop(step) = info.solve_time_ms;
            costs_prop(step) = info.cost;

            err = x_curr_prop(1:n) - X_horizon(1:n, 1);
            errors_prop(step) = norm(err);

            % Step forward in simulator
            acc = full(dyn_prop.forward_dyn_fun(signs.*x_curr_prop(1:n), signs.*x_curr_prop(n+1:end), signs.*tau_opt));
            acc = signs .* acc;
            v_next = x_curr_prop(n+1:end) + 0.02 * acc;
            q_next = x_curr_prop(1:n) + 0.02 * v_next;
            x_curr_prop = [q_next; v_next];
        end

        rob_res = struct();
        rob_res.name = name;
        rob_res.base_mean_ms   = mean(times_base);
        rob_res.prop_mean_ms   = mean(times_prop);
        rob_res.base_median_ms = median(times_base);
        rob_res.prop_median_ms = median(times_prop);
        rob_res.base_p95_ms    = prctile(times_base, 95);
        rob_res.prop_p95_ms    = prctile(times_prop, 95);
        rob_res.base_p99_ms    = prctile(times_base, 99);
        rob_res.prop_p99_ms    = prctile(times_prop, 99);
        rob_res.speedup        = rob_res.base_mean_ms / rob_res.prop_mean_ms;
        rob_res.rmse_base      = sqrt(mean(errors_base.^2));
        rob_res.rmse_prop      = sqrt(mean(errors_prop.^2));
        rob_res.cost_base      = mean(costs_base);
        rob_res.cost_prop      = mean(costs_prop);

        results.(name) = rob_res;
    end

    fprintf('\n====================================================================\n');
    fprintf('  TABLE 6 NMPC BENCHMARK WITH OBSTACLES SUMMARY DASHBOARD           \n');
    fprintf('====================================================================\n');
    fprintf('%-10s | %-14s | %-10s | %-10s | %-10s | %-8s | %-10s\n', ...
        'Robot', 'Formulation', 'Mean (ms)', 'p95 (ms)', 'p99 (ms)', 'Speedup', 'RMSE (rad)');
    fprintf('%s\n', repmat('-', 1, 85));

    names = fieldnames(results);
    for k = 1:length(names)
        r = results.(names{k});
        fprintf('%-10s | %-14s | %-10.1f | %-10.1f | %-10.1f | %-8s | %-10.4f\n', ...
            r.name, 'Baseline', r.base_mean_ms, r.base_p95_ms, r.base_p99_ms, '1.00x', r.rmse_base);
        fprintf('%-10s | %-14s | %-10.1f | %-10.1f | %-10.1f | %-8.2fx | %-10.4f\n', ...
            '', 'Proposed', r.prop_mean_ms, r.prop_p95_ms, r.prop_p99_ms, sprintf('%.2fx', r.speedup), r.rmse_prop);
    end
    fprintf('====================================================================\n\n');
end
