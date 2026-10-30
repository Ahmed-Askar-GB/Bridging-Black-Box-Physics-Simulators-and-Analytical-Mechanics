function results = benchmark_trajectory_opt_15bvp(robot_list, num_bvps)
% BENCHMARK_TRAJECTORY_OPT_15BVP  Replicates Table 4 of CADENCE manuscript.
%
% Solves N random boundary-value problems per platform (UR5, UR16e, KUKA iiwa 7)
% under:
%   1. Exact Symbolic Lagrangian Hessian (max 100 iterations)
%   2. Quasi-Newton L-BFGS (500 iterations cutoff)
%
% Validates that exact second-order curvature achieves 100% convergence across
% all problems, whereas L-BFGS experiences timeout failures on redundant or
% heavy-payload manipulators near obstacle boundaries.
%
% =========================================================================

    if nargin < 1 || isempty(robot_list)
        robot_list = {'UR5', 'UR16e', 'KUKA_iiwa'};
    end
    if nargin < 2 || isempty(num_bvps)
        num_bvps = 15;
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
    fprintf('  CADENCE OFFLINE TRAJECTORY OPTIMIZATION BENCHMARK (TABLE 4 PARITY)\n');
    fprintf('  Evaluating Exact Hessian vs Quasi-Newton L-BFGS across %d BVPs    \n', num_bvps);
    fprintf('====================================================================\n\n');

    results = struct();

    for r = 1:length(robot_list)
        name = robot_list{r};
        fprintf('\n>>> [PLATFORM %d/%d] Benchmarking Robot: %s <<<\n', r, length(robot_list), name);

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

        dyn = dynamics_casadi(dh, inertia, chain, struct('use_momentum_coriolis', true, 'use_cholesky', true, 'verbose', false));
        fk  = fk_casadi(dh, chain, struct('verbose', false));
        n   = chain.n;

        % Torque bounds
        if strcmpi(name, 'UR16e')
            tau_max = 150.0;
        elseif strcmpi(name, 'KUKA_iiwa')
            tau_max = [176; 176; 110; 110; 110; 40; 40];
        else
            tau_max = 150.0;
        end

        % Obstacle sphere at nominal location
        obs_center = [0.35; 0.10; 0.50];

        ocp_opts = struct('N', 10, 'dt', 0.04, 'tau_max', tau_max, ...
                          'obstacle_center', obs_center, 'obstacle_radius', 0.12, ...
                          'safety_margin', 0.04, 'verbose', false);

        ocp_opts.hessian_mode = 'exact';
        ocp_exact = trajectory_optimization_casadi(dyn, fk, ocp_opts);

        ocp_opts.hessian_mode = 'limited-memory';
        ocp_lbfgs = trajectory_optimization_casadi(dyn, fk, ocp_opts);

        % Predefined reproducible random seed
        rng(42);

        exact_conv_count = 0;
        lbfgs_conv_count = 0;
        exact_iters = [];
        lbfgs_iters = [];
        exact_times = [];
        lbfgs_times = [];

        ctc_rmse_list = [];
        open_rmse_list = [];

        signs = detect_joint_signs(chain, fk.fk_link_funs);

        for seed = 1:num_bvps
            q_init   = -0.5 + 1.0 * rand(n, 1);
            q_target = -0.5 + 1.0 * rand(n, 1);
            x_init   = [q_init; zeros(n, 1)];
            x_target = [q_target; zeros(n, 1)];

            % 1. Solve under Exact Hessian
            sol_ex = ocp_exact.solve_bvp(x_init, x_target);
            if sol_ex.success
                exact_conv_count = exact_conv_count + 1;
                exact_iters(end+1) = sol_ex.iterations;
                exact_times(end+1) = sol_ex.solve_time_s;

                % Replay trajectory in forward simulation
                traj_t = (0:ocp_opts.N) * ocp_opts.dt;
                q_ref = sol_ex.X_opt(1:n, :)';
                v_ref = sol_ex.X_opt(n+1:end, :)';
                ddq_ref = zeros(size(q_ref));

                % CTC closed-loop replay
                ctc_opts = struct('kp', 200.0, 'kd', 30.0, 'joint_signs', signs);
                sim_res_ctc = forward_dynamics_sim(dyn, x_init, traj_t, q_ref, v_ref, ddq_ref, ctc_opts, struct('dt', 0.001, 't_dur', traj_t(end)));
                err_ctc = sim_res_ctc.q_sim - q_ref;
                ctc_rmse_list(end+1) = sqrt(mean(err_ctc(:).^2));

                % Open-loop torque replay
                sim_res_open = forward_dynamics_sim(dyn, x_init, traj_t, q_ref, v_ref, ddq_ref, struct(), struct('dt', 0.001, 't_dur', traj_t(end)));
                err_open = sim_res_open.q_sim - q_ref;
                open_rmse_list(end+1) = sqrt(mean(err_open(:).^2));
            end

            % 2. Solve under L-BFGS
            sol_lb = ocp_lbfgs.solve_bvp(x_init, x_target);
            if sol_lb.success
                lbfgs_conv_count = lbfgs_conv_count + 1;
                lbfgs_iters(end+1) = sol_lb.iterations;
                lbfgs_times(end+1) = sol_lb.solve_time_s;
            end

            fprintf('    BVP %2d/%2d: Exact Conv=%d (iter=%d, %.2fs) | L-BFGS Conv=%d (iter=%d, %.2fs)\n', ...
                seed, num_bvps, sol_ex.success, sol_ex.iterations, sol_ex.solve_time_s, ...
                sol_lb.success, sol_lb.iterations, sol_lb.solve_time_s);
        end

        rob_res = struct();
        rob_res.name = name;
        rob_res.exact_conv_rate = (exact_conv_count / num_bvps) * 100;
        rob_res.lbfgs_conv_rate = (lbfgs_conv_count / num_bvps) * 100;
        rob_res.mean_exact_iters = mean(exact_iters);
        rob_res.mean_lbfgs_iters = mean(lbfgs_iters);
        rob_res.mean_exact_time  = mean(exact_times);
        rob_res.mean_lbfgs_time  = mean(lbfgs_times);
        rob_res.speedup          = rob_res.mean_lbfgs_time / rob_res.mean_exact_time;
        rob_res.ctc_rmse         = mean(ctc_rmse_list);
        rob_res.open_rmse        = mean(open_rmse_list);

        results.(name) = rob_res;
    end

    fprintf('\n====================================================================\n');
    fprintf('  TABLE 4 TRAJECTORY OPTIMIZATION BENCHMARK SUMMARY                 \n');
    fprintf('====================================================================\n');
    fprintf('%-10s | %-11s | %-11s | %-12s | %-12s | %-10s | %-14s\n', ...
        'Robot', 'Exact Conv', 'L-BFGS Conv', 'Exact Iters', 'L-BFGS Iters', 'Speedup', 'CTC/Open RMSE');
    fprintf('%s\n', repmat('-', 1, 90));

    names = fieldnames(results);
    for k = 1:length(names)
        r = results.(names{k});
        fprintf('%-10s | %-10.1f%% | %-10.1f%% | %-12.1f | %-12.1f | %-9.2fx | %.4f / %.4f\n', ...
            r.name, r.exact_conv_rate, r.lbfgs_conv_rate, r.mean_exact_iters, ...
            r.mean_lbfgs_iters, r.speedup, r.ctc_rmse, r.open_rmse);
    end
    fprintf('====================================================================\n\n');
end
