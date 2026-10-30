function results = benchmark_multi_dof_scalability(dof_list)
% BENCHMARK_MULTI_DOF_SCALABILITY  Replicates Table 3 of CADENCE manuscript.
%
% Generates synthetic serial manipulator chains across n in {3, 4, 6, 7, 10, 12, 14}
% and benchmarks:
%   1. Mass Matrix M(q) and Gravity G(q) opcodes
%   2. Coriolis Bias: Baseline Christoffel (n^3.25) vs Proposed Momentum (n^2.29)
%   3. Forward Dynamics: Baseline Inversion (n^3.22) vs Proposed Cholesky (n^2.32)
%   4. Derivation latency (s) and single-step evaluation latency (us)
%
% =========================================================================

    if nargin < 1 || isempty(dof_list)
        dof_list = [3, 4, 6, 7, 10, 12, 14];
    end

    this_dir = fileparts(mfilename('fullpath'));
    base_dir = fileparts(this_dir);

    addpath(base_dir);
    addpath(fullfile(base_dir, 'parsing'));
    addpath(fullfile(base_dir, 'kinematics'));
    addpath(fullfile(base_dir, 'dynamics'));

    fprintf('\n====================================================================\n');
    fprintf('  CADENCE MULTI-DOF SCALABILITY BENCHMARK (TABLE 3 PARITY)          \n');
    fprintf('  Evaluating Asymptotic Scaling across n in [%s] DOFs              \n', num2str(dof_list));
    fprintf('====================================================================\n\n');

    import casadi.*

    results = struct();

    for idx = 1:length(dof_list)
        n = dof_list(idx);
        fprintf('>>> Benchmarking Chain with n = %d DOFs <<<\n', n);

        % Synthesize canonical synthetic DH table and link inertias
        DHpram = zeros(n, 5);
        sigma  = ones(n, 1); % all revolute
        m_vec  = 2.5 * ones(n, 1);
        P_Ci   = zeros(n, 3);
        I_ten  = zeros(3, 3, n);

        for i = 1:n
            DHpram(i, :) = [0.25, (-1)^i * pi/2, 0.0, 0.15, 1.0];
            P_Ci(i, :)   = [0.10, 0.0, 0.05];
            I_ten(:, :, i) = diag([0.02, 0.03, 0.02]);
        end

        dh = struct('DHpram', DHpram, 'sigma', sigma);
        inertia = struct('m', m_vec, 'P_Ci', P_Ci, 'I', I_ten);
        chain = struct('n', n);

        % 1. Compile Proposed Formulation (Momentum + Cholesky)
        t0 = tic;
        dyn_prop = dynamics_casadi(dh, inertia, chain, ...
            struct('use_momentum_coriolis', true, 'use_cholesky', true, 'verbose', false));
        t_deriv = toc(t0);

        % 2. Compile Baseline Formulation (Christoffel + Inversion)
        dyn_base = dynamics_casadi(dh, inertia, chain, ...
            struct('use_momentum_coriolis', false, 'use_cholesky', false, 'verbose', false));

        % Extract DAG instruction / operation counts via CasADi n_instructions
        n_op_M = dyn_prop.M_fun.n_instructions();
        n_op_G = dyn_prop.G_fun.n_instructions();
        n_op_c_prop = dyn_prop.c_bias_fun.n_instructions();
        n_op_c_base = dyn_base.C_fun.n_instructions();
        n_op_fd_prop = dyn_prop.forward_dyn_fun.n_instructions();
        n_op_fd_base = dyn_base.forward_dyn_fun.n_instructions();

        c_red_pct  = (1.0 - n_op_c_prop / n_op_c_base) * 100.0;
        fd_red_pct = (1.0 - n_op_fd_prop / n_op_fd_base) * 100.0;

        % Evaluate latency over 1000 steps
        q_test   = 0.1 * ones(n, 1);
        dq_test  = 0.05 * ones(n, 1);
        tau_test = 1.0 * ones(n, 1);

        t_bench_prop = tic;
        for k = 1:1000
            res = full(dyn_prop.forward_dyn_fun(q_test, dq_test, tau_test));
        end
        lat_prop_us = (toc(t_bench_prop) / 1000.0) * 1e6;

        t_bench_base = tic;
        for k = 1:1000
            res = full(dyn_base.forward_dyn_fun(q_test, dq_test, tau_test));
        end
        lat_base_us = (toc(t_bench_base) / 1000.0) * 1e6;

        speedup = lat_base_us / lat_prop_us;

        entry = struct();
        entry.n = n;
        entry.M_op = n_op_M;
        entry.G_op = n_op_G;
        entry.c_base_op = n_op_c_base;
        entry.c_prop_op = n_op_c_prop;
        entry.c_reduction_pct = c_red_pct;
        entry.fd_base_op = n_op_fd_base;
        entry.fd_prop_op = n_op_fd_prop;
        entry.fd_reduction_pct = fd_red_pct;
        entry.derivation_s = t_deriv;
        entry.lat_prop_us = lat_prop_us;
        entry.lat_base_us = lat_base_us;
        entry.speedup = speedup;

        field_name = sprintf('DOF_%d', n);
        results.(field_name) = entry;

        fprintf('  [n=%2d] Coriolis Red: %+.1f%% | FD Red: %+.1f%% | Latency: %.1f us (Speedup: %.2fx)\n', ...
            n, c_red_pct, fd_red_pct, lat_prop_us, speedup);
    end

    fprintf('\n=========================================================================================\n');
    fprintf('  TABLE 3 QUANTITATIVE MULTI-DOF COMPLEXITY & SCALABILITY DASHBOARD                      \n');
    fprintf('=========================================================================================\n');
    fprintf('%-4s | %-7s | %-6s | %-10s | %-10s | %-8s | %-10s | %-10s | %-8s\n', ...
        'DOFs', 'M(q) Op', 'G(q) Op', 'C Base Op', 'C Prop Op', 'C Red %', 'FD Base Op', 'FD Prop Op', 'Speedup');
    fprintf('%s\n', repmat('-', 1, 95));

    fields = fieldnames(results);
    for k = 1:length(fields)
        e = results.(fields{k});
        fprintf('%-4d | %-7d | %-6d | %-10d | %-10d | %+7.1f%% | %-10d | %-10d | %6.2fx\n', ...
            e.n, e.M_op, e.G_op, e.c_base_op, e.c_prop_op, e.c_reduction_pct, ...
            e.fd_base_op, e.fd_prop_op, e.speedup);
    end
    fprintf('=========================================================================================\n\n');
end
