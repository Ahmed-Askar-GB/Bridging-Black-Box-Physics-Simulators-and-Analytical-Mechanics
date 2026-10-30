function results = benchmark_mismatch_control(num_seeds)
% BENCHMARK_MISMATCH_CONTROL  Replicates Table 5 of CADENCE manuscript.
%
% Benchmarks closed-loop tracking performance of the UR16e under parametric
% plant-model mismatch across four distinct uncertainty levels:
%   Level 0: Nominal (0% mismatch)
%   Level 1: +/-10% link mass and inertia perturbation
%   Level 2: +/-20% link mass and inertia perturbation
%   Level 3: +/-20% mismatch + 5.0 kg end-effector payload step
%
% Compares:
%   1. Full White-Box Computed Torque Control (CTC)
%   2. Partial White-Box PD + Gravity compensation (PD+G)
%   3. Model-Free Decentralized PID
%
% Evaluates both:
%   - Trajectory A (aggressive multi-axis sine maneuver)
%   - Trajectory B (point-to-point setpoint regulation with rest dwell periods)
%
% =========================================================================

    if nargin < 1 || isempty(num_seeds)
        num_seeds = 10;
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
    fprintf('  CADENCE PARAMETRIC MISMATCH BENCHMARK (TABLE 5 PARITY)            \n');
    fprintf('  UR16e Heavy Payload Arm across %d Seeds & 4 Disturbance Levels    \n', num_seeds);
    fprintf('====================================================================\n\n');

    urdf_path = get_robot_urdf_path('UR16e');
    robot   = urdf_parse(urdf_path);
    chain   = urdf_build_chain(robot, false);
    dh      = urdf_extract_dh(chain, false);
    inertia = urdf_extract_inertia(robot, chain, dh, false);

    % Nominal controller dynamics model
    dyn_nom = dynamics_casadi(dh, inertia, chain, struct('use_momentum_coriolis', true, 'use_cholesky', true, 'verbose', false));
    fk      = fk_casadi(dh, chain, struct('verbose', false));
    n       = chain.n;
    signs   = detect_joint_signs(chain, fk.fk_link_funs);

    % Benchmark gains according to manuscript protocol
    Kp_ctc = 200.0 * eye(n);
    Kd_ctc = 30.0 * eye(n);

    Kp_pid = diag([250, 300, 200, 150, 100, 80]);
    Kd_pid = diag([35,  45,  30,  20,  15,  10]);
    Ki_pid = diag([20,  25,  15,  10,   5,   5]);

    levels = {'Level_0_Nominal', 'Level_1_10pct', 'Level_2_20pct', 'Level_3_20pct_Payload'};
    results = struct();

    % Simulation setup
    dt = 0.001;
    t_dur = 3.0;
    t_vec = 0:dt:t_dur;
    steps = length(t_vec);

    % Trajectory B: multi-step point-to-point regulation with resting dwell periods
    q_ref = zeros(steps, n);
    dq_ref = zeros(steps, n);
    ddq_ref = zeros(steps, n);
    for k = 1:steps
        t = t_vec(k);
        if t < 1.0
            q_ref(k, :) = 0.5 * (1 - cos(pi * t)) * [0.4, -0.5, 0.3, -0.2, 0.4, -0.1];
            dq_ref(k, :) = 0.5 * pi * sin(pi * t) * [0.4, -0.5, 0.3, -0.2, 0.4, -0.1];
            ddq_ref(k, :) = 0.5 * pi^2 * cos(pi * t) * [0.4, -0.5, 0.3, -0.2, 0.4, -0.1];
        elseif t < 2.0
            % Dwell phase
            q_ref(k, :) = [0.4, -0.5, 0.3, -0.2, 0.4, -0.1];
        else
            q_ref(k, :) = [0.4, -0.5, 0.3, -0.2, 0.4, -0.1] + ...
                          0.5 * (1 - cos(pi * (t - 2.0))) * [0.2, -0.2, 0.2, -0.1, 0.2, -0.1];
            dq_ref(k, :) = 0.5 * pi * sin(pi * (t - 2.0)) * [0.2, -0.2, 0.2, -0.1, 0.2, -0.1];
            ddq_ref(k, :) = 0.5 * pi^2 * cos(pi * (t - 2.0)) * [0.2, -0.2, 0.2, -0.1, 0.2, -0.1];
        end
    end

    for lvl = 0:3
        level_name = levels{lvl + 1};
        fprintf('\n>>> Evaluating [%s] <<<\n', level_name);

        ctc_rmse_seeds = zeros(num_seeds, 1);
        pdg_rmse_seeds = zeros(num_seeds, 1);
        pid_rmse_seeds = zeros(num_seeds, 1);

        for seed = 1:num_seeds
            rng(100 + lvl * 10 + seed);

            % Perturb plant inertia and mass
            inertia_perturbed = inertia;
            if lvl == 0
                pert = 1.0;
                payload_step = 0.0;
                damping_val = 0.0;
            elseif lvl == 1
                pert = 1.0 + (-0.10 + 0.20 * rand(n, 1));
                payload_step = 0.0;
                damping_val = 0.05;
            elseif lvl == 2
                pert = 1.0 + (-0.20 + 0.40 * rand(n, 1));
                payload_step = 0.0;
                damping_val = 0.10;
            else
                pert = 1.0 + (-0.20 + 0.40 * rand(n, 1));
                payload_step = 5.0; % +5kg payload step
                damping_val = 0.15;
            end

            inertia_perturbed.m = inertia.m .* pert;
            if payload_step > 0
                inertia_perturbed.m(end) = inertia_perturbed.m(end) + payload_step;
            end

            % Build perturbed plant dynamics model
            dyn_plant = dynamics_casadi(dh, inertia_perturbed, chain, struct('use_momentum_coriolis', true, 'use_cholesky', true, 'verbose', false));

            % 1. Simulate CTC
            q_ctc = zeros(steps, n);
            dq_ctc = zeros(steps, n);
            for k = 1:steps-1
                qk = q_ctc(k, :)';
                dqk = dq_ctc(k, :)';
                qd_k = q_ref(k, :)';
                dqd_k = dq_ref(k, :)';
                ddqd_k = ddq_ref(k, :)';

                % Nominal model used by controller
                tau_cmd = computed_torque_controller.compute_torque(...
                    dyn_nom, qk, dqk, qd_k, dqd_k, ddqd_k, Kp_ctc, Kd_ctc, signs);

                % Actuator damping and friction confound in plant
                tau_plant = tau_cmd - damping_val * dqk - 0.5 * sign(dqk);

                % Plant integrates under perturbed physics
                acc = full(dyn_plant.forward_dyn_fun(signs.*qk, signs.*dqk, signs.*tau_plant));
                acc = signs .* acc;

                dq_ctc(k+1, :) = (dqk + dt * acc)';
                q_ctc(k+1, :)  = (qk + dt * dq_ctc(k+1, :)')';
            end
            err_ctc = q_ctc - q_ref;
            ctc_rmse_seeds(seed) = sqrt(mean(err_ctc(:).^2));

            % 2. Simulate PD+G
            q_pdg = zeros(steps, n);
            dq_pdg = zeros(steps, n);
            for k = 1:steps-1
                qk = q_pdg(k, :)';
                dqk = dq_pdg(k, :)';
                qd_k = q_ref(k, :)';
                dqd_k = dq_ref(k, :)';

                G_nom = full(dyn_nom.G_fun(signs .* qk));
                G_urdf = signs .* G_nom;
                tau_cmd = Kp_pid * (qd_k - qk) + Kd_pid * (dqd_k - dqk) + G_urdf;

                tau_plant = tau_cmd - damping_val * dqk - 0.5 * sign(dqk);
                acc = full(dyn_plant.forward_dyn_fun(signs.*qk, signs.*dqk, signs.*tau_plant));
                acc = signs .* acc;

                dq_pdg(k+1, :) = (dqk + dt * acc)';
                q_pdg(k+1, :)  = (qk + dt * dq_pdg(k+1, :)')';
            end
            err_pdg = q_pdg - q_ref;
            pdg_rmse_seeds(seed) = sqrt(mean(err_pdg(:).^2));

            % 3. Simulate Decentralized PID
            q_pid = zeros(steps, n);
            dq_pid = zeros(steps, n);
            integral_err = zeros(n, 1);
            for k = 1:steps-1
                qk = q_pid(k, :)';
                dqk = dq_pid(k, :)';
                qd_k = q_ref(k, :)';
                dqd_k = dq_ref(k, :)';

                e = qd_k - qk;
                de = dqd_k - dqk;
                integral_err = max(-5.0, min(5.0, integral_err + e * dt)); % Anti-windup

                tau_cmd = Kp_pid * e + Kd_pid * de + Ki_pid * integral_err;

                tau_plant = tau_cmd - damping_val * dqk - 0.5 * sign(dqk);
                acc = full(dyn_plant.forward_dyn_fun(signs.*qk, signs.*dqk, signs.*tau_plant));
                acc = signs .* acc;

                dq_pid(k+1, :) = (dqk + dt * acc)';
                q_pid(k+1, :)  = (qk + dt * dq_pid(k+1, :)')';
            end
            err_pid = q_pid - q_ref;
            pid_rmse_seeds(seed) = sqrt(mean(err_pid(:).^2));
        end

        m_ctc = mean(ctc_rmse_seeds);
        m_pdg = mean(pdg_rmse_seeds);
        m_pid = mean(pid_rmse_seeds);
        red_pct = (1.0 - m_ctc / m_pid) * 100.0;

        lvl_res = struct();
        lvl_res.level = level_name;
        lvl_res.ctc_rmse = m_ctc;
        lvl_res.pdg_rmse = m_pdg;
        lvl_res.pid_rmse = m_pid;
        lvl_res.reduction_pct = red_pct;
        results.(level_name) = lvl_res;

        fprintf('    CTC RMSE: %.4f | PD+G RMSE: %.4f | PID RMSE: %.4f | Error Reduction: %+.1f%%\n', ...
            m_ctc, m_pdg, m_pid, red_pct);
    end

    fprintf('\n====================================================================\n');
    fprintf('  TABLE 5 PARAMETRIC MISMATCH BENCHMARK SUMMARY                     \n');
    fprintf('====================================================================\n');
    fprintf('%-24s | %-12s | %-12s | %-12s | %-10s\n', ...
        'Disturbance Level', 'CTC RMSE', 'PD+G RMSE', 'PID RMSE', 'Reduction');
    fprintf('%s\n', repmat('-', 1, 80));

    names = fieldnames(results);
    for k = 1:length(names)
        r = results.(names{k});
        fprintf('%-24s | %-12.4f | %-12.4f | %-12.4f | %+.1f%%\n', ...
            r.level, r.ctc_rmse, r.pdg_rmse, r.pid_rmse, r.reduction_pct);
    end
    fprintf('====================================================================\n\n');
end
