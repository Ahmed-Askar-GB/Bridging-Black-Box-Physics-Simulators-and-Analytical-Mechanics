function results = run_multirobot_5stage_verification(robot_list)
% RUN_MULTIROBOT_5STAGE_VERIFICATION  Automated 5-stage verification suite.
%
% Replicates Table 2 of the CADENCE Scientific Reports manuscript across
% diverse industrial manipulators:
%   1. UR5 (6-DOF non-spherical wrist)
%   2. UR16e (6-DOF heavy payload)
%   3. ABB IRB 120 (6-DOF spherical wrist)
%   4. Puma 560 (6-DOF classic benchmark with spectral sanitization)
%   5. KUKA LBR iiwa 7 (7-DOF redundant chain)
%
% =========================================================================

    if nargin < 1 || isempty(robot_list)
        robot_list = {'UR5', 'UR16e', 'ABB_IRB120', 'Puma560', 'KUKA_iiwa'};
    end

    this_dir = fileparts(mfilename('fullpath'));
    base_dir = fileparts(this_dir);

    addpath(base_dir);
    addpath(fullfile(base_dir, 'parsing'));
    addpath(fullfile(base_dir, 'kinematics'));
    addpath(fullfile(base_dir, 'dynamics'));
    addpath(fullfile(base_dir, 'control'));
    addpath(fullfile(base_dir, 'verification'));
    addpath(fullfile(base_dir, 'utils'));

    fprintf('\n====================================================================\n');
    fprintf('  CADENCE 5-STAGE MULTI-ROBOT VERIFICATION SUITE (MATLAB-CASADI)   \n');
    fprintf('  Target Publication: Nature Scientific Reports                     \n');
    fprintf('====================================================================\n\n');

    results = struct();

    for r = 1:length(robot_list)
        name = robot_list{r};
        fprintf('\n>>> [PLATFORM %d/%d] Ingesting Robot: %s <<<\n', r, length(robot_list), name);

        try
            urdf_path = get_robot_urdf_path(name);
            fprintf('    URDF Path: %s\n', urdf_path);
        catch ME
            fprintf('    [SKIP] %s not found: %s\n', name, ME.message);
            continue;
        end

        robot_res = struct();
        robot_res.name = name;
        robot_res.urdf_path = urdf_path;

        % ---------------------------------------------------------------------
        % STAGE 1: Model Ingestion, Inertial Projection & Invariant Audit
        % ---------------------------------------------------------------------
        fprintf('  --- Stage 1: Structural & Inertial Consistency ---\n');
        t0 = tic;
        robot   = urdf_parse(urdf_path);
        chain   = urdf_build_chain(robot, false);
        dh      = urdf_extract_dh(chain, false);
        inertia = urdf_extract_inertia(robot, chain, dh, false);
        t_parse = toc(t0);

        n = chain.n;
        robot_res.n = n;
        robot_res.t_parse = t_parse;

        % Physical realizability check (triangle inequalities)
        triangle_violations = 0;
        for i = 1:n
            if iscell(inertia.I)
                I_i = inertia.I{i};
            else
                I_i = squeeze(inertia.I(:, :, i));
            end
            e = sort(eig(0.5*(I_i + I_i')));
            if (e(1) + e(2)) < e(3) - 1e-6
                triangle_violations = triangle_violations + 1;
            end
        end

        % Compile CasADi Model with momentum Coriolis and Cholesky
        t1 = tic;
        dyn = dynamics_casadi(dh, inertia, chain, struct('use_momentum_coriolis', true, 'use_cholesky', true, 'verbose', false));
        fk  = fk_casadi(dh, chain, struct('verbose', false));
        jac = jacobian_casadi(dh, inertia, chain, struct('verbose', false));
        t_model = toc(t1);
        robot_res.t_model = t_model;

        % Mass matrix SPD check across 50 random configurations
        min_eig_M = inf;
        sym_M_err = 0;
        for s = 1:50
            q_rand = -pi + 2*pi*rand(n, 1);
            M_val  = full(dyn.M_fun(q_rand));
            e_M    = eig(0.5*(M_val + M_val'));
            min_eig_M = min(min_eig_M, min(e_M));
            sym_M_err = max(sym_M_err, norm(M_val - M_val', 'fro'));
        end
        robot_res.min_eig_M = min_eig_M;
        robot_res.sym_M_err = sym_M_err;
        robot_res.stage1_pass = (min_eig_M > 0) && (sym_M_err < 1e-10);
        fprintf('    [OK] DOFs: %d | M(q) SPD min eig: %.4e | Triangle Violations: %d | Pass: %d\n', ...
            n, min_eig_M, triangle_violations, robot_res.stage1_pass);

        % ---------------------------------------------------------------------
        % STAGE 2: Forward Kinematics & Quaternion Parity
        % ---------------------------------------------------------------------
        fprintf('  --- Stage 2: Forward Kinematics Verification ---\n');
        max_quat_err = 0;
        max_so3_err  = 0;
        for s = 1:50
            q_rand = -pi + 2*pi*rand(n, 1);
            T_ee = full(fk.fk_ee_fun(q_rand));
            R_ee = T_ee(1:3, 1:3);
            max_so3_err = max(max_so3_err, norm(R_ee * R_ee' - eye(3), 'fro'));
            max_so3_err = max(max_so3_err, abs(det(R_ee) - 1.0));
            quat = R_to_quat(R_ee);
            max_quat_err = max(max_quat_err, abs(norm(quat) - 1.0));
        end
        robot_res.max_quat_err = max_quat_err;
        robot_res.max_so3_err = max_so3_err;
        robot_res.stage2_pass = (max_so3_err < 1e-10) && (max_quat_err < 1e-10);
        fprintf('    [OK] SO(3) Det/Orth Err: %.2e | Quat Unit Err: %.2e | Pass: %d\n', ...
            max_so3_err, max_quat_err, robot_res.stage2_pass);

        % ---------------------------------------------------------------------
        % STAGE 3: Differential Kinematics & Sign Parity
        % ---------------------------------------------------------------------
        fprintf('  --- Stage 3: Differential Kinematics & Sign Parity ---\n');
        signs = detect_joint_signs(chain, fk.fk_link_funs);
        robot_res.signs = signs;

        % Differential velocity verification via central difference
        max_vel_err = 0;
        eps_step = 1e-7;
        for s = 1:20
            q_rand  = -pi + 2*pi*rand(n, 1);
            dq_rand = -1.0 + 2.0*rand(n, 1);
            T_plus  = full(fk.fk_ee_fun(q_rand + eps_step * dq_rand));
            T_minus = full(fk.fk_ee_fun(q_rand - eps_step * dq_rand));
            p_dot_fd = (T_plus(1:3, 4) - T_minus(1:3, 4)) / (2.0 * eps_step);

            Jv_val = full(jac.Jv_fun(q_rand));
            p_dot_ana = Jv_val * dq_rand;
            max_vel_err = max(max_vel_err, norm(p_dot_fd - p_dot_ana));
        end
        robot_res.max_vel_err = max_vel_err;
        robot_res.stage3_pass = (max_vel_err < 1e-6);
        fprintf('    [OK] Sign Parity: [%s] | Diff Vel Err: %.2e m/s | Pass: %d\n', ...
            num2str(signs', '%+d '), max_vel_err, robot_res.stage3_pass);

        % ---------------------------------------------------------------------
        % STAGE 4: Dynamics Consistency & Matrix-Free HVP Verification
        % ---------------------------------------------------------------------
        fprintf('  --- Stage 4: Dynamics Consistency & Matrix-Free HVP ---\n');
        % Check static gravity balance: tau(q, 0, 0) == G(q)
        q_rand = -pi + 2*pi*rand(n, 1);
        G_val = full(dyn.G_fun(q_rand));
        tau_static = full(dyn.tau_fun(q_rand, zeros(n,1), zeros(n,1)));
        grav_balance_err = norm(G_val - tau_static);

        % Forward dynamics roundtrip: ddq_fwd(q, dq, tau_id(q, dq, ddq_true)) == ddq_true
        dq_rand  = -1.0 + 2.0*rand(n, 1);
        ddq_rand = -2.0 + 4.0*rand(n, 1);
        tau_id_val = full(dyn.tau_fun(q_rand, dq_rand, ddq_rand));
        ddq_recovered = full(dyn.forward_dyn_fun(q_rand, dq_rand, tau_id_val));
        fwd_roundtrip_err = norm(ddq_recovered - ddq_rand);

        % Matrix-free HVP verification against central difference
        v_rand = randn(n, 1); v_rand = v_rand / norm(v_rand);
        hvp_ana = full(dyn.hvp_G_fun(q_rand, v_rand));
        G_plus  = full(dyn.G_fun(q_rand + eps_step * v_rand));
        G_minus = full(dyn.G_fun(q_rand - eps_step * v_rand));
        hvp_fd  = (G_plus - G_minus) / (2.0 * eps_step);
        hvp_err = norm(hvp_ana - hvp_fd);

        robot_res.grav_balance_err = grav_balance_err;
        robot_res.fwd_roundtrip_err = fwd_roundtrip_err;
        robot_res.hvp_err = hvp_err;
        robot_res.stage4_pass = (grav_balance_err < 1e-10) && (fwd_roundtrip_err < 1e-10) && (hvp_err < 1e-6);
        fprintf('    [OK] Static Balance Err: %.2e | Fwd Roundtrip: %.2e | HVP Err: %.2e | Pass: %d\n', ...
            grav_balance_err, fwd_roundtrip_err, hvp_err, robot_res.stage4_pass);

        % ---------------------------------------------------------------------
        % STAGE 5: Closed-Loop Computed Torque Control Trajectory Tracking
        % ---------------------------------------------------------------------
        fprintf('  --- Stage 5: Closed-Loop Computed Torque Verification ---\n');
        q0 = zeros(n, 1);
        qf = 0.5 * ones(n, 1);
        [traj_t, q_ref, dq_ref, ddq_ref] = traj_poly5(q0, qf, 1.5, 0.002);

        % Run closed-loop CTC forward simulation
        ctc_opts = struct('kp', 250.0, 'kd', 30.0, 'joint_signs', signs);
        sim_opts = struct('dt', 0.002, 't_dur', 1.5, 'verbose', false);
        sim_res = forward_dynamics_sim(dyn, [q0; zeros(n, 1)], traj_t, q_ref, dq_ref, ddq_ref, ctc_opts, sim_opts);

        tracking_err = sim_res.q_sim - q_ref;
        rmse = sqrt(mean(tracking_err(:).^2));
        max_err = max(abs(tracking_err(:)));

        robot_res.ctc_rmse = rmse;
        robot_res.ctc_max_err = max_err;
        robot_res.stage5_pass = (rmse < 1e-3);
        fprintf('    [OK] CTC Tracking RMSE: %.4e rad | Max Err: %.4e rad | Pass: %d\n', ...
            rmse, max_err, robot_res.stage5_pass);

        robot_res.all_passed = robot_res.stage1_pass && robot_res.stage2_pass && ...
                               robot_res.stage3_pass && robot_res.stage4_pass && robot_res.stage5_pass;
        results.(name) = robot_res;
        fprintf('>>> PLATFORM [%s] 5-STAGE VERIFICATION: %s <<<\n', name, ...
            char(ternary(robot_res.all_passed, '100% PASSED', 'FAILED')));
    end

    fprintf('\n====================================================================\n');
    fprintf('  5-STAGE VERIFICATION SUMMARY DASHBOARD                            \n');
    fprintf('====================================================================\n');
    fprintf('%-12s | %-4s | %-12s | %-12s | %-12s | %-10s\n', ...
        'Robot', 'DOFs', 'M(q) SPD', 'HVP Err', 'CTC RMSE (rad)', 'Pass Rate');
    fprintf('%s\n', repmat('-', 1, 72));

    names = fieldnames(results);
    for k = 1:length(names)
        r = results.(names{k});
        fprintf('%-12s | %-4d | %-12.2e | %-12.2e | %-12.4e | %-10s\n', ...
            r.name, r.n, r.min_eig_M, r.hvp_err, r.ctc_rmse, ...
            char(ternary(r.all_passed, '100% PASS', 'FAIL')));
    end
    fprintf('====================================================================\n\n');
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
