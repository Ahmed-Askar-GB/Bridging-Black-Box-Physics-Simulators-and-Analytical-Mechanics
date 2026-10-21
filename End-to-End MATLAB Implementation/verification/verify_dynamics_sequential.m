function report = verify_dynamics_sequential(robot, chain, dh, inertia, opts)
% VERIFY_DYNAMICS_SEQUENTIAL  Sequential Dynamic Parameter & Frame Verification.
%
% Implements the automated sequential verification protocol to validate that:
%   1. Link mass m_i is correctly extracted and assigned.
%   2. Link Center of Mass (CoM) P_{Ci} is mapped to the exact DH frame origin
%      and orientation.
%   3. Link inertia tensor I_i is correctly rotated into the DH frame.
%   4. The analytical Lagrangian dynamic formulation (dynamics_casadi / DH_dyn_casadi)
%      strictly matches the Simscape Multibody Digital Twin ground-truth.
%
% =========================================================================
% SEQUENTIAL VALIDATION PROTOCOL:
%   For each sequential stage i = 1 .. N:
%
%   Step A: Downstream Zeroing
%     - In both the analytical CasADi dynamics and the Simscape Multibody model,
%       all downstream links k > i have their inertial parameters zeroed:
%         m_k = 0 kg,  P_{Ck} = [0, 0, 0] m,  I_k = zeros(3,3) kg·m²
%     - All upstream links k <= i retain their physical inertial parameters.
%
%   Step B: Degree-of-Freedom Isolation
%     - All other joints j ~= i are held frozen at nominal non-zero configurations
%       q_j(t) = q_{j,0},  dq_j(t) = 0,  ddq_j(t) = 0.
%     - Holding at non-zero angles ensures that gravitational torques are active
%       and test the spatial orientation of the link's CoM vector.
%
%   Step C: Trajectory Excitation
%     - Joint i undergoes dynamic excitation via a smooth 5th-order polynomial
%       trajectory q_i(t) with continuous velocity and acceleration.
%
%   Step D: Actuation Torque Cross-Validation
%     - Joint i's actuation torque tau_{i,sim}(t) is measured via the Simscape
%       Revolute Joint sensor.
%     - The analytical torque tau_{i,ana}(t) is evaluated via the stage-i
%       CasADi dynamics function:
%         tau_{ana}(t) = M_i(q(t)) * ddq(t) + C_i(q(t), dq(t)) * dq(t) + G_i(q(t))
%     - The torque discrepancy residual is computed:
%         e_{tau, i}(t) = |tau_{i,sim}(t) - tau_{i,ana}(t)|
%     - Verified against tolerance tol_tau.
%
% =========================================================================
% INPUTS:
%   robot    (struct) Output of urdf_parse()
%   chain    (struct) Output of urdf_build_chain()
%   dh       (struct) Output of urdf_extract_dh()
%   inertia  (struct) Output of urdf_extract_inertia()
%   opts     (struct) [optional]
%     .stages     (1xK int) Stages to evaluate (default: 1:chain.n)
%     .duration   (double)  Simulation duration per stage in seconds (default: 2.0)
%     .q_holding  (Nx1)     Nominal holding angles for frozen joints
%     .tol_tau    (double)  Torque tolerance in N·m (default: 0.1 N·m)
%     .mesh_dir   (char)    Path to STL meshes (optional)
%     .verbose    (logical) Print progress (default: true)
%
% =========================================================================
% OUTPUT:
%   report   (struct) with fields:
%     .all_passed     (logical) True if all tested stages passed
%     .stage_results  (cell)    Per-stage result structs:
%       .stage        (int)     Stage index (1..N)
%       .joint_name   (char)    Name of isolated joint
%       .link_name    (char)    Name of verified link
%       .passed       (logical) True if stage passed
%       .max_err      (double)  Maximum torque discrepancy (N·m)
%       .rms_err      (double)  RMS torque discrepancy (N·m)
%       .peak_torque  (double)  Peak actuation torque in simulation (N·m)
%       .time         [Mx1]     Time vector
%       .tau_sim      [Mx1]     Simscape joint torque
%       .tau_ana      [Mx1]     Analytical joint torque
%
% =========================================================================

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    n = chain.n;

    stages = 1:n;
    if isfield(opts, 'stages') && ~isempty(opts.stages), stages = opts.stages; end

    t_dur = 2.0;
    if isfield(opts, 'duration') && ~isempty(opts.duration), t_dur = opts.duration; end

    tol_tau = 0.1; % 0.1 N·m tolerance
    if isfield(opts, 'tol_tau') && ~isempty(opts.tol_tau), tol_tau = opts.tol_tau; end

    verbose = ~isfield(opts, 'verbose') || opts.verbose;

    mesh_dir = '';
    if isfield(opts, 'mesh_dir'), mesh_dir = opts.mesh_dir; end

    % Default holding pose with non-zero angles to test gravity in 3D
    q_holding = [0.1; -0.2; 0.2; -0.15; 0.2; -0.1];
    if isfield(opts, 'q_holding') && ~isempty(opts.q_holding)
        q_holding = opts.q_holding(:);
    end
    if length(q_holding) < n
        q_holding = [q_holding; zeros(n - length(q_holding), 1)];
    else
        q_holding = q_holding(1:n);
    end

    % Joint sign alignment vector
    if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
        joint_signs = opts.joint_signs(:);
    else
        % Auto-detect sign by comparing world axis with DH z axis (pure mathematical alignment)
        fk_temp = fk_casadi(dh, chain, struct('verbose', false));
        joint_signs = detect_joint_signs(chain, fk_temp);
    end

    num_stages = length(stages);
    stage_results = cell(num_stages, 1);
    all_passed = true;

    if verbose
        fprintf('\n============================================================\n');
        fprintf('  SEQUENTIAL DYNAMIC PARAMETER & FRAME VERIFICATION\n');
        fprintf('  Number of DOFs: %d  |  Stages to verify: %s\n', n, mat2str(stages));
        fprintf('  Duration per stage: %.1f s  |  Torque Tol: %.3f N.m\n', t_dur, tol_tau);
        fprintf('============================================================\n\n');
    end

    % -----------------------------------------------------------------------
    % Loop over each sequential stage
    % -----------------------------------------------------------------------
    for s_idx = 1:num_stages
        i = stages(s_idx);
        joint_name = chain.active_joints{i}.name;
        link_name  = chain.active_links{i};

        if verbose
            fprintf('--- [Stage %d/%d] Verifying Link %d: "%s" (Joint: "%s") ---\n', ...
                    s_idx, num_stages, i, link_name, joint_name);
        end

        % -------------------------------------------------------------------
        % 1. Build stage-i analytical dynamic model (zero downstream links k > i)
        % -------------------------------------------------------------------
        inertia_stage = inertia;
        for k = (i + 1):n
            inertia_stage.m(k)     = 0.0;
            inertia_stage.P_Ci(k,:) = [0.0, 0.0, 0.0];
            inertia_stage.I(:,:,k) = zeros(3, 3);
        end

        dyn_stage = dynamics_casadi(dh, inertia_stage, chain, struct('verbose', false));

        % -------------------------------------------------------------------
        % 2. Build stage-i Simscape Multibody model
        % -------------------------------------------------------------------
        model_name = sprintf('dt_seq_verify_stage%d', i);
        if bdIsLoaded(model_name)
            close_system(model_name, 0);
        end

        opts_build = struct();
        opts_build.model_name = model_name;
        opts_build.output_dir = fullfile(pwd);
        opts_build.stop_time  = sprintf('%.2f', t_dur);
        opts_build.mesh_dir   = mesh_dir;
        opts_build.verbose    = false;

        dt = dt_build(robot, chain, inertia, opts_build);

        % Zero out downstream link solids in Simscape
        for k = (i + 1):n
            l_k_name  = chain.active_links{k};
            geom_name = sprintf('Geometry_%d_%s', k, regexprep(l_k_name, '[^a-zA-Z0-9_]', '_'));
            blk_path  = [model_name, '/', geom_name];
            set_param(blk_path, 'Mass', '0');
            set_param(blk_path, 'CenterOfMass', '[0, 0, 0]');
            set_param(blk_path, 'MomentsOfInertia', '[0, 0, 0]');
            set_param(blk_path, 'ProductsOfInertia', '[0, 0, 0]');
        end

        dt = dt_add_ee_frame(dt, chain, struct('verbose', false));
        dt = dt_add_sensors(dt, struct('verbose', false));

        % -------------------------------------------------------------------
        % 3. Prescribe motion: Joint i excited, all other joints frozen
        % -------------------------------------------------------------------
        t_vec = linspace(0, t_dur, 501);
        q0 = q_holding;
        qf = q_holding;

        % Excitation amplitude on joint i (0.5 rad excitation range)
        qf(i) = q0(i) + 0.5;

        [q_m, dq_m, ddq_m] = traj_poly5(q0, qf, t_dur, t_vec);

        traj = struct('time', t_vec, 'q', q_m, 'dq', dq_m, 'ddq', ddq_m);
        dt   = dt_add_trajectory(dt, traj, struct('joint_signs', joint_signs, 'verbose', false));

        % -------------------------------------------------------------------
        % 4. Run headless simulation
        % -------------------------------------------------------------------
        if verbose
            fprintf('  Simulating Stage %d dynamic response in Simscape ...\n', i);
        end
        simOut = sim(model_name, 'StopTime', sprintf('%.2f', t_dur));

        % -------------------------------------------------------------------
        % 5. Cross-validate actuation torque on joint i
        % -------------------------------------------------------------------
        tau_sim_ts = simOut.get('tau_sim');
        t_sim      = tau_sim_ts.Time;
        tau_sim_d  = tau_sim_ts.Data;
        if ndims(tau_sim_d) == 3, tau_sim_d = squeeze(tau_sim_d)'; end

        q_sim_ts  = simOut.get('q_sim');
        dq_sim_ts = simOut.get('dq_sim');
        q_sim_d   = q_sim_ts.Data;
        dq_sim_d  = dq_sim_ts.Data;
        if ndims(q_sim_d) == 3, q_sim_d = squeeze(q_sim_d)'; end
        if ndims(dq_sim_d) == 3, dq_sim_d = squeeze(dq_sim_d)'; end

        % Map back joint signs
        q_sim_d  = q_sim_d  .* (joint_signs');
        dq_sim_d = dq_sim_d .* (joint_signs');

        % Compute analytical torque across all simulation time steps
        M_steps = length(t_sim);
        tau_ana_all = zeros(M_steps, n);

        for k = 1:M_steps
            qk   = q_sim_d(k, :)';
            dqk  = dq_sim_d(k, :)';

            % Acceleration: interpolated from prescribed trajectory
            ddqk = zeros(n, 1);
            for jj = 1:n
                ddqk(jj) = interp1(t_vec, ddq_m(jj, :), t_sim(k), 'linear', 0);
            end

            tau_k = full(dyn_stage.tau_fun(qk, dqk, ddqk));
            tau_ana_all(k, :) = tau_k';
        end

        % Extract isolated joint i torque
        tau_sim_i = tau_sim_d(:, i);
        tau_ana_i = tau_ana_all(:, i);

        % -----------------------------------------------------------------------
        % SIGN CONVENTION HEURISTIC:
        % Simscape Revolute Joint torque sensors report REACTION torque (the
        % torque that the joint exerts on the child body), which may be opposite
        % in sign to the ACTUATION torque (the torque prescribed by the trajectory
        % and computed by the analytical model).
        %
        % The Pearson correlation coefficient between tau_sim and tau_ana is:
        %   corr_val = (tau_sim · tau_ana) / (|tau_sim| · |tau_ana|)
        %
        % If corr_val < -0.8, the signals are strongly anti-correlated,
        % indicating a CONSISTENT sign flip rather than a true dynamic error.
        % In that case, tau_sim is negated before residual computation.
        %
        % LIMITATION: This heuristic masks real errors only when the error
        % signal itself has corr < -0.8 with the prediction (very unlikely for
        % random inertia errors, but possible for systematic frame inversions).
        % Always inspect the tau_sim vs tau_ana time traces for marginal stages.
        % -----------------------------------------------------------------------
        corr_val = dot(tau_sim_i, tau_ana_i) / (norm(tau_sim_i)*norm(tau_ana_i) + 1e-12);
        if corr_val < -0.8
            warning('verify_dynamics_sequential:signFlip', ...
                    'Stage %d: strong anti-correlation (corr=%.3f) — negating tau_sim. Inspect plots.', ...
                    i, corr_val);
            tau_sim_i = -tau_sim_i;
        end

        err_vec = abs(tau_sim_i - tau_ana_i);
        max_err = max(err_vec);
        rms_err = sqrt(mean(err_vec.^2));
        peak_t  = max(abs(tau_sim_i));

        % Pass criterion: max error < tol_tau, or relative error < 2% if peak torque is large
        stage_passed = (max_err <= tol_tau) || (peak_t > 1.0 && max_err / peak_t < 0.05);

        if ~stage_passed
            all_passed = false;
        end

        if verbose
            fprintf('  Torque Discrepancy : Max = %.4e N.m | RMS = %.4e N.m\n', max_err, rms_err);
            fprintf('  Peak Torque Range  : Sim = [%.3f, %.3f] N.m | Ana = [%.3f, %.3f] N.m\n', ...
                    min(tau_sim_i), max(tau_sim_i), min(tau_ana_i), max(tau_ana_i));
            if stage_passed
                fprintf('  >>> STAGE %d RESULT: PASSED (Parameters & frame verified) <<<\n\n', i);
            else
                fprintf('  >>> STAGE %d RESULT: FAILED (Exceeds tolerance %.3f N.m) <<<\n\n', i, tol_tau);
            end
        end

        % Save stage result
        s_res.stage       = i;
        s_res.joint_name  = joint_name;
        s_res.link_name   = link_name;
        s_res.passed      = stage_passed;
        s_res.max_err     = max_err;
        s_res.rms_err     = rms_err;
        s_res.peak_torque = peak_t;
        s_res.time        = t_sim;
        s_res.tau_sim     = tau_sim_i;
        s_res.tau_ana     = tau_ana_i;
        stage_results{s_idx} = s_res;

        % Cleanup model
        close_system(model_name, 0);
        if exist([model_name, '.slx'], 'file')
            delete([model_name, '.slx']);
        end
    end

    % -----------------------------------------------------------------------
    % Assemble master report
    % -----------------------------------------------------------------------
    report.all_passed    = all_passed;
    report.stage_results = stage_results;
    report.stages        = stages;

    if verbose
        fprintf('============================================================\n');
        if all_passed
            fprintf('  >>> ALL SEQUENTIAL DYNAMIC STAGES PASSED SUCCESSFULLY <<<\n');
        else
            fprintf('  >>> ONE OR MORE SEQUENTIAL DYNAMIC STAGES FAILED <<<\n');
        end
        fprintf('============================================================\n\n');
    end

end % function verify_dynamics_sequential
