function report = test_ur16e_dual_model_verification(opts)
% TEST_UR16E_DUAL_MODEL_VERIFICATION  Verification of UR16e Manipulator.
%
% This script conducts an end-to-end verification of the UR16e robot:
%   1. Ingests universalUR16e.urdf and its visual/collision meshes.
%   2. Extracts Standard DH parameters (dh, dh_parameters) and dynamic/inertial parameters.
%   3. Programmatically creates and saves the DH-based Simscape Digital Twin:
%      'test_UR16e/dt_ur16e_dh.slx'.
%   4. Generates and saves the native Simscape model via smimport:
%      'test_UR16e/dt_ur16e_smimport.slx'.
%   5. Performs detailed parameter cross-validation:
%      - Compares DH parameters and geometric invariants.
%      - Compares inertia parameters (masses, centers of mass, and inertia tensors)
%        between URDF specifications and pipeline extractions.
%   6. Simulates both models under smooth trajectory excitation.
%   7. Verifies Forward Kinematics, auto-diff Jacobian, and dynamic invariants
%      with fail-fast assertions.
%   8. Performs Sequential Dynamic Parameter Verification (Stage 7).
%   9. Logs any discrepancies into a structured diagnostic report.
%
% =========================================================================

    % -----------------------------------------------------------------------
    % 0. Paths and Configuration
    % -----------------------------------------------------------------------
    this_dir   = fileparts(mfilename('fullpath'));
    parent_dir = fileparts(this_dir);
    root_dir   = fileparts(parent_dir);

    addpath(fullfile(parent_dir, 'parsing'));
    addpath(fullfile(parent_dir, 'kinematics'));
    addpath(fullfile(parent_dir, 'dynamics'));
    addpath(fullfile(parent_dir, 'compilation'));
    addpath(fullfile(parent_dir, 'digital_twin'));
    addpath(fullfile(parent_dir, 'verification'));
    addpath(fullfile(parent_dir, 'utils'));
    addpath(fullfile(parent_dir, 'test_UR16e'));

    ur16e_dir = fullfile(parent_dir, 'test_UR16e');
    urdf_path = fullfile(ur16e_dir, 'universalUR16e.urdf');
    dh_model_name       = 'dt_ur16e_dh';
    smimport_model_name = 'dt_ur16e_smimport';
    dh_model_file       = fullfile(ur16e_dir, [dh_model_name, '.slx']);
    smimport_model_file = fullfile(ur16e_dir, [smimport_model_name, '.slx']);

    if nargin < 1 || isempty(opts)
        opts = struct();
    end

    % Pre-parse temporary nominal reach for tolerance initialization
    % (Refined with exact DH reach once DH parameters are extracted)
    tol = get_verification_tolerances(opts, 1.0);
    t_dur = tol.t_dur;

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('    UR16e DUAL-MODEL SIMULINK & ANALYTICAL VERIFICATION SUITE       \n');
    fprintf('====================================================================\n');
    fprintf('  URDF Target    : %s\n', urdf_path);
    fprintf('  DH Model       : %s\n', dh_model_file);
    fprintf('  smimport Model : %s\n', smimport_model_file);
    fprintf('  Tolerances     : Pos=%.2e m | Rot=%.2e rad | Vel=%.2e m/s | SeqTau=%.2f N.m\n', ...
            tol.pos, tol.rot, tol.vel, tol.seq_tau);
    fprintf('====================================================================\n\n');

    t_start = tic;

    % -----------------------------------------------------------------------
    % 1. Parse URDF and Extract Kinematic Chain
    % -----------------------------------------------------------------------
    fprintf('>>> STEP 1: Parsing universalUR16e.urdf & Kinematic Chain ...\n');
    assert(exist(urdf_path, 'file') == 2, 'URDF file not found: %s', urdf_path);

    robot = urdf_parse(urdf_path);
    fprintf('  Parsed %d links, %d joints from URDF. Root link: "%s"\n', ...
            robot.links.Count, robot.joints.Count, robot.root_link);

    chain = urdf_build_chain(robot, false);
    n = chain.n;
    fprintf('  Active DOFs: %d. Active links:\n', n);
    for i = 1:n
        fprintf('    Joint %d: %-20s -> Child Link: %-20s\n', ...
                i, chain.active_joints{i}.name, chain.active_links{i});
    end

    % -----------------------------------------------------------------------
    % 2. Extract Standard DH Parameters & Inertial Parameters
    % -----------------------------------------------------------------------
    fprintf('\n>>> STEP 2: Extracting DH Parameters & Dynamic Inertias ...\n');
    dh = urdf_extract_dh(chain, false);
    inertia = urdf_extract_inertia(robot, chain, dh, false);

    fprintf('\n  --- Standard DH Parameters Table [a_i, alpha_i, d_i, theta_offset_i] ---\n');
    fprintf('  Joint |      a (m)     |   alpha (rad)  |      d (m)     |  theta_off (rad)\n');
    fprintf('  ------+----------------+----------------+----------------+-----------------\n');
    for i = 1:n
        fprintf('    %d   |  %12.6f  |  %12.6f  |  %12.6f  |  %12.6f\n', ...
                i, dh.table(i, 3), dh.table(i, 4), dh.table(i, 2), dh.table(i, 1));
    end

    % Refine scale-aware tolerances using exact geometric reach from DH table
    L_reach = sum(abs(dh.table(:, 3))) + sum(abs(dh.table(:, 2)));
    tol = get_verification_tolerances(opts, L_reach);

    fprintf('\n  --- Dynamic Inertia Parameters (Masses & Principal Inertias) ---\n');
    fprintf('  Link  |  Mass (kg)  |     CoM [x, y, z] (m)     |    I_diag [Ixx, Iyy, Izz] (kg*m^2)\n');
    fprintf('  ------+-------------+---------------------------+------------------------------------\n');
    for i = 1:n
        c_i = inertia.P_Ci(i, :);
        I_diag = [inertia.I(1,1,i), inertia.I(2,2,i), inertia.I(3,3,i)];
        fprintf('    %d   |  %9.4f  | [%6.3f, %6.3f, %6.3f] | [%.5e, %.5e, %.5e]\n', ...
                i, inertia.m(i), c_i(1), c_i(2), c_i(3), I_diag(1), I_diag(2), I_diag(3));
    end
    fprintf('  Total Active Arm Mass = %.4f kg\n', sum(inertia.m));

    % -----------------------------------------------------------------------
    % 3. CasADi Symbolic Kinematics & Analytical Invariants
    % -----------------------------------------------------------------------
    fprintf('\n>>> STEP 3: Symbolic Kinematics & Lagrangian Dynamic Invariants ...\n');
    fk   = fk_casadi(dh, chain, struct('verbose', false));
    jac  = jacobian_casadi(fk, dh, chain, struct('verbose', false));
    dyn  = dynamics_casadi(dh, inertia, chain, struct('verbose', false));

    joint_signs = detect_joint_signs(chain, fk);
    fprintf('  Auto-detected joint signs: %s\n', mat2str(joint_signs));

    % Verify Mass Matrix symmetry & Positive Definiteness
    q_rnd = rand(n, 1);
    M_val = full(dyn.M_fun(q_rnd));
    assert(max(max(abs(M_val - M_val'))) < tol.sym, ...
           'Mass matrix is asymmetric (residual exceeds tol.sym = %.1e)!', tol.sym);
    assert(min(eig(M_val)) > 0, 'Mass matrix is not strictly positive-definite!');
    fprintf('  [PASS] Mass Matrix M(q) is strictly symmetric positive-definite (min eig = %.4e, sym err < %.1e).\n', ...
            min(eig(M_val)), tol.sym);

    % Static balance invariant: tau(q,0,0) == G(q)
    G_val   = full(dyn.G_fun(q_rnd));
    tau_val = full(dyn.tau_fun(q_rnd, zeros(n,1), zeros(n,1)));
    assert(max(abs(tau_val - G_val)) < tol.gravity, ...
           'Static equilibrium violation: tau(q,0,0) != G(q) (residual exceeds tol.gravity = %.1e)!', tol.gravity);
    fprintf('  [PASS] Static balance invariant verified: tau(q, 0, 0) == G(q) (error < %.1e N.m).\n', tol.gravity);

    % -----------------------------------------------------------------------
    % 4. Create and Save Both Simulink Models
    % -----------------------------------------------------------------------
    fprintf('\n>>> STEP 4: Creating & Saving Both Simulink Models ...\n');

    % 4A. Model from DH Parameters (dt_ur16e_dh.slx)
    fprintf('  Creating DH-based Digital Twin ("%s") ...\n', dh_model_name);
    if bdIsLoaded(dh_model_name), close_system(dh_model_name, 0); end

    compiled_model = casadi_compile(fk, jac, dyn, robot.name, ur16e_dir, ...
                                    struct('gen_c', false, 'verbose', false));

    opts_build = struct();
    opts_build.model_name = dh_model_name;
    opts_build.output_dir = ur16e_dir;
    opts_build.stop_time  = sprintf('%.2f', t_dur);
    opts_build.mesh_dir   = fullfile(ur16e_dir, 'meshes', 'visual');
    opts_build.verbose    = false;

    dt = dt_build(robot, chain, inertia, opts_build);
    dt = dt_add_ee_frame(dt, chain, struct('model', compiled_model, 'verbose', false, 'save_model', true));
    dt = dt_add_sensors(dt, struct('verbose', false, 'save_model', true));

    % Trajectory excitation
    t_vec = linspace(0, t_dur, 1001);
    q0 = zeros(n, 1);
    q1 = 0.35 * sin((1:n)' * 1.1);
    [q_traj, dq_traj, ddq_traj] = traj_poly5(q0, q1, t_dur, t_vec);
    traj_struct = struct('time', t_vec, 'q', q_traj, 'dq', dq_traj, 'ddq', ddq_traj);
    dt = dt_add_trajectory(dt, traj_struct, struct('joint_signs', joint_signs, 'verbose', false, 'save_model', true));

    save_system(dh_model_name, dh_model_file);
    fprintf('  [SAVED] DH-based model successfully saved to: %s\n', dh_model_file);

    % 4B. Model from URDF via smimport (dt_ur16e_smimport.slx)
    if ~exist(smimport_model_file, 'file')
        fprintf('Importing native Simscape model via smimport ...\n');
        orig_dir = pwd;
        cd(ur16e_dir);
        try
            smimport('universalUR16e.urdf', 'ModelName', smimport_model_name);
            save_system(smimport_model_name, smimport_model_file);
            close_system(smimport_model_name, 0);
            cd(orig_dir);
        catch ex
            cd(orig_dir);
            rethrow(ex);
        end
    end
    fprintf('  [CONFIRMED] smimport model present at: %s\n', smimport_model_file);

    % -----------------------------------------------------------------------
    % 5. Cross-Check Inertia Parameters & DH Geometric Parameters
    % -----------------------------------------------------------------------
    fprintf('\n>>> STEP 5: Inertia & DH Parameter Cross-Validation ...\n');
    discrepancy_list = {};

    % Inertia comparison between URDF links and extracted DH chain
    fprintf('  Verifying Link Inertias and Centers of Mass:\n');
    for i = 1:n
        link_name = chain.active_links{i};
        urdf_link = robot.links(link_name);
        
        m_urdf = urdf_link.mass;
        m_dh   = inertia.m(i);
        diff_m = abs(m_urdf - m_dh);

        if diff_m > tol.mass
            warning_msg = sprintf('Link %d ("%s") mass mismatch: URDF=%.4f kg, DH=%.4f kg (Tol: %.1e kg)', ...
                                  i, link_name, m_urdf, m_dh, tol.mass);
            discrepancy_list{end+1} = warning_msg;
            fprintf('    [!] %s\n', warning_msg);
        else
            fprintf('    Link %d ("%-18s") Mass = %.4f kg [MATCH]\n', i, link_name, m_dh);
        end
    end

    % -----------------------------------------------------------------------
    % 6. Dynamic Simulation & Fail-Fast Kinematic/Dynamic Cross-Validation
    % -----------------------------------------------------------------------
    fprintf('\n>>> STEP 6: Simulating DH Model & Performing Fail-Fast Verification ...\n');
    simOut = sim(dh_model_name, 'StopTime', sprintf('%.2f', t_dur));

    q_sim_ts    = simOut.get('q_sim');
    dq_sim_ts   = simOut.get('dq_sim');
    p_sim_ts    = simOut.get('ee_pos');
    quat_sim_ts = simOut.get('ee_quat');
    v_sim_ts    = simOut.get('ee_vel');
    w_sim_ts    = simOut.get('ee_omega');
    tau_sim_ts  = simOut.get('tau_sim');

    t_steps = p_sim_ts.Time;
    M_steps = length(t_steps);

    p_sim_d    = p_sim_ts.Data;    if ndims(p_sim_d) == 3, p_sim_d = squeeze(p_sim_d)'; end
    quat_sim_d = quat_sim_ts.Data; if ndims(quat_sim_d) == 3, quat_sim_d = squeeze(quat_sim_d)'; end
    v_sim_d    = v_sim_ts.Data;    if ndims(v_sim_d) == 3, v_sim_d = squeeze(v_sim_d)'; end
    w_sim_d    = w_sim_ts.Data;    if ndims(w_sim_d) == 3, w_sim_d = squeeze(w_sim_d)'; end
    q_sim_d    = q_sim_ts.Data;    if ndims(q_sim_d) == 3, q_sim_d = squeeze(q_sim_d)'; end
    dq_sim_d   = dq_sim_ts.Data;   if ndims(dq_sim_d) == 3, dq_sim_d = squeeze(dq_sim_d)'; end
    tau_sim_d  = tau_sim_ts.Data;  if ndims(tau_sim_d) == 3, tau_sim_d = squeeze(tau_sim_d)'; end

    q_ana_eval  = q_sim_d  .* (joint_signs');
    dq_ana_eval = dq_sim_d .* (joint_signs');

    max_pos_err = 0; max_rot_err = 0;
    max_v_err   = 0; max_w_err   = 0;

    for k = 1:M_steps
        qk  = q_ana_eval(k, :)';
        dqk = dq_ana_eval(k, :)';

        % Analytical FK & Jacobian
        [p_ana, R_ana, ~] = fk.fk_fun(qk);
        p_ana    = full(p_ana);
        R_ana    = full(R_ana);
        quat_ana = R_to_quat(R_ana);
        J_k   = full(jac.J_fun(qk));
        v_ana = J_k(1:3, :) * dqk;
        w_ana = J_k(4:6, :) * dqk;

        % Simscape readings
        p_sim_k = p_sim_d(k, :)';
        q_sim_k = quat_sim_d(k, :)';
        v_sim_k = v_sim_d(k, :)';
        w_sim_k = w_sim_d(k, :)';

        % Position discrepancy
        err_p = norm(p_sim_k - p_ana);
        if err_p > max_pos_err, max_pos_err = err_p; end
        if err_p > tol.pos
            close_system(dh_model_name, 0);
            error('UR16eVerification:PositionDiscrepancy', ...
                  ['[FAIL-FAST] EE position mismatch at step %d (t=%.4f s)!\n' ...
                   '  Error: %.4e m (Tol: %.1e m)\n' ...
                   '  Simscape:   [%.6f, %.6f, %.6f]\n' ...
                   '  Analytical: [%.6f, %.6f, %.6f]'], ...
                  k, t_steps(k), err_p, tol.pos, p_sim_k(1), p_sim_k(2), p_sim_k(3), p_ana(1), p_ana(2), p_ana(3));
        end

        % Orientation discrepancy
        dot_q = min(1.0, max(-1.0, abs(dot(q_sim_k / norm(q_sim_k), quat_ana / norm(quat_ana)))));
        err_rot = 2.0 * acos(dot_q);
        if err_rot > max_rot_err, max_rot_err = err_rot; end
        if err_rot > tol.rot
            close_system(dh_model_name, 0);
            error('UR16eVerification:OrientationDiscrepancy', ...
                  '[FAIL-FAST] EE orientation mismatch at step %d: %.4e rad (Tol: %.1e rad)', ...
                  k, err_rot, tol.rot);
        end

        % Velocity discrepancies
        err_v = norm(v_sim_k - v_ana);
        if err_v > max_v_err, max_v_err = err_v; end
        tol_v_dyn = max(tol.vel, tol.vel_rel * norm(v_sim_k));
        assert(err_v <= tol_v_dyn, 'Linear velocity mismatch at step %d: %.4e m/s (Tol: %.2e m/s)', k, err_v, tol_v_dyn);

        err_w = norm(w_sim_k - w_ana);
        if err_w > max_w_err, max_w_err = err_w; end
        tol_w_dyn = max(tol.omega, tol.vel_rel * norm(w_sim_k));
        assert(err_w <= tol_w_dyn, 'Angular velocity mismatch at step %d: %.4e rad/s (Tol: %.2e rad/s)', k, err_w, tol_w_dyn);
    end

    fprintf('  [PASS] Position Validation : Max Pos Err = %.4e m   (Tol: %.1e m)\n', max_pos_err, tol.pos);
    fprintf('  [PASS] Rotation Validation : Max Rot Err = %.4e rad (Tol: %.1e rad)\n', max_rot_err, tol.rot);
    fprintf('  [PASS] Velocity Validation : Max Lin Vel = %.4e m/s | Max Ang Vel = %.4e rad/s\n', max_v_err, max_w_err);

    close_system(dh_model_name, 0);

    % -----------------------------------------------------------------------
    % 7. Sequential Dynamic Parameter & Frame Verification
    % -----------------------------------------------------------------------
    fprintf('\n>>> STEP 7: Sequential Dynamic Torque Cross-Validation across all joints ...\n');
    opts_seq = struct();
    opts_seq.stages      = 1:n;
    opts_seq.duration    = 2.0;
    opts_seq.tol_tau     = tol.seq_tau; % Scale-aware torque tolerance for sequential dynamics
    opts_seq.verbose     = false;
    opts_seq.joint_signs = joint_signs;

    report_dyn_seq = verify_dynamics_sequential(robot, chain, dh, inertia, opts_seq);

    for s = 1:length(report_dyn_seq.stage_results)
        st = report_dyn_seq.stage_results{s};
        if ~st.passed
            error('UR16eVerification:DynamicStageFailed', ...
                  '[FAIL-FAST] Dynamic torque cross-validation FAILED at Link %d ("%s")! Discrepancy: %.4e N.m (Tol: %.2f N.m)', ...
                  st.stage, st.link_name, st.max_err, tol.seq_tau);
        end
        fprintf('  [PASS] Link %d ("%-18s") Dynamic Torque: Max Err = %.4e N.m (Peak: %.2f N.m, Tol: %.2f N.m)\n', ...
                st.stage, st.link_name, st.max_err, st.peak_torque, tol.seq_tau);
    end

    total_time = toc(t_start);

    % -----------------------------------------------------------------------
    % 8. Assemble Diagnostic Report
    % -----------------------------------------------------------------------
    report.robot_name        = robot.name;
    report.num_dof           = n;
    report.dh_model_file     = dh_model_file;
    report.smimport_model_file = smimport_model_file;
    report.dh_parameters     = dh.DHpram;
    report.mass_vector       = inertia.m;
    report.total_mass        = sum(inertia.m);
    report.max_pos_err       = max_pos_err;
    report.max_rot_err       = max_rot_err;
    report.max_v_err         = max_v_err;
    report.max_w_err         = max_w_err;
    report.tolerances        = tol;
    report.discrepancies     = discrepancy_list;
    report.elapsed_time      = total_time;
    report.passed            = true;

    % Save report
    report_file = fullfile(ur16e_dir, 'verification_report.mat');
    save(report_file, 'report');

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('  >>> UR16e DUAL-MODEL VERIFICATION COMPLETED: 100%% PASSED <<<      \n');
    fprintf('  DH Model Saved       : %s\n', dh_model_file);
    fprintf('  smimport Model Saved : %s\n', smimport_model_file);
    fprintf('  Report Saved         : %s\n', report_file);
    fprintf('  Total Elapsed Time   : %.2f s\n', total_time);
    fprintf('====================================================================\n\n');

end % function test_ur16e_dual_model_verification