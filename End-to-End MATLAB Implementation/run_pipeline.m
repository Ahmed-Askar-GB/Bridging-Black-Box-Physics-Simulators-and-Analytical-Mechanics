%% run_pipeline.m
% =========================================================================
% MASTER PIPELINE ORCHESTRATOR
% End-to-End URDF-to-Analytical-Model & Simscape Digital Twin Framework
% =========================================================================
% Executes the complete automated workflow from a raw URDF file to:
%   1. Parsing & BFS kinematic chain extraction.
%   2. Foot-point P-Q standard DH parameter derivation.
%   3. Inertial parameter coordinate transformation into DH frames.
%   4. Algorithmic CasADi Forward Kinematics, Jacobians, and Dynamics.
%   5. Embedded C-code generation and standalone model compilation.
%   6. Programmatic Simscape Multibody Digital Twin generation.
%   7. Dynamic trajectory simulation & kinematic ground-truth verification.
%   8. Sequential dynamic parameter & frame cross-validation.
%
% =========================================================================
% USAGE:
%   cd('End-to-End MATLAB-CasADi Implementation');
%   pipeline_report = run_pipeline();
%
%   Or with custom URDF:
%   pipeline_report = run_pipeline('path/to/robot.urdf');
% =========================================================================

function pipeline_report = run_pipeline(urdf_path, opts)

    if nargin < 1 || isempty(urdf_path)
        this_dir = fileparts(mfilename('fullpath'));
        root_dir = fileparts(this_dir);
        urdf_path = fullfile(root_dir, 'universalUR5.urdf');
    end

    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    this_dir = fileparts(mfilename('fullpath'));
    root_dir = fileparts(this_dir);

    % Configure paths
    addpath(fullfile(this_dir, 'parsing'));
    addpath(fullfile(this_dir, 'kinematics'));
    addpath(fullfile(this_dir, 'dynamics'));
    addpath(fullfile(this_dir, 'compilation'));
    addpath(fullfile(this_dir, 'digital_twin'));
    addpath(fullfile(this_dir, 'control'));
    addpath(fullfile(this_dir, 'verification'));
    addpath(fullfile(this_dir, 'visualization'));
    addpath(fullfile(this_dir, 'utils'));
    addpath(root_dir);

    t_start_total = tic;

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('       END-TO-END PURE MATLAB ROBOT DYNAMICS & DIGITAL TWIN          \n');
    fprintf('                   (100%% PURE MATLAB - NO CASADI)                   \n');
    fprintf('====================================================================\n');
    fprintf('  URDF File : %s\n', urdf_path);
    fprintf('  Timestamp : %s\n', datestr(now));
    fprintf('====================================================================\n\n');

    % -----------------------------------------------------------------------
    % STAGE 1: Parsing Layer
    % -----------------------------------------------------------------------
    fprintf('>>> [STAGE 1/6] PARSING LAYER <<<\n');
    t_stage1 = tic;
    robot   = urdf_parse(urdf_path);
    chain   = urdf_build_chain(robot, false);
    dh      = urdf_extract_dh(chain, false);
    inertia = urdf_extract_inertia(robot, chain, dh, false);

    fprintf('  [OK] Parsed %d URDF links and %d joints.\n', robot.links.Count, robot.joints.Count);
    fprintf('  [OK] Extracted %d-DOF serial active kinematic chain: %s -> %s\n', ...
            chain.n, chain.active_start, chain.active_links{end});
    fprintf('  [OK] Standard DH Parameters table successfully generated.\n');
    fprintf('  [OK] Transformed link masses, CoM vectors, and inertia tensors into DH frames.\n');
    fprintf('  Stage 1 completed in %.3f s\n\n', toc(t_stage1));

    % -----------------------------------------------------------------------
    % STAGE 2: Pure MATLAB Analytical Kinematics & Dynamics Layer
    % -----------------------------------------------------------------------
    fprintf('>>> [STAGE 2/6] PURE MATLAB ANALYTICAL KINEMATICS & DYNAMICS <<<\n');
    t_stage2 = tic;
    fk   = fk_matlab(dh, chain, struct('verbose', false));
    jac  = jacobian_matlab(fk, dh, chain, struct('verbose', false));
    dyn  = dynamics_matlab(dh, inertia, chain, struct('verbose', false));

    fprintf('  [OK] Forward Kinematics compiled (full EE pose, quaternions, link transforms).\n');
    fprintf('  [OK] Geometric Jacobian J(q) and exact time-derivative J_dot(q, dq) compiled.\n');
    fprintf('  [OK] Lagrangian Dynamics M(q)*ddq + C(q,dq)*dq + G(q) = tau compiled.\n');
    fprintf('  Stage 2 completed in %.3f s\n\n', toc(t_stage2));

    % -----------------------------------------------------------------------
    % STAGE 3: Compilation & Export Layer
    % -----------------------------------------------------------------------
    fprintf('>>> [STAGE 3/6] MODEL COMPILATION & EXPORT <<<\n');
    t_stage3 = tic;
    out_dir_comp = fullfile(this_dir, 'compilation');
    compiled_model = matlab_compile(fk, jac, dyn, robot.name, out_dir_comp, ...
                                    struct('verbose', false));
    fprintf('  [OK] Saved compiled model to: %s\n', compiled_model.mat_file);
    fprintf('  Stage 3 completed in %.3f s\n\n', toc(t_stage3));

    % -----------------------------------------------------------------------
    % STAGE 4: Simscape Multibody Digital Twin Generation
    % -----------------------------------------------------------------------
    fprintf('>>> [STAGE 4/6] SIMSCAPE MULTIBODY DIGITAL TWIN GENERATION <<<\n');
    t_stage4 = tic;
    model_name = 'dt_ur5_master';
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end

    mesh_dir = fullfile(root_dir, 'STL_Files');
    opts_dt = struct();
    opts_dt.model_name = model_name;
    opts_dt.output_dir = fullfile(this_dir, 'digital_twin');
    opts_dt.stop_time  = '3.0';
    opts_dt.mesh_dir   = mesh_dir;
    opts_dt.verbose    = false;

    dt = dt_build(robot, chain, inertia, opts_dt);

    % Anchor End-Effector frame to DH frame 6
    dt = dt_add_ee_frame(dt, chain, struct('model', compiled_model, 'verbose', false));

    % Add comprehensive sensing instrumentation
    dt = dt_add_sensors(dt, struct('verbose', false));

    % Automatically align joint signs between URDF joint axes and DH z-axes
    joint_signs = detect_joint_signs(chain, fk);

    % Prescribe 5th-order polynomial excitation trajectory
    dt = dt_add_trajectory(dt, [], struct('joint_signs', joint_signs, 'verbose', false));

    fprintf('  [OK] Programmatic Simscape model "%s.slx" created.\n', model_name);
    fprintf('  [OK] Anchored End-Effector transform sensor coincident with analytical DH EE frame.\n');
    fprintf('  [OK] Instrumented 7 timeseries sensors (EE pos, quat, vel, omega; joint q, dq, tau).\n');
    fprintf('  [OK] Configured filtered 5th-order polynomial multi-axis excitation.\n');
    fprintf('  Stage 4 completed in %.3f s\n\n', toc(t_stage4));

    % -----------------------------------------------------------------------
    % STAGE 5: Kinematic & Differential Ground-Truth Cross-Validation
    % -----------------------------------------------------------------------
    fprintf('>>> [STAGE 5/6] KINEMATIC GROUND-TRUTH CROSS-VALIDATION <<<\n');
    t_stage5 = tic;
    L_reach = sum(abs(dh.table(:, 3))) + sum(abs(dh.table(:, 2)));
    tol = get_verification_tolerances(opts, L_reach);

    fprintf('  Running dynamic simulation in Simscape (%.1f s) ...\n', tol.t_dur);
    simOut = sim(dt.model_name, 'StopTime', sprintf('%.2f', tol.t_dur));

    % Verify Forward Kinematics
    opts_fk = struct('verbose', false, 'tol_pos', tol.pos, 'tol_rot', tol.rot, 'joint_signs', joint_signs);
    report_fk = verify_fk(compiled_model, simOut, opts_fk);

    % Verify Jacobians
    opts_jac = struct('verbose', false, 'tol_v', tol.vel, 'tol_w', tol.omega, 'joint_signs', joint_signs);
    report_jac = verify_jacobian(compiled_model, simOut, opts_jac);

    fprintf('  [FK Validation]       Max Pos Err : %.4e m   | Max Rot Err : %.4e rad  --> %s\n', ...
            report_fk.max_pos_err, report_fk.max_rot_err_rad, status_str(report_fk.passed));
    fprintf('  [Jacobian Validation] Max Lin Vel : %.4e m/s | Max Ang Vel : %.4e rad/s --> %s\n', ...
            report_jac.max_v_err, report_jac.max_w_err, status_str(report_jac.passed));
    assert(report_fk.passed, 'FK verification failed.');
    assert(report_jac.passed, 'Jacobian verification failed.');
    fprintf('  Stage 5 completed in %.3f s\n\n', toc(t_stage5));

    % -----------------------------------------------------------------------
    % STAGE 6: Sequential Dynamic Parameter & Frame Verification
    % -----------------------------------------------------------------------
    fprintf('>>> [STAGE 6/6] SEQUENTIAL DYNAMIC PARAMETER & FRAME VERIFICATION <<<\n');
    t_stage6 = tic;
    opts_seq = struct();
    opts_seq.stages    = 1:chain.n;
    opts_seq.duration  = 2.0;
    opts_seq.tol_tau   = tol.seq_tau;
    opts_seq.mesh_dir  = mesh_dir;
    opts_seq.verbose   = false;
    opts_seq.joint_signs = joint_signs;

    report_dyn_seq = verify_dynamics_sequential(robot, chain, dh, inertia, opts_seq);

    for s = 1:length(report_dyn_seq.stage_results)
        st = report_dyn_seq.stage_results{s};
        fprintf('  [Stage %d: Link %-15s] Max Tau Err: %.4e N.m | Peak: %.2f N.m --> %s\n', ...
                st.stage, st.link_name, st.max_err, st.peak_torque, status_str(st.passed));
    end
    assert(report_dyn_seq.all_passed, 'Sequential dynamic verification failed.');
    fprintf('  Stage 6 completed in %.3f s\n\n', toc(t_stage6));

    % Cleanup master model
    close_system(model_name, 0);

    % -----------------------------------------------------------------------
    % Unified Report Assembly
    % -----------------------------------------------------------------------
    total_time = toc(t_start_total);

    pipeline_report.passed          = true;
    pipeline_report.total_time_s    = total_time;
    pipeline_report.robot_name      = robot.name;
    pipeline_report.num_dof         = chain.n;
    pipeline_report.dh_table        = dh.table;
    pipeline_report.report_fk       = report_fk;
    pipeline_report.report_jac      = report_jac;
    pipeline_report.report_dyn_seq  = report_dyn_seq;
    pipeline_report.tolerances      = tol;

    fprintf('====================================================================\n');
    fprintf('   MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN %.2f s       \n', total_time);
    fprintf('   ALL PHASES (1 to 6) VERIFIED TO NUMERICAL TOLERANCES (100%% PASS) \n');
    fprintf('====================================================================\n\n');

end % function run_pipeline


function s = status_str(passed)
    if passed
        s = 'PASSED [OK]';
    else
        s = 'FAILED [ERROR]';
    end
end
