function report = main_test_workflow(urdf_path, user_opts)
% MAIN_TEST_WORKFLOW  End-to-end automated verification pipeline for URDF models.
%
% This pipeline encapsulates the dual-model verification architecture into
% modular, sequential execution steps:
%
%   Step 0: Setup environment, paths, dynamic naming, and scale-aware tolerances.
%   Step 1: Parse URDF and extract the active kinematic chain.
%   Step 2: Extract standard DH parameters and inertial parameters.
%   Step 3: Build CasADi symbolic models and verify Lagrangian invariants.
%   Step 4: Generate DH Simulink twin and native smimport Simulink twin.
%   Step 5: Cross-validate inertial parameters (mass, CoM, inertia).
%   Step 6: Dynamic simulation and fail-fast kinematic/differential verification.
%   Step 7: Sequential dynamic torque cross-validation across all joints.
%   Step 8: Assemble diagnostic report, save to disk, and present summary.
%
% =========================================================================
% USAGE:
%   % 1. Run directly with default UR16e robot:
%   main_test_workflow;
%
%   % 2. Run with custom URDF path and custom tolerance overrides:
%   opts = struct();
%   opts.tol_pos     = 5e-4;   % 0.5 mm
%   opts.tol_rot     = 1e-4;   % 0.1 mrad
%   opts.tol_seq_tau = 10.0;   % 10 N.m torque tolerance
%   report = main_test_workflow('path/to/my_robot.urdf', opts);
% =========================================================================

    % ---------------------------------------------------------------------
    % Default Configuration & Adjustable Tolerances
    % ---------------------------------------------------------------------
    if nargin < 1 || isempty(urdf_path)
        this_dir   = fileparts(mfilename('fullpath'));
        parent_dir = fileparts(this_dir);
        urdf_path  = fullfile(parent_dir, 'test_UR16e', 'universalUR16e.urdf');
    end

    if nargin < 2 || isempty(user_opts)
        user_opts = struct();
        % Default adjustable parameters (can be modified here directly):
        % user_opts.tol_pos     = 1e-3;   % EE position tolerance [m]
        % user_opts.tol_rot     = 5e-3;   % EE rotation geodesic tolerance [rad]
        % user_opts.tol_vel     = 1e-2;   % EE linear velocity tolerance [m/s]
        % user_opts.tol_omega   = 1e-2;   % EE angular velocity tolerance [rad/s]
        % user_opts.tol_seq_tau = 10.0;   % Sequential dynamics torque tolerance [N.m]
        % user_opts.t_dur       = 5.0;    % Trajectory simulation duration [s]
    end

    % ---------------------------------------------------------------------
    % Step 0: Setup Environment, Dynamic Naming, and Tolerances
    % ---------------------------------------------------------------------
    cfg = step0_setup_config(urdf_path, user_opts);

    % ---------------------------------------------------------------------
    % Step 1: Parse URDF and Extract Kinematic Chain
    % ---------------------------------------------------------------------
    [robot, chain] = step1_parse_chain(cfg);

    % ---------------------------------------------------------------------
    % Step 2: Extract Standard DH Parameters & Inertial Parameters
    % ---------------------------------------------------------------------
    [dh, inertia, cfg] = step2_extract_parameters(robot, chain, cfg);

    % ---------------------------------------------------------------------
    % Step 3: CasADi Symbolic Kinematics, Dynamics & Invariants
    % ---------------------------------------------------------------------
    [fk, jac, dyn, joint_signs] = step3_symbolic_invariants(dh, chain, inertia, cfg);

    % ---------------------------------------------------------------------
    % Step 4: Build & Save Triple Models (DH Twin, smimport, smimport Sensed)
    % ---------------------------------------------------------------------
    [dt, compiled_model, dt_smimport_sensed] = step4_build_models(robot, chain, inertia, fk, jac, dyn, joint_signs, cfg);

    % ---------------------------------------------------------------------
    % Step 5: Cross-Validate Extracted Inertial Parameters
    % ---------------------------------------------------------------------
    discrepancy_list = step5_cross_validate_parameters(robot, chain, inertia, cfg);

    % ---------------------------------------------------------------------
    % Step 6: Simulate DH Model & Perform Fail-Fast Verification
    % ---------------------------------------------------------------------
    sim_results = step6_simulate_and_verify(dt, fk, jac, joint_signs, cfg);

    % ---------------------------------------------------------------------
    % Step 7: Sequential Dynamic Torque Cross-Validation
    % ---------------------------------------------------------------------
    dyn_report = step7_sequential_dynamics(robot, chain, dh, inertia, joint_signs, cfg);

    % ---------------------------------------------------------------------
    % Step 8: Assemble Diagnostic Report & Final Summary
    % ---------------------------------------------------------------------
    total_time = toc(cfg.t_start);
    report = step8_assemble_report(robot, chain, dh, inertia, sim_results, dyn_report, discrepancy_list, total_time, cfg);

end
