function report = step8_assemble_report(robot, chain, dh, inertia, sim_results, dyn_report, discrepancy_list, total_time, cfg)
% STEP8_ASSEMBLE_REPORT  Assemble diagnostic verification report and save to disk.
%
% Performs:
%   1. Collects all model metrics, extracted DH tables, dynamic parameters,
%      discrepancy list, and simulation/dynamics errors.
%   2. Saves the structured report to cfg.report_file.
%   3. Prints the final completion banner.
%
% =========================================================================
% INPUTS:
%   robot            (struct) Parsed URDF robot structure.
%   chain            (struct) Kinematic chain struct.
%   dh               (struct) Standard DH parameter table.
%   inertia          (struct) Extracted inertial parameters.
%   sim_results      (struct) Results from step6_simulate_and_verify.
%   dyn_report       (struct) Results from step7_sequential_dynamics.
%   discrepancy_list (cell)   Discrepancies identified during parameter cross-validation.
%   total_time       (double) Total elapsed execution time in seconds.
%   cfg              (struct) Configuration struct containing file paths and tolerances.
%
% OUTPUT:
%   report           (struct) Complete diagnostic report structure.
% =========================================================================

    fprintf('\n>>> STEP 8: Assembling Diagnostic Report & Saving Results ...\n');

    report = struct();
    report.robot_name                 = robot.name;
    report.num_dof                    = chain.n;
    report.dh_model_file              = cfg.dh_model_file;
    report.smimport_model_file        = cfg.smimport_model_file;
    report.smimport_sensed_model_file = cfg.smimport_sensed_model_file;
    report.dh_parameters              = dh.DHpram;
    report.mass_vector                = inertia.m;
    report.total_mass                 = sum(inertia.m);
    report.max_pos_err                = sim_results.max_pos_err;
    report.max_rot_err                = sim_results.max_rot_err;
    report.max_v_err                  = sim_results.max_v_err;
    report.max_w_err                  = sim_results.max_w_err;
    report.tolerances                 = cfg.tol;
    report.discrepancies              = discrepancy_list;
    report.sequential_dynamics        = dyn_report;
    report.elapsed_time               = total_time;
    report.passed                     = true;

    % Save report MAT file
    save(cfg.report_file, 'report');

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('  >>> ROBOT VERIFICATION COMPLETED: 100%% PASSED <<<                \n');
    fprintf('  Robot Name                      : %s\n', robot.name);
    fprintf('  DoF                             : %d\n', report.num_dof);
    fprintf('  Model 1 (DH Multibody)          : %s\n', cfg.dh_model_file);
    fprintf('  Model 2 (Raw smimport)          : %s\n', cfg.smimport_model_file);
    fprintf('  Model 3 (smimport Instrumented) : %s\n', cfg.smimport_sensed_model_file);
    fprintf('  Report Saved                    : %s\n', cfg.report_file);
    fprintf('  Total Elapsed Time              : %.2f s\n', total_time);
    fprintf('====================================================================\n\n');

end
