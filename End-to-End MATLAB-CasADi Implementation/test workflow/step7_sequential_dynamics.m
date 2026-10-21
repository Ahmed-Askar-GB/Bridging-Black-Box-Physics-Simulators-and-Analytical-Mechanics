function dyn_report = step7_sequential_dynamics(robot, chain, dh, inertia, joint_signs, cfg)
% STEP7_SEQUENTIAL_DYNAMICS  Sequential dynamic torque cross-validation across all joints.
%
% Performs:
%   1. Dynamic torque verification for each joint sequentially using
%      verify_dynamics_sequential.
%   2. Fail-fast error assertion if any link exceeds cfg.tol.seq_tau.
%   3. Returns detailed stage-by-stage dynamic torque reports.
%
% =========================================================================
% INPUTS:
%   robot       (struct) Parsed URDF robot structure.
%   chain       (struct) Kinematic chain struct.
%   dh          (struct) Standard DH parameter table and frames.
%   inertia     (struct) Extracted inertial parameters (mass, CoM, inertia).
%   joint_signs [Nx1]    Actuation sign vector.
%   cfg         (struct) Configuration struct containing tolerances.
%
% OUTPUT:
%   dyn_report  (struct) Report from verify_dynamics_sequential.
% =========================================================================

    fprintf('\n>>> STEP 7: Sequential Dynamic Torque Cross-Validation across all joints ...\n');
    n = chain.n;

    opts_seq = struct();
    opts_seq.stages      = 1:n;
    opts_seq.duration    = 2.0;
    opts_seq.tol_tau     = cfg.tol.seq_tau; % Scale-aware torque tolerance
    opts_seq.verbose     = false;
    opts_seq.joint_signs = joint_signs;

    dyn_report = verify_dynamics_sequential(robot, chain, dh, inertia, opts_seq);

    for s = 1:length(dyn_report.stage_results)
        st = dyn_report.stage_results{s};
        if ~st.passed
            error('step7_sequential_dynamics:DynamicStageFailed', ...
                  '[FAIL-FAST] Dynamic torque cross-validation FAILED at Link %d ("%s")! Discrepancy: %.4e N.m (Tol: %.2f N.m)', ...
                  st.stage, st.link_name, st.max_err, cfg.tol.seq_tau);
        end
        fprintf('  [PASS] Link %d ("%-18s") Dynamic Torque: Max Err = %.4e N.m (Peak: %.2f N.m, Tol: %.2f N.m)\n', ...
                st.stage, st.link_name, st.max_err, st.peak_torque, cfg.tol.seq_tau);
    end

end
