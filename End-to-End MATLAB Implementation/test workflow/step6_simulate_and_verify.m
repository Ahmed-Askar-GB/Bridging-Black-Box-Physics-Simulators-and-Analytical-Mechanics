function sim_results = step6_simulate_and_verify(dt, fk, jac, joint_signs, cfg)
% STEP6_SIMULATE_AND_VERIFY  Dynamic simulation and fail-fast kinematic/differential cross-check.
%
% Performs:
%   1. Dynamic simulation of the DH-based Digital Twin under excitation.
%   2. Reading and re-mapping of Simscape joint timeseries into analytical frames.
%   3. Point-by-point cross-validation:
%      - End-Effector position error vs analytical FK (Fail-Fast: tol.pos).
%      - End-Effector quaternion geodesic error vs analytical FK (Fail-Fast: tol.rot).
%      - End-Effector linear velocity vs analytical Jacobian (Fail-Fast: tol.vel).
%      - End-Effector angular velocity vs analytical Jacobian (Fail-Fast: tol.omega).
%
% =========================================================================
% INPUTS:
%   dt          (struct) Digital Twin model struct.
%   fk          (struct) CasADi Forward Kinematics function object.
%   jac         (struct) CasADi Jacobian function object.
%   joint_signs [Nx1]    Actuation sign vector.
%   cfg         (struct) Configuration struct with tolerances and model names.
%
% OUTPUT:
%   sim_results (struct) Simulation validation metrics (max errors and timeseries).
% =========================================================================

    fprintf('\n>>> STEP 6: Simulating DH Model & Performing Fail-Fast Verification ...\n');

    simOut = sim(cfg.dh_model_name, 'StopTime', sprintf('%.2f', cfg.tol.t_dur));

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

        % Analytical FK & Jacobian evaluation
        [p_ana, R_ana, ~] = fk.fk_fun(qk);
        p_ana    = full(p_ana);
        R_ana    = full(R_ana);
        quat_ana = R_to_quat(R_ana);
        J_k      = full(jac.J_fun(qk));
        v_ana    = J_k(1:3, :) * dqk;
        w_ana    = J_k(4:6, :) * dqk;

        % Simscape readings
        p_sim_k = p_sim_d(k, :)';
        q_sim_k = quat_sim_d(k, :)';
        v_sim_k = v_sim_d(k, :)';
        w_sim_k = w_sim_d(k, :)';

        % 1. Position discrepancy check
        err_p = norm(p_sim_k - p_ana);
        if err_p > max_pos_err, max_pos_err = err_p; end
        if err_p > cfg.tol.pos
            close_system(cfg.dh_model_name, 0);
            error('step6_simulate_and_verify:PositionDiscrepancy', ...
                  ['[FAIL-FAST] EE position mismatch at step %d (t=%.4f s)!\n' ...
                   '  Error: %.4e m (Tol: %.1e m)\n' ...
                   '  Simscape:   [%.6f, %.6f, %.6f]\n' ...
                   '  Analytical: [%.6f, %.6f, %.6f]'], ...
                  k, t_steps(k), err_p, cfg.tol.pos, ...
                  p_sim_k(1), p_sim_k(2), p_sim_k(3), p_ana(1), p_ana(2), p_ana(3));
        end

        % 2. Orientation discrepancy check
        dot_q = min(1.0, max(-1.0, abs(dot(q_sim_k / norm(q_sim_k), quat_ana / norm(quat_ana)))));
        err_rot = 2.0 * acos(dot_q);
        if err_rot > max_rot_err, max_rot_err = err_rot; end
        if err_rot > cfg.tol.rot
            close_system(cfg.dh_model_name, 0);
            error('step6_simulate_and_verify:OrientationDiscrepancy', ...
                  '[FAIL-FAST] EE orientation mismatch at step %d: %.4e rad (Tol: %.1e rad)', ...
                  k, err_rot, cfg.tol.rot);
        end

        % 3. Linear velocity discrepancy check
        err_v = norm(v_sim_k - v_ana);
        if err_v > max_v_err, max_v_err = err_v; end
        tol_v_dyn = max(cfg.tol.vel, cfg.tol.vel_rel * norm(v_sim_k));
        assert(err_v <= tol_v_dyn, ...
               'step6_simulate_and_verify:LinearVelocityMismatch', ...
               'Linear velocity mismatch at step %d: %.4e m/s (Tol: %.2e m/s)', k, err_v, tol_v_dyn);

        % 4. Angular velocity discrepancy check
        err_w = norm(w_sim_k - w_ana);
        if err_w > max_w_err, max_w_err = err_w; end
        tol_w_dyn = max(cfg.tol.omega, cfg.tol.vel_rel * norm(w_sim_k));
        assert(err_w <= tol_w_dyn, ...
               'step6_simulate_and_verify:AngularVelocityMismatch', ...
               'Angular velocity mismatch at step %d: %.4e rad/s (Tol: %.2e rad/s)', k, err_w, tol_w_dyn);
    end

    fprintf('  [PASS] Position Validation : Max Pos Err = %.4e m   (Tol: %.1e m)\n', max_pos_err, cfg.tol.pos);
    fprintf('  [PASS] Rotation Validation : Max Rot Err = %.4e rad (Tol: %.1e rad)\n', max_rot_err, cfg.tol.rot);
    fprintf('  [PASS] Velocity Validation : Max Lin Vel = %.4e m/s | Max Ang Vel = %.4e rad/s\n', max_v_err, max_w_err);

    close_system(cfg.dh_model_name, 0);

    % Assemble simulation output results
    sim_results = struct();
    sim_results.max_pos_err = max_pos_err;
    sim_results.max_rot_err = max_rot_err;
    sim_results.max_v_err   = max_v_err;
    sim_results.max_w_err   = max_w_err;
    sim_results.time        = t_steps;
    sim_results.p_sim       = p_sim_d;
    sim_results.q_sim       = q_sim_d;
    sim_results.dq_sim      = dq_sim_d;
    sim_results.tau_sim     = tau_sim_d;

end
