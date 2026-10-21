function report = verify_jacobian(model, sim_data, opts)
% VERIFY_JACOBIAN  Quantitative Differential Kinematics Cross-Validation.
%
% Compares the End-Effector spatial velocities (linear and angular) logged
% by the Simscape Multibody Digital Twin against the analytical Geometric
% Jacobian J(q) model derived via CasADi (jacobian_casadi).
%
% =========================================================================
% VERIFICATION PROTOCOL:
%   At each simulation time step k = 1..M:
%     1. Read joint state [q_sim(k,:); dq_sim(k,:)] from Simscape simulation.
%     2. Evaluate analytical Jacobian:
%          J_num = full(jac.J_fun(q_sim(k,:)'))   % [6xN]
%          v_ana = J_num(1:3, :) * dq_sim(k,:)'   % [3x1] linear velocity
%          w_ana = J_num(4:6, :) * dq_sim(k,:)'   % [3x1] angular velocity
%     3. Read ground-truth Simscape Transform Sensor velocity readings:
%          v_sim = ee_vel(k,:)'   [3x1] (m/s)
%          w_sim = ee_omega(k,:)' [3x1] (rad/s)
%     4. Compute discrepancy metrics:
%          e_v(k) = ||v_sim(k) - v_ana(k)||_2
%          e_w(k) = ||w_sim(k) - w_ana(k)||_2
%
% =========================================================================
% TOLERANCES & PASS CRITERIA:
%   - Maximum Linear Velocity Error:  < 1.0e-4 m/s
%   - Maximum Angular Velocity Error: < 1.0e-4 rad/s
%
% =========================================================================
% INPUTS:
%   model     (struct) Output of casadi_compile() or jacobian_casadi().
%                      Must contain: .jac (or be jac struct itself)
%   sim_data  (struct or Simulink.SimulationOutput)
%                      Contains: 'q_sim', 'dq_sim', 'ee_vel', 'ee_omega'
%   opts      (struct) [optional]
%     .tol_v    (double)  Linear velocity tolerance in m/s (default: 1e-4)
%     .tol_w    (double)  Angular velocity tolerance in rad/s (default: 1e-4)
%     .verbose  (logical) Print summary to console (default: true)
%     .plot     (logical) Generate validation figure (default: false)
%
% =========================================================================
% OUTPUT:
%   report    (struct) with fields:
%     .passed        (logical) True if all metrics within tolerances
%     .max_v_err     (double)  Max linear velocity error (m/s)
%     .rms_v_err     (double)  RMS linear velocity error (m/s)
%     .max_w_err     (double)  Max angular velocity error (rad/s)
%     .rms_w_err     (double)  RMS angular velocity error (rad/s)
%     .time          [Mx1]     Time vector
%     .v_err_vec     [Mx1]     Linear velocity error vs time
%     .w_err_vec     [Mx1]     Angular velocity error vs time
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m
%   test_phase4_verification.m

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    tol_v = 1e-4;
    if isfield(opts, 'tol_v') && ~isempty(opts.tol_v), tol_v = opts.tol_v; end

    tol_w = 1e-4;
    if isfield(opts, 'tol_w') && ~isempty(opts.tol_w), tol_w = opts.tol_w; end

    verbose = ~isfield(opts, 'verbose') || opts.verbose;
    do_plot = isfield(opts, 'plot') && opts.plot;

    % Extract jac struct
    if isfield(model, 'jac')
        jac = model.jac;
    else
        jac = model;
    end
    assert(isfield(jac, 'J_fun'), 'jac struct must contain compiled J_fun.');

    % -----------------------------------------------------------------------
    % 1. Extract simulation signals
    % -----------------------------------------------------------------------
    q_data   = get_data(sim_data, 'q_sim');    % [MxN]
    dq_data  = get_data(sim_data, 'dq_sim');   % [MxN]
    v_data   = get_data(sim_data, 'ee_vel');   % [Mx3]
    w_data   = get_data(sim_data, 'ee_omega'); % [Mx3]
    time_vec = get_time(sim_data, 'ee_vel');   % [Mx1]

    % Apply joint sign mapping if specified
    if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
        q_data  = q_data  .* (opts.joint_signs(:)');
        dq_data = dq_data .* (opts.joint_signs(:)');
    end

    M_steps = size(q_data, 1);
    assert(M_steps > 0, 'Simulation data contains 0 time steps.');

    % -----------------------------------------------------------------------
    % 2. Evaluate Analytical Jacobian & Compute Residuals
    % -----------------------------------------------------------------------
    v_err = zeros(M_steps, 1);
    w_err = zeros(M_steps, 1);

    for k = 1:M_steps
        qk  = q_data(k, :)';   % [Nx1]
        dqk = dq_data(k, :)';  % [Nx1]

        % Analytical Jacobian evaluation
        J_num = full(jac.J_fun(qk));   % [6xN]
        v_ana = J_num(1:3, :) * dqk;   % [3x1]
        w_ana = J_num(4:6, :) * dqk;   % [3x1]

        % Ground-truth Simscape Transform Sensor readings
        v_sim = v_data(k, :)';         % [3x1]
        w_sim = w_data(k, :)';         % [3x1]

        % Discrepancies
        v_err(k) = norm(v_sim - v_ana);
        w_err(k) = norm(w_sim - w_ana);
    end

    % -----------------------------------------------------------------------
    % 3. Aggregate Metrics & Pass/Fail Evaluation
    % -----------------------------------------------------------------------
    max_v = max(v_err);
    rms_v = sqrt(mean(v_err.^2));
    max_w = max(w_err);
    rms_w = sqrt(mean(w_err.^2));

    passed = (max_v <= tol_v) && (max_w <= tol_w);

    if verbose
        fprintf('\n============================================================\n');
        fprintf('  DIFFERENTIAL KINEMATICS REPORT: Jacobian vs Digital Twin\n');
        fprintf('============================================================\n');
        fprintf('  Time Steps Evaluated : %d (duration: %.2f s)\n', M_steps, time_vec(end));
        fprintf('  Linear Vel Error (m/s)  : Max = %.4e  |  RMS = %.4e  (Tol: %.1e)\n', ...
                max_v, rms_v, tol_v);
        fprintf('  Angular Vel Error(rad/s): Max = %.4e  |  RMS = %.4e  (Tol: %.1e)\n', ...
                max_w, rms_w, tol_w);
        if passed
            fprintf('  >>> STATUS: PASSED (Ground-truth velocities match Jacobian) <<<\n');
        else
            fprintf('  >>> STATUS: FAILED (Velocity discrepancy exceeds tolerance) <<<\n');
        end
        fprintf('============================================================\n\n');
    end

    % -----------------------------------------------------------------------
    % 4. Assemble Report Struct
    % -----------------------------------------------------------------------
    report.passed    = passed;
    report.max_v_err = max_v;
    report.rms_v_err = rms_v;
    report.max_w_err = max_w;
    report.rms_w_err = rms_w;
    report.time      = time_vec;
    report.v_err_vec = v_err;
    report.w_err_vec = w_err;

    % Optional plot
    if do_plot
        figure('Name', 'Jacobian Velocity Residuals', 'Color', 'w');
        subplot(2, 1, 1);
        plot(time_vec, v_err * 1e3, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
        yline(tol_v * 1e3, '--r', 'Tolerance');
        ylabel('Linear Vel Error (mm/s)');
        grid on; title('Digital Twin vs Analytical Geometric Jacobian');

        subplot(2, 1, 2);
        plot(time_vec, rad2deg(w_err), 'LineWidth', 1.5, 'Color', [0.85 0.325 0.098]);
        yline(rad2deg(tol_w), '--r', 'Tolerance');
        ylabel('Angular Vel Error (deg/s)'); xlabel('Time (s)');
        grid on;
    end

end % function verify_jacobian


% =========================================================================
%  LOCAL HELPERS
% =========================================================================

function data = get_data(sim_data, vname)
    if isstruct(sim_data) && isfield(sim_data, vname)
        raw = sim_data.(vname);
    elseif isa(sim_data, 'Simulink.SimulationOutput')
        raw = sim_data.get(vname);
    elseif evalin('base', sprintf('exist(''%s'', ''var'')', vname))
        raw = evalin('base', vname);
    else
        error('verify_jacobian:signalNotFound', 'Signal "%s" not found.', vname);
    end

    if isa(raw, 'timeseries')
        val = raw.Data;
    else
        val = raw;
    end

    if ndims(val) == 3
        data = squeeze(val)';
    else
        data = val;
    end
end

function t = get_time(sim_data, vname)
    if isstruct(sim_data) && isfield(sim_data, vname)
        raw = sim_data.(vname);
    elseif isa(sim_data, 'Simulink.SimulationOutput')
        raw = sim_data.get(vname);
    elseif evalin('base', sprintf('exist(''%s'', ''var'')', vname))
        raw = evalin('base', vname);
    else
        t = []; return;
    end

    if isa(raw, 'timeseries')
        t = raw.Time;
    else
        t = (1:size(raw, 1))';
    end
end
