function report = verify_fk(model, sim_data, opts)
% VERIFY_FK  Quantitative Kinematic Ground-Truth Cross-Validation.
%
% Compares the End-Effector (EE) trajectory logged by the Simscape Multibody
% Digital Twin against the analytical Forward Kinematics (FK) model derived
% via CasADi (fk_casadi).
%
% =========================================================================
% VERIFICATION PROTOCOL:
%   At each simulation time step k = 1..M:
%     1. Read joint state q_sim(k,:) from Simscape simulation.
%     2. Evaluate analytical FK:  [p_ana, R_ana, quat_ana] = fk.fk_fun(q_sim(k,:))
%     3. Read ground-truth Simscape Transform Sensor readings:
%        - Position:    p_sim(k,:)   [3x1] (m)
%        - Quaternion:  quat_sim(k,:) [4x1] [qw; qx; qy; qz]
%     4. Compute discrepancy metrics:
%        - Euclidean position error:  e_pos(k) = ||p_sim(k) - p_ana(k)||_2
%        - Quaternion geodesic orientation error angle:
%            theta_err(k) = 2 * acos( min(1.0, |quat_sim' * quat_ana|) )
%          (Note: Geodesic error is chosen over Frobenius matrix error as it is
%           numerically well-behaved across all SO(3) rotations including near π).
%
% =========================================================================
% TOLERANCES & PASS CRITERIA:
%   - Maximum Position Error:    < 1.0e-4 m  (0.1 mm, set by ODE integrator)
%   - Maximum Orientation Error: < 1.0e-4 rad (0.006 deg)
%
% =========================================================================
% INPUTS:
%   model     (struct) Output of casadi_compile() or fk_casadi().
%                      Must contain: .fk (or be fk struct itself)
%   sim_data  (struct or Simulink.SimulationOutput)
%                      Contains timeseries: 'q_sim', 'ee_pos', 'ee_quat'
%   opts      (struct) [optional]
%     .tol_pos  (double)  Position tolerance in meters (default: 1e-4)
%     .tol_rot  (double)  Rotation angle tolerance in radians (default: 1e-4)
%     .verbose  (logical) Print summary to console (default: true)
%     .plot     (logical) Generate validation figure (default: false)
%
% =========================================================================
% OUTPUT:
%   report    (struct) with fields:
%     .passed          (logical) True if all metrics within tolerances
%     .max_pos_err     (double)  Max Euclidean position error (m)
%     .rms_pos_err     (double)  RMS position error (m)
%     .max_rot_err_rad (double)  Max angular orientation error (rad)
%     .rms_rot_err_rad (double)  RMS angular orientation error (rad)
%     .time            [Mx1]     Time vector
%     .pos_err_vec     [Mx1]     Position error vs time
%     .rot_err_vec     [Mx1]     Rotation error vs time
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m
%   test_phase4_verification.m

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    tol_pos = 1e-4;
    if isfield(opts, 'tol_pos') && ~isempty(opts.tol_pos), tol_pos = opts.tol_pos; end

    tol_rot = 1e-4;
    if isfield(opts, 'tol_rot') && ~isempty(opts.tol_rot), tol_rot = opts.tol_rot; end

    verbose = ~isfield(opts, 'verbose') || opts.verbose;
    do_plot = isfield(opts, 'plot') && opts.plot;

    % Extract fk struct
    if isfield(model, 'fk')
        fk = model.fk;
    else
        fk = model;
    end
    assert(isfield(fk, 'fk_fun'), 'fk struct must contain compiled fk_fun.');

    % -----------------------------------------------------------------------
    % 1. Extract simulation signals
    % -----------------------------------------------------------------------
    q_data   = get_data(sim_data, 'q_sim');    % [MxN]
    p_data   = get_data(sim_data, 'ee_pos');   % [Mx3]
    q_quat   = get_data(sim_data, 'ee_quat');  % [Mx4]
    time_vec = get_time(sim_data, 'ee_pos');   % [Mx1]

    % Apply joint sign mapping if specified
    if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
        q_data = q_data .* (opts.joint_signs(:)');
    end

    M_steps = size(q_data, 1);
    assert(M_steps > 0, 'Simulation data contains 0 time steps.');

    % -----------------------------------------------------------------------
    % 2. Evaluate Analytical FK & Compute Residuals
    % -----------------------------------------------------------------------
    pos_err = zeros(M_steps, 1);
    rot_err = zeros(M_steps, 1);

    for k = 1:M_steps
        qk = q_data(k, :)';  % [Nx1]

        % Analytical FK
        [p_ana, ~, quat_ana] = fk.fk_fun(qk);
        % Note: R_ana (rotation matrix) is not used here — orientation error is
        % computed via the quaternion geodesic distance (more numerically stable
        % than Frobenius norm near θ = ±180°).  See docstring metric #2.
        p_ana    = full(p_ana);
        quat_ana = full(quat_ana);   % [qw; qx; qy; qz]
        % Simscape Transform Sensor ground-truth
        p_sim    = p_data(k, :)';
        quat_sim = q_quat(k, :)';    % [qw; qx; qy; qz]

        % 1. Position error
        pos_err(k) = norm(p_sim - p_ana);

        % 2. Quaternion orientation error
        % Normalize quaternions
        q_s_n = quat_sim / norm(quat_sim);
        q_a_n = quat_ana / norm(quat_ana);

        % Geodesic distance on SO(3): theta = 2 * acos(|q1 · q2|)
        dot_q = abs(dot(q_s_n, q_a_n));
        dot_q = min(1.0, max(-1.0, dot_q));
        rot_err(k) = 2.0 * acos(dot_q);
    end

    % -----------------------------------------------------------------------
    % 3. Aggregate Metrics & Pass/Fail Evaluation
    % -----------------------------------------------------------------------
    max_pos = max(pos_err);
    rms_pos = sqrt(mean(pos_err.^2));
    max_rot = max(rot_err);
    rms_rot = sqrt(mean(rot_err.^2));

    passed = (max_pos <= tol_pos) && (max_rot <= tol_rot);

    if verbose
        fprintf('\n============================================================\n');
        fprintf('  KINEMATIC VERIFICATION REPORT: FK vs Digital Twin\n');
        fprintf('============================================================\n');
        fprintf('  Time Steps Evaluated : %d (duration: %.2f s)\n', M_steps, time_vec(end));
        fprintf('  Position Error (m)   : Max = %.4e  |  RMS = %.4e  (Tol: %.1e)\n', ...
                max_pos, rms_pos, tol_pos);
        fprintf('  Rotation Error (rad) : Max = %.4e  |  RMS = %.4e  (Tol: %.1e)\n', ...
                max_rot, rms_rot, tol_rot);
        fprintf('  Rotation Error (deg) : Max = %.4f° |  RMS = %.4f°\n', ...
                rad2deg(max_rot), rad2deg(rms_rot));
        if passed
            fprintf('  >>> STATUS: PASSED (Ground-truth matches analytical FK) <<<\n');
        else
            fprintf('  >>> STATUS: FAILED (Discrepancy exceeds tolerance) <<<\n');
        end
        fprintf('============================================================\n\n');
    end

    % -----------------------------------------------------------------------
    % 4. Assemble Report Struct
    % -----------------------------------------------------------------------
    report.passed          = passed;
    report.max_pos_err     = max_pos;
    report.rms_pos_err     = rms_pos;
    report.max_rot_err_rad = max_rot;
    report.rms_rot_err_rad = rms_rot;
    report.time            = time_vec;
    report.pos_err_vec     = pos_err;
    report.rot_err_vec     = rot_err;

    % Optional plot
    if do_plot
        figure('Name', 'FK Verification Residuals', 'Color', 'w');
        subplot(2, 1, 1);
        plot(time_vec, pos_err * 1e3, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
        yline(tol_pos * 1e3, '--r', 'Tolerance');
        ylabel('Position Error (mm)');
        grid on; title('Digital Twin vs Analytical Forward Kinematics');

        subplot(2, 1, 2);
        plot(time_vec, rad2deg(rot_err), 'LineWidth', 1.5, 'Color', [0.85 0.325 0.098]);
        yline(rad2deg(tol_rot), '--r', 'Tolerance');
        ylabel('Orientation Error (deg)'); xlabel('Time (s)');
        grid on;
    end

end % function verify_fk


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
        error('verify_fk:signalNotFound', 'Signal "%s" not found.', vname);
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
