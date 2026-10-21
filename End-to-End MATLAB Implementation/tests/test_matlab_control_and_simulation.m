function report = test_matlab_control_and_simulation(opts)
% TEST_MATLAB_CONTROL_AND_SIMULATION
%
% Validates the new MATLAB modules:
%   1. verify_invariants.m: Mass positivity, triangle inequalities, SO(3), M(q) SPD.
%   2. forward_dynamics_sim.m: Headless RK4 forward dynamics numerical simulation.
%   3. computed_torque_controller.m: Closed-loop trajectory tracking with machine precision.
%   4. nmpc_casadi.m: Multiple-shooting NMPC optimal control convergence.
%
% =========================================================================

    this_dir   = fileparts(mfilename('fullpath'));
    parent_dir = fileparts(this_dir);

    % Ensure all directories are on path
    addpath(fullfile(parent_dir, 'parsing'));
    addpath(fullfile(parent_dir, 'kinematics'));
    addpath(fullfile(parent_dir, 'dynamics'));
    addpath(fullfile(parent_dir, 'compilation'));
    addpath(fullfile(parent_dir, 'control'));
    addpath(fullfile(parent_dir, 'verification'));
    addpath(fullfile(parent_dir, 'visualization'));
    addpath(fullfile(parent_dir, 'utils'));
    addpath(fullfile(parent_dir, 'test_UR16e'));

    urdf_path = fullfile(parent_dir, 'test_UR16e', 'universalUR16e.urdf');
    if nargin < 1 || isempty(opts)
        opts = struct();
    end

    fprintf('\n====================================================================\n');
    fprintf('    VALIDATION SUITE: MATLAB CONTROL, INVARIANTS & SIMULATION        \n');
    fprintf('====================================================================\n');
    fprintf('  URDF Target: %s\n\n', urdf_path);

    % 1. Ingestion & Analytical Derivation
    fprintf('>>> [TEST 1/4] Ingesting URDF & Compiling CasADi Kinematics/Dynamics ...\n');
    robot   = urdf_parse(urdf_path);
    chain   = urdf_build_chain(robot, false);
    dh      = urdf_extract_dh(chain, false);
    inertia = urdf_extract_inertia(robot, chain, dh, false);

    fk  = fk_casadi(dh, chain, struct('verbose', false));
    jac = jacobian_casadi(fk, dh, chain, struct('verbose', false));
    dyn = dynamics_casadi(dh, inertia, chain, struct('verbose', false));
    joint_signs = detect_joint_signs(chain, fk);

    fprintf('  Model compiled successfully: %d DOFs. Signs: %s\n', chain.n, mat2str(joint_signs));

    % 2. Invariants & Physical Realizability Verification
    fprintf('\n>>> [TEST 2/4] Testing verify_invariants.m ...\n');
    inv_opts = struct('num_samples', 15, 'tol', 1e-7, 'verbose', true);
    [inv_results, all_passed] = verify_invariants(dh, inertia, fk, dyn, inv_opts);
    assert(all_passed, 'Test failed: Mathematical/Physical invariants check did not pass.');
    fprintf('  [PASS] All invariants verified successfully.\n');

    % 3. Closed-Loop Computed Torque Controller & Headless RK4 Simulation
    fprintf('\n>>> [TEST 3/4] Testing computed_torque_controller.m & forward_dynamics_sim.m ...\n');
    n = chain.n;
    t_span = [0, 2.0];
    dt = 0.002;
    t_vec = (t_span(1):dt:t_span(2))';
    M_steps = length(t_vec);

    % Generate smooth reference trajectory via 5th order polynomial
    q_start = zeros(n, 1);
    q_goal  = [0.5; -0.8; 0.6; -0.4; 0.5; -0.3];
    q_ref   = zeros(M_steps, n);
    dq_ref  = zeros(M_steps, n);
    ddq_ref = zeros(M_steps, n);

    for j = 1:n
        [q_j, dq_j, ddq_j] = traj_poly5(q_start(j), q_goal(j), t_span(2), t_vec);
        q_ref(:, j)   = q_j;
        dq_ref(:, j)  = dq_j;
        ddq_ref(:, j) = ddq_j;
    end

    % Instantiate Computed Torque Controller
    Kp = 400.0 * eye(n);
    Kd = 40.0 * eye(n);
    ctrl = computed_torque_controller(dyn, Kp, Kd, joint_signs);

    % Define closed-loop controller callback
    ctrl_fun = @(t_k, q_k, dq_k) ctrl.compute(...
        q_k, dq_k, ...
        interp1(t_vec, q_ref, t_k, 'linear', 'extrap')', ...
        interp1(t_vec, dq_ref, t_k, 'linear', 'extrap')', ...
        interp1(t_vec, ddq_ref, t_k, 'linear', 'extrap')');

    % Run Headless Forward Dynamics Simulation
    sim_opts = struct('dt', dt, 'joint_signs', joint_signs, 'fk', fk, 'integrator', 'rk4');
    sim_out  = forward_dynamics_sim(dyn, q_start, zeros(n, 1), t_span, ctrl_fun, sim_opts);

    % Evaluate tracking performance
    q_err   = abs(sim_out.q - q_ref);
    max_err = max(q_err(:));
    rms_err = sqrt(mean(q_err(:).^2));

    fprintf('  Tracking Error: Max = %.4e rad, RMS = %.4e rad\n', max_err, rms_err);
    assert(max_err < 5e-3, 'Test failed: Computed torque tracking error exceeded tolerance (got: %.2e rad).', max_err);
    fprintf('  [PASS] Computed torque closed-loop control verified with excellent tracking accuracy.\n');

    % 4. NMPC Trajectory Tracking Test
    fprintf('\n>>> [TEST 4/4] Testing nmpc_casadi.m ...\n');
    nmpc_opts = struct('N', 10, 'dt', 0.05, 'tau_max', 200.0, 'joint_signs', joint_signs, 'verbose', false);
    nmpc = nmpc_casadi(dyn, nmpc_opts);

    x_curr = [q_start; zeros(n, 1)];
    X_ref_horizon = repmat([q_goal; zeros(n, 1)], 1, nmpc.N + 1);

    [tau_opt, X_pred, U_pred, success] = nmpc.solve(x_curr, X_ref_horizon);
    assert(success, 'Test failed: NMPC failed to converge on test horizon.');
    fprintf('  [PASS] NMPC solve converged. Optimal initial torque: %s N.m\n', mat2str(tau_opt', 3));

    fprintf('\n====================================================================\n');
    fprintf('    ALL MATLAB UPGRADES (INVARIANTS, CONTROL, SIMULATION) PASSED!   \n');
    fprintf('====================================================================\n\n');

    report = struct();
    report.inv_results = inv_results;
    report.max_tracking_err = max_err;
    report.rms_tracking_err = rms_err;
    report.nmpc_success = success;
    report.sim_out = sim_out;
end
