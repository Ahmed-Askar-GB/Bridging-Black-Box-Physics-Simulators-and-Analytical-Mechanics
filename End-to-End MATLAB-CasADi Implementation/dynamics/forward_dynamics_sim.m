function sim_out = forward_dynamics_sim(dyn, q0, dq0, t_span, controller, opts)
% FORWARD_DYNAMICS_SIM  Headless numerical simulation of robot forward dynamics.
%
% Integrates the forward dynamics equations:
%   ddq = forward_dyn_fun(q, dq, tau) = M(q)^{-1} * (tau - C(q, dq)*dq - G(q))
%
% Uses 4th-order Runge-Kutta (RK4) integration in pure MATLAB without requiring
% Simulink or Simscape Multibody.
%
% =========================================================================
% INPUTS:
%   dyn        (struct)          CasADi dynamics struct with forward_dyn_fun.
%   q0         [Nx1 double]      Initial joint position in URDF frame.
%   dq0        [Nx1 double]      Initial joint velocity in URDF frame.
%   t_span     [1x2] or [Mx1]    Simulation time interval [0, T_final] or vector.
%   controller (optional)        One of:
%                                  - Function handle: @(t, q, dq) tau_cmd
%                                  - Controller object with compute() method
%                                  - Constant [Nx1] torque vector
%                                  - [] or omitted: unactuated (tau = 0)
%   opts       (struct) [opt]    Simulation options:
%                                  .dt          Integration timestep (default: 0.001 s)
%                                  .joint_signs Actuation sign vector (default: ones(n,1))
%                                  .fk          CasADi FK struct for EE logging
%                                  .integrator  'rk4' (default) or 'euler'
%                                  .verbose     Print progress (default: false)
%
% OUTPUT:
%   sim_out    (struct) with fields:
%                .t        [Mx1] Time vector
%                .q        [M x N] Joint positions (URDF frame)
%                .dq       [M x N] Joint velocities (URDF frame)
%                .ddq      [M x N] Joint accelerations (URDF frame)
%                .tau      [M x N] Applied torques (URDF frame)
%                .p_ee     [M x 3] EE Cartesian positions (if fk supplied)
%                .quat_ee  [M x 4] EE quaternions (if fk supplied)
% =========================================================================

    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    dt = 0.001; if isfield(opts, 'dt'), dt = opts.dt; end
    integrator = 'rk4'; if isfield(opts, 'integrator'), integrator = opts.integrator; end
    verbose = false; if isfield(opts, 'verbose'), verbose = opts.verbose; end

    q0  = double(q0(:));
    dq0 = double(dq0(:));
    n   = length(q0);

    if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
        signs = double(opts.joint_signs(:));
    else
        signs = ones(n, 1);
    end

    if length(t_span) == 2
        t = (t_span(1):dt:t_span(2))';
    else
        t = t_span(:);
        if length(t) > 1
            dt = t(2) - t(1);
        end
    end
    M_steps = length(t);

    % Preallocate history arrays
    q_hist   = zeros(M_steps, n);
    dq_hist  = zeros(M_steps, n);
    ddq_hist = zeros(M_steps, n);
    tau_hist = zeros(M_steps, n);

    q_curr  = q0;
    dq_curr = dq0;

    fk_provided = isfield(opts, 'fk') && ~isempty(opts.fk);
    if fk_provided
        p_ee_hist    = zeros(M_steps, 3);
        quat_ee_hist = zeros(M_steps, 4);
    end

    % Helper for evaluating acceleration in DH frame then converting back
    function ddq_urdf = eval_forward_acc(q_u, dq_u, tau_u)
        q_dh   = signs .* q_u;
        dq_dh  = signs .* dq_u;
        tau_dh = signs .* tau_u;

        ddq_dh = full(dyn.forward_dyn_fun(q_dh, dq_dh, tau_dh));
        ddq_urdf = signs .* ddq_dh;
    end

    % Helper to evaluate controller
    function tau_cmd = eval_control(t_k, q_k, dq_k)
        if isempty(controller)
            tau_cmd = zeros(n, 1);
        elseif isa(controller, 'function_handle')
            tau_cmd = double(controller(t_k, q_k, dq_k));
        elseif isobject(controller) && ismethod(controller, 'compute')
            tau_cmd = double(controller.compute(q_k, dq_k));
        elseif isvector(controller) && length(controller) == n
            tau_cmd = double(controller(:));
        else
            tau_cmd = zeros(n, 1);
        end
        tau_cmd = tau_cmd(:);
    end

    if verbose
        fprintf('\n>>> Running Headless Forward Dynamics Simulation (%d steps, dt=%.4f s) ...\n', ...
                M_steps, dt);
    end

    % Main integration loop
    for k = 1:M_steps
        t_k = t(k);

        % Evaluate control command
        tau_k = eval_control(t_k, q_curr, dq_curr);

        % Evaluate acceleration
        ddq_k = eval_forward_acc(q_curr, dq_curr, tau_k);

        % Record state
        q_hist(k, :)   = q_curr';
        dq_hist(k, :)  = dq_curr';
        ddq_hist(k, :) = ddq_k';
        tau_hist(k, :) = tau_k';

        if fk_provided
            q_dh = signs .* q_curr;
            [p_val, ~, quat_val] = opts.fk.fk_fun(q_dh);
            p_ee_hist(k, :)    = full(p_val)';
            quat_ee_hist(k, :) = full(quat_val)';
        end

        % Time-step integration
        if k < M_steps
            if strcmpi(integrator, 'euler')
                q_curr  = q_curr + dt * dq_curr;
                dq_curr = dq_curr + dt * ddq_k;
            else
                % 4th-Order Runge-Kutta (RK4)
                f1_v = dq_curr;
                f1_a = ddq_k;

                q_mid1  = q_curr + 0.5 * dt * f1_v;
                dq_mid1 = dq_curr + 0.5 * dt * f1_a;
                f2_v    = dq_mid1;
                f2_a    = eval_forward_acc(q_mid1, dq_mid1, tau_k);

                q_mid2  = q_curr + 0.5 * dt * f2_v;
                dq_mid2 = dq_curr + 0.5 * dt * f2_a;
                f3_v    = dq_mid2;
                f3_a    = eval_forward_acc(q_mid2, dq_mid2, tau_k);

                q_end   = q_curr + dt * f3_v;
                dq_end  = dq_curr + dt * f3_a;
                f4_v    = dq_end;
                f4_a    = eval_forward_acc(q_end, dq_end, tau_k);

                q_curr  = q_curr  + (dt / 6.0) * (f1_v + 2*f2_v + 2*f3_v + f4_v);
                dq_curr = dq_curr + (dt / 6.0) * (f1_a + 2*f2_a + 2*f3_a + f4_a);
            end
        end
    end

    sim_out = struct();
    sim_out.t    = t;
    sim_out.q    = q_hist;
    sim_out.dq   = dq_hist;
    sim_out.ddq  = ddq_hist;
    sim_out.tau  = tau_hist;

    if fk_provided
        sim_out.p_ee    = p_ee_hist;
        sim_out.quat_ee = quat_ee_hist;
    end

    if verbose
        fprintf('  [OK] Simulation complete. Final time: %.3f s\n', t(end));
    end
end
