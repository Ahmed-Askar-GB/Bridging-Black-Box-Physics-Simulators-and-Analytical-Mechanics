function dt = dt_add_trajectory(dt, traj, opts)
% DT_ADD_TRAJECTORY  Prescribes excitation trajectory to the Digital Twin.
%
% Configures joint motion actuation across all N degrees of freedom in the
% Simscape Multibody model. Injects smooth trajectory signals (positions,
% velocities, accelerations) from the MATLAB workspace via "From Workspace"
% sources and Simulink-PS Converter blocks.
%
% =========================================================================
% KEY IMPLEMENTATION DETAILS:
%   1. Motion Actuation Mode:
%      Each Revolute Joint is configured with:
%        MotionActuationMode = 'InputMotion'
%        TorqueActuationMode = 'NoTorque'
%      This creates an actuation physical signal input port: LConn(2).
%
%   2. Input Filtering & Derivatives:
%      The Simulink-PS Converter is configured with 2nd-order input filtering
%      (FilteringAndDerivatives = 'filter') to ensure $C^2$ continuity for
%      velocities and accelerations, preventing Simscape solver chattering.
%
%   3. Workspace Trajectory Injection:
%      If a trajectory struct is provided (e.g., from traj_poly5), its data
%      is pushed to the MATLAB base workspace:
%        - 'sim_traj_q'    : [time, q_1, q_2, ..., q_N]
%        - 'sim_traj_time' : time vector
%      If traj is empty or omitted, a default smooth 5th-order multi-axis
%      excitation trajectory is automatically generated over [0, stop_time].
%
% =========================================================================
% INPUTS:
%   dt    (struct)  Output of dt_build(). Required fields:
%                     .model_name    (char)
%                     .joint_blocks  {Nx1 cell} Paths of Revolute Joints
%                     .n_dof         (int)
%                     .y_main        (numeric)
%   traj  (struct)  [optional] Trajectory struct with fields:
%                     .time  [1xM or Mx1] Time vector (s)
%                     .q     [NxM]        Joint positions (rad)
%                     .dq    [NxM]        Joint velocities (rad/s)
%                     .ddq   [NxM]        Joint accelerations (rad/s²)
%   opts  (struct)  [optional] Options:
%     .save_model (logical) Save model when done (default: true)
%     .verbose    (logical) Print progress messages (default: false)
%
% =========================================================================
% OUTPUT:
%   dt    (struct)  Updated dt struct with trajectory metadata:
%                     .trajectory_injected (logical)
%                     .traj_data           (struct) Injected trajectory
%
% =========================================================================
% USAGE:
%   dt = dt_build(robot, chain, inertia);
%   dt = dt_add_ee_frame(dt, chain);
%   dt = dt_add_sensors(dt);
%   dt = dt_add_trajectory(dt);   % uses default polynomial trajectory
%
%   % Run simulation headless:
%   simOut = sim(dt.model_name);
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m
%   verify_fk.m
%   verify_jacobian.m
%   verify_dynamics_sequential.m

    if nargin < 2 || isempty(traj)
        traj = struct();
    end
    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    save_model = ~isfield(opts, 'save_model') || opts.save_model;
    verbose    = isfield(opts, 'verbose') && opts.verbose;

    model_name = dt.model_name;
    n          = dt.n_dof;

    % -----------------------------------------------------------------------
    % 1. Generate or validate trajectory
    % -----------------------------------------------------------------------
    stop_time_val = str2double(get_param(model_name, 'StopTime'));
    if isnan(stop_time_val) || stop_time_val <= 0
        stop_time_val = 5.0;
    end

    if ~isfield(traj, 'time') || isempty(traj.time) || ~isfield(traj, 'q') || isempty(traj.q)
        % Default: 5th-order polynomial excitation trajectory
        t_vec = linspace(0, stop_time_val, 2001);

        % Excitation targets (non-zero angles across all joints)
        q0 = zeros(n, 1);
        q1 = [0.4; -0.3; 0.5; -0.2; 0.3; -0.4];
        if n > 6
            q1 = [q1; 0.2 * ones(n - 6, 1)];
        else
            q1 = q1(1:n);
        end

        % Evaluate polynomial
        tau_norm = t_vec / stop_time_val;
        s  = 10*tau_norm.^3 - 15*tau_norm.^4 + 6*tau_norm.^5;
        ds = (30*tau_norm.^2 - 60*tau_norm.^3 + 30*tau_norm.^4) / stop_time_val;
        dds= (60*tau_norm - 180*tau_norm.^2 + 120*tau_norm.^3) / (stop_time_val^2);

        delta_q = q1 - q0;
        q_mat   = q0 + delta_q * s;
        dq_mat  = delta_q * ds;
        ddq_mat = delta_q * dds;

        traj.time = t_vec;
        traj.q    = q_mat;
        traj.dq   = dq_mat;
        traj.ddq  = ddq_mat;
    end

    % -----------------------------------------------------------------------
    % 2. Push trajectory variables to base workspace for Simulink
    % -----------------------------------------------------------------------
    % NOTE: sim_traj_q is injected into the MATLAB base workspace because
    % Simulink's 'From Workspace' block evaluates in the base workspace context.
    % Do NOT clear the base workspace (e.g. `clear` or `clearvars`) between
    % calling dt_add_trajectory() and executing sim(model_name).
    %
    % Check for joint sign alignment (URDF joint axis vs DH coordinate axis)
    if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
        joint_signs = opts.joint_signs(:);
    elseif isfield(dt, 'joint_signs') && ~isempty(dt.joint_signs)
        joint_signs = dt.joint_signs(:);
    else
        joint_signs = ones(n, 1);
    end
    dt.joint_signs = joint_signs;

    % Format for Simulink 'From Workspace': [time, q1, q2, ..., qN]
    time_col   = traj.time(:);
    q_cols     = traj.q';  % [M x N] in analytical coordinate frame
    q_cols_sim = q_cols .* (joint_signs');  % mapped to Simscape joint direction
    sim_traj_q = [time_col, q_cols_sim];

    assignin('base', 'sim_traj_q', sim_traj_q);
    assignin('base', 'sim_traj_time', time_col);

    if verbose
        fprintf('[dt_add_trajectory] Assigned sim_traj_q [%dx%d] to base workspace.\n', ...
                size(sim_traj_q, 1), size(sim_traj_q, 2));
    end

    % -----------------------------------------------------------------------
    % 3. Create From Workspace block & Demux in Simulink
    % -----------------------------------------------------------------------
    fws_path = [model_name, '/FromWS_Trajectory'];
    h_fws = add_block('simulink/Sources/From Workspace', fws_path, ...
        'Position', [60, 420, 150, 460]);
    set_param(fws_path, 'VariableName', 'sim_traj_q');
    set_param(fws_path, 'Interpolate',  'on');
    set_param(fws_path, 'ZeroCross',    'on');
    p_fws = get_param(h_fws, 'PortHandles');

    demux_path = [model_name, '/Demux_Trajectory'];
    h_dmx = add_block('simulink/Commonly Used Blocks/Demux', demux_path, ...
        'Position', [180, 400, 185, 400 + n*30]);
    set_param(demux_path, 'Outputs', num2str(n));
    p_dmx = get_param(h_dmx, 'PortHandles');

    add_line(model_name, p_fws.Outport(1), p_dmx.Inport(1), 'autorouting', 'on');

    % -----------------------------------------------------------------------
    % 4. Connect each joint to its trajectory input
    % -----------------------------------------------------------------------
    for i = 1:n
        jt_path = dt.joint_blocks{i};

        % Configure joint for prescribed motion input
        set_param(jt_path, 'MotionActuationMode', 'InputMotion');
        set_param(jt_path, 'TorqueActuationMode', 'ComputedTorque');

        % Get updated port handles (LConn(2) is the motion actuation port)
        p_jt = get_param(jt_path, 'PortHandles');

        % Simulink-PS Converter for joint i
        s2p_path = [model_name, sprintf('/S2P_Joint_%d', i)];
        h_s2p = add_block('nesl_utility/Simulink-PS Converter', s2p_path, ...
            'Position', [220, 400 + (i-1)*30, 260, 420 + (i-1)*30]);

        set_param(s2p_path, 'Unit', 'rad');
        set_param(s2p_path, 'FilteringAndDerivatives', 'filter');
        set_param(s2p_path, 'SimscapeFilterOrder',     '2');
        set_param(s2p_path, 'InputFilterTimeConstant', '0.0002');

        p_s2p = get_param(h_s2p, 'PortHandles');

        % Wire: Demux Outport(i) -> Simulink-PS Converter Inport(1)
        add_line(model_name, p_dmx.Outport(i), p_s2p.Inport(1), 'autorouting', 'on');

        % Wire: Simulink-PS Converter RConn(1) -> Joint LConn(2)
        add_line(model_name, p_s2p.RConn(1), p_jt.LConn(2), 'autorouting', 'on');
    end

    if verbose
        fprintf('[dt_add_trajectory] Motion actuation wired across all %d joints.\n', n);
    end

    % -----------------------------------------------------------------------
    % 5. Update dt struct and save
    % -----------------------------------------------------------------------
    dt.trajectory_injected = true;
    dt.traj_data           = traj;

    if save_model
        save_system(model_name);
    end

end % function dt_add_trajectory
