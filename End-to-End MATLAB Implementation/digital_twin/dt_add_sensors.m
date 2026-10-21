function dt = dt_add_sensors(dt, opts)
% DT_ADD_SENSORS  Instruments the Digital Twin with EE and joint sensors.
%
% Adds Simscape Multibody Transform Sensors and Revolute Joint measurement
% interfaces, converting physical signals (PS) into standard Simulink signals
% that stream directly into the MATLAB workspace via "To Workspace" sinks.
%
% =========================================================================
% SENSORS INSTANTIATED:
%   1. End-Effector (EE) Translation Sensor:
%      - Position [x; y; z] (m) in World Frame      -> workspace: 'ee_pos'
%      - Linear velocity [vx; vy; vz] (m/s) in World -> workspace: 'ee_vel'
%
%   2. End-Effector (EE) Rotation Sensor:
%      - Quaternion [qw; qx; qy; qz] (scalar-first) -> workspace: 'ee_quat'
%      - Angular velocity [wx; wy; wz] (rad/s)      -> workspace: 'ee_omega'
%
%   3. Joint Sensors (for each active joint i = 1..N):
%      - Joint angle q_i (rad)                      -> muxed: 'q_sim'   [time x N]
%      - Joint velocity dq_i (rad/s)                -> muxed: 'dq_sim'  [time x N]
%      - Joint actuator torque tau_i (N·m)          -> muxed: 'tau_sim' [time x N]
%
% =========================================================================
% INPUTS:
%   dt    (struct)  Output of dt_build() and dt_add_ee_frame(). Required:
%                     .model_name    (char)
%                     .world_port    (handle)  World frame port
%                     .last_port     (handle)  Terminal frame port
%                     .joint_blocks  {Nx1 cell} Paths of Revolute Joint blocks
%                     .x_col_end     (numeric) Coordinate for layout
%   opts  (struct)  [optional] Options:
%     .save_model (logical) Save model when done (default: true)
%     .verbose    (logical) Print progress messages (default: false)
%
% =========================================================================
% OUTPUT:
%   dt    (struct)  Updated dt struct with sensor metadata:
%                     .sensors_added (logical)
%                     .sensor_blocks {cell}
%
% =========================================================================
% USAGE:
%   dt = dt_build(robot, chain, inertia);
%   dt = dt_add_ee_frame(dt, chain);
%   dt = dt_add_sensors(dt);
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m
%   verify_fk.m
%   verify_jacobian.m
%   verify_dynamics_sequential.m

    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    save_model = ~isfield(opts, 'save_model') || opts.save_model;
    verbose    = isfield(opts, 'verbose') && opts.verbose;

    model_name = dt.model_name;
    n          = dt.n_dof;

    % Frame port to measure: use EE port if present, else terminal link port
    if isfield(dt, 'ee_port') && ~isempty(dt.ee_port)
        target_port = dt.ee_port;
    else
        target_port = dt.last_port;
    end
    world_port = dt.world_port;

    x_s = dt.x_col_end + 60;
    y_s = 60;

    % -----------------------------------------------------------------------
    % 1. End-Effector Translation Sensor (Position & Linear Velocity)
    % -----------------------------------------------------------------------
    ts_pos_path = [model_name, '/EE_Translation_Sensor'];
    h_ts_pos = add_block('sm_lib/Frames and Transforms/Transform Sensor', ...
        ts_pos_path, 'Position', [x_s, y_s, x_s + 60, y_s + 60]);
    set_param(ts_pos_path, 'SenseXYZ',    'on');
    set_param(ts_pos_path, 'SenseXYZDot', 'on');

    p_tsp = get_param(h_ts_pos, 'PortHandles');
    add_line(model_name, world_port,   p_tsp.LConn(1), 'autorouting', 'on');
    add_line(model_name, target_port,  p_tsp.RConn(1), 'autorouting', 'on');

    % Connect Position: RConn(2) -> PS-Conv -> To Workspace 'ee_pos'
    add_sink_chain(model_name, p_tsp.RConn(2), 'ee_pos', [x_s + 100, y_s - 15]);

    % Connect Velocity: RConn(3) -> PS-Conv -> To Workspace 'ee_vel'
    add_sink_chain(model_name, p_tsp.RConn(3), 'ee_vel', [x_s + 100, y_s + 35]);

    y_s = y_s + 120;

    % -----------------------------------------------------------------------
    % 2. End-Effector Rotation Sensor (Quaternion & Angular Velocity)
    % -----------------------------------------------------------------------
    ts_rot_path = [model_name, '/EE_Rotation_Sensor'];
    h_ts_rot = add_block('sm_lib/Frames and Transforms/Transform Sensor', ...
        ts_rot_path, 'Position', [x_s, y_s, x_s + 60, y_s + 60]);
    set_param(ts_rot_path, 'SenseQ',     'on');
    set_param(ts_rot_path, 'SenseOmega', 'on');

    p_tsr = get_param(h_ts_rot, 'PortHandles');
    add_line(model_name, world_port,   p_tsr.LConn(1), 'autorouting', 'on');
    add_line(model_name, target_port,  p_tsr.RConn(1), 'autorouting', 'on');

    % Connect Quaternion: RConn(2) -> PS-Conv -> To Workspace 'ee_quat'
    add_sink_chain(model_name, p_tsr.RConn(2), 'ee_quat', [x_s + 100, y_s - 15]);

    % Connect Angular Velocity: RConn(3) -> PS-Conv -> To Workspace 'ee_omega'
    add_sink_chain(model_name, p_tsr.RConn(3), 'ee_omega', [x_s + 100, y_s + 35]);

    y_s = y_s + 140;

    % -----------------------------------------------------------------------
    % 3. Joint Sensors (Position, Velocity, Actuator Torque)
    % -----------------------------------------------------------------------
    % We enable sensing on each Revolute Joint block:
    %   SensePosition: 'on'      -> RConn(2)
    %   SenseVelocity: 'on'      -> RConn(3)
    %   SenseTorqueForce: 'on'   -> RConn(4)
    %
    % Then convert each through PS-Simulink Converter, and Mux them into
    % composite vectors: q_sim [Nx1], dq_sim [Nx1], tau_sim [Nx1].
    % -----------------------------------------------------------------------
    mux_q_path   = [model_name, '/Mux_q'];
    mux_dq_path  = [model_name, '/Mux_dq'];
    mux_tau_path = [model_name, '/Mux_tau'];

    h_mq   = add_block('simulink/Commonly Used Blocks/Mux', mux_q_path,   'Position', [x_s + 200, y_s,       x_s + 205, y_s + 60]);
    h_mdq  = add_block('simulink/Commonly Used Blocks/Mux', mux_dq_path,  'Position', [x_s + 200, y_s + 80,  x_s + 205, y_s + 140]);
    h_mtau = add_block('simulink/Commonly Used Blocks/Mux', mux_tau_path, 'Position', [x_s + 200, y_s + 160, x_s + 205, y_s + 220]);

    set_param(mux_q_path,   'Inputs', num2str(n));
    set_param(mux_dq_path,  'Inputs', num2str(n));
    set_param(mux_tau_path, 'Inputs', num2str(n));

    p_mq   = get_param(h_mq,   'PortHandles');
    p_mdq  = get_param(h_mdq,  'PortHandles');
    p_mtau = get_param(h_mtau, 'PortHandles');

    for i = 1:n
        jt_path = dt.joint_blocks{i};
        set_param(jt_path, 'SensePosition',    'on');
        set_param(jt_path, 'SenseVelocity',    'on');
        set_param(jt_path, 'SenseTorqueForce', 'on');

        p_jt = get_param(jt_path, 'PortHandles');

        % Joint Position PS -> Simulink
        conv_q_path = [model_name, sprintf('/PS_Conv_q_%d', i)];
        h_cq = add_block('nesl_utility/PS-Simulink Converter', conv_q_path, ...
            'Position', [x_s + 80, y_s - 20 + i*10, x_s + 110, y_s - 10 + i*10]);
        p_cq = get_param(h_cq, 'PortHandles');
        add_line(model_name, p_jt.RConn(2), p_cq.LConn(1), 'autorouting', 'on');
        add_line(model_name, p_cq.Outport(1), p_mq.Inport(i), 'autorouting', 'on');

        % Joint Velocity PS -> Simulink
        conv_dq_path = [model_name, sprintf('/PS_Conv_dq_%d', i)];
        h_cdq = add_block('nesl_utility/PS-Simulink Converter', conv_dq_path, ...
            'Position', [x_s + 80, y_s + 60 + i*10, x_s + 110, y_s + 70 + i*10]);
        p_cdq = get_param(h_cdq, 'PortHandles');
        add_line(model_name, p_jt.RConn(3), p_cdq.LConn(1), 'autorouting', 'on');
        add_line(model_name, p_cdq.Outport(1), p_mdq.Inport(i), 'autorouting', 'on');

        % Joint Torque PS -> Simulink
        conv_tau_path = [model_name, sprintf('/PS_Conv_tau_%d', i)];
        h_ctau = add_block('nesl_utility/PS-Simulink Converter', conv_tau_path, ...
            'Position', [x_s + 80, y_s + 140 + i*10, x_s + 110, y_s + 150 + i*10]);
        p_ctau = get_param(h_ctau, 'PortHandles');
        add_line(model_name, p_jt.RConn(4), p_ctau.LConn(1), 'autorouting', 'on');
        add_line(model_name, p_ctau.Outport(1), p_mtau.Inport(i), 'autorouting', 'on');
    end

    % To Workspace for Muxed signals
    add_workspace_sink(model_name, p_mq.Outport(1),   'q_sim',   [x_s + 240, y_s + 15]);
    add_workspace_sink(model_name, p_mdq.Outport(1),  'dq_sim',  [x_s + 240, y_s + 95]);
    add_workspace_sink(model_name, p_mtau.Outport(1), 'tau_sim', [x_s + 240, y_s + 175]);

    if verbose
        fprintf('[dt_add_sensors] Sensors connected: ee_pos, ee_vel, ee_quat, ee_omega, q_sim, dq_sim, tau_sim\n');
    end

    dt.sensors_added = true;

    if save_model
        save_system(model_name);
    end

end % function dt_add_sensors


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function add_sink_chain(model_name, ps_port, var_name, pos)
% Connects a Physical Signal port through PS-Simulink Converter to To Workspace
    conv_path = [model_name, '/PS_Conv_', var_name];
    h_conv = add_block('nesl_utility/PS-Simulink Converter', conv_path, ...
        'Position', [pos(1), pos(2), pos(1) + 30, pos(2) + 25]);
    p_conv = get_param(h_conv, 'PortHandles');
    add_line(model_name, ps_port, p_conv.LConn(1), 'autorouting', 'on');

    ws_path = [model_name, '/ToWS_', var_name];
    h_ws = add_block('simulink/Sinks/To Workspace', ws_path, ...
        'Position', [pos(1) + 60, pos(2), pos(1) + 120, pos(2) + 25]);
    set_param(ws_path, 'VariableName', var_name);
    set_param(ws_path, 'SaveFormat',   'Timeseries');
    p_ws = get_param(h_ws, 'PortHandles');
    add_line(model_name, p_conv.Outport(1), p_ws.Inport(1), 'autorouting', 'on');
end

function add_workspace_sink(model_name, outport, var_name, pos)
% Connects a Simulink outport directly to a To Workspace block
    ws_path = [model_name, '/ToWS_', var_name];
    h_ws = add_block('simulink/Sinks/To Workspace', ws_path, ...
        'Position', [pos(1), pos(2), pos(1) + 60, pos(2) + 25]);
    set_param(ws_path, 'VariableName', var_name);
    set_param(ws_path, 'SaveFormat',   'Timeseries');
    p_ws = get_param(h_ws, 'PortHandles');
    add_line(model_name, outport, p_ws.Inport(1), 'autorouting', 'on');
end
