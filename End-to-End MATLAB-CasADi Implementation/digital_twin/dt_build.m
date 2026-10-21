function dt = dt_build(robot, chain, inertia, opts)
% DT_BUILD  Programmatically constructs a Simscape Multibody Digital Twin.
%
% Generates an exact, headless Simscape Multibody / Simulink model (.slx)
% of a serial robot from the parsed URDF and kinematic chain structs.
%
% This function refactors and modularizes the monolithic build_dt_simulink.m
% script into a clean, reusable functional interface that operates on any
% serial robot extracted by urdf_parse and urdf_build_chain.
%
% =========================================================================
% KEY ARCHITECTURAL FEATURES:
%   1. Automatic Joint Axis Alignment:
%      Simscape Revolute Joints always rotate about the local Z-axis.
%      URDF joints may rotate about arbitrary axes (e.g., [0 1 0] for Y).
%      This function automatically detects the axis direction and introduces
%      pre-joint and post-joint Rigid Transform blocks (e.g., Rx(-90°) / Rx(+90°))
%      to align the rotation axis without altering physical kinematics.
%
%   2. Rigid Body Dynamics & Inertia:
%      Link visual geometry and inertial parameters are populated using the
%      URDF-frame mass, CoM, and moments/products of inertia (File Solid).
%      If STL mesh files exist, they are linked for 3D visualization in
%      Mechanics Explorer.
%
%   3. Headless Operation:
%      The model is created, configured, and saved programmatically without
%      requiring GUI interaction, enabling automated test pipelines.
%
% =========================================================================
% INPUTS:
%   robot   (struct)  Output of urdf_parse(). Contains all links and joints.
%   chain   (struct)  Output of urdf_build_chain(). Contains ordered serial chain.
%   inertia (struct)  Output of urdf_extract_inertia(). Contains link masses/inertias.
%   opts    (struct)  [optional] Options:
%     .model_name  (char)     Name of Simulink model (default: 'dt_<robot_name>')
%     .output_dir  (char)     Output directory to save .slx (default: current dir)
%     .stop_time   (char/num) Simulation stop time (default: '5')
%     .solver      (char)     Solver name (default: 'daessc' / variable-step)
%     .gravity     [3x1]      Gravity vector in world frame (default: [0; 0; -9.81])
%     .mesh_dir    (char)     Directory containing STL mesh files (optional)
%     .verbose     (logical)  Print progress messages (default: false)
%
% =========================================================================
% OUTPUT:
%   dt      (struct) with fields:
%     .model_name    (char)       Name of the generated Simulink model
%     .slx_path      (char)       Full path to the saved .slx file
%     .n_dof         (int)        Number of degrees of freedom (joints)
%     .joint_blocks  {Nx1 cell}   Simulink block paths of Revolute Joints
%     .link_blocks   {Nx1 cell}   Simulink block paths of Solid Body blocks
%     .world_port    (handle)     PortHandle of World Frame RConn(1)
%     .last_port     (handle)     PortHandle of the terminal link frame
%
% =========================================================================
% USAGE:
%   robot   = urdf_parse('universalUR5.urdf');
%   chain   = urdf_build_chain(robot);
%   dh      = urdf_extract_dh(chain);
%   inertia = urdf_extract_inertia(robot, chain, dh);
%   dt      = dt_build(robot, chain, inertia);
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m
%   verify_fk.m
%   verify_dynamics_sequential.m

    % -----------------------------------------------------------------------
    % 0. Parse optional parameters
    % -----------------------------------------------------------------------
    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    assert(isstruct(robot) && isfield(robot, 'links') && isfield(robot, 'root_link'), ...
           'dt_build:invalidRobot', 'robot struct missing .links or .root_link from urdf_parse.');
    assert(isstruct(chain) && isfield(chain, 'n') && isfield(chain, 'active_joints'), ...
           'dt_build:invalidChain', 'chain struct missing required fields (.n, .active_joints).');
    assert(isstruct(inertia) && isfield(inertia, 'm') && length(inertia.m) == chain.n, ...
           'dt_build:mismatchDOF', 'inertia.m length must match chain.n.');

    r_name = robot.name;
    if isempty(r_name), r_name = 'robot'; end
    r_name = regexprep(r_name, '[^a-zA-Z0-9_]', '_');

    model_name = sprintf('dt_%s', r_name);
    if isfield(opts, 'model_name') && ~isempty(opts.model_name)
        model_name = opts.model_name;
    end

    output_dir = pwd;
    if isfield(opts, 'output_dir') && ~isempty(opts.output_dir)
        output_dir = opts.output_dir;
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    stop_time = '5';
    if isfield(opts, 'stop_time') && ~isempty(opts.stop_time)
        if isnumeric(opts.stop_time)
            stop_time = num2str(opts.stop_time);
        else
            stop_time = char(opts.stop_time);
        end
    end

    gravity_vec = [0; 0; -9.81];
    if isfield(opts, 'gravity') && ~isempty(opts.gravity)
        gravity_vec = opts.gravity(:);
    end

    mesh_dir = '';
    if isfield(opts, 'mesh_dir') && ~isempty(opts.mesh_dir)
        mesh_dir = opts.mesh_dir;
    end

    verbose = isfield(opts, 'verbose') && opts.verbose;

    % -----------------------------------------------------------------------
    % 1. Initialize Simulink / Simscape Environment
    % -----------------------------------------------------------------------
    % If a system with this name is already loaded, close it without saving.
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end

    if verbose
        fprintf('[dt_build] Loading Simscape Multibody libraries...\n');
    end
    load_system('sm_lib');
    load_system('nesl_utility');

    if verbose
        fprintf('[dt_build] Creating Simulink system: %s ...\n', model_name);
    end
    new_system(model_name);
    set_param(model_name, 'StopTime', stop_time);
    set_param(model_name, 'Solver', 'daessc');

    % -----------------------------------------------------------------------
    % 2. Create World Frame, Solver Configuration, Mechanism Configuration
    % -----------------------------------------------------------------------
    h_wf = add_block('sm_lib/Frames and Transforms/World Frame', ...
        [model_name, '/World Frame'], 'Position', [50, 200, 90, 240]);
    h_sc = add_block('nesl_utility/Solver Configuration', ...
        [model_name, '/Solver Configuration'], 'Position', [50, 60, 90, 100]);
    h_mc = add_block('sm_lib/Utilities/Mechanism Configuration', ...
        [model_name, '/Mechanism Configuration'], 'Position', [50, 320, 90, 360]);

    % Set gravity in Mechanism Configuration
    grav_str = sprintf('[%.10f, %.10f, %.10f]', gravity_vec(1), gravity_vec(2), gravity_vec(3));
    set_param([model_name, '/Mechanism Configuration'], 'GravityVector', grav_str);

    p_wf = get_param(h_wf, 'PortHandles');
    p_sc = get_param(h_sc, 'PortHandles');
    p_mc = get_param(h_mc, 'PortHandles');

    add_line(model_name, p_wf.RConn(1), p_sc.RConn(1), 'autorouting', 'on');
    add_line(model_name, p_wf.RConn(1), p_mc.RConn(1), 'autorouting', 'on');

    % -----------------------------------------------------------------------
    % 3. Create Base Link Geometry / Inertia
    % -----------------------------------------------------------------------
    % Connect base link to World Frame
    root_link_name = robot.root_link;
    base_mass = 4.0;
    base_com  = [0; 0; 0];
    base_I    = [0.00443; 0.00443; 0.0072];

    % Search for base mesh file
    base_mesh = resolve_mesh_path('base', mesh_dir);
    if isempty(base_mesh)
        base_mesh = resolve_mesh_path('base_link', mesh_dir);
    end

    if isKey(robot.links, root_link_name)
        blk = robot.links(root_link_name);
        if blk.mass > 0
            base_mass = blk.mass;
            base_com = blk.com;
            base_I   = [blk.inertia(1,1); blk.inertia(2,2); blk.inertia(3,3)];
        else
            if verbose
                warning('dt_build:zeroBaseMass', ...
                        'Root link "%s" has zero mass. Using nominal base_mass=4.0 kg.', root_link_name);
            end
        end
    end

    base_path = [model_name, '/Base_Geometry'];
    if ~isempty(base_mesh) && exist(base_mesh, 'file')
        h_base = add_block('sm_lib/Body Elements/File Solid', ...
            base_path, 'Position', [150, 190, 230, 250]);
        set_param(base_path, 'ExtGeomFileName', base_mesh);
        set_param(base_path, 'UnitType', 'Custom');
        set_param(base_path, 'ExtGeomFileUnits', 'm');
    else
        h_base = add_block('sm_lib/Body Elements/Brick Solid', ...
            base_path, 'Position', [150, 190, 230, 250]);
        set_param(base_path, 'BrickDimensions', '[0.1, 0.1, 0.1]');
        set_param(base_path, 'BrickDimensionsUnits', 'm');
    end

    set_param(base_path, 'InertiaType', 'Custom');
    set_param(base_path, 'Mass', sprintf('%.8f', base_mass));
    set_param(base_path, 'MassUnits', 'kg');
    set_param(base_path, 'CenterOfMass', sprintf('[%.8f, %.8f, %.8f]', base_com(1), base_com(2), base_com(3)));
    set_param(base_path, 'CenterOfMassUnits', 'm');
    set_param(base_path, 'MomentsOfInertia', sprintf('[%.8f, %.8f, %.8f]', base_I(1), base_I(2), base_I(3)));
    set_param(base_path, 'MomentsOfInertiaUnits', 'kg*m^2');
    set_param(base_path, 'ProductsOfInertia', '[0, 0, 0]');
    set_param(base_path, 'ProductsOfInertiaUnits', 'kg*m^2');

    p_base = get_param(h_base, 'PortHandles');
    add_line(model_name, p_wf.RConn(1), p_base.RConn(1), 'autorouting', 'on');

    % -----------------------------------------------------------------------
    % 4. Build Kinematic Chain
    % -----------------------------------------------------------------------
    n = chain.n;
    joint_blocks = cell(n, 1);
    link_blocks  = cell(n, 1);

    last_port = p_wf.RConn(1);
    x_col = 280;
    y_main = 200;

    % Optional Root-to-Joint1 fixed transform (e.g., base_link to base_link_inertia)
    if isfield(chain, 'T_root_to_j1') && ~isempty(chain.T_root_to_j1) && norm(chain.T_root_to_j1 - eye(4)) > 1e-6
        tf_root_path = [model_name, '/RootToJ1_Transform'];
        h_tf_root = add_block('sm_lib/Frames and Transforms/Rigid Transform', ...
            tf_root_path, 'Position', [x_col, y_main - 30, x_col + 70, y_main + 30]);
        R_root = chain.T_root_to_j1(1:3, 1:3);
        p_root = chain.T_root_to_j1(1:3, 4);
        set_param(tf_root_path, 'TranslationMethod', 'Cartesian');
        set_param(tf_root_path, 'TranslationCartesianOffset', ...
            sprintf('[%.10f, %.10f, %.10f]', p_root(1), p_root(2), p_root(3)));
        set_param(tf_root_path, 'TranslationCartesianOffsetUnits', 'm');
        set_rotation_matrix(tf_root_path, R_root);
        p_tf_root = get_param(h_tf_root, 'PortHandles');
        add_line(model_name, last_port, p_tf_root.LConn(1), 'autorouting', 'on');
        last_port = p_tf_root.RConn(1);
        x_col = x_col + 100;
    end

    for i = 1:n
        j_struct = chain.active_joints{i};
        l_name   = chain.active_links{i};
        l_struct = robot.links(l_name);

        if verbose
            fprintf('[dt_build] Building joint %d: "%s" (child: "%s")...\n', ...
                    i, j_struct.name, l_name);
        end

        % -------------------------------------------------------------------
        % 4A. Joint Origin Rigid Transform (xyz + rpy from URDF joint)
        % -------------------------------------------------------------------
        tf_pre_path = [model_name, sprintf('/JointOrigin_%d', i)];
        h_tf_pre = add_block('sm_lib/Frames and Transforms/Rigid Transform', ...
            tf_pre_path, 'Position', [x_col, y_main - 30, x_col + 70, y_main + 30]);

        set_param(tf_pre_path, 'TranslationMethod', 'Cartesian');
        set_param(tf_pre_path, 'TranslationCartesianOffset', ...
            sprintf('[%.10f, %.10f, %.10f]', j_struct.xyz(1), j_struct.xyz(2), j_struct.xyz(3)));
        set_param(tf_pre_path, 'TranslationCartesianOffsetUnits', 'm');

        % Rotation from URDF rpy
        set_rotation_rpy(tf_pre_path, j_struct.rpy);

        p_tf_pre = get_param(h_tf_pre, 'PortHandles');
        add_line(model_name, last_port, p_tf_pre.LConn(1), 'autorouting', 'on');
        x_col = x_col + 100;

        % -------------------------------------------------------------------
        % 4B. Pre-Joint Axis Alignment
        % -------------------------------------------------------------------
        % Simscape Revolute Joint always rotates about local Z.
        % If URDF joint axis is not +Z, we apply alignment rotation.
        [pre_rpy, post_rpy, needs_align] = compute_axis_alignment(j_struct.axis);

        if needs_align
            tf_aln_path = [model_name, sprintf('/AxisPre_%d', i)];
            h_tf_aln = add_block('sm_lib/Frames and Transforms/Rigid Transform', ...
                tf_aln_path, 'Position', [x_col, y_main - 30, x_col + 70, y_main + 30]);
            set_param(tf_aln_path, 'TranslationMethod', 'None');
            set_rotation_rpy(tf_aln_path, pre_rpy);
            p_tf_aln = get_param(h_tf_aln, 'PortHandles');
            add_line(model_name, p_tf_pre.RConn(1), p_tf_aln.LConn(1), 'autorouting', 'on');
            joint_base_port = p_tf_aln.RConn(1);
            x_col = x_col + 100;
        else
            joint_base_port = p_tf_pre.RConn(1);
        end

        % -------------------------------------------------------------------
        % 4C. Revolute Joint Block
        % -------------------------------------------------------------------
        jt_name = sprintf('Revolute_%d_%s', i, regexprep(l_name, '[^a-zA-Z0-9_]', '_'));
        jt_path = [model_name, '/', jt_name];
        h_jt = add_block('sm_lib/Joints/Revolute Joint', ...
            jt_path, 'Position', [x_col, y_main - 25, x_col + 60, y_main + 25]);
        p_jt = get_param(h_jt, 'PortHandles');
        add_line(model_name, joint_base_port, p_jt.LConn(1), 'autorouting', 'on');
        joint_blocks{i} = jt_path;
        x_col = x_col + 100;

        % -------------------------------------------------------------------
        % 4D. Post-Joint Axis Restoration
        % -------------------------------------------------------------------
        if needs_align
            tf_post_path = [model_name, sprintf('/AxisPost_%d', i)];
            h_tf_post = add_block('sm_lib/Frames and Transforms/Rigid Transform', ...
                tf_post_path, 'Position', [x_col, y_main - 30, x_col + 70, y_main + 30]);
            set_param(tf_post_path, 'TranslationMethod', 'None');
            set_rotation_rpy(tf_post_path, post_rpy);
            p_tf_post = get_param(h_tf_post, 'PortHandles');
            add_line(model_name, p_jt.RConn(1), p_tf_post.LConn(1), 'autorouting', 'on');
            link_origin_port = p_tf_post.RConn(1);
            x_col = x_col + 100;
        else
            link_origin_port = p_jt.RConn(1);
        end

        % -------------------------------------------------------------------
        % 4E. Link Body (Geometry & Inertia)
        % -------------------------------------------------------------------
        geom_name = sprintf('Geometry_%d_%s', i, regexprep(l_name, '[^a-zA-Z0-9_]', '_'));
        geom_path = [model_name, '/', geom_name];

        % Mesh resolution based on link name
        link_mesh = resolve_mesh_path(l_name, mesh_dir);
        if ~isempty(link_mesh) && exist(link_mesh, 'file')
            h_geom = add_block('sm_lib/Body Elements/File Solid', ...
                geom_path, 'Position', [x_col, y_main - 85, x_col + 70, y_main - 15]);
            set_param(geom_path, 'ExtGeomFileName',  link_mesh);
            set_param(geom_path, 'UnitType',         'Custom');
            set_param(geom_path, 'ExtGeomFileUnits', 'm');
        else
            h_geom = add_block('sm_lib/Body Elements/Brick Solid', ...
                geom_path, 'Position', [x_col, y_main - 85, x_col + 70, y_main - 15]);
            set_param(geom_path, 'BrickDimensions', '[0.1, 0.1, 0.1]');
            set_param(geom_path, 'BrickDimensionsUnits', 'm');
        end

        % Link inertia in URDF link frame
        l_mass = l_struct.mass;
        l_com  = l_struct.com;
        l_I    = [l_struct.inertia(1,1); l_struct.inertia(2,2); l_struct.inertia(3,3)];

        set_param(geom_path, 'InertiaType',        'Custom');
        set_param(geom_path, 'Mass',               sprintf('%.8f', l_mass));
        set_param(geom_path, 'MassUnits',          'kg');
        set_param(geom_path, 'CenterOfMass',       sprintf('[%.8f, %.8f, %.8f]', l_com(1), l_com(2), l_com(3)));
        set_param(geom_path, 'CenterOfMassUnits',  'm');
        set_param(geom_path, 'MomentsOfInertia',   sprintf('[%.8f, %.8f, %.8f]', l_I(1), l_I(2), l_I(3)));
        set_param(geom_path, 'MomentsOfInertiaUnits', 'kg*m^2');
        % Use actual off-diagonal inertia products from URDF.
        % l_struct.inertia is the full 3x3 tensor (symmetric), so:
        %   ixy = I(1,2),  ixz = I(1,3),  iyz = I(2,3)
        % Simscape convention: [Ixy, Ixz, Iyz] (all with the sign from the tensor).
        l_ixy = l_struct.inertia(1,2);
        l_ixz = l_struct.inertia(1,3);
        l_iyz = l_struct.inertia(2,3);
        set_param(geom_path, 'ProductsOfInertia',  sprintf('[%.8f, %.8f, %.8f]', l_ixy, l_ixz, l_iyz));
        set_param(geom_path, 'ProductsOfInertiaUnits', 'kg*m^2');

        p_geom = get_param(h_geom, 'PortHandles');
        add_line(model_name, link_origin_port, p_geom.RConn(1), 'autorouting', 'on');
        link_blocks{i} = geom_path;

        last_port = link_origin_port;
        x_col = x_col + 120;
    end

    % -----------------------------------------------------------------------
    % 5. Save System
    % -----------------------------------------------------------------------
    slx_file = fullfile(output_dir, [model_name, '.slx']);
    save_system(model_name, slx_file);

    if verbose
        fprintf('[dt_build] Digital Twin created successfully: %s\n', slx_file);
    end

    % -----------------------------------------------------------------------
    % 6. Assemble Output Struct
    % -----------------------------------------------------------------------
    dt.model_name   = model_name;
    dt.slx_path     = slx_file;
    dt.n_dof        = n;
    dt.joint_blocks = joint_blocks;
    dt.link_blocks  = link_blocks;
    dt.world_port   = p_wf.RConn(1);
    dt.last_port    = last_port;
    dt.x_col_end    = x_col;
    dt.y_main       = y_main;

end % function dt_build


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function [pre_rpy, post_rpy, needs_align] = compute_axis_alignment(axis_vec)
% COMPUTE_AXIS_ALIGNMENT  Determines rotation to align URDF joint axis to Simscape Z.
    ax = axis_vec(:) / norm(axis_vec);
    needs_align = true;

    if norm(ax - [0; 0; 1]) < 1e-4
        % Already aligned with +Z
        pre_rpy  = [0, 0, 0];
        post_rpy = [0, 0, 0];
        needs_align = false;
    elseif norm(ax - [0; 1; 0]) < 1e-4
        % Y-axis rotation: Rx(-90°) maps Z to Y (so joint Z is along URDF Y)
        pre_rpy  = [-pi/2, 0, 0];
        post_rpy = [ pi/2, 0, 0];
    elseif norm(ax - [0; -1; 0]) < 1e-4
        % -Y-axis rotation: Rx(+90°) maps Z to -Y
        pre_rpy  = [ pi/2, 0, 0];
        post_rpy = [-pi/2, 0, 0];
    elseif norm(ax - [1; 0; 0]) < 1e-4
        % X-axis rotation: Ry(+90°) maps Z to X
        pre_rpy  = [0,  pi/2, 0];
        post_rpy = [0, -pi/2, 0];
    elseif norm(ax - [-1; 0; 0]) < 1e-4
        % -X-axis rotation: Ry(-90°) maps Z to -X
        pre_rpy  = [0, -pi/2, 0];
        post_rpy = [0,  pi/2, 0];
    elseif norm(ax - [0; 0; -1]) < 1e-4
        % -Z-axis rotation: Rx(180°)
        pre_rpy  = [pi, 0, 0];
        post_rpy = [pi, 0, 0];
    else
        % General axis-angle alignment to [0; 0; 1]
        z_target = [0; 0; 1];
        rot_axis = cross(ax, z_target);
        rot_ang  = acos(dot(ax, z_target));
        if norm(rot_axis) > 1e-5
            rot_axis = rot_axis / norm(rot_axis);
        else
            rot_axis = [1; 0; 0];
        end
        % Convert to rotation matrix and RPY
        K = [0 -rot_axis(3) rot_axis(2); rot_axis(3) 0 -rot_axis(1); -rot_axis(2) rot_axis(1) 0];
        R_align = eye(3) + sin(rot_ang)*K + (1-cos(rot_ang))*(K*K);
        pre_rpy  = rotm2rpy_local(R_align);
        post_rpy = rotm2rpy_local(R_align');
    end
end

function rpy = rotm2rpy_local(R)
% Extracts Roll-Pitch-Yaw from 3x3 rotation matrix
    sy = -R(3, 1);
    pitch = asin(max(-1, min(1, sy)));
    if abs(cos(pitch)) > 1e-6
        roll = atan2(R(3, 2), R(3, 3));
        yaw  = atan2(R(2, 1), R(1, 1));
    else
        roll = atan2(-R(2, 3), R(2, 2));
        yaw  = 0;
    end
    rpy = [roll, pitch, yaw];
end

function set_rotation_rpy(blk, rpy)
% Sets rotation of a Rigid Transform block using ArbitraryAxis
    r = rpy(1);  p = rpy(2);  y = rpy(3);
    cr = cos(r); sr = sin(r);
    cp = cos(p); sp = sin(p);
    cy = cos(y); sy = sin(y);

    R = [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr ;
          sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr ;
         -sp,     cp*sr,             cp*cr            ];

    cosT = (trace(R) - 1) / 2;
    th   = acos(max(-1, min(1, cosT)));

    if th < 1e-9
        set_param(blk, 'RotationMethod', 'None');
        return;
    end

    if abs(th - pi) < 1e-6
        M = R + eye(3);
        [~, col] = max(sum(M.^2));
        ax = M(:, col);
    else
        ax = [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)] / (2*sin(th));
    end
    ax = ax / norm(ax);

    set_param(blk, 'RotationMethod',        'ArbitraryAxis');
    set_param(blk, 'RotationArbitraryAxis', sprintf('[%.10f, %.10f, %.10f]', ax(1), ax(2), ax(3)));
    set_param(blk, 'RotationAngle',         sprintf('%.10f', th));
    set_param(blk, 'RotationAngleUnits',    'rad');
end

function set_rotation_matrix(blk, R)
% Sets rotation of a Rigid Transform block from a 3x3 rotation matrix
    cosT = (trace(R) - 1) / 2;
    th   = acos(max(-1, min(1, cosT)));

    if th < 1e-9
        set_param(blk, 'RotationMethod', 'None');
        return;
    end

    if abs(th - pi) < 1e-6
        M = R + eye(3);
        [~, col] = max(sum(M.^2));
        ax = M(:, col);
    else
        ax = [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)] / (2*sin(th));
    end
    ax = ax / norm(ax);

    set_param(blk, 'RotationMethod',        'ArbitraryAxis');
    set_param(blk, 'RotationArbitraryAxis', sprintf('[%.10f, %.10f, %.10f]', ax(1), ax(2), ax(3)));
    set_param(blk, 'RotationAngle',         sprintf('%.10f', th));
    set_param(blk, 'RotationAngleUnits',    'rad');
end

function mesh_path = resolve_mesh_path(link_name, mesh_dir)
% Resolves mesh path for a link from STL_Files directory
    mesh_path = '';
    if isempty(link_name), return; end

    % Clean link name: remove '_link', remove underscores
    s1 = regexprep(link_name, '_link$', '');
    s2 = regexprep(s1, '_', '');

    candidates = {
        fullfile(pwd, 'STL_Files', [s1, '.stl']);
        fullfile(pwd, 'STL_Files', [s2, '.stl']);
        fullfile(pwd, '..', 'STL_Files', [s1, '.stl']);
        fullfile(pwd, '..', 'STL_Files', [s2, '.stl']);
    };
    if ~isempty(mesh_dir)
        candidates{end+1} = fullfile(mesh_dir, [s1, '.stl']);
        candidates{end+1} = fullfile(mesh_dir, [s2, '.stl']);
        candidates{end+1} = fullfile(mesh_dir, [link_name, '.stl']);
    end

    for k = 1:length(candidates)
        if exist(candidates{k}, 'file')
            mesh_path = candidates{k};
            return;
        end
    end
end
