function fig_handle = animate_robot_matlab(fk, t, q_actual, q_desired, opts)
% ANIMATE_ROBOT_MATLAB  Interactive 3D robot trajectory visualizer with analytical overlay.
%
% Renders the animated 3D motion of the robot manipulator:
%   - Per-link kinematic skeleton with cylindrical segments and joint spheres.
%   - 3D coordinate triads (RGB: Red=X, Green=Y, Blue=Z) at each joint frame.
%   - End-Effector 3D trajectory trail.
%   - Analytical CasADi End-Effector target overlay (green marker) when q_desired
%     is provided to visually confirm zero-lag coincidence with the simulated robot.
%
% =========================================================================
% INPUTS:
%   fk          (struct)             Output from fk_casadi().
%   t           [Mx1 double]         Time vector.
%   q_actual    [M x N double]       Joint position history (URDF frame).
%   q_desired   [M x N double] [opt] Desired reference joint trajectory (optional).
%   opts        (struct)       [opt] Visualization options:
%                                      .fps         Frames per second (default: 30)
%                                      .playback_speed Speed multiplier (default: 1.0)
%                                      .joint_signs Actuation sign vector (default: ones(n,1))
%                                      .trail_color Color for EE trail (default: [0 0.4470 0.7410])
%                                      .robot_name  Title string (default: 'Manipulator')
%                                      .loop        Loop animation (default: false)
%
% OUTPUT:
%   fig_handle  Figure handle to the interactive animation window.
% =========================================================================

    if nargin < 4
        q_desired = [];
    end
    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    fps            = 30;    if isfield(opts, 'fps'),            fps            = opts.fps; end
    playback_speed = 1.0;   if isfield(opts, 'playback_speed'), playback_speed = opts.playback_speed; end
    trail_color    = [0.0, 0.45, 0.74]; if isfield(opts, 'trail_color'), trail_color = opts.trail_color; end
    robot_name     = 'Manipulator'; if isfield(opts, 'robot_name'), robot_name = opts.robot_name; end
    loop_anim      = false; if isfield(opts, 'loop'),           loop_anim      = opts.loop; end

    [M_steps, n] = size(q_actual);
    if isfield(opts, 'joint_signs') && ~isempty(opts.joint_signs)
        signs = double(opts.joint_signs(:)');
    else
        signs = ones(1, n);
    end

    has_ref = ~isempty(q_desired) && isequal(size(q_desired), [M_steps, n]);

    % Downsample if trajectory rate is faster than display fps
    dt_data = t(2) - t(1);
    step_stride = max(1, round((1.0 / fps) / dt_data));
    idx_frames  = 1:step_stride:M_steps;
    N_frames    = length(idx_frames);

    % Pre-compute all link frame positions for bounds sizing
    P_all = zeros(M_steps, n + 2, 3);
    for k = 1:M_steps
        qk_dh = (q_actual(k, :) .* signs)';
        % Base origin
        P_all(k, 1, :) = [0, 0, 0];
        % Intermediate links
        for j = 1:n
            T_j = full(fk.fk_link_funs{j}(qk_dh));
            P_all(k, j + 1, :) = T_j(1:3, 4)';
        end
        % EE
        T_ee = full(fk.fk_ee_fun(qk_dh));
        P_all(k, n + 2, :) = T_ee(1:3, 4)';
    end

    % Spatial workspace limits
    all_x = P_all(:, :, 1);
    all_y = P_all(:, :, 2);
    all_z = P_all(:, :, 3);
    max_reach = max(sqrt(all_x(:).^2 + all_y(:).^2 + all_z(:).^2));
    axis_lim = max(0.5, max_reach * 1.25);
    axis_bounds = [-axis_lim, axis_lim, -axis_lim, axis_lim, -0.1, axis_lim];

    % Create figure window
    fig = figure('Name', sprintf('%s - Interactive 3D Trajectory Visualizer', robot_name), ...
                 'NumberTitle', 'off', ...
                 'Color', [0.12, 0.13, 0.15], ...
                 'Position', [150, 100, 1000, 750]);

    ax = axes('Parent', fig, 'Color', [0.15, 0.16, 0.19]);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');
    axis(ax, axis_bounds);
    view(ax, 45, 30);
    set(ax, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], 'ZColor', [0.7 0.7 0.7], ...
            'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.8);
    xlabel(ax, 'X (m)', 'Color', 'w');
    ylabel(ax, 'Y (m)', 'Color', 'w');
    zlabel(ax, 'Z (m)', 'Color', 'w');
    title_handle = title(ax, sprintf('%s | t = 0.00 s', robot_name), ...
                         'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

    % Render ground grid plane
    [X_plane, Y_plane] = meshgrid(linspace(-axis_lim, axis_lim, 11), linspace(-axis_lim, axis_lim, 11));
    Z_plane = zeros(size(X_plane));
    mesh(ax, X_plane, Y_plane, Z_plane, 'FaceColor', 'none', 'EdgeColor', [0.25 0.25 0.28], 'LineWidth', 0.5);

    % Graphical object handles
    % 1. Robot kinematic links (lines with markers)
    h_links = plot3(ax, 0, 0, 0, '-o', 'Color', [0.9 0.7 0.2], 'LineWidth', 4, ...
                    'MarkerFaceColor', [0.9 0.4 0.1], 'MarkerSize', 8, 'MarkerEdgeColor', 'w');

    % 2. Actual EE trail
    h_trail = plot3(ax, nan, nan, nan, '-', 'Color', trail_color, 'LineWidth', 2.0);

    % 3. Actual EE marker
    h_ee = plot3(ax, nan, nan, nan, 'p', 'MarkerFaceColor', [1.0 0.3 0.3], ...
                 'MarkerEdgeColor', 'w', 'MarkerSize', 12);

    % 4. Desired / Analytical EE target marker (if reference provided)
    if has_ref
        h_target = plot3(ax, nan, nan, nan, 's', 'MarkerFaceColor', [0.1 0.9 0.3], ...
                         'MarkerEdgeColor', 'k', 'MarkerSize', 10);
        legend([h_links, h_trail, h_ee, h_target], ...
               {'Manipulator Links', 'EE Trail', 'Simulated EE', 'Analytical CasADi Target'}, ...
               'TextColor', 'w', 'Color', [0.2 0.2 0.25], 'EdgeColor', [0.4 0.4 0.4], ...
               'Location', 'northeast');
    else
        legend([h_links, h_trail, h_ee], ...
               {'Manipulator Links', 'EE Trail', 'End-Effector'}, ...
               'TextColor', 'w', 'Color', [0.2 0.2 0.25], 'EdgeColor', [0.4 0.4 0.4], ...
               'Location', 'northeast');
    end

    % Triad axis scale
    triad_len = max_reach * 0.10;
    h_triads_x = [];
    h_triads_y = [];
    h_triads_z = [];

    % Initialize triads for each frame (base + n joints + EE)
    n_frames_total = n + 2;
    for j = 1:n_frames_total
        h_triads_x(j) = quiver3(ax, 0,0,0, 0,0,0, 'r', 'LineWidth', 1.8, 'MaxHeadSize', 0.5);
        h_triads_y(j) = quiver3(ax, 0,0,0, 0,0,0, 'g', 'LineWidth', 1.8, 'MaxHeadSize', 0.5);
        h_triads_z(j) = quiver3(ax, 0,0,0, 0,0,0, 'b', 'LineWidth', 1.8, 'MaxHeadSize', 0.5);
    end

    trail_pts = zeros(N_frames, 3);
    trail_count = 0;

    % Animation loop
    keep_running = true;
    while keep_running && isvalid(fig)
        trail_count = 0;

        for f_idx = 1:N_frames
            if ~isvalid(fig), break; end
            k = idx_frames(f_idx);
            t_curr = t(k);

            % Current joint angles in DH frame
            qk_dh = (q_actual(k, :) .* signs)';

            % Compute all frames
            origins = zeros(n_frames_total, 3);
            axes_x  = zeros(n_frames_total, 3);
            axes_y  = zeros(n_frames_total, 3);
            axes_z  = zeros(n_frames_total, 3);

            % Frame 0 (Base)
            origins(1, :) = [0, 0, 0];
            axes_x(1, :)  = [1, 0, 0] * triad_len;
            axes_y(1, :)  = [0, 1, 0] * triad_len;
            axes_z(1, :)  = [0, 0, 1] * triad_len;

            % Frames 1..n
            for j = 1:n
                T_j = full(fk.fk_link_funs{j}(qk_dh));
                origins(j + 1, :) = T_j(1:3, 4)';
                axes_x(j + 1, :)  = T_j(1:3, 1)' * triad_len;
                axes_y(j + 1, :)  = T_j(1:3, 2)' * triad_len;
                axes_z(j + 1, :)  = T_j(1:3, 3)' * triad_len;
            end

            % EE Frame
            T_ee = full(fk.fk_ee_fun(qk_dh));
            origins(end, :) = T_ee(1:3, 4)';
            axes_x(end, :)  = T_ee(1:3, 1)' * triad_len;
            axes_y(end, :)  = T_ee(1:3, 2)' * triad_len;
            axes_z(end, :)  = T_ee(1:3, 3)' * triad_len;

            % Update links
            set(h_links, 'XData', origins(:, 1), 'YData', origins(:, 2), 'ZData', origins(:, 3));

            % Update triads
            for j = 1:n_frames_total
                set(h_triads_x(j), 'XData', origins(j, 1), 'YData', origins(j, 2), 'ZData', origins(j, 3), ...
                                   'UData', axes_x(j, 1),  'VData', axes_x(j, 2),  'WData', axes_x(j, 3));
                set(h_triads_y(j), 'XData', origins(j, 1), 'YData', origins(j, 2), 'ZData', origins(j, 3), ...
                                   'UData', axes_y(j, 1),  'VData', axes_y(j, 2),  'WData', axes_y(j, 3));
                set(h_triads_z(j), 'XData', origins(j, 1), 'YData', origins(j, 2), 'ZData', origins(j, 3), ...
                                   'UData', axes_z(j, 1),  'VData', axes_z(j, 2),  'WData', axes_z(j, 3));
            end

            % Update EE marker & trail
            ee_pos = origins(end, :);
            set(h_ee, 'XData', ee_pos(1), 'YData', ee_pos(2), 'ZData', ee_pos(3));

            trail_count = trail_count + 1;
            trail_pts(trail_count, :) = ee_pos;
            set(h_trail, 'XData', trail_pts(1:trail_count, 1), ...
                         'YData', trail_pts(1:trail_count, 2), ...
                         'ZData', trail_pts(1:trail_count, 3));

            % Update analytical target overlay
            if has_ref
                qd_dh = (q_desired(k, :) .* signs)';
                T_ref = full(fk.fk_ee_fun(qd_dh));
                p_ref = T_ref(1:3, 4)';
                set(h_target, 'XData', p_ref(1), 'YData', p_ref(2), 'ZData', p_ref(3));
            end

            set(title_handle, 'String', sprintf('%s | t = %.2f / %.2f s', robot_name, t_curr, t(end)));
            drawnow limitrate;

            % Timing pause for real-time synchronization
            pause_time = (1.0 / fps) / playback_speed;
            pause(max(0.001, pause_time));
        end

        if ~loop_anim
            keep_running = false;
        end
    end

    fig_handle = fig;
end
