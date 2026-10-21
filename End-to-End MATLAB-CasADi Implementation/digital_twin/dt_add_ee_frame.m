function dt = dt_add_ee_frame(dt, chain, opts)
% DT_ADD_EE_FRAME  Anchors the End-Effector reference frame to the Digital Twin.
%
% Explicitly connects a Rigid Transform block representing the End-Effector (EE)
% offset to the terminal link frame of the Simscape Multibody model.
%
% This ensures that:
%   1. The Digital Twin possesses an identifiable EE reference frame coincident
%      with the terminal tool point (or flange).
%   2. Transform Sensors can directly measure the EE pose and velocities in the
%      world frame without ambiguity.
%   3. Kinematic ground-truth from the Digital Twin exactly aligns with the
%      analytical FK and Jacobians derived by fk_casadi and jacobian_casadi.
%
% =========================================================================
% INPUTS:
%   dt    (struct)  Output of dt_build(). Required fields:
%                     .model_name  (char)
%                     .last_port   (handle)
%                     .x_col_end   (numeric)
%                     .y_main      (numeric)
%   chain (struct)  Output of urdf_build_chain(). Required fields:
%                     .T_ee        [4x4] fixed EE offset transform
%   opts  (struct)  [optional] Options:
%     .save_model (logical) Save model after adding EE frame (default: true)
%     .verbose    (logical) Print progress message (default: false)
%
% =========================================================================
% OUTPUT:
%   dt    (struct)  Updated dt struct with additional fields:
%                     .ee_block  (char)   Path to EE Transform block
%                     .ee_port   (handle) Port handle of EE follower frame
%
% =========================================================================
% USAGE:
%   dt = dt_build(robot, chain, inertia);
%   dt = dt_add_ee_frame(dt, chain);
%
% =========================================================================
% CALLED BY:
%   dt_add_sensors.m
%   verify_fk.m
%   verify_jacobian.m

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    save_model = ~isfield(opts, 'save_model') || opts.save_model;
    verbose    = isfield(opts, 'verbose') && opts.verbose;

    model_name = dt.model_name;
    x_col      = dt.x_col_end;
    y_main     = dt.y_main;

    % -----------------------------------------------------------------------
    % 1. Add Rigid Transform block for End-Effector
    % -----------------------------------------------------------------------
    ee_block_path = [model_name, '/EE_Transform'];
    h_ee = add_block('sm_lib/Frames and Transforms/Rigid Transform', ...
        ee_block_path, 'Position', [x_col, y_main - 30, x_col + 70, y_main + 30]);

    % Determine End-Effector offset transform T_target.
    % Priority for determining T_target:
    %   1. opts.T_ee     — explicit override; caller supplies the exact [4x4] transform
    %   2. opts.model    — auto-compute T_rel = T_urdf{end} \ T_dh(q=0)
    %                      (reconciles Simscape terminal frame with DH frame N)
    %   3. chain.T_rel   — precomputed relative transform already stored in chain
    %   4. chain.T_ee    — raw URDF EE fixed-joint transform (default / fallback)
    if isfield(opts, 'T_ee') && ~isempty(opts.T_ee)
        T_target = opts.T_ee;
    elseif isfield(opts, 'model') && ~isempty(opts.model)
        % Compute relative transform from terminal link frame to DH Frame 6
        T_u0 = chain.T_urdf{end};
        if isfield(opts.model, 'fk')
            fk_ee = opts.model.fk.fk_ee_fun;
        else
            fk_ee = opts.model.fk_ee_fun;
        end
        T_dh0 = full(fk_ee(zeros(chain.n, 1)));
        T_target = T_u0 \ T_dh0;
    elseif isfield(chain, 'T_rel') && ~isempty(chain.T_rel)
        T_target = chain.T_rel;
    else
        T_target = chain.T_ee;
    end

    % Translation and rotation from T_target
    t_ee = T_target(1:3, 4);
    R_ee = T_target(1:3, 1:3);

    set_param(ee_block_path, 'TranslationMethod', 'Cartesian');
    set_param(ee_block_path, 'TranslationCartesianOffset', ...
        sprintf('[%.10f, %.10f, %.10f]', t_ee(1), t_ee(2), t_ee(3)));
    set_param(ee_block_path, 'TranslationCartesianOffsetUnits', 'm');

    % Set rotation via ArbitraryAxis
    set_rotation_matrix(ee_block_path, R_ee);

    % Connect terminal link frame to EE transform base port
    p_ee = get_param(h_ee, 'PortHandles');
    add_line(model_name, dt.last_port, p_ee.LConn(1), 'autorouting', 'on');

    if verbose
        fprintf('[dt_add_ee_frame] EE frame anchored: %s\n', ee_block_path);
        fprintf('  Offset translation: [%.4f, %.4f, %.4f] m\n', t_ee(1), t_ee(2), t_ee(3));
    end

    % -----------------------------------------------------------------------
    % 2. Update dt struct
    % -----------------------------------------------------------------------
    dt.ee_block  = ee_block_path;
    dt.ee_port   = p_ee.RConn(1);
    dt.x_col_end = x_col + 100;

    if save_model
        save_system(model_name);
    end

end % function dt_add_ee_frame


% =========================================================================
%  LOCAL HELPER
% =========================================================================

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
