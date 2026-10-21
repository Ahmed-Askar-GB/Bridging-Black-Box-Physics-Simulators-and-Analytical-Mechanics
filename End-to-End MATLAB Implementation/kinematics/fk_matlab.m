function fk = fk_matlab(dh, chain, opts)
% FK_MATLAB  Pure MATLAB Forward Kinematics via standard Denavit-Hartenberg parameters.
%
% Computes:
%   - Per-link transformation matrices T_0^i(q)
%   - Terminal End-Effector transformation T_0^ee(q) = T_0^n(q)
%   - End-Effector position p_ee(q)
%   - End-Effector rotation matrix R_ee(q)
%   - Unit quaternion quat_ee(q) via Shepperd's numerically stable algorithm
%
% =========================================================================

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    verbose = false; if isfield(opts, 'verbose'), verbose = opts.verbose; end

    n   = chain.n;
    DHp = dh.DHpram;   % [Nx5]: [a, alpha, theta_off, d, sigma]

    % -------------------------------------------------------------------------
    % Core Forward Kinematics Evaluation Function
    % -------------------------------------------------------------------------
    function [T0i, T0e] = compute_all_transforms(q)
        q = double(q(:));
        T0i = cell(n, 1);

        if isfield(opts, 'T_base') && ~isempty(opts.T_base)
            T_running = opts.T_base;
        else
            T_running = eye(4);
        end

        for i = 1:n
            a_i     = DHp(i, 1);
            alpha_i = DHp(i, 2);
            th_off  = DHp(i, 3);
            d_off   = DHp(i, 4);
            sigma_i = DHp(i, 5);

            if sigma_i == 1
                theta_i = th_off + q(i);
                d_i     = d_off;
            else
                theta_i = th_off;
                d_i     = d_off + q(i);
            end

            ct = cos(theta_i); st = sin(theta_i);
            ca = cos(alpha_i); sa = sin(alpha_i);

            A_i = [ct,  -st*ca,   st*sa,   a_i*ct;
                   st,   ct*ca,  -ct*sa,   a_i*st;
                    0,      sa,      ca,      d_i;
                    0,       0,       0,       1 ];

            T_running = T_running * A_i;
            T0i{i}    = T_running;
        end

        if isfield(opts, 'T_tool') && ~isempty(opts.T_tool)
            T0e = T0i{n} * opts.T_tool;
        else
            T0e = T0i{n};
        end
    end

    % -------------------------------------------------------------------------
    % Primary Callable Function Handles
    % -------------------------------------------------------------------------
    % 1. fk_fun: Evaluates [p_ee, R_ee, quat_ee]
    fk_fun = @(q) eval_fk_fun(q);
    function [p_ee, R_ee, quat_ee] = eval_fk_fun(q)
        [~, T0e] = compute_all_transforms(q);
        p_ee = T0e(1:3, 4);
        R_ee = T0e(1:3, 1:3);
        quat_ee = R_to_quat(R_ee);
    end

    % 2. fk_ee_fun: Evaluates 4x4 End-Effector pose matrix
    fk_ee_fun = @(q) eval_ee_transform(q);
    function T0e = eval_ee_transform(q)
        [~, T0e] = compute_all_transforms(q);
    end

    % 3. fk_link_funs: Evaluates 4x4 pose for link i
    fk_link_funs = cell(n, 1);
    for i = 1:n
        fk_link_funs{i} = @(q) eval_link_transform(q, i);
    end
    function T_i = eval_link_transform(q, idx)
        [T0i, ~] = compute_all_transforms(q);
        T_i = T0i{idx};
    end

    % Package output struct
    fk = struct();
    fk.n            = n;
    fk.fk_fun       = fk_fun;
    fk.fk_ee_fun    = fk_ee_fun;
    fk.fk_link_funs = fk_link_funs;
    fk.fk_link_fun  = fk_link_funs;  % Alias for compatibility
    fk.dh           = dh;
    fk.chain        = chain;

    if verbose
        fprintf('  [OK] Pure MATLAB Forward Kinematics compiled (%d DOFs).\n', n);
    end
end
