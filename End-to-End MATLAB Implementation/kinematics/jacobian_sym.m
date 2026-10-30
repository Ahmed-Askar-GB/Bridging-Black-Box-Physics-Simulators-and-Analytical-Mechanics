function jac = jacobian_sym(dh, inertia, chain, opts)
% JACOBIAN_SYM  Pure MATLAB Symbolic Geometric & CoM Jacobians.
%
% Implements textbook formulations for link CoM Jacobians and End-Effector Jacobians:
%   For revolute joints (sigma_j = 1):
%     J_{v,i}(:,j) = z_{j-1} \times (p_{C_i} - O_{j-1})
%     J_{\omega,i}(:,j) = z_{j-1}
%   For prismatic joints (sigma_j = 0 or 2):
%     J_{v,i}(:,j) = z_{j-1}
%     J_{\omega,i}(:,j) = 0_{3\times 1}
%
% =========================================================================
% INPUTS:
%   dh      (struct) Standard DH parameter table
%   inertia (struct) Link masses and CoM locations
%   chain   (struct) Active kinematic chain definition
%   opts    (struct) [optional]
%                      .export_matlabFunction (logical) Export function handles (default: true)
%                      .verbose               (logical) Print progress (default: false)
%
% OUTPUT:
%   jac     (struct) Containing:
%             .q          Symbolic joint variables [Nx1 sym]
%             .dq         Symbolic joint velocities [Nx1 sym]
%             .J_ee       Symbolic End-Effector Geometric Jacobian [6xN sym]
%             .dJ_ee      Symbolic EE Jacobian derivative dJ/dt [6xN sym]
%             .J_v_Ci     Cell array {Nx1 sym} of link linear CoM Jacobians [3xN sym]
%             .J_w_Ci     Cell array {Nx1 sym} of link angular CoM Jacobians [3xN sym]
%             .J_Ci       Cell array {Nx1 sym} of full link CoM Jacobians [6xN sym]
%             .J_ee_fun   (optional) matlabFunction handle @(q) -> J_ee
%             .dJ_ee_fun  (optional) matlabFunction handle @(q, dq) -> dJ_ee
% =========================================================================

    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    export_fun = true;  if isfield(opts, 'export_matlabFunction'), export_fun = opts.export_matlabFunction; end
    verbose    = false; if isfield(opts, 'verbose'),               verbose    = opts.verbose; end

    n = chain.n;

    % Build Forward Kinematics symbolically
    fk = fk_sym(dh, chain, struct('export_matlabFunction', false, 'verbose', false));
    q = fk.q;
    dq = sym('dq', [n, 1], 'real');

    % Extract frame origins O_j and axes z_j
    T_root = sym(chain.T_root_to_j1);
    if isempty(T_root), T_root = eye(4); end

    O_all = cell(n + 1, 1);
    z_all = cell(n + 1, 1);

    % Base frame (index 0)
    O_all{1} = T_root(1:3, 4);
    z_all{1} = T_root(1:3, 3);

    for i = 1:n
        Ti = fk.T0i{i};
        O_all{i + 1} = Ti(1:3, 4);
        z_all{i + 1} = Ti(1:3, 3);
    end

    % Standardize revolute flags (1 = revolute, 0 or 2 = prismatic)
    if isfield(dh, 'sigma')
        sigma = dh.sigma == 1;
    elseif isfield(dh, 'DHpram')
        sigma = dh.DHpram(:, 5) == 1;
    else
        sigma = true(n, 1);
    end

    % End-Effector position & Jacobian
    p_ee = fk.p_ee;
    J_v_ee = sym(zeros(3, n));
    J_w_ee = sym(zeros(3, n));

    for j = 1:n
        z_prev = z_all{j};
        O_prev = O_all{j};

        if sigma(j)
            % Revolute joint: J_v = z_{j-1} x (p_ee - O_{j-1}), J_w = z_{j-1}
            J_v_ee(:, j) = cross(z_prev, p_ee - O_prev);
            J_w_ee(:, j) = z_prev;
        else
            % Prismatic joint: J_v = z_{j-1}, J_w = 0
            J_v_ee(:, j) = z_prev;
            J_w_ee(:, j) = sym([0; 0; 0]);
        end
    end
    J_ee = [J_v_ee; J_w_ee];

    % Time-derivative of End-Effector Jacobian: dJ/dt = sum_k (dJ/dq_k * dq_k)
    dJ_ee = sym(zeros(6, n));
    for k = 1:n
        dJ_ee = dJ_ee + diff(J_ee, q(k)) * dq(k);
    end

    % Link CoM Jacobians for Dynamics
    P_Ci = inertia.P_Ci;
    J_v_Ci = cell(n, 1);
    J_w_Ci = cell(n, 1);
    J_Ci   = cell(n, 1);

    for i = 1:n
        Ti = fk.T0i{i};
        Ri = Ti(1:3, 1:3);
        Oi = Ti(1:3, 4);

        % World position of CoM of link i: p_{C_i} = O_i + R_i * P_{C_i}
        p_Ci_world = Oi + Ri * sym(P_Ci(i, :)');

        Jv_i = sym(zeros(3, n));
        Jw_i = sym(zeros(3, n));

        for j = 1:i
            z_prev = z_all{j};
            O_prev = O_all{j};

            if sigma(j)
                % Revolute joint: z_{j-1} x (p_{C_i} - O_{j-1})
                Jv_i(:, j) = cross(z_prev, p_Ci_world - O_prev);
                Jw_i(:, j) = z_prev;
            else
                % Prismatic joint: z_{j-1}
                Jv_i(:, j) = z_prev;
                Jw_i(:, j) = sym([0; 0; 0]);
            end
        end

        J_v_Ci{i} = simplify(Jv_i);
        J_w_Ci{i} = simplify(Jw_i);
        J_Ci{i}   = [J_v_Ci{i}; J_w_Ci{i}];

        if verbose
            fprintf('  [JACOBIAN-SYM] Link %d CoM Jacobian synthesized.\n', i);
        end
    end

    jac = struct();
    jac.q      = q;
    jac.dq     = dq;
    jac.J_ee   = J_ee;
    jac.dJ_ee  = dJ_ee;
    jac.J_v_Ci = J_v_Ci;
    jac.J_w_Ci = J_w_Ci;
    jac.J_Ci   = J_Ci;

    if export_fun
        if verbose, fprintf('  [JACOBIAN-SYM] Exporting matlabFunction handles...\n'); end
        jac.J_ee_fun  = matlabFunction(J_ee, 'Vars', {q});
        jac.dJ_ee_fun = matlabFunction(dJ_ee, 'Vars', {q, dq});
    end
end
