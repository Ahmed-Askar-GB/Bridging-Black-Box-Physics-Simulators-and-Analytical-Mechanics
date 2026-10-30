function fk = fk_sym(dh, chain, opts)
% FK_SYM  Pure MATLAB Symbolic Forward Kinematics via standard DH convention.
%
% Implements the textbook kinematic formulation:
%   T_0^i(q) = T_0^{i-1} * A_i(q_i)
% where:
%   A_i(q_i) = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
%
% =========================================================================
% INPUTS:
%   dh     (struct) Output from urdf_extract_dh:
%                     .table  [Nx4] [theta_offset, d, a, alpha]
%                     .sigma  [Nx1] 1 = revolute, 2 = prismatic
%                     .DHpram [Nx5] [a, alpha, theta_offset, d, sigma] (optional)
%   chain  (struct) Output from urdf_build_chain:
%                     .n            Number of DOFs
%                     .T_root_to_j1 [4x4] Fixed transform from base to joint 1
%                     .T_ee         [4x4] Fixed transform from link n to EE
%   opts   (struct) [optional]
%                     .export_matlabFunction (logical) Generate numeric function handles (default: true)
%                     .verbose               (logical) Print progress (default: false)
%
% OUTPUT:
%   fk     (struct) Containing:
%            .q            Symbolic joint variable vector [Nx1 sym]
%            .T0i          Cell array {Nx1 sym} of cumulative transforms T_0^i(q)
%            .T0e          Full symbolic End-Effector transform T_0^ee(q) [4x4 sym]
%            .p_ee         Symbolic EE position vector [3x1 sym]
%            .R_ee         Symbolic EE orientation matrix [3x3 sym]
%            .fk_fun       (optional) matlabFunction handle @(q) -> [p_ee, R_ee]
%            .fk_ee_fun    (optional) matlabFunction handle @(q) -> T_0^ee
% =========================================================================

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    export_fun = true;  if isfield(opts, 'export_matlabFunction'), export_fun = opts.export_matlabFunction; end
    verbose    = false; if isfield(opts, 'verbose'),               verbose    = opts.verbose; end

    n = chain.n;

    % Standardize DH parameters
    if isfield(dh, 'table')
        table = dh.table;
    elseif isfield(dh, 'DHpram')
        table = [dh.DHpram(:, 3), dh.DHpram(:, 4), dh.DHpram(:, 1), dh.DHpram(:, 2)];
    else
        error('dh struct must contain .table [Nx4] or .DHpram [Nx5]');
    end

    if isfield(dh, 'sigma')
        sigma = dh.sigma;
    elseif isfield(dh, 'DHpram')
        sigma = dh.DHpram(:, 5);
    else
        sigma = ones(n, 1);
    end

    T_root_to_j1 = chain.T_root_to_j1;
    if isempty(T_root_to_j1), T_root_to_j1 = eye(4); end
    T_ee = chain.T_ee;
    if isempty(T_ee), T_ee = eye(4); end

    % Initialize symbolic joint variables
    q = sym('q', [n, 1], 'real');

    % Cumulative frame transformations
    T0i = cell(n, 1);
    T_curr = sym(T_root_to_j1);

    for i = 1:n
        th_off = table(i, 1);
        d_val  = table(i, 2);
        a_val  = table(i, 3);
        al_val = table(i, 4);

        if sigma(i) == 1
            % Revolute joint: theta_i = th_off + q_i, d_i is constant
            theta_i = th_off + q(i);
            d_i     = d_val;
        else
            % Prismatic joint: theta_i is constant, d_i = d_val + q_i
            theta_i = th_off;
            d_i     = d_val + q(i);
        end

        % Elementary Standard DH transformation:
        % A_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
        ct = cos(theta_i);
        st = sin(theta_i);
        ca = cos(al_val);
        sa = sin(al_val);

        Ai = [ct,  -st*ca,   st*sa,  a_val*ct;
              st,   ct*ca,  -ct*sa,  a_val*st;
               0,      sa,      ca,     d_i;
               0,       0,       0,       1];

        % T_0^i = T_0^{i-1} * A_i(q_i)
        T_curr = T_curr * Ai;
        T0i{i} = simplify(T_curr);

        if verbose
            fprintf('  [FK-SYM] Link %d transform T_0^%d synthesized.\n', i, i);
        end
    end

    % End-Effector transformation: T_0^ee = T_0^n * T_ee
    T0e  = simplify(T_curr * sym(T_ee));
    p_ee = T0e(1:3, 4);
    R_ee = T0e(1:3, 1:3);

    fk = struct();
    fk.n    = n;
    fk.q    = q;
    fk.T0i  = T0i;
    fk.T0e  = T0e;
    fk.p_ee = p_ee;
    fk.R_ee = R_ee;
    fk.dh   = dh;
    fk.chain = chain;

    % Optional high-speed matlabFunction compilation
    if export_fun
        if verbose, fprintf('  [FK-SYM] Exporting matlabFunction handles...\n'); end
        fk.fk_ee_fun = matlabFunction(T0e, 'Vars', {q});
        fk.fk_fun    = matlabFunction(p_ee, R_ee, 'Vars', {q});
    end
end
