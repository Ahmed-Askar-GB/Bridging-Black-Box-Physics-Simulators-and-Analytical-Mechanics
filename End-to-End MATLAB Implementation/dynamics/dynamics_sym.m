function dyn = dynamics_sym(dh, inertia, chain, opts)
% DYNAMICS_SYM  Pure MATLAB Symbolic Euler-Lagrange Robot Dynamics.
%
% Implements the textbook Lagrangian mechanics formulations:
%   1. Mass Matrix:
%      M(q) = \sum_{i=1}^n [m_i J_{v,i}^T J_{v,i} + J_{\omega,i}^T I_i^0 J_{\omega,i}]
%      where I_i^0 = R_i * I_i * R_i^T (Inertia tensor in world/base orientation)
%
%   2. Coriolis & Centrifugal Matrix via Christoffel Symbols of the 1st kind:
%      c_{ijk} = 0.5 * [ \partial M_{ij}/\partial q_k + \partial M_{ik}/\partial q_j - \partial M_{kj}/\partial q_i ]
%      C_{ij}(q, \dot{q}) = \sum_{k=1}^n c_{ijk} * \dot{q}_k
%
%   3. Gravity Torque Vector:
%      G(q) = -\sum_{i=1}^n J_{v,i}^T * (m_i * g)
%
%   4. Euler-Lagrange Equations of Motion:
%      \tau(q, \dot{q}, \ddot{q}) = M(q)*\ddot{q} + C(q, \dot{q})*\dot{q} + G(q)
%
% =========================================================================
% INPUTS:
%   dh      (struct) Standard DH parameters
%   inertia (struct) Link masses, CoM vectors, and 3x3 inertia tensors
%   chain   (struct) Kinematic chain definition
%   opts    (struct) [optional]
%                      .g_vec                 [3x1] (default: [0; 0; -9.81])
%                      .export_matlabFunction (logical) (default: true)
%                      .verbose               (logical) (default: false)
%
% OUTPUT:
%   dyn     (struct) Containing:
%             .q, .dq, .ddq    Symbolic coordinate variables
%             .M_sym           Symbolic Mass Matrix [NxN sym]
%             .C_sym           Symbolic Coriolis Matrix [NxN sym]
%             .G_sym           Symbolic Gravity Vector [Nx1 sym]
%             .tau_sym         Symbolic Inverse Dynamics Vector [Nx1 sym]
%             .c_ijk           Cell/Tensor of Christoffel symbols
%             .M_fun           @(q) -> M [NxN] (matlabFunction)
%             .C_fun           @(q, dq) -> C [NxN] (matlabFunction)
%             .G_fun           @(q) -> G [Nx1] (matlabFunction)
%             .tau_fun         @(q, dq, ddq) -> tau [Nx1] (matlabFunction)
%             .forward_dyn_fun @(q, dq, tau) -> ddq [Nx1]
% =========================================================================

    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    g_vec      = [0; 0; -9.81]; if isfield(opts, 'g_vec'),                 g_vec      = opts.g_vec(:); end
    export_fun = true;          if isfield(opts, 'export_matlabFunction'), export_fun = opts.export_matlabFunction; end
    verbose    = false;         if isfield(opts, 'verbose'),               verbose    = opts.verbose; end

    n = chain.n;

    % Obtain symbolic Jacobians and kinematics
    jac = jacobian_sym(dh, inertia, chain, struct('export_matlabFunction', false, 'verbose', false));
    fk  = fk_sym(dh, chain, struct('export_matlabFunction', false, 'verbose', false));

    q   = jac.q;
    dq  = jac.dq;
    ddq = sym('ddq', [n, 1], 'real');

    m_vec = inertia.m(:);

    % Normalize inertia tensors
    I_tensors = cell(n, 1);
    for i = 1:n
        if iscell(inertia.I)
            I_tensors{i} = inertia.I{i};
        elseif ndims(inertia.I) == 3 && size(inertia.I, 3) == n
            I_tensors{i} = inertia.I(:, :, i);
        elseif ndims(inertia.I) == 3 && size(inertia.I, 1) == n
            I_tensors{i} = squeeze(inertia.I(i, :, :));
        else
            I_tensors{i} = inertia.I(:, :, i);
        end
    end

    % -------------------------------------------------------------------------
    % Step 1: Compute Mass Matrix M(q)
    % M(q) = sum_{i=1}^n (m_i * J_{v,i}^T * J_{v,i} + J_{w,i}^T * I_i^0 * J_{w,i})
    % -------------------------------------------------------------------------
    if verbose, fprintf('  [DYN-SYM] Computing Generalized Mass Matrix M(q)...\n'); end
    M_sym = sym(zeros(n, n));

    for i = 1:n
        mi   = m_vec(i);
        Jvi  = jac.J_v_Ci{i};
        Jwi  = jac.J_w_Ci{i};
        Ri   = fk.T0i{i}(1:3, 1:3);
        Ii_0 = Ri * sym(I_tensors{i}) * Ri.'; % Inertia in base frame coordinates

        M_sym = M_sym + mi * (Jvi.' * Jvi) + (Jwi.' * Ii_0 * Jwi);
    end
    M_sym = simplify(M_sym);

    % -------------------------------------------------------------------------
    % Step 2: Compute Coriolis Effects
    % Mode A: Classical Christoffel Symbols (O(n^3))
    % Mode B: Generalized Momentum Directional Derivative (O(n^2))
    %         c_bias = d(M*dq)/dt - 0.5 * grad_q(dq'*M*dq)
    % -------------------------------------------------------------------------
    if verbose, fprintf('  [DYN-SYM] Computing Coriolis terms (Momentum & Christoffel)...\n'); end

    % Mode B: Momentum Directional Derivative
    p_mom       = M_sym * dq;
    dp_dt       = jacobian(p_mom, q) * dq;  % d(M*dq)/dt
    T_kin       = 0.5 * (dq.' * M_sym * dq);
    grad_T      = jacobian(T_kin, q).';     % grad_q(T)
    c_bias_sym  = simplify(dp_dt - grad_T);

    % Mode A: Classical Christoffel symbols
    C_sym = sym(zeros(n, n));
    c_ijk = cell(n, n, n);

    dM_dq = cell(n, 1);
    for k = 1:n
        dM_dq{k} = diff(M_sym, q(k));
    end

    for i = 1:n
        for j = 1:n
            c_ij = sym(0);
            for k = 1:n
                c_ijk{i, j, k} = 0.5 * (dM_dq{k}(i, j) + dM_dq{j}(i, k) - dM_dq{i}(k, j));
                c_ij = c_ij + c_ijk{i, j, k} * dq(k);
            end
            C_sym(i, j) = c_ij;
        end
    end
    C_sym = simplify(C_sym);

    % -------------------------------------------------------------------------
    % Step 3: Compute Gravity Vector G(q)
    % G(q) = -sum_{i=1}^n J_{v,i}^T * (m_i * g)
    % -------------------------------------------------------------------------
    if verbose, fprintf('  [DYN-SYM] Computing Gravity Vector G(q)...\n'); end
    G_sym = sym(zeros(n, 1));
    sym_g = sym(g_vec);

    for i = 1:n
        mi  = m_vec(i);
        Jvi = jac.J_v_Ci{i};
        G_sym = G_sym - Jvi.' * (mi * sym_g);
    end
    G_sym = simplify(G_sym);

    % Directional Hessian-Vector Product for G: \nabla_q^2 V(q)*v = \partial G/\partial q * v
    v_dir     = sym('v_dir', [n, 1], 'real');
    hvp_G_sym = jacobian(G_sym, q) * v_dir;

    % -------------------------------------------------------------------------
    % Step 4: Inverse Dynamics tau(q, dq, ddq)
    % -------------------------------------------------------------------------
    tau_sym = simplify(M_sym * ddq + c_bias_sym + G_sym);

    dyn = struct();
    dyn.n          = n;
    dyn.q          = q;
    dyn.dq         = dq;
    dyn.ddq        = ddq;
    dyn.v_dir      = v_dir;
    dyn.M_sym      = M_sym;
    dyn.C_sym      = C_sym;
    dyn.G_sym      = G_sym;
    dyn.c_bias_sym = c_bias_sym;
    dyn.hvp_G_sym  = hvp_G_sym;
    dyn.tau_sym    = tau_sym;
    dyn.c_ijk      = c_ijk;
    dyn.dh         = dh;
    dyn.inertia    = inertia;
    dyn.chain      = chain;

    % Export high-speed matlabFunction handles
    if export_fun
        if verbose, fprintf('  [DYN-SYM] Generating matlabFunction handles...\n'); end
        dyn.M_fun      = matlabFunction(M_sym, 'Vars', {q});
        dyn.C_fun      = matlabFunction(C_sym, 'Vars', {q, dq});
        dyn.G_fun      = matlabFunction(G_sym, 'Vars', {q});
        dyn.c_bias_fun = matlabFunction(c_bias_sym, 'Vars', {q, dq});
        dyn.hvp_G_fun  = matlabFunction(hvp_G_sym, 'Vars', {q, v_dir});
        dyn.tau_fun    = matlabFunction(tau_sym, 'Vars', {q, dq, ddq});

        % Forward dynamics evaluation function with Cholesky solve
        dyn.forward_dyn_fun = @(q_val, dq_val, tau_val) ...
            chol_solve_sym(dyn.M_fun(q_val(:)), tau_val(:) - dyn.c_bias_fun(q_val(:), dq_val(:)) - dyn.G_fun(q_val(:)));
    end
end

function x = chol_solve_sym(A, b)
    % Helper: Fast Cholesky solve for SPD system A*x = b
    try
        R = chol(A);
        x = R \ (R' \ b);
    catch
        x = A \ b;
    end
end
end
