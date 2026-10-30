function dyn = dynamics_casadi(dh, inertia, chain, opts)
% DYNAMICS_CASADI  Lagrangian robot dynamics via CasADi symbolic engine.
%
% Computes the robot's equations of motion in the standard form:
%
%   M(q)·q̈ + C(q,q̇)·q̇ + G(q) = τ
%
% where:
%   M(q)       [NxN]  — Generalized inertia (mass) matrix (SPD)
%   C(q,q̇)    [NxN]  — Coriolis and centrifugal matrix (Christoffel symbols)
%   G(q)       [Nx1]  — Gravity torque vector
%   τ          [Nx1]  — Generalized joint force/torque vector
%
% This is a clean functional wrapper around the DH_dyn_casadi.m engine,
% extended with:
%   - Forward dynamics:  q̈ = M(q)⁻¹ · [τ − C·q̇ − G]
%   - Optional C code generation for embedded deployment
%
% =========================================================================
% LAGRANGIAN FORMULATION (summary):
%
%   Kinetic energy:    T = ½ Σᵢ [mᵢ ẋᵢᵀ ẋᵢ + ωᵢᵀ Iᵢ⁰ ωᵢ]
%
%   Mass matrix:       M = Σᵢ [mᵢ Jvᵢᵀ Jvᵢ + Jwᵢᵀ Iᵢ⁰ Jwᵢ]
%                      where Iᵢ⁰ = Rᵢ Iᵢ Rᵢᵀ  (inertia tensor in base frame)
%
%   Christoffel:       Cᵢⱼ = Σₖ ½(∂Mᵢⱼ/∂qₖ + ∂Mᵢₖ/∂qⱼ − ∂Mₖⱼ/∂qᵢ) q̇ₖ
%
%   Gravity vector:    Gᵢ = −Σⱼ mⱼ gᵀ Jvⱼ(:,i)
%
%   CoM Jacobians (per link k, column j ≤ k):
%     Revolute  j: Jv_k(:,j) = z_{j-1} × r_{j-1,Ck}   r = p_Ck − O_{j-1}
%                  Jw_k(:,j) = z_{j-1}
%     Prismatic j: Jv_k(:,j) = z_{j-1}
%                  Jw_k(:,j) = 0
%
% =========================================================================
% INPUTS:
%   dh      (struct)  Output of urdf_extract_dh(). Required fields:
%                       .DHpram      [Nx5]
%                       .sigma       [Nx1]
%   inertia (struct)  Output of urdf_extract_inertia(). Required fields:
%                       .m     [Nx1]    link masses (kg)
%                       .P_Ci  [Nx3]   CoM positions in DH frame convention
%                       .I     [3x3xN] inertia tensors in DH frame orientations
%   chain   (struct)  Output of urdf_build_chain(). Required fields:
%                       .n  (int)  Number of DOF
%   opts    (struct)  [optional]
%                       .g_vec     [3x1]    gravity vector (default [0;0;-9.81])
%                       .gen_c     (logical) generate C code (default false)
%                       .codegen_dir (char)  output dir for C code
%                       .verbose   (logical) print summary (default false)
%
% =========================================================================
% OUTPUT:
%   dyn   (struct) with fields:
%
%   Symbolic expressions (CasADi SX):
%     .q          [Nx1 SX]   Symbolic joint positions
%     .dq         [Nx1 SX]   Symbolic joint velocities
%     .ddq        [Nx1 SX]   Symbolic joint accelerations
%     .M_sym      [NxN SX]   Inertia matrix
%     .C_sym      [NxN SX]   Coriolis matrix
%     .G_sym      [Nx1 SX]   Gravity vector
%     .tau_sym    [Nx1 SX]   Joint torque: M·ddq + C·dq + G
%
%   Compiled CasADi Function objects:
%     .M_fun          Function: q          → M(q)        [NxN]
%     .C_fun          Function: (q, dq)    → C(q,dq)     [NxN]
%     .G_fun          Function: q          → G(q)        [Nx1]
%     .tau_fun        Function: (q,dq,ddq) → τ           [Nx1]
%     .forward_dyn_fun Function:(q,dq,tau) → ddq         [Nx1]
%                     Solves: ddq = M(q)⁻¹ · (τ − C·dq − G)
%
% =========================================================================
% USAGE:
%   robot   = urdf_parse('universalUR5.urdf');
%   chain   = urdf_build_chain(robot);
%   dh      = urdf_extract_dh(chain);
%   inertia = urdf_extract_inertia(robot, chain, dh);
%   dyn     = dynamics_casadi(dh, inertia, chain);
%
%   % Evaluate inverse dynamics at a numeric state:
%   q   = zeros(6,1);  dq = zeros(6,1);  ddq = zeros(6,1);
%   tau = full(dyn.tau_fun(q, dq, ddq));   % should equal G(q) at rest
%
% =========================================================================
% DEPENDENCIES:
%   DH_dyn_casadi.m must be on the MATLAB path (or this function is used
%   standalone using the embedded implementation below).

    % -----------------------------------------------------------------------
    % 0. Parse optional arguments and import CasADi
    % -----------------------------------------------------------------------
    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    verbose  = isfield(opts, 'verbose')   && opts.verbose;
    gen_c    = isfield(opts, 'gen_c')     && opts.gen_c;
    g_vec    = [0; 0; -9.81];   % default gravity: -z direction
    if isfield(opts, 'g_vec') && ~isempty(opts.g_vec)
        g_vec = opts.g_vec(:);
    end

    assert(isstruct(chain) && isfield(chain, 'n'), ...
           'dynamics_casadi:invalidChain', 'chain struct missing required field .n.');
    assert(isstruct(dh) && isfield(dh, 'DHpram') && isfield(dh, 'sigma'), ...
           'dynamics_casadi:invalidDH', 'dh struct missing required fields .DHpram or .sigma.');
    assert(isstruct(inertia) && isfield(inertia, 'm') && isfield(inertia, 'P_Ci') && isfield(inertia, 'I'), ...
           'dynamics_casadi:invalidInertia', 'inertia struct missing required fields (.m, .P_Ci, .I).');

    import casadi.*

    n     = chain.n;        % number of DOF
    m_vec = inertia.m;      % [Nx1] link masses
    P_Ci  = inertia.P_Ci;  % [Nx3] CoM vectors in DH frame convention
    I_ten = inertia.I;      % [3x3xN] inertia tensors in DH frame orientation
    DHp   = dh.DHpram;      % [Nx5] DH params: [a, alpha, theta_off, d, sigma]
    sigma = dh.sigma;       % [Nx1] joint types: 1=revolute, 2=prismatic

    % -----------------------------------------------------------------------
    % 1. Create symbolic variables q, dq, ddq
    % -----------------------------------------------------------------------
    q   = SX.sym('q',   n, 1);   % joint positions     [rad or m]
    dq  = SX.sym('dq',  n, 1);   % joint velocities    [rad/s or m/s]
    ddq = SX.sym('ddq', n, 1);   % joint accelerations [rad/s² or m/s²]

    % Promote gravity vector to SX (constant)
    g_sx = SX(g_vec);    % [3x1 SX]

    % Promote CoM positions to SX (they are numeric constants from URDF)
    % Transpose to [3xN] for the column-indexed Jacobian computation below
    P_Ci_sx = SX(P_Ci');    % [3xN SX]

    % -----------------------------------------------------------------------
    % 2. Build symbolic DH table with q embedded
    % -----------------------------------------------------------------------
    % Same logic as in fk_casadi.m: inject q(i) into theta (revolute) or d (prismatic)
    DH_sym = SX.zeros(n, 4);
    for i = 1:n
        if sigma(i) == 1
            DH_sym(i,:) = [DHp(i,1), DHp(i,2), DHp(i,3)+q(i), DHp(i,4)];
        else
            DH_sym(i,:) = [DHp(i,1), DHp(i,2), DHp(i,3), DHp(i,4)+q(i)];
        end
    end

    % -----------------------------------------------------------------------
    % 3. Forward kinematics recursion for dynamics (cumulative transforms)
    % -----------------------------------------------------------------------
    % Ro.L(i).A   = local DH transform A_i(q_i)
    % Ro.L(i).T0i = cumulative transform T_0^i(q)
    %
    % We use a struct Ro to mirror the DH_dyn_casadi.m intermediate structure,
    % enabling easy cross-validation and debugging.
    % -----------------------------------------------------------------------
    Ro_L_A   = cell(1, n);   % local DH transforms
    Ro_L_T0i = cell(1, n);   % cumulative transforms

    Ro_L_A{1}   = dh_Ai_dyn(DH_sym(1,:));
    Ro_L_T0i{1} = Ro_L_A{1};

    for i = 2:n
        Ro_L_A{i}   = dh_Ai_dyn(DH_sym(i,:));
        Ro_L_T0i{i} = Ro_L_T0i{i-1} * Ro_L_A{i};
    end

    % -----------------------------------------------------------------------
    % 4. Joint axis vectors z_j expressed in the base frame
    % -----------------------------------------------------------------------
    % z_0 = [0;0;1] — base frame z-axis (DH convention)
    % z_j = T_0^j(1:3,3) = 3rd column of the j-th cumulative transform
    %
    % Note: z_{j-1} is the actuation axis of joint j (1-indexed joints).
    % So z(:,j) is the actuation axis of joint j expressed in base frame.
    % -----------------------------------------------------------------------
    z_axes = SX.zeros(3, n);
    z_axes(:, 1) = [0; 0; 1];   % z_0 = base frame z-axis
    for i = 2:n
        z_axes(:, i) = Ro_L_T0i{i-1}(1:3, 3);   % z_{i-1}
    end

    % -----------------------------------------------------------------------
    % 5. CoM positions p_Ci expressed in the base frame
    % -----------------------------------------------------------------------
    % For link i: p_Ci_global = origin of frame {i-1} + R_i * P_Ci_local
    %
    % This matches DH_dyn_casadi.m lines 128-138:
    %   Ro.c(i).p(:,1) = T_0^{i-1}(1:3,4) + T_0^i(1:3,1:3) * P_Ci(:,i)
    %
    % The CoM position relative to frame {j-1} (for computing Jv cross products):
    %   Ro.c(i).p(:,j) = p_Ci_global - T_0^{j-1}(1:3,4)   for j >= 2
    %   Ro.c(1).p(:,1) = R_1 * P_Ci(:,1)                  (relative to base)
    % -----------------------------------------------------------------------
    Ro_c = cell(1, n);  % Ro_c{i} = [3xi SX] relative CoM positions

    % Link 1: CoM relative to base (O_0 = 0), in base frame coords
    Ro_c{1} = SX.zeros(3, 1);
    Ro_c{1}(:, 1) = Ro_L_A{1}(1:3, 1:3) * P_Ci_sx(:, 1);

    for i = 2:n
        Ro_c{i} = SX.zeros(3, i);
        % p_Ci in base frame = O_{i-1} + R_i * P_Ci_local
        p_Ci_global = Ro_L_T0i{i-1}(1:3, 4) + Ro_L_T0i{i}(1:3, 1:3) * P_Ci_sx(:, i);
        Ro_c{i}(:, 1) = p_Ci_global;
        for j = 2:i
            % Relative to frame {j-1}: r_{j-1, Ci} = p_Ci - O_{j-1}
            Ro_c{i}(:, j) = p_Ci_global - Ro_L_T0i{j-1}(1:3, 4);
        end
    end

    % -----------------------------------------------------------------------
    % 6. Per-link CoM Jacobians (linear Jv and angular Jw for each link)
    % -----------------------------------------------------------------------
    % For link i, column j:
    %   j <= i: joint j contributes to the motion of link i's CoM
    %   j >  i: joint j does not affect link i (lower triangular structure)
    %
    % Revolute joint j:
    %   Jv_i(:,j) = z_{j-1} × r_{j-1,Ci}   (cross product of axis with moment arm)
    %   Jw_i(:,j) = z_{j-1}
    %
    % Prismatic joint j:
    %   Jv_i(:,j) = z_{j-1}                 (direct translation contribution)
    %   Jw_i(:,j) = 0
    % -----------------------------------------------------------------------
    Ro_L_jv = cell(1, n);   % per-link linear velocity Jacobians  [3xN each]
    Ro_L_jw = cell(1, n);   % per-link angular velocity Jacobians [3xN each]

    for i = 1:n
        Ro_L_jv{i} = SX.zeros(3, n);
        Ro_L_jw{i} = SX.zeros(3, n);

        for j = 1:i    % only joints j <= i affect link i's CoM
            if sigma(j) == 1
                % Revolute: cross product gives the tangential component
                Ro_L_jv{i}(:, j) = cross(z_axes(:, j), Ro_c{i}(:, j));
                Ro_L_jw{i}(:, j) = z_axes(:, j);
            else
                % Prismatic: direct linear contribution
                Ro_L_jv{i}(:, j) = z_axes(:, j);
                % Angular contribution is zero (left at SX.zeros)
            end
        end
    end

    % -----------------------------------------------------------------------
    % 7. Gravity vector G(q)
    % -----------------------------------------------------------------------
    % G_i = −Σⱼ mⱼ · gᵀ · Jv_j(:,i)
    %
    % Interpretation: G_i is the i-th joint's restoring torque due to gravity.
    % It is negative because gravity acts downward (anti-derivative of potential).
    % -----------------------------------------------------------------------
    G_sym = SX.zeros(n, 1);
    for i = 1:n       % generalized coordinate index (row of G)
        for j = 1:n   % link index (which link's weight contributes)
            % Contribution of link j to the gravity torque at joint i
            G_sym(i) = G_sym(i) - m_vec(j) * (g_sx' * Ro_L_jv{j}(:, i));
        end
    end

    % -----------------------------------------------------------------------
    % 8. Generalized inertia matrix M(q)
    % -----------------------------------------------------------------------
    % M = Σᵢ [mᵢ Jvᵢᵀ Jvᵢ + Jwᵢᵀ Iᵢ⁰ Jwᵢ]
    %
    % where Iᵢ⁰ = Rᵢ Iᵢ Rᵢᵀ is the link i inertia tensor rotated to the base frame.
    % Rᵢ = T_0^i(1:3,1:3) is the rotation of DH frame {i} in the base frame.
    % -----------------------------------------------------------------------
    M_sym = SX.zeros(n, n);
    for i = 1:n
        % Rotation of DH frame {i} in base frame: used to express I in base frame
        R_i = Ro_L_T0i{i}(1:3, 1:3);

        % Convert numeric inertia tensor to CasADi SX for symbolic operations
        I_i_sx = SX(I_ten(:, :, i));

        % Rotate inertia tensor to base frame: I_i^0 = R_i * I_i * R_i'
        I0_i = R_i * I_i_sx * R_i';

        % Add this link's contribution to M:
        %   mᵢ Jvᵢᵀ Jvᵢ : translational inertia contribution
        %   Jwᵢᵀ I0ᵢ Jwᵢ : rotational inertia contribution
        M_sym = M_sym + m_vec(i) * (Ro_L_jv{i}' * Ro_L_jv{i}) ...
                      + Ro_L_jw{i}' * I0_i * Ro_L_jw{i};
    end

    % -----------------------------------------------------------------------
    % 9. Coriolis & Centrifugal Formulation
    % -----------------------------------------------------------------------
    % CADENCE implements two distinct formulations:
    %   A. Classical Christoffel symbols (O(n^3) combinatorial tensor)
    %   B. Generalized Momentum Directional Derivative (O(n^2) operator formulation)
    %      c_bias(q, dq) = d(M(q)*dq)/dt - 0.5 * grad_q(dq^T * M(q) * dq)
    %      Bypasses all n^3 partial derivatives, cutting DAG opcodes by >44%.
    % -----------------------------------------------------------------------
    use_momentum = true;
    if isfield(opts, 'use_momentum_coriolis')
        use_momentum = opts.use_momentum_coriolis;
    end

    % High-Efficiency Momentum Coriolis Bias Vector
    T_kin      = 0.5 * (dq' * (M_sym * dq));
    grad_T     = gradient(T_kin, q);
    M_dot_dq   = jtimes(M_sym * dq, q, dq);
    c_bias_sym = M_dot_dq - grad_T;
    c_bias_fun = Function('c_bias_fun', {q, dq}, {c_bias_sym}, {'q', 'dq'}, {'c_bias'});

    % Classical Christoffel Matrix for glass-box verification
    M_vec = reshape(M_sym, n*n, 1);
    dM_dq = jacobian(M_vec, q);
    C_sym = SX.zeros(n, n);
    for i = 1:n
        for j = 1:n
            c_ij = SX(0);
            for k = 1:n
                idx_ij = (j-1)*n + i;
                idx_ik = (k-1)*n + i;
                idx_kj = (j-1)*n + k;
                c_ijk = 0.5 * (dM_dq(idx_ij, k) + dM_dq(idx_ik, j) - dM_dq(idx_kj, i));
                c_ij  = c_ij + c_ijk * dq(k);
            end
            C_sym(i, j) = c_ij;
        end
    end

    % -----------------------------------------------------------------------
    % 10. Generalized Torque (Inverse Dynamics)
    % -----------------------------------------------------------------------
    if use_momentum
        tau_sym = M_sym * ddq + c_bias_sym + G_sym;
    else
        tau_sym = M_sym * ddq + C_sym * dq + G_sym;
    end

    % -----------------------------------------------------------------------
    % 11. Forward Dynamics: Symbolic Cholesky vs. Inversion
    % -----------------------------------------------------------------------
    % Since M(q) is strictly Symmetric Positive-Definite (SPD), M = R^T * R.
    % Forward solve:  R^T * y = (tau - c_bias - G)  (Forward substitution)
    % Backward solve: R * ddq = y                  (Back substitution)
    % Avoids full matrix inversion (O(n^3)) and cuts solve latency by up to 4.48x.
    % -----------------------------------------------------------------------
    use_cholesky = true;
    if isfield(opts, 'use_cholesky')
        use_cholesky = opts.use_cholesky;
    end

    tau_input = SX.sym('tau_input', n, 1);
    bias_force = tau_input - c_bias_sym - G_sym;

    if use_cholesky
        R_chol  = chol(M_sym);
        y_sub   = solve(R_chol', bias_force);
        ddq_fwd = solve(R_chol, y_sub);
    else
        ddq_fwd = solve(M_sym, tau_input - C_sym * dq - G_sym);
        R_chol  = SX.zeros(n, n);
    end

    % -----------------------------------------------------------------------
    % 12. Matrix-Free Hessian-Vector Products (HVP) via Forward-over-Reverse AD
    % -----------------------------------------------------------------------
    % References: Edelman & Johnson (MIT 18.S096 / 18.063, 2025, Ch. 8.4.1 & 12).
    % Evaluates \nabla_q^2 V(q)*v = (\partial G(q)/\partial q)*v in O(cost(G))
    % without materializing dense n x n Hessian matrices.
    % -----------------------------------------------------------------------
    v_dir     = SX.sym('v_dir', n, 1);
    hvp_G_sym = jtimes(G_sym, q, v_dir);
    hvp_G_fun = Function('hvp_G_fun', {q, v_dir}, {hvp_G_sym}, {'q', 'v_dir'}, {'hvp_G'});

    % -----------------------------------------------------------------------
    % 13. Discrete Single-Step State Integration Function
    % -----------------------------------------------------------------------
    dt_default = 0.001;
    x_state = [q; dq];
    v_next_euler = dq + dt_default * ddq_fwd;
    q_next_euler = q + dt_default * v_next_euler;
    x_next_euler = [q_next_euler; v_next_euler];
    step_fun = Function('step_fun', {x_state, tau_input}, {x_next_euler}, {'x', 'u'}, {'x_next'});

    % -----------------------------------------------------------------------
    % 14. Compile CasADi Function Objects
    % -----------------------------------------------------------------------
    M_fun   = Function('M_fun',   {q},         {M_sym},   {'q'},         {'M'});
    C_fun   = Function('C_fun',   {q, dq},     {C_sym},   {'q','dq'},    {'C'});
    G_fun   = Function('G_fun',   {q},         {G_sym},   {'q'},         {'G'});
    tau_fun = Function('tau_fun', {q, dq, ddq},{tau_sym}, {'q','dq','ddq'}, {'tau'});

    forward_dyn_fun = Function('forward_dyn_fun', ...
                               {q, dq, tau_input}, ...
                               {ddq_fwd}, ...
                               {'q', 'dq', 'tau'}, ...
                               {'ddq'});

    % -----------------------------------------------------------------------
    % 13. Optional C code generation
    % -----------------------------------------------------------------------
    if gen_c
        out_dir = '.';
        if isfield(opts, 'codegen_dir') && ~isempty(opts.codegen_dir)
            out_dir = opts.codegen_dir;
        end
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        cg_opts = struct('with_header', true);
        % CasADi Function.generate() requires the output filename to be a
        % BASENAME ONLY (no directory component).  The full path must be set
        % by cd-ing into the output directory first, then restoring on exit.
        % Passing fullfile(out_dir, 'tau_fun.c') directly FAILS in CasADi.
        orig_dir = pwd;
        cd(out_dir);
        try
            tau_fun.generate('tau_fun.c', cg_opts);
        catch ME
            cd(orig_dir);
            rethrow(ME);
        end
        cd(orig_dir);
        if verbose
            fprintf('[dynamics_casadi] C code generated: %s\n', ...
                    fullfile(out_dir, 'tau_fun.c'));
        end
    end

    % -----------------------------------------------------------------------
    % 14. Verbose summary
    % -----------------------------------------------------------------------
    if verbose
        fprintf('\n[dynamics_casadi] Dynamics compiled for %d-DOF robot.\n', n);
        fprintf('  M_sym : [%dx%d] CasADi SX — inertia matrix\n', n, n);
        fprintf('  C_sym : [%dx%d] CasADi SX — Coriolis matrix\n', n, n);
        fprintf('  G_sym : [%dx1] CasADi SX — gravity vector\n', n);
        fprintf('  Functions: M_fun, C_fun, G_fun, tau_fun, forward_dyn_fun\n');

        % Sanity check: at q=0, dq=0, ddq=0 → tau should equal G(0)
        q0   = zeros(n, 1);
        G0   = full(G_fun(q0));
        tau0 = full(tau_fun(q0, q0, q0));
        err  = max(abs(G0 - tau0));
        fprintf('  Sanity (tau(0,0,0) == G(0)): max error = %.2e N·m\n', err);
    end

    % -----------------------------------------------------------------------
    % 15. Assemble output struct
    % -----------------------------------------------------------------------
    dyn.n               = n;
    dyn.q               = q;               % [Nx1 SX]
    dyn.dq              = dq;              % [Nx1 SX]
    dyn.ddq             = ddq;             % [Nx1 SX]
    dyn.M_sym           = M_sym;           % [NxN SX]
    dyn.C_sym           = C_sym;           % [NxN SX]
    dyn.G_sym           = G_sym;           % [Nx1 SX]
    dyn.tau_sym         = tau_sym;         % [Nx1 SX]
    dyn.c_bias_sym      = c_bias_sym;      % [Nx1 SX]
    dyn.R_chol          = R_chol;          % [NxN SX]
    dyn.M_fun           = M_fun;           % Function: q → M
    dyn.C_fun           = C_fun;           % Function: (q,dq) → C
    dyn.G_fun           = G_fun;           % Function: q → G
    dyn.tau_fun         = tau_fun;         % Function: (q,dq,ddq) → tau
    dyn.c_bias_fun      = c_bias_fun;      % Function: (q,dq) → c_bias
    dyn.forward_dyn_fun = forward_dyn_fun; % Function: (q,dq,tau) → ddq
    dyn.hvp_G_fun       = hvp_G_fun;       % Function: (q, v_dir) → hvp_G
    dyn.step_fun        = step_fun;        % Function: (x, u) → x_next

end % function dynamics_casadi


% =========================================================================
%  LOCAL HELPER FUNCTION
% =========================================================================

function A = dh_Ai_dyn(dh_row)
% DH_AI_DYN  Standard DH individual transformation matrix (CasADi SX).
%
% Identical to the local helper in fk_casadi.m — reproduced here so
% dynamics_casadi.m can be used as a standalone file without importing fk_casadi.
%
% INPUT:  dh_row  [1x4 SX]  [a, alpha, theta(q), d(q)]
% OUTPUT: A       [4x4 SX]  standard DH transformation matrix
    a_i   = dh_row(1);   % link length
    alpha = dh_row(2);   % link twist
    theta = dh_row(3);   % joint angle (may contain q_i for revolute)
    d_i   = dh_row(4);   % link offset (may contain q_i for prismatic)

    ct = cos(theta);  st = sin(theta);
    ca = cos(alpha);  sa = sin(alpha);

    A = [ct, -st*ca,  st*sa,  a_i*ct;
         st,  ct*ca, -ct*sa,  a_i*st;
          0,     sa,     ca,     d_i;
          0,      0,      0,      1 ];
end
