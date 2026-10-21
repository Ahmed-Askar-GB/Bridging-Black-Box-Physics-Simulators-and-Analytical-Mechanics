function jac = jacobian_casadi(fk, dh, chain, opts)
% JACOBIAN_CASADI  Geometric and analytical Jacobians via CasADi auto-diff.
%
% Computes the full 6xN geometric Jacobian J(q) mapping joint velocities
% to end-effector spatial velocities, plus the time-derivative J̇(q, q̇)
% needed for EE acceleration computation and operational-space control.
%
%   [ v_e ]   [ Jv(q) ]             [ v̇_e ]   [ J̇v(q,q̇) ]        [ Jv(q) ]
%   [ ω_e ] = [ Jw(q) ] · q̇  ,     [ ω̇_e ] = [ J̇w(q,q̇) ] · q̇ + [ Jw(q) ] · q̈
%
% COMPUTATION STRATEGY FOR J̇:
%   CasADi algorithmic auto-differentiation (exact & symbolic):
%     Reshape J to a vector [6N x 1], differentiate w.r.t. q ([6N x N] result),
%     then contract with q̇:  J̇_vec = dJ_dq * q̇,  reshape back to [6 x N].
%     This is exact (no finite differences or truncation error) and preserves
%     symbolic graph structure for code generation and higher derivatives.
%
% =========================================================================
% GEOMETRIC JACOBIAN CONSTRUCTION:
%
%   For joint j (base-frame axis z_{j-1}, origin O_{j-1}):
%     Revolute:  Jv(:,j) = z_{j-1} × (p_e − O_{j-1})   (cross product)
%                Jw(:,j) = z_{j-1}
%     Prismatic: Jv(:,j) = z_{j-1}
%                Jw(:,j) = 0
%
%   Where:
%     z_{j-1} = T_0^{j-1}(1:3,3)     (3rd column = z-axis of frame {j-1})
%     O_{j-1} = T_0^{j-1}(1:3,4)     (4th column = origin of frame {j-1})
%     p_e     = T_0^EE(1:3,4)         (EE origin, from fk.T0e)
%
%   The Jacobian is evaluated at the EE position (NOT at CoM positions).
%   For dynamics, see dynamics_casadi.m which builds per-link CoM Jacobians.
%
% =========================================================================
% INPUTS:
%   fk     (struct)   Output of fk_casadi(). Required fields:
%                       .q        [Nx1 SX]  symbolic joint vector
%                       .T0i      {Nx1}     per-link cumulative transforms
%                       .T0e      [4x4 SX]  EE transform
%                       .p_ee     [3x1 SX]  EE position
%   dh     (struct)   Output of urdf_extract_dh(). Required fields:
%                       .sigma    [Nx1]     1=revolute, 2=prismatic
%   chain  (struct)   Output of urdf_build_chain(). Required fields:
%                       .n        (int)     Number of DOF
%   opts   (struct)   [optional]
%                       .verbose  (logical)  Print summary (default false)
%
% =========================================================================
% OUTPUT:
%   jac   (struct) with fields:
%
%   Symbolic expressions (CasADi SX):
%     .q         [Nx1 SX]    Symbolic joint coordinates (shared with fk.q)
%     .dq        [Nx1 SX]    Symbolic joint velocities
%     .J_sym     [6xN SX]    Geometric Jacobian expression [Jv; Jw]
%     .Jv_sym    [3xN SX]    Linear velocity Jacobian
%     .Jw_sym    [3xN SX]    Angular velocity Jacobian
%     .Jdot_sym  [6xN SX]    Time-derivative J̇(q, q̇) expression
%
%   Compiled CasADi Function objects:
%     .J_fun      Function: q(Nx1) → J(6xN)
%                 Maps joint positions to geometric Jacobian.
%     .Jv_fun     Function: q(Nx1) → Jv(3xN)
%     .Jw_fun     Function: q(Nx1) → Jw(3xN)
%     .Jdot_fun   Function: (q(Nx1), dq(Nx1)) → Jdot(6xN)
%                 Maps joint positions+velocities to Jacobian time-derivative.
%
% =========================================================================
% USAGE:
%   robot = urdf_parse('universalUR5.urdf');
%   chain = urdf_build_chain(robot);
%   dh    = urdf_extract_dh(chain);
%   fk    = fk_casadi(dh, chain);
%   jac   = jacobian_casadi(fk, dh, chain);
%
%   % Evaluate Jacobian at a numeric configuration:
%   q_num = [0.1; -0.5; 0.3; -0.2; 0.4; 0.1];
%   J_num = full(jac.J_fun(q_num));
%
%   % Evaluate J-dot at a numeric state:
%   dq_num = [0.1; 0.2; -0.1; 0.0; 0.3; -0.2];
%   Jdot_num = full(jac.Jdot_fun(q_num, dq_num));
%
% =========================================================================
% CALLED BY:
%   dynamics_casadi.m  (Phase 2 — uses per-CoM Jacobians built similarly)
%   verify_jacobian.m  (Phase 4)
%   casadi_compile.m   (Phase 2)

    % -----------------------------------------------------------------------
    % 0. Parse optional arguments and import CasADi
    % -----------------------------------------------------------------------
    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    verbose = isfield(opts, 'verbose') && opts.verbose;

    assert(isstruct(fk) && isfield(fk, 'q') && isfield(fk, 'T0i') && isfield(fk, 'p_ee'), ...
           'jacobian_casadi:invalidFK', 'fk struct missing required fields (.q, .T0i, .p_ee).');
    assert(isstruct(dh) && isfield(dh, 'sigma'), ...
           'jacobian_casadi:invalidDH', 'dh struct missing required field .sigma.');
    assert(isstruct(chain) && isfield(chain, 'n'), ...
           'jacobian_casadi:invalidChain', 'chain struct missing required field .n.');

    import casadi.*

    n   = chain.n;         % number of DOF
    q   = fk.q;            % [Nx1 SX] symbolic joint vector (reuse from fk)
    dq  = SX.sym('dq', n, 1);   % [Nx1 SX] symbolic joint velocity vector

    % -----------------------------------------------------------------------
    % 1. Extract per-frame axes and origins from FK transforms
    % -----------------------------------------------------------------------
    % z_{j-1} is the joint j actuation axis expressed in the BASE frame.
    %   z_0 = [0;0;1] (base frame z-axis — world frame z)
    %   z_{j-1} = T_0^{j-1}(1:3, 3)  for j >= 2
    %
    % O_{j-1} is the origin of frame {j-1} expressed in the base frame.
    %   O_0 = [0;0;0]  (base frame origin)
    %   O_{j-1} = T_0^{j-1}(1:3, 4)  for j >= 2
    % -----------------------------------------------------------------------
    z = SX.zeros(3, n);   % [3xN] joint axes in base frame
    O = SX.zeros(3, n);   % [3xN] joint origins in base frame

    % Base frame seed: z_0 = world z-axis, O_0 = world origin
    z(:, 1) = [0; 0; 1];
    O(:, 1) = [0; 0; 0];

    for j = 2:n
        % z_{j-1} = 3rd column of T_0^{j-1}  (z-axis of the previous frame)
        z(:, j) = fk.T0i{j-1}(1:3, 3);
        % O_{j-1} = 4th column of T_0^{j-1}  (origin of the previous frame)
        O(:, j) = fk.T0i{j-1}(1:3, 4);
    end

    % EE position in base frame (from the FK symbolic expression)
    p_e = fk.p_ee;   % [3x1 SX]

    % -----------------------------------------------------------------------
    % 2. Assemble the geometric Jacobian column by column
    % -----------------------------------------------------------------------
    % For each joint j, the Jacobian column depends on the joint type:
    %
    %   Revolute  (sigma=1):
    %     Jv(:,j) = z_{j-1} × (p_e − O_{j-1})   [cross product]
    %     Jw(:,j) = z_{j-1}
    %
    %   Prismatic (sigma=2):
    %     Jv(:,j) = z_{j-1}                       [translation direction]
    %     Jw(:,j) = 0                              [no angular contribution]
    %
    % Note: All joints j contribute to the EE Jacobian because we are
    % computing the Jacobian at the EE (tip of the chain), not at an
    % intermediate link.  This means ALL N columns are non-zero.
    % -----------------------------------------------------------------------
    Jv = SX.zeros(3, n);   % [3xN] linear velocity Jacobian
    Jw = SX.zeros(3, n);   % [3xN] angular velocity Jacobian

    for j = 1:n
        % Moment arm: vector from joint j-1 origin to EE position
        r_j = p_e - O(:, j);   % [3x1 SX]

        if dh.sigma(j) == 1
            % --- Revolute joint contribution ---
            % Linear: z_{j-1} × r_j  (cross product gives tangential velocity)
            Jv(:, j) = cross(z(:, j), r_j);
            % Angular: z_{j-1} directly (rotation axis contributes to ω_e)
            Jw(:, j) = z(:, j);
        else
            % --- Prismatic joint contribution ---
            % Linear: z_{j-1} (pure translation along joint axis)
            Jv(:, j) = z(:, j);
            % Angular: zero (prismatic joints don't contribute to angular velocity)
            Jw(:, j) = SX.zeros(3, 1);
        end
    end

    % Stack into the full 6xN geometric Jacobian [linear; angular]
    J_sym = [Jv; Jw];   % [6xN SX]

    % -----------------------------------------------------------------------
    % 3. Compute J̇(q, q̇) via CasADi matrix-valued auto-differentiation
    % -----------------------------------------------------------------------
    % J̇ is the time derivative of J(q) along the trajectory q(t), dq(t).
    %
    % Using the chain rule:
    %   dJ/dt = (∂J/∂q) · dq/dt = sum_k [ (∂J/∂q_k) · q̇_k ]
    %
    % CasADi approach:
    %   1. Reshape J to a vector j_vec [6N x 1] (column-major stack)
    %   2. Compute dJ_dq = jacobian(j_vec, q)  → [6N x N] matrix
    %      where dJ_dq[i,k] = d(j_vec[i])/d(q[k])
    %   3. Contract with q̇:
    %      Jdot_vec = dJ_dq * dq  → [6N x 1]
    %   4. Reshape back to [6 x N]
    %
    % This is exact (no finite differences, no approximations) and produces
    % a CasADi expression that can be compiled and differentiated further.
    % -----------------------------------------------------------------------

    % Reshape J [6xN] to a column vector [6N x 1] (MATLAB column-major order)
    j_vec = reshape(J_sym, 6*n, 1);   % [6N x 1 SX]

    % Differentiate j_vec w.r.t. the joint vector q
    % Result: dJ_dq is [6N x N], where dJ_dq(i,k) = d(j_vec(i))/d(q(k))
    dJ_dq = jacobian(j_vec, q);       % [6N x N SX]

    % Multiply by q̇ to get the time-derivative of J in vectorized form
    Jdot_vec = dJ_dq * dq;            % [6N x 1 SX]

    % Reshape back to [6 x N]
    Jdot_sym = reshape(Jdot_vec, 6, n);   % [6xN SX] = J̇(q, q̇)

    % -----------------------------------------------------------------------
    % 4. Compile CasADi Function objects
    % -----------------------------------------------------------------------
    % These are pre-compiled graphs for fast numeric evaluation.
    % -----------------------------------------------------------------------

    % Geometric Jacobian: q → J(q)  [6xN]
    J_fun = Function('J_fun', ...
                     {q}, ...
                     {J_sym}, ...
                     {'q'}, ...
                     {'J'});

    % Linear velocity Jacobian: q → Jv(q)  [3xN]
    Jv_fun = Function('Jv_fun', ...
                      {q}, ...
                      {Jv}, ...
                      {'q'}, ...
                      {'Jv'});

    % Angular velocity Jacobian: q → Jw(q)  [3xN]
    Jw_fun = Function('Jw_fun', ...
                      {q}, ...
                      {Jw}, ...
                      {'q'}, ...
                      {'Jw'});

    % Jacobian time-derivative: (q, q̇) → J̇(q, q̇)  [6xN]
    Jdot_fun = Function('Jdot_fun', ...
                        {q, dq}, ...
                        {Jdot_sym}, ...
                        {'q', 'dq'}, ...
                        {'Jdot'});

    % -----------------------------------------------------------------------
    % 5. Verbose summary
    % -----------------------------------------------------------------------
    if verbose
        fprintf('\n[jacobian_casadi] Jacobian compiled for %d-DOF robot.\n', n);
        fprintf('  J_sym    : [6 x %d] CasADi SX — geometric Jacobian\n', n);
        fprintf('  Jdot_sym : [6 x %d] CasADi SX — Jacobian time-derivative\n', n);
        fprintf('  CasADi Functions: J_fun, Jv_fun, Jw_fun, Jdot_fun\n');

        % Quick numeric check at q=0: print Jacobian condition number
        q_zero = zeros(n, 1);
        J_zero = full(J_fun(q_zero));
        fprintf('  Jacobian at q=0: cond(J) = %.4e\n', cond(J_zero));
    end

    % -----------------------------------------------------------------------
    % 6. Assemble output struct
    % -----------------------------------------------------------------------
    jac.q        = q;           % [Nx1 SX]  (same symbolic as fk.q)
    jac.dq       = dq;          % [Nx1 SX]  symbolic joint velocities
    jac.J_sym    = J_sym;       % [6xN SX]  full geometric Jacobian
    jac.Jv_sym   = Jv;          % [3xN SX]  linear velocity Jacobian
    jac.Jw_sym   = Jw;          % [3xN SX]  angular velocity Jacobian
    jac.Jdot_sym = Jdot_sym;    % [6xN SX]  J̇(q, q̇)
    jac.J_fun    = J_fun;       % Function: q → J  [6xN]
    jac.Jv_fun   = Jv_fun;      % Function: q → Jv [3xN]
    jac.Jw_fun   = Jw_fun;      % Function: q → Jw [3xN]
    jac.Jdot_fun = Jdot_fun;    % Function: (q,dq) → Jdot [6xN]

end % function jacobian_casadi
