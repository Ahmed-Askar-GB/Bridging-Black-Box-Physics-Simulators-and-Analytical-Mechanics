function fk = fk_casadi(dh, chain, opts)
% FK_CASADI  CasADi symbolic forward kinematics for all links and the EE.
%
% Constructs CasADi symbolic expressions for the full forward kinematics
% of a serial robot described by standard DH parameters.  Returns both
% the symbolic expressions (for further differentiation or composition)
% and compiled CasADi Function objects (for fast numeric evaluation).
%
% This function extends the numeric-only DH_fk.m to the CasADi symbolic
% domain, enabling:
%   - Automatic differentiation (for Jacobians, J-dot, etc.)
%   - C code generation for embedded deployment
%   - Fast repeated evaluation with pre-compiled Function objects
%
% =========================================================================
% STANDARD DH TRANSFORMATION (reminder):
%
%   A_i(q_i) = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
%
%   For revolute  joints: theta_i = theta_offset_i + q_i   (q_i is variable)
%   For prismatic joints: d_i     = d_offset_i     + q_i   (q_i is variable)
%
%   Cumulative FK:
%     T_0^i(q) = A_1(q_1) * A_2(q_2) * ... * A_i(q_i)
%
%   End-effector:
%     T_0^EE(q) = T_0^N(q) * T_ee    (T_ee is the fixed EE offset from chain)
%
% =========================================================================
% INPUTS:
%   dh    (struct)  Output of urdf_extract_dh(). Required fields:
%                     .DHpram      [Nx5]  [a, alpha, theta_offset, d, sigma]
%                     .joint_types {Nx1}  'revolute' | 'prismatic'
%                     .sigma       [Nx1]  1=revolute, 2=prismatic
%   chain (struct)  Output of urdf_build_chain(). Required fields:
%                     .n           (int)    Number of DOF
%                     .T_ee        [4x4]   Fixed EE offset transform
%   opts  (struct)  [optional] Configuration options:
%                     .verbose  (logical)  Print summary (default false)
%
% =========================================================================
% OUTPUT:
%   fk    (struct) with fields:
%
%   Symbolic expressions (CasADi SX):
%     .q          [Nx1 SX]  Symbolic joint coordinate vector
%     .T0i        {Nx1}     Per-link cumulative transforms:
%                            T0i{i} is 4x4 SX: T_0^i(q)
%     .T0e        [4x4 SX]  Full EE transform:  T_0^EE = T_0^N * T_ee
%     .p_ee       [3x1 SX]  EE position (m)
%     .R_ee       [3x3 SX]  EE rotation matrix
%     .quat_ee    [4x1 SX]  EE quaternion [qw; qx; qy; qz] (scalar-first)
%
%   Compiled CasADi Function objects:
%     .fk_fun        Function: q(Nx1) → {p_ee(3x1), R_ee(3x3), quat_ee(4x1)}
%                    Call as: [p, R, quat] = fk.fk_fun(q_numeric)
%     .fk_link_fun   {Nx1 cell} of Function: q(Nx1) → T0i{i}(4x4)
%                    Call as: T_i = fk.fk_link_fun{i}(q_numeric)
%     .fk_ee_fun     Function: q(Nx1) → T0e(4x4) full EE transform
%
% =========================================================================
% USAGE:
%   robot   = urdf_parse('universalUR5.urdf');
%   chain   = urdf_build_chain(robot);
%   dh      = urdf_extract_dh(chain);
%   fk      = fk_casadi(dh, chain);
%
%   % Numeric evaluation at a specific configuration:
%   q_test = [0.1; -0.5; 0.3; -0.2; 0.4; 0.1];
%   [p, R, quat] = fk.fk_fun(q_test);
%
% =========================================================================
% DEPENDENCIES:
%   CasADi (https://web.casadi.org/) must be installed and on the MATLAB path.
%   Import is done locally: import casadi.*
%
% =========================================================================
% CALLED BY:
%   jacobian_casadi.m    (Phase 2)
%   dynamics_casadi.m    (Phase 2, via fk for CoM kinematics)
%   casadi_compile.m     (Phase 2)
%   verify_fk.m          (Phase 4)

    % -----------------------------------------------------------------------
    % 0. Parse optional arguments and import CasADi
    % -----------------------------------------------------------------------
    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    verbose = isfield(opts, 'verbose') && opts.verbose;

    assert(isstruct(chain) && isfield(chain, 'n'), ...
           'fk_casadi:invalidChain', 'chain struct missing required field .n.');
    assert(isstruct(dh) && isfield(dh, 'DHpram') && isfield(dh, 'sigma'), ...
           'fk_casadi:invalidDH', 'dh struct missing required fields .DHpram or .sigma.');

    % CasADi must be available — import into the local scope
    import casadi.*

    n = chain.n;   % number of DOF

    % -----------------------------------------------------------------------
    % 1. Create symbolic joint variable vector q
    % -----------------------------------------------------------------------
    % q is an [Nx1] CasADi SX symbolic vector.
    % q(i) is added to theta (revolute) or d (prismatic) for joint i.
    q = SX.sym('q', n, 1);   % symbolic joint coordinates [rad or m]

    % -----------------------------------------------------------------------
    % 2. Build the symbolic DH table with q embedded
    % -----------------------------------------------------------------------
    % For each joint i, inject the symbolic variable q(i) into the
    % appropriate DH parameter column:
    %   revolute  (sigma=1): theta_i = theta_offset + q_i
    %   prismatic (sigma=2): d_i     = d_offset     + q_i
    %
    % Resulting DH_sym row i: [a_i, alpha_i, theta_i(q_i), d_i(q_i)]
    % (where theta_i or d_i is a CasADi expression, the other is a constant)
    % -----------------------------------------------------------------------
    DHp = dh.DHpram;    % [Nx5] numeric DH params: [a, alpha, theta_off, d, sigma]

    % Pre-allocate the symbolic DH table (N rows x 4 cols: [a, alpha, theta, d])
    DH_sym = SX.zeros(n, 4);

    for i = 1:n
        a_i     = DHp(i, 1);    % link length (constant)
        alpha_i = DHp(i, 2);    % link twist  (constant)
        th_off  = DHp(i, 3);    % theta offset (constant zero-config value)
        d_off   = DHp(i, 4);    % d offset     (constant zero-config value)
        sigma_i = DHp(i, 5);    % joint type: 1=revolute, 2=prismatic

        if sigma_i == 1
            % Revolute joint: q_i adds to theta
            theta_i = th_off + q(i);
            d_i     = d_off;
        else
            % Prismatic joint: q_i adds to d
            theta_i = th_off;
            d_i     = d_off + q(i);
        end

        DH_sym(i, 1) = a_i;
        DH_sym(i, 2) = alpha_i;
        DH_sym(i, 3) = theta_i;
        DH_sym(i, 4) = d_i;
    end

    % -----------------------------------------------------------------------
    % 3. Forward kinematic recursion — compute cumulative transforms T0i
    % -----------------------------------------------------------------------
    % T0i{i} = A_1(q1) * A_2(q2) * ... * A_i(qi)
    %
    % This is the standard "product of exponentials" DH recursion.
    % Each A_i is a 4x4 CasADi SX matrix.
    % -----------------------------------------------------------------------
    T0i       = cell(n, 1);   % cell array to store cumulative transforms
    if isfield(opts, 'T_base') && ~isempty(opts.T_base)
        T_running = SX(opts.T_base);
    else
        T_running = SX.eye(4);    % identity: frame 0 (base frame)
    end

    for i = 1:n
        % Compute A_i(q_i) — the local DH transform for joint i
        A_i       = dh_Ai_sym(DH_sym(i, :));

        % Accumulate: T_0^i = T_0^{i-1} * A_i
        T_running = T_running * A_i;
        T0i{i}    = T_running;    % store cumulative transform up to link i
    end

    % -----------------------------------------------------------------------
    % 4. Compute end-effector transform T0e
    % -----------------------------------------------------------------------
    % By construction in urdf_extract_dh.m, the N-th DH frame already
    % incorporates the end-effector translation (d_ee and a_ee from chain.T_ee).
    % Therefore, T0i{n} is already the terminal frame of the serial DH chain.
    % If an optional additional tool transform opts.T_tool is passed, apply it;
    % otherwise T_0^EE = T_0^N.
    % -----------------------------------------------------------------------
    if isfield(opts, 'T_tool') && ~isempty(opts.T_tool)
        T_tool_sx = SX(opts.T_tool);
        T0e = T0i{n} * T_tool_sx;
    else
        T0e = T0i{n};
    end

    % Extract EE position and orientation
    p_ee = T0e(1:3, 4);            % [3x1 SX] EE position
    R_ee = T0e(1:3, 1:3);          % [3x3 SX] EE rotation matrix

    % -----------------------------------------------------------------------
    % 5. Compute EE quaternion from rotation matrix
    % -----------------------------------------------------------------------
    % Uses the Shepperd algorithm (single positive-trace branch) adapted for
    % CasADi SX symbolic computation. A branching Shepperd is impossible here
    % because CasADi cannot auto-differentiate through MATLAB if/switch.
    %
    % Single-branch formula (valid when trace(R) > -1, i.e., |θ_rot| < 180°):
    %   tr = R11 + R22 + R33
    %   qw = 0.5 * sqrt(1 + tr)            ← scalar quaternion component
    %   qx = (R32 - R23) / (4*qw)
    %   qy = (R13 - R31) / (4*qw)
    %   qz = (R21 - R12) / (4*qw)
    %
    % *** SINGULARITY LIMITATION ***
    % When the EE rotation angle θ ≈ ±180°, qw ≈ 0 and qx,qy,qz diverge.
    % For UR5 this is safe because joint limits prevent θ = 180° at the EE.
    % For general robots or full 4π coverage, use R_to_quat.m (numeric, 4-branch).
    % -----------------------------------------------------------------------
    tr   = R_ee(1,1) + R_ee(2,2) + R_ee(3,3);    % matrix trace
    qw   = 0.5 * sqrt(1.0 + tr);                  % scalar part
    inv4qw = 1.0 / (4.0 * qw);                    % precompute 1/(4*qw)
    qx   = (R_ee(3,2) - R_ee(2,3)) * inv4qw;
    qy   = (R_ee(1,3) - R_ee(3,1)) * inv4qw;
    qz   = (R_ee(2,1) - R_ee(1,2)) * inv4qw;

    quat_ee = [qw; qx; qy; qz];    % [4x1 SX] quaternion (scalar-first)

    % -----------------------------------------------------------------------
    % 6. Compile CasADi Function objects
    % -----------------------------------------------------------------------
    % CasADi Function objects are JIT-compiled computational graphs.
    % Once compiled, they evaluate orders of magnitude faster than
    % re-evaluating the symbolic expression with subs().
    %
    % Naming convention:  Function(name, {inputs}, {outputs}, {in_names}, {out_names})
    % -----------------------------------------------------------------------

    % -- 6a. FK function: maps q → (p_ee, R_ee, quat_ee)
    fk_fun = Function('fk_fun', ...
                      {q}, ...
                      {p_ee, R_ee, quat_ee}, ...
                      {'q'}, ...
                      {'p_ee', 'R_ee', 'quat_ee'});

    % -- 6b. Full EE transform function: maps q → T0e (4x4)
    fk_ee_fun = Function('fk_ee_fun', ...
                         {q}, ...
                         {T0e}, ...
                         {'q'}, ...
                         {'T0e'});

    % -- 6c. Per-link FK functions: one per link, maps q → T0i{i}
    fk_link_fun = cell(n, 1);
    for i = 1:n
        fk_link_fun{i} = Function(sprintf('fk_link_%d', i), ...
                                  {q}, ...
                                  {T0i{i}}, ...
                                  {'q'}, ...
                                  {sprintf('T0_%d', i)});
    end

    % -----------------------------------------------------------------------
    % 7. Verbose summary
    % -----------------------------------------------------------------------
    if verbose
        fprintf('\n[fk_casadi] Forward kinematics compiled for %d-DOF robot.\n', n);
        fprintf('  Symbolic variables: q [%dx1]\n', n);
        fprintf('  EE position expression:   p_ee [3x1 SX]\n');
        fprintf('  EE rotation expression:   R_ee [3x3 SX]\n');
        fprintf('  EE quaternion expression: quat_ee [4x1 SX]\n');
        fprintf('  CasADi Functions compiled: fk_fun, fk_ee_fun, fk_link_fun{1..%d}\n', n);

        % Quick sanity check: evaluate at q=0 and print EE position
        q_zero   = zeros(n, 1);
        T0e_zero = full(fk_ee_fun(q_zero));   % returns DM [4x4]
        fprintf('  EE position at q=0: [%.4f, %.4f, %.4f] m\n', ...
                T0e_zero(1,4), T0e_zero(2,4), T0e_zero(3,4));
    end

    % -----------------------------------------------------------------------
    % 8. Assemble output struct
    % -----------------------------------------------------------------------
    fk.q           = q;            % [Nx1 SX]  symbolic joint coordinates
    fk.T0i         = T0i;          % {Nx1}     per-link cumulative transforms (SX)
    fk.T0e         = T0e;          % [4x4 SX]  full EE transform
    fk.p_ee        = p_ee;         % [3x1 SX]  EE position
    fk.R_ee        = R_ee;         % [3x3 SX]  EE rotation matrix
    fk.quat_ee     = quat_ee;      % [4x1 SX]  EE quaternion [qw;qx;qy;qz]
    fk.fk_fun      = fk_fun;       % Function: q → {p_ee, R_ee, quat_ee}
    fk.fk_ee_fun   = fk_ee_fun;    % Function: q → T0e
    fk.fk_link_fun = fk_link_fun;  % {Nx1}: Function q → T0i{i}

end % function fk_casadi


% =========================================================================
%  LOCAL HELPER FUNCTION
% =========================================================================

function A = dh_Ai_sym(dh_row)
% DH_AI_SYM  Standard DH local transformation matrix (CasADi SX version).
%
% Computes the 4x4 homogeneous transformation A_i from a single DH row:
%   A_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
%
% Expanded form:
%   A = [ cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ) ]
%       [ sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ) ]
%       [   0        sin(α)          cos(α)           d     ]
%       [   0          0               0              1     ]
%
% INPUT:
%   dh_row  [1x4 SX or numeric]  [a, alpha, theta, d]
%           (row of the DH_sym table where theta and/or d may be SX expressions)
%
% OUTPUT:
%   A       [4x4 SX]  Standard DH transformation matrix
    a_i    = dh_row(1);   % link length  (constant numeric)
    alpha  = dh_row(2);   % link twist   (constant numeric)
    theta  = dh_row(3);   % joint angle  (SX for revolute: theta_off + q_i)
    d_i    = dh_row(4);   % link offset  (SX for prismatic: d_off + q_i)

    % Trig functions applied to SX expressions → SX outputs
    ct = cos(theta);   st = sin(theta);
    ca = cos(alpha);   sa = sin(alpha);

    % Standard DH matrix — each element is a CasADi SX scalar expression
    A = [ct,  -st*ca,   st*sa,   a_i*ct;
         st,   ct*ca,  -ct*sa,   a_i*st;
          0,      sa,      ca,      d_i;
          0,       0,       0,       1 ];
end
