function dh = urdf_extract_dh(chain, verbose)
% URDF_EXTRACT_DH  Extract Standard Denavit-Hartenberg parameters from the chain.
%
% This is a pure functional refactor of the P-Q algorithm embedded in:
%   GenerateDhParams.calculate_dh_and_dynamics()   (My Implementation)
%   GenerateDhParams.calculate_dh_params()          (Report version)
%
% The P-Q algorithm (Hartenberg & Denavit, 1955) extracts standard DH
% parameters by finding the common perpendicular (shortest connector)
% between consecutive joint axes.  No prior knowledge of the robot's
% geometry is required beyond the world-frame joint axis vectors.
%
% =========================================================================
% STANDARD DH CONVENTION REMINDER:
%   Frame {i} is attached to link i.  The transform from frame {i-1} to {i}:
%
%     A_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
%
%   where:
%     theta_i     — joint angle (rotation about z_{i-1});   VARIABLE for revolute
%     d_i         — link offset (translation along z_{i-1}); VARIABLE for prismatic
%     a_i         — link length (translation along x_i);     always constant
%     alpha_i     — link twist  (rotation about x_i);        always constant
%
% =========================================================================
% INPUTS:
%   chain    (struct)   Output of urdf_build_chain(). Must contain:
%                         .n               (int)     Number of DOF
%                         .active_joints   (1xN cell) Each with .type and .T_effective
%                         .O               (3xN)     Joint origins at q=0
%                         .Z               (3xN)     Joint unit axes at q=0
%                         .T_root_to_j1    (4x4)
%                         .d_ee, .a_ee     (double)  EE offsets
%   verbose  (logical)  [optional, default false]
%
% OUTPUT:
%   dh      (struct) with fields:
%     .table        [Nx4]   Standard DH table:
%                           column 1: theta_offset (rad)  — zero-config offset on theta_i
%                           column 2: d             (m)   — link offset along z_{i-1}
%                           column 3: a             (m)   — link length along x_i
%                           column 4: alpha         (rad) — twist about x_i
%                           For revolute joints: theta_i = theta_offset + q_i
%                           For prismatic joints: d_i    = d + q_i
%     .DHpram       [Nx5]   DH table reformatted for DH_dyn.m / DH_dyn_casadi.m:
%                           column 1: a       (m)
%                           column 2: alpha   (rad)
%                           column 3: theta   (rad, = theta_offset)
%                           column 4: d       (m)
%                           column 5: sigma   (1 = revolute, 2 = prismatic)
%     .joint_types  {Nx1}   Cell array: 'revolute' or 'prismatic' per joint
%     .sigma        [Nx1]   Numeric: 1 = revolute, 2 = prismatic
%     .T_dh         {Nx1}   Per-link cumulative standard DH transforms at q=0:
%                           T_dh{i} = A_1(0) * A_2(0) * ... * A_i(0)
%                           Shape: [4x4]
%
% =========================================================================
% ALGORITHM (P-Q common-normal method):
%
%   For each pair of consecutive joint axes (z_{i-1}, z_i):
%
%   CASE A — Parallel or anti-parallel axes (|z_{i-1} · z_i| ≈ 1):
%     d_i = 0 (convention)
%     Choose x_i from the vector between the axis origins, projected onto
%     the plane perpendicular to z_i.
%
%   CASE B — Skew or intersecting axes:
%     Find the foot-points P_i (on z_{i-1}) and Q_i (on z_i) of the common
%     perpendicular using the parametric line formula:
%       t1 = (δO · z_{i-1} − c · δO · z_i) / (1 − c²)
%       t2 = (c · δO · z_{i-1} − δO · z_i) / (1 − c²)
%     where c = z_{i-1} · z_i, δO = O_i − O_{i-1}
%     d_i = (P_i − Q_{i-1}) · z_{i-1}
%     Sub-case B1 (skew):       x_i+1 = (Q_i − P_i) / |Q_i − P_i|
%     Sub-case B2 (intersect):  x_i+1 = (z_{i-1} × z_i) / |z_{i-1} × z_i|
%
%   After computing Q_i and x_{i+1}:
%     a_i     = |Q_i − P_i|
%     alpha_i = atan2((z_{i-1} × z_i) · x_{i+1},  z_{i-1} · z_i)
%     theta_i_offset = atan2((x_{i-1} × x_i) · z_{i-1},  x_{i-1} · x_i)
%
% =========================================================================
% USAGE:
%   robot = urdf_parse('universalUR5.urdf');
%   chain = urdf_build_chain(robot);
%   dh    = urdf_extract_dh(chain);
%   dh    = urdf_extract_dh(chain, true);   % verbose
%
% =========================================================================
% CALLED BY:
%   urdf_extract_inertia.m (Phase 1, Step 4)
%   run_pipeline.m         (Master orchestrator)

    % -----------------------------------------------------------------------
    % 0. Default arguments
    % -----------------------------------------------------------------------
    if nargin < 2
        verbose = false;
    end

    assert(isstruct(chain) && isfield(chain, 'n'), 'urdf_extract_dh:invalidChain', ...
           'Input chain must be a struct with field .n output from urdf_build_chain.');
    assert(chain.n > 0, 'urdf_extract_dh:zeroDOF', ...
           'Chain has 0 DOF. At least 1 active joint is required for DH extraction.');

    n = chain.n;   % number of active DOF

    % -----------------------------------------------------------------------
    % 1. Build the augmented world-frame axis/origin arrays
    % -----------------------------------------------------------------------
    % The P-Q algorithm operates on N+1 axis-origin pairs:
    %   Indices 1..N  correspond to the N active joints
    %   Index  N+1    is the virtual EE node (same axis as joint N,
    %                  origin at the EE tip in world coords)
    %
    % chain.O  [3xN] and chain.Z [3xN] are the world-frame origins and
    % axes of all active joints at q=0.  We append the EE node:
    % -----------------------------------------------------------------------

    % Compute world-frame EE origin:
    % T_urdf{n} is the global transform at the last active link.
    % Multiplying by chain.T_ee gives the EE tip in world frame.
    T_ee_world = chain.T_urdf{n} * chain.T_ee;
    O_ee       = T_ee_world(1:3, 4);   % EE origin position [3x1]

    % Append EE to the O and Z arrays (EE has the same axis as the last joint)
    O = [chain.O, O_ee];           % [3 x (N+1)]
    Z = [chain.Z, chain.Z(:, n)];  % [3 x (N+1)],  Z(:,N+1) = Z(:,N) by DH convention

    total_nodes = n + 1;  % N active joints + 1 virtual EE node

    % -----------------------------------------------------------------------
    % 2. Initialize P-Q algorithm state arrays
    % -----------------------------------------------------------------------
    % Q(:,i) — foot-point on axis i-1 (the DH frame {i-1} origin projected
    %          onto the common perpendicular)
    % X(:,i) — unit x-axis of DH frame {i} (direction of common perpendicular)
    %
    % Seed values for the base frame (frame 0):
    %   Q(:,1) = origin of the world frame = [0;0;0]
    %   X(:,1) = x-axis of the world frame = [1;0;0]
    % -----------------------------------------------------------------------
    Q         = zeros(3, total_nodes);  % foot-points on axes
    X         = zeros(3, total_nodes);  % DH x-axis directions
    Q(:, 1)   = [0; 0; 0];             % world frame origin
    X(:, 1)   = [1; 0; 0];             % world frame x-axis

    % Output arrays
    % Note on column conventions:
    %   DH_table : [theta_offset, d, a, alpha] — standard textbook order (θ, d, a, α)
    %   DHpram   : [a, alpha, theta, d, sigma] — legacy DH_dyn.m / DH_dyn_casadi.m format
    DH_table    = zeros(n, 4);    % [theta_offset, d, a, alpha] per row
    joint_types = cell(n, 1);     % 'revolute' or 'prismatic' per row

    % -----------------------------------------------------------------------
    % 3. Main P-Q loop — process each pair of consecutive joint axes
    % -----------------------------------------------------------------------
    for i = 1:n

        % --- Retrieve axis and origin vectors for this step -----------------
        prev_Z  = Z(:, i);     % z_{i-1}: axis of joint i   (world frame)
        curr_Z  = Z(:, i+1);   % z_i:     axis of joint i+1 (world frame, EE at last)
        prev_O  = O(:, i);     % O_{i-1}: origin of joint i
        curr_O  = O(:, i+1);   % O_i:     origin of joint i+1

        delta_O = curr_O - prev_O;    % vector between consecutive origins [3x1]
        c       = dot(prev_Z, curr_Z); % cosine of angle between consecutive axes

        % ===================================================================
        % CASE A — Parallel (or anti-parallel) consecutive axes
        % Condition: |c| ≈ 1  (axes are parallel within numerical tolerance)
        %
        % Convention: d_i = 0 for parallel axes (the last row is overridden
        % below with the physically correct d_ee / a_ee from the EE transform)
        % ===================================================================
        if abs(abs(c) - 1) < 1e-5

            d_val = 0;
            P_i   = Q(:, i);   % foot-point stays at the current Q node

            % Project curr_O onto z_i to find the next foot-point Q(:,i+1)
            % (foot-point of curr_O on the next axis line)
            Q(:, i+1) = curr_O + dot(P_i - curr_O, curr_Z) * curr_Z;

            % Choose x_{i+1} as the component of delta_O perpendicular to z_i
            % (i.e., the cross-section direction in the plane perp. to the axis)
            X_dir = delta_O - dot(delta_O, curr_Z) * curr_Z;
            if norm(X_dir) > 1e-5
                X(:, i+1) = X_dir / norm(X_dir);   % normalize
            else
                % Collinear axes with no offset — inherit previous x-axis
                X(:, i+1) = X(:, i);
            end

        % ===================================================================
        % CASE B — Skew or intersecting consecutive axes
        % Condition: |c| is NOT ≈ 1
        %
        % Find the common perpendicular between the two axis lines using
        % the parametric formula for the closest-point pair on two skew lines.
        % ===================================================================
        else

            % Denominator of the line-distance formula
            denom = 1 - c^2;   % = sin²(angle) > 0 for non-parallel axes

            % Parameters t1 and t2 locate the foot-points:
            %   P_i = prev_O + t1 * prev_Z   (foot on line through O_{i-1} along z_{i-1})
            %   Q_i = curr_O + t2 * curr_Z   (foot on line through O_i   along z_i)
            t1 = (dot(delta_O, prev_Z) - c * dot(delta_O, curr_Z)) / denom;
            t2 = (c * dot(delta_O, prev_Z) - dot(delta_O, curr_Z)) / denom;

            P_i        = prev_O + t1 * prev_Z;  % foot-point on previous axis
            Q(:, i+1)  = curr_O + t2 * curr_Z;  % foot-point on current axis

            % d_i = signed distance from Q_{i-1} to P_i along z_{i-1}
            d_val = dot(P_i - Q(:, i), prev_Z);

            % x_{i+1} = direction of the common perpendicular = (Q_i - P_i)
            n_vec = Q(:, i+1) - P_i;

            if norm(n_vec) > 1e-5
                % Sub-case B1 — SKEW axes: unique common normal direction
                X(:, i+1) = n_vec / norm(n_vec);

            else
                % Sub-case B2 — INTERSECTING axes: the common perpendicular
                % has zero length (axes meet at a point), so we use the
                % cross product to define x_{i+1} direction.
                X_dir = cross(prev_Z, curr_Z);
                X_dir = X_dir / norm(X_dir);

                % Enforce a consistent sign convention: flip if the cross product
                % points in the opposite direction to what we expect.
                % (This matches the sign choice in GenerateDhParams line 431)
                if dot(cross(prev_Z, curr_Z), X_dir) > 0
                    X_dir = -X_dir;
                end
                X(:, i+1) = X_dir;
            end

        end % end CASE A / CASE B

        % ===================================================================
        % Extract the four DH parameters from the geometric results
        % ===================================================================

        % d_i (link offset) — already computed above (CASE A: 0, CASE B: t1-based)
        DH_table(i, 2) = d_val;

        % a_i (link length) = distance between foot-points on adjacent axes
        %   a_i = |Q(:,i+1) - P_i|
        DH_table(i, 3) = norm(Q(:, i+1) - P_i);

        % alpha_i (link twist) = angle from z_{i-1} to z_i about x_i
        %   Computed via atan2 to get the correct signed angle.
        %   The triple product (z_{i-1} × z_i) · x_{i+1} gives the sign.
        alpha_val = atan2(dot(cross(prev_Z, curr_Z), X(:, i+1)), ...
                          dot(prev_Z, curr_Z));
        DH_table(i, 4) = alpha_val;

        % theta_i (joint angle offset) = angle from x_{i-1} to x_i about z_{i-1}
        %   This is the zero-configuration offset for revolute joints.
        %   For prismatic joints it is still the frame orientation offset.
        theta_val = atan2(dot(cross(X(:, i), X(:, i+1)), prev_Z), ...
                          dot(X(:, i), X(:, i+1)));

        % Wrap to [-pi, pi] for canonical representation
        theta_val = mod(theta_val + pi, 2*pi) - pi;

        % Round to zero if below numerical noise (avoids spurious tiny offsets)
        % Threshold 1e-4 rad ≈ 0.006° — safely below any physical DH offset
        if abs(theta_val) < 1e-4 || abs(abs(theta_val) - pi) < 1e-4
            theta_val = 0;
        end
        DH_table(i, 1) = theta_val;

        % Record joint type based on the URDF joint type string
        if strcmp(chain.active_joints{i}.type, 'prismatic')
            joint_types{i} = 'prismatic';  % d_i is the motion variable
        else
            joint_types{i} = 'revolute';   % theta_i is the motion variable
        end

    end % end main P-Q loop

    % -----------------------------------------------------------------------
    % 4. Override last-row d and a with physically correct EE values
    % -----------------------------------------------------------------------
    % The Case A logic sets d=0 for parallel axes (the last pair is usually
    % parallel for robots like the UR5 wrist).  The true d_ee and a_ee are
    % derived from the accumulated fixed-joint EE transform (chain.T_ee),
    % which accurately captures the physical distance to the EE tip.
    % theta and alpha for the last row are still correctly computed above.
    % -----------------------------------------------------------------------
    DH_table(end, 2) = chain.d_ee;  % override d for the last joint
    DH_table(end, 3) = chain.a_ee;  % override a for the last joint

    % -----------------------------------------------------------------------
    % 5. Build sigma vector and the DHpram matrix (for DH_dyn / DH_dyn_casadi)
    % -----------------------------------------------------------------------
    % DH_dyn.m and DH_dyn_casadi.m expect parameters in the order:
    %   [a, alpha, theta_offset, d, sigma]
    %   sigma = 1 for revolute  (theta is the variable column)
    %   sigma = 2 for prismatic (d is the variable column)
    %
    % NOTE: This differs from the raw DH_table column order [theta, d, a, alpha]!
    % The reordering is intentional to match the legacy DH_dyn.m interface.
    % -----------------------------------------------------------------------
    sigma   = ones(n, 1);   % default: all revolute
    DHpram  = zeros(n, 5);

    for i = 1:n
        if strcmp(joint_types{i}, 'prismatic')
            sigma(i) = 2;   % prismatic joint → d is variable
        end

        % Reorder columns from [theta, d, a, alpha] to [a, alpha, theta, d, sigma]
        DHpram(i, :) = [DH_table(i, 3), ...  % a
                        DH_table(i, 4), ...  % alpha
                        DH_table(i, 1), ...  % theta_offset
                        DH_table(i, 2), ...  % d
                        sigma(i)];           % sigma (1=rev, 2=pris)
    end

    % -----------------------------------------------------------------------
    % 6. Compute cumulative standard DH transforms at q = 0
    % -----------------------------------------------------------------------
    % T_dh{i} = A_1(q1=0) * A_2(q2=0) * ... * A_i(qi=0)
    %
    % These are the DH frame poses in world coordinates at zero configuration.
    % They are needed by urdf_extract_inertia() to transform CoM and inertia
    % tensors into the DH frame orientations.
    % -----------------------------------------------------------------------
    T_dh       = cell(n, 1);
    T_dh_accum = eye(4);

    for i = 1:n
        % Build the standard DH A_i matrix for joint i at q_i = 0.
        % At zero config: theta = theta_offset, d = d_nom (no variable added)
        A_i        = dh_Ai(DHpram(i, :));
        T_dh_accum = T_dh_accum * A_i;
        T_dh{i}    = T_dh_accum;
    end

    % -----------------------------------------------------------------------
    % 7. Verbose output
    % -----------------------------------------------------------------------
    if verbose
        fprintf('\n[urdf_extract_dh] === Standard DH Parameters ===\n');
        fprintf('  %-3s | %-10s | %-16s | %-16s | %-12s | %-12s\n', ...
                '#', 'Type', 'theta_off (rad)', 'd (m)', 'a (m)', 'alpha (rad)');
        fprintf('  %s\n', repmat('=', 1, 80));
        for i = 1:n
            mt = ' ';  md = ' ';
            if strcmp(joint_types{i}, 'revolute'),  mt = '*'; end
            if strcmp(joint_types{i}, 'prismatic'), md = '*'; end
            fprintf('  %-3d | %-10s | %s%15.6f | %s%15.6f | %12.6f | %12.6f\n', ...
                    i, joint_types{i}, ...
                    mt, DH_table(i, 1), ...   % theta_offset  (* = variable for revolute)
                    md, DH_table(i, 2), ...   % d             (* = variable for prismatic)
                    DH_table(i, 3), ...        % a
                    DH_table(i, 4));           % alpha
        end
        fprintf('  (* marks the joint-variable column)\n\n');
    end

    % -----------------------------------------------------------------------
    % 8. Assemble the output struct
    % -----------------------------------------------------------------------
    dh.table       = DH_table;    % [Nx4]: [theta_offset, d, a, alpha]
    dh.DHpram      = DHpram;      % [Nx5]: [a, alpha, theta, d, sigma]
    dh.joint_types = joint_types; % {Nx1}: 'revolute' | 'prismatic'
    dh.sigma       = sigma;       % [Nx1]: 1=revolute, 2=prismatic
    dh.T_dh        = T_dh;        % {Nx1}: cumulative DH transforms at q=0

end % function urdf_extract_dh


% =========================================================================
%  LOCAL HELPER FUNCTION
% =========================================================================

function A = dh_Ai(Dh_row)
% DH_AI  Compute the standard DH individual link transformation matrix.
%
% Standard DH:
%   A_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
%
%   Expanded form (4x4 homogeneous matrix):
%   A = [ cos(th)  -sin(th)*cos(al)   sin(th)*sin(al)   a*cos(th) ]
%       [ sin(th)   cos(th)*cos(al)  -cos(th)*sin(al)   a*sin(th) ]
%       [    0          sin(al)           cos(al)            d     ]
%       [    0            0                  0               1     ]
%
% INPUT:
%   Dh_row  [1x4] or [1x5]: [a, alpha, theta, d]  (or [a, alpha, theta, d, sigma])
%           This is the DHpram row format (NOT the DH_table format).
%
% OUTPUT:
%   A  [4x4]  Standard DH homogeneous transformation matrix

    ai     = Dh_row(1);  % link length (m)
    alphai = Dh_row(2);  % link twist  (rad)
    si     = Dh_row(3);  % joint angle / theta (rad) — zero-config offset
    di     = Dh_row(4);  % link offset / d    (m)

    % Precompute trig values for efficiency
    c_th = cos(si);  s_th = sin(si);
    c_al = cos(alphai); s_al = sin(alphai);

    A = [c_th, -s_th*c_al,  s_th*s_al,  ai*c_th;
         s_th,  c_th*c_al, -c_th*s_al,  ai*s_th;
            0,       s_al,       c_al,       di;
            0,          0,          0,        1];
end
