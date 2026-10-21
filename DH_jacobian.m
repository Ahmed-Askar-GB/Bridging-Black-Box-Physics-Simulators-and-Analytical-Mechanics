function [J, Jv, Jw, T0e, Ro] = DH_jacobian(DHpram, q, joint_types, P_point)
% DH_JACOBIAN Computes the 6xn geometric Jacobian matrix of a serial robot 
% from standard Denavit-Hartenberg (DH) parameters.
%
% Inspired by the kinematic propagation in DH_dyn.m, this function constructs
% the full geometric Jacobian mapping joint velocities to end-effector spatial
% velocities (linear and angular):
%
%       [ v_e ]   [ Jv ]
%       [ w_e ] = [ Jw ] * dq = J(q) * dq
%
% Inputs:
%   - DHpram: Matrix of standard DH parameters (n x 4 or n x 5):
%             Columns: [a, alpha, theta_offset, d] or [a, alpha, theta_offset, d, joint_type]
%             a [m]: link length along x_i
%             alpha [rad]: link twist about x_i
%             theta_offset [rad]: nominal joint angle offset about z_(i-1)
%             d [m]: link offset along z_(i-1)
%             joint_type: 1 for revolute, 0 for prismatic (optional 5th column)
%   - q: Generalized joint coordinate vector (n x 1 or 1 x n), numeric or symbolic.
%        For revolute joints, q_i adds to theta_offset.
%        For prismatic joints, q_i adds to d.
%   - joint_types: (Optional) Vector of length n with joint types (1 = revolute, 0 = prismatic).
%                  Ignored if DHpram has 5 columns. Defaults to all revolute (ones(n,1)).
%   - P_point: (Optional) 3x1 offset vector of the interest point expressed in the
%              terminal frame n (defaults to [0; 0; 0] for the end-effector origin).
%
% Outputs:
%   - J: Full 6 x n geometric Jacobian matrix [Jv; Jw]
%   - Jv: 3 x n linear velocity Jacobian matrix (v_e = Jv * dq)
%   - Jw: 3 x n angular velocity Jacobian matrix (w_e = Jw * dq)
%   - T0e: 4 x 4 homogeneous transformation from base frame to the interest point
%   - Ro: Diagnostic structure containing intermediate frame transformations,
%         actuation axes, and link positions.
%
% Example:
%   % 2-DOF planar revolute arm
%   syms q1 q2 real
%   DH = [0.5, 0, 0, 0; 0.5, 0, 0, 0];
%   [J, Jv, Jw] = DH_jacobian(DH, [q1; q2]);

    n = length(q);
    q = q(:); % Ensure column vector

    % Parse joint types
    if size(DHpram, 2) >= 5
        types = DHpram(:, 5);
    elseif nargin >= 3 && ~isempty(joint_types)
        types = joint_types(:);
    else
        types = ones(n, 1); % Default: all joints are revolute
    end

    % Default interest point offset
    if nargin < 4 || isempty(P_point)
        P_point = zeros(3, 1);
    else
        P_point = P_point(:);
    end

    % Check if inputs are symbolic
    is_symbolic = isa(q, 'sym') || isa(DHpram, 'sym') || isa(P_point, 'sym');

    % Step 1: Forward Kinematic Recursion
    % Compute per-link transformation matrices A_i(q_i)
    if is_symbolic
        T0i = sym(zeros(4, 4, n));
    else
        T0i = zeros(4, 4, n);
    end

    for i = 1:n
        ai = DHpram(i, 1);
        alphai = DHpram(i, 2);
        theta_nom = DHpram(i, 3);
        d_nom = DHpram(i, 4);

        if types(i) == 1
            % Revolute joint: coordinate adds to theta
            theta_i = theta_nom + q(i);
            d_i = d_nom;
        else
            % Prismatic joint: coordinate adds to d
            theta_i = theta_nom;
            d_i = d_nom + q(i);
        end

        Ai = compute_Ai(ai, alphai, theta_i, d_i);
        Ro.L(i).A = Ai;

        if i == 1
            T0i(:, :, 1) = Ai;
        else
            T0i(:, :, i) = T0i(:, :, i-1) * Ai;
        end
        Ro.L(i).T0i = T0i(:, :, i);
    end

    % Step 2: Actuation Axes and End-Effector Localization
    % z_(j-1) is the actuation axis of joint j expressed in the base frame
    if is_symbolic
        z = sym(zeros(3, n));
    else
        z = zeros(3, n);
    end
    z(:, 1) = [0; 0; 1]; % Base frame z-axis (z_0)
    for j = 2:n
        z(:, j) = T0i(1:3, 3, j-1); % z_(j-1) extracted from column 3 of T_0^(j-1)
    end
    Ro.z = z;

    % Target end-effector point in base frame
    T_end = T0i(:, :, n);
    p_e = T_end(1:3, 4) + T_end(1:3, 1:3) * P_point;
    Ro.p_e = p_e;

    % Complete transformation to the interest point
    T0e = T_end;
    T0e(1:3, 4) = p_e;

    % Step 3: Column-wise Geometric Jacobian Assembly
    if is_symbolic
        Jv = sym(zeros(3, n));
        Jw = sym(zeros(3, n));
    else
        Jv = zeros(3, n);
        Jw = zeros(3, n);
    end

    for j = 1:n
        % Origin of frame (j-1) in base coordinates
        if j == 1
            p_prev = zeros(3, 1);
        else
            p_prev = T0i(1:3, 4, j-1);
        end

        % Moment arm vector from joint origin (j-1) to the end-effector point
        r_j = p_e - p_prev;

        if types(j) == 1
            % Revolute Joint contribution
            Jv(:, j) = cross(z(:, j), r_j);
            Jw(:, j) = z(:, j);
        else
            % Prismatic Joint contribution
            Jv(:, j) = z(:, j);
            if is_symbolic
                Jw(:, j) = sym(zeros(3, 1));
            else
                Jw(:, j) = zeros(3, 1);
            end
        end
    end

    % Stack linear and angular Jacobians
    J = [Jv; Jw];
    Ro.Jv = Jv;
    Ro.Jw = Jw;
    Ro.J = J;
end

% Helper function: Standard DH homogeneous transformation matrix
function A = compute_Ai(ai, alphai, thetai, di)
    c_th = cos(thetai);
    s_th = sin(thetai);
    c_al = cos(alphai);
    s_al = sin(alphai);

    A = [c_th,  -s_th * c_al,   s_th * s_al,  ai * c_th;
         s_th,   c_th * c_al,  -c_th * s_al,  ai * s_th;
            0,          s_al,          c_al,         di;
            0,             0,             0,          1];
end
