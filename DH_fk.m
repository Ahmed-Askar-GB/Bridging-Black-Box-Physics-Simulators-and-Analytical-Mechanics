function [pose, T0e, Ro] = DH_fk(DHpram, q, joint_types, P_point)
% DH_FK Computes serial robot forward kinematics from Standard-DH parameters
% and returns a comprehensive RobotPose object supporting multiple orientation
% parameterizations (Rotation matrix, ZYZ Euler angles, Roll-Pitch-Yaw, Quaternions).
%
% transformations, applies tool point offsets, and packages the complete pose.
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
%   - joint_types: (Optional) Vector of length n with joint types (1 = revolute, 0 = prismatic).
%                  Defaults to all revolute.
%   - P_point: (Optional) 3x1 offset vector of the interest point in terminal frame n.
%              Defaults to [0; 0; 0].
%
% Outputs:
%   - pose: Instance of RobotPose class with properties:
%           * position: [x; y; z] in meters
%           * R: 3 x 3 rotation matrix in SO(3)
%           * rpy: [roll; pitch; yaw] in radians
%           * zyz: [phi; theta; psi] Euler angles in radians
%           * quaternion: [qw; qx; qy; qz] (scalar-first)
%           * quaternion_xyzw: [qx; qy; qz; qw] (ROS format)
%           * T: 4 x 4 homogeneous transformation matrix in SE(3)
%           * T_all: 4 x 4 x n intermediate link transforms
%           * is_gimbal_lock: boolean singularity indicator
%   - T0e: 4 x 4 transformation matrix from base to the interest point.
%   - Ro: Diagnostic structure with per-link matrices and joint axes.
%
% Example:
%   DH = [0, pi/2, 0, 0.089159; -0.425, 0, 0, 0];
%   q = [0.1; -0.5];
%   pose = DH_fk(DH, q);
%   disp(pose);

    n = length(q);
    q = q(:);

    % Parse joint types
    if size(DHpram, 2) >= 5
        types = DHpram(:, 5);
    elseif nargin >= 3 && ~isempty(joint_types)
        types = joint_types(:);
    else
        types = ones(n, 1);
    end

    % Parse interest point
    if nargin < 4 || isempty(P_point)
        P_point = zeros(3, 1);
    else
        P_point = P_point(:);
    end

    is_symbolic = isa(q, 'sym') || isa(DHpram, 'sym') || isa(P_point, 'sym');

    % Step 1: Forward Kinematic Recursion
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
            theta_i = theta_nom + q(i);
            d_i = d_nom;
        else
            theta_i = theta_nom;
            d_i = d_nom + q(i);
        end

        c_th = cos(theta_i);
        s_th = sin(theta_i);
        c_al = cos(alphai);
        s_al = sin(alphai);

        Ai = [c_th,  -s_th * c_al,   s_th * s_al,  ai * c_th;
             s_th,   c_th * c_al,  -c_th * s_al,  ai * s_th;
                0,          s_al,          c_al,         d_i;
                0,             0,             0,           1];

        Ro.L(i).A = Ai;

        if i == 1
            T0i(:, :, 1) = Ai;
        else
            T0i(:, :, i) = T0i(:, :, i-1) * Ai;
        end
        Ro.L(i).T0i = T0i(:, :, i);
    end

    % Step 2: Final End-Effector Localization with Interest Point
    T_end = T0i(:, :, n);
    p_e = T_end(1:3, 4) + T_end(1:3, 1:3) * P_point;
    
    T0e = T_end;
    T0e(1:3, 4) = p_e;
    Ro.T0e = T0e;

    % Step 3: Instantiate RobotPose Class
    pose = RobotPose(T0e, T0i);
end
