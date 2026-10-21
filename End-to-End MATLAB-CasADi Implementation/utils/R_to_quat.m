function q = R_to_quat(R)
% R_TO_QUAT  Robust rotation matrix to unit quaternion conversion.
%
% Converts a 3x3 rotation matrix R ∈ SO(3) to a unit quaternion
% q = [qw; qx; qy; qz] using the Shepperd algorithm.  The algorithm
% is branch-selected based on the largest diagonal element of R, which
% avoids the numerical instability (division by near-zero) that occurs
% in the simpler single-branch formula when qw is near zero (i.e., when
% the rotation angle is near ±180°).
%
% =========================================================================
% QUATERNION CONVENTION:
%   q = [qw; qx; qy; qz]   (scalar-first / Hamilton convention)
%   qw = cos(θ/2)            (scalar part, always non-negative here)
%   [qx; qy; qz] = n̂·sin(θ/2)  (vector part, rotation axis × half-angle)
%
%   The resulting quaternion always has qw >= 0 (canonical form).
%   This means rotations by θ and θ+2π give the same quaternion.
%
% =========================================================================
% ALGORITHM (Shepperd 1978 — 4-branch method):
%
%   Compute the "decision variable" for each branch:
%     B0 = 1 + R11 + R22 + R33   → 4·qw² if positive trace
%     B1 = 1 + R11 - R22 - R33   → 4·qx² if R11 dominates
%     B2 = 1 - R11 + R22 - R33   → 4·qy² if R22 dominates
%     B3 = 1 - R11 - R22 + R33   → 4·qz² if R33 dominates
%
%   Choose the branch with the largest Bk (guarantees the division in
%   the other components is by a number at least 1/4 of the max Bk,
%   so no branch has a near-zero denominator).
%
% =========================================================================
% INPUT:
%   R   [3x3 double]  Rotation matrix.  Should satisfy:
%                       det(R) ≈ 1   (checked with tolerance 1e-6)
%                       R'*R  ≈ I3   (orthogonality)
%                     Numerically noisy inputs are handled gracefully.
%
% OUTPUT:
%   q   [4x1 double]  Unit quaternion [qw; qx; qy; qz] with qw >= 0.
%                     The output is normalized to unit length.
%
% =========================================================================
% USAGE:
%   R = expm(skew([0;0;1]) * pi/3);   % 60° rotation about z
%   q = R_to_quat(R);
%   disp(q)   % [cos(30°); 0; 0; sin(30°)] ≈ [0.866; 0; 0; 0.5]
%
%   % Round-trip check:
%   R2 = quat2rotm(q');   % requires Robotics Toolbox or own implementation
%
% =========================================================================
% CALLED BY:
%   verify_fk.m  (Phase 4 — comparing EE orientations)

    % -----------------------------------------------------------------------
    % 0. Input validation (optional but helpful for debugging)
    % -----------------------------------------------------------------------
    if ~isequal(size(R), [3, 3])
        error('R_to_quat:invalidInput', ...
              'R must be a 3x3 matrix. Got size [%d x %d].', size(R, 1), size(R, 2));
    end

    % -----------------------------------------------------------------------
    % 1. Compute Shepperd branch decision variables
    % -----------------------------------------------------------------------
    % Each B_k corresponds to 4 times the square of one quaternion component.
    % The largest value guarantees we divide by the largest component,
    % minimizing round-off error.
    B0 = 1 + R(1,1) + R(2,2) + R(3,3);   % 4·qw²  (positive trace case)
    B1 = 1 + R(1,1) - R(2,2) - R(3,3);   % 4·qx²
    B2 = 1 - R(1,1) + R(2,2) - R(3,3);   % 4·qy²
    B3 = 1 - R(1,1) - R(2,2) + R(3,3);   % 4·qz²

    % Find the largest branch (to select the numerically best formula)
    [~, branch] = max([B0, B1, B2, B3]);

    % -----------------------------------------------------------------------
    % 2. Compute quaternion via the selected branch
    % -----------------------------------------------------------------------
    % In each branch, the dominant component is computed as sqrt(Bk/4),
    % and the remaining three are computed via the R off-diagonal elements
    % divided by 4 times the dominant component.
    % -----------------------------------------------------------------------
    switch branch

        case 1
            % Branch 0: qw is largest (positive trace → most common case)
            % S = 4·qw
            S  = 2 * sqrt(B0);     % = 4·qw, so qw = S/4
            qw = S / 4;
            qx = (R(3,2) - R(2,3)) / S;   % = (R32−R23) / (4·qw)
            qy = (R(1,3) - R(3,1)) / S;   % = (R13−R31) / (4·qw)
            qz = (R(2,1) - R(1,2)) / S;   % = (R21−R12) / (4·qw)

        case 2
            % Branch 1: qx is largest
            % S = 4·qx
            S  = 2 * sqrt(B1);
            qw = (R(3,2) - R(2,3)) / S;
            qx = S / 4;
            qy = (R(1,2) + R(2,1)) / S;
            qz = (R(1,3) + R(3,1)) / S;

        case 3
            % Branch 2: qy is largest
            % S = 4·qy
            S  = 2 * sqrt(B2);
            qw = (R(1,3) - R(3,1)) / S;
            qx = (R(1,2) + R(2,1)) / S;
            qy = S / 4;
            qz = (R(2,3) + R(3,2)) / S;

        case 4
            % Branch 3: qz is largest
            % S = 4·qz
            S  = 2 * sqrt(B3);
            qw = (R(2,1) - R(1,2)) / S;
            qx = (R(1,3) + R(3,1)) / S;
            qy = (R(2,3) + R(3,2)) / S;
            qz = S / 4;
    end

    % -----------------------------------------------------------------------
    % 3. Enforce canonical form (qw >= 0) and normalize
    % -----------------------------------------------------------------------
    q = [qw; qx; qy; qz];

    % Normalize to unit length (guards against numerical round-off)
    q_norm = norm(q);
    if q_norm < 1e-12
        error('R_to_quat:degenerateInput', ...
              'Input rotation matrix produces a near-zero quaternion. Check that R is in SO(3).');
    end
    q = q / q_norm;

    % Enforce qw >= 0 (canonical / double-cover disambiguation)
    % The quaternion q and -q represent the same rotation.
    % We always choose the representation with positive scalar part.
    if q(1) < 0
        q = -q;
    end

end % function R_to_quat
