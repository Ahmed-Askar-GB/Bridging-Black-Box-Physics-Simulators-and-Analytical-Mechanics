function [q_t, dq_t, ddq_t] = traj_poly5(q0, q1, T_dur, t)
% TRAJ_POLY5  Generate a 5th-order minimum-jerk polynomial trajectory.
%
% Computes joint positions, velocities, and accelerations for a smooth
% point-to-point motion from q0 to q1 over duration T_dur.
%
% The 5th-order polynomial satisfies ZERO velocity AND ZERO acceleration
% at both the start (t=0) and end (t=T_dur), making it "minimum-jerk"
% and suitable for excitation trajectories without impulsive loads.
%
% =========================================================================
% POLYNOMIAL COEFFICIENTS (normalized time s = t/T):
%
%   s(τ) = 10τ³ − 15τ⁴ + 6τ⁵          where τ = t/T
%   ṡ(τ) = (30τ² − 60τ³ + 30τ⁴) / T
%   s̈(τ) = (60τ  − 180τ² + 120τ³) / T²
%
%   q(t)   = q0 + (q1 − q0) · s(t/T)
%   q̇(t)  = (q1 − q0) · ṡ(t/T)
%   q̈(t)  = (q1 − q0) · s̈(t/T)
%
% Boundary conditions verified:
%   s(0)=0, ṡ(0)=0, s̈(0)=0   (rest at start)
%   s(1)=1, ṡ(1)=0, s̈(1)=0   (rest at end)
%
% =========================================================================
% INPUTS:
%   q0     [Nx1 or 1x1]  Starting joint configuration (rad or m)
%                         If scalar, applied to all joints.
%   q1     [Nx1 or 1x1]  Ending joint configuration (rad or m)
%                         If scalar, applied to all joints.
%   T_dur  (double)       Total trajectory duration (s). Must be > 0.
%   t      [1xT or Tx1]  Time vector (s). All values should be in [0, T_dur].
%                         Values outside this range will be clamped to the
%                         boundary values (q0 or q1).
%
% =========================================================================
% OUTPUTS:
%   q_t    [NxT]  Joint position trajectory (rad or m)
%   dq_t   [NxT]  Joint velocity trajectory (rad/s or m/s)
%   ddq_t  [NxT]  Joint acceleration trajectory (rad/s² or m/s²)
%
%   where N = number of joints, T = length(t)
%
% =========================================================================
% USAGE:
%   % Single joint, 3-second trajectory
%   t = linspace(0, 3, 1000);
%   [q, dq, ddq] = traj_poly5(0, pi/4, 3, t);
%
%   % 6-DOF robot: from rest to a target pose
%   q0  = zeros(6,1);
%   q1  = [0.5; -0.3; 0.4; -0.2; 0.6; -0.1];
%   t   = linspace(0, 5, 5000);
%   [q, dq, ddq] = traj_poly5(q0, q1, 5, t);
%
%   % Check boundary conditions
%   assert(norm(q(:,1) - q0)   < 1e-14);
%   assert(norm(q(:,end) - q1) < 1e-14);
%   assert(norm(dq(:,1))       < 1e-14);
%   assert(norm(dq(:,end))     < 1e-14);
%
% =========================================================================
% CALLED BY:
%   dt_add_trajectory.m     (Phase 3 — Digital Twin excitation)
%   verify_dynamics_sequential.m (Phase 5 — per-link isolation test)

    % -----------------------------------------------------------------------
    % 0. Input validation and formatting
    % -----------------------------------------------------------------------
    if T_dur <= 0
        error('traj_poly5:invalidDuration', ...
              'T_dur must be positive. Got T_dur = %.4f', T_dur);
    end

    % Force column vectors
    q0 = q0(:);   % [Nx1]
    q1 = q1(:);   % [Nx1]
    t  = t(:)';   % [1xT] ensure row vector for broadcasting

    % Handle scalar inputs: replicate to match each other
    if isscalar(q0) && ~isscalar(q1)
        q0 = repmat(q0, length(q1), 1);
    elseif isscalar(q1) && ~isscalar(q0)
        q1 = repmat(q1, length(q0), 1);
    end

    if length(q0) ~= length(q1)
        error('traj_poly5:sizeMismatch', ...
              'q0 and q1 must have the same length. Got %d and %d.', ...
              length(q0), length(q1));
    end

    N = length(q0);   % number of joints
    T = length(t);    % number of time samples

    % -----------------------------------------------------------------------
    % 1. Compute normalized time τ = t/T_dur and clamp to [0, 1]
    % -----------------------------------------------------------------------
    tau = t / T_dur;           % [1xT] normalized time in [0, 1] (ideally)
    tau = max(0, min(1, tau)); % clamp: before t=0 stays at q0, after T at q1

    % -----------------------------------------------------------------------
    % 2. Evaluate the 5th-order polynomial and its derivatives
    % -----------------------------------------------------------------------
    % Shape function s(τ) and its derivatives (scalar, then broadcast to [NxT])

    % s(τ) = 10τ³ − 15τ⁴ + 6τ⁵  : normalized position profile ∈ [0, 1]
    s    = 10*tau.^3 - 15*tau.^4 + 6*tau.^5;            % [1xT]

    % ṡ(τ) = (30τ² − 60τ³ + 30τ⁴) / T  : normalized velocity profile
    % (derivative w.r.t. t, not τ, so divide by T_dur)
    ds   = (30*tau.^2 - 60*tau.^3 + 30*tau.^4) / T_dur; % [1xT]

    % s̈(τ) = (60τ − 180τ² + 120τ³) / T²  : normalized acceleration profile
    dds  = (60*tau - 180*tau.^2 + 120*tau.^3) / T_dur^2; % [1xT]

    % -----------------------------------------------------------------------
    % 3. Scale by the joint displacement (q1 - q0) and add offset q0
    % -----------------------------------------------------------------------
    % Using broadcasting: (q1-q0) is [Nx1], s is [1xT]
    % Their product is [NxT] via MATLAB's implicit broadcasting.

    delta_q = q1 - q0;     % [Nx1] total joint displacement

    q_t   = q0  + delta_q .* s;    % [NxT] position
    dq_t  =       delta_q .* ds;   % [NxT] velocity  (zero offset)
    ddq_t =       delta_q .* dds;  % [NxT] acceleration (zero offset)

end % function traj_poly5
