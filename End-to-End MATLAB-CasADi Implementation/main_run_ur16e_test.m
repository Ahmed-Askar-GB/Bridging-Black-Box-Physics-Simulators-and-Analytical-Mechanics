%% main_run_ur16e_test.m
% =========================================================================
% Main Driver Script for UR16e Dual-Model Verification with Custom Tolerances
% =========================================================================
% Demonstrates how tolerance parameters can be easily adjusted from outside
% the test function in the main script.
%
% Tolerances can be passed as a flat struct or nested struct.
% All fields are optional; sensible defaults are automatically applied.
% =========================================================================

clear; clc;
this_dir = fileparts(mfilename('fullpath'));
cd(this_dir);
addpath(genpath(this_dir));

% -------------------------------------------------------------------------
% 1. User-Configurable Tolerances & Simulation Parameters
% -------------------------------------------------------------------------
opts = struct();

% Kinematic tolerances (meters and radians)
opts.tol_pos       = 0.025;   % EE position tolerance (m) [e.g. 25 mm]
opts.tol_rot       = 0.030;   % EE orientation tolerance (rad) [~1.7 deg]

% Velocity tolerances (m/s and rad/s)
opts.tol_vel       = 0.25;    % EE linear velocity absolute floor (m/s)
opts.tol_omega     = 0.25;    % EE angular velocity absolute floor (rad/s)
opts.tol_vel_rel   = 0.05;    % Relative velocity scaling (5%)

% Dynamic & torque tolerances (N*m)
opts.tol_tau       = 0.50;    % Simscape vs CasADi torque tolerance (N*m)
opts.tol_seq_tau   = 5.0;     % Sequential dynamics torque tolerance across joints (N*m)

% Invariant & model parameter tolerances
opts.tol_mass      = 1e-6;    % URDF vs DH link mass cross-check tolerance (kg)
opts.tol_sym       = 1e-11;   % Mass matrix symmetry residual (|M - M^T|)
opts.tol_gravity   = 1e-11;   % Static gravitational balance residual (|tau - G|)

% Simulation duration (seconds)
opts.t_dur         = 2.5;

% -------------------------------------------------------------------------
% 2. Run Verification Suite
% -------------------------------------------------------------------------
fprintf('\n>>> Executing UR16e verification with user-defined tolerances ...\n');
report = test_ur16e_dual_model_verification(opts);

% -------------------------------------------------------------------------
% 3. Inspect Results & Active Tolerances
% -------------------------------------------------------------------------
disp('Verification completed. Active tolerances used:');
disp(report.tolerances);
