%% test_phase5_sequential_dynamics.m
% =========================================================================
% INTEGRATION TEST — Phase 5: Sequential Dynamic Parameter & Frame Verification
% =========================================================================
% Verifies that link inertial parameters (mass, CoM, and inertia tensor)
% and coordinate frame transformations are sequentially consistent between
% the analytical CasADi dynamic model and the Simscape Digital Twin.
%
% Verification Stages:
%   For link i = 1..6:
%     - Downstream links k > i zeroed out (m_k = 0, I_k = 0, CoM = 0)
%     - All other joints frozen (q_j = const, dq_j = 0, ddq_j = 0)
%     - Joint i dynamically excited
%     - Actuation torque tau_i(t) cross-validated against analytical tau
%
% =========================================================================
% PASS criterion: "ALL PHASE 5 TESTS PASSED SUCCESSFULLY" printed.
% =========================================================================

clear; clc;

% -------------------------------------------------------------------------
% 0. Setup paths
% -------------------------------------------------------------------------
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
root_dir   = fileparts(parent_dir);

addpath(fullfile(parent_dir, 'parsing'));
addpath(fullfile(parent_dir, 'kinematics'));
addpath(fullfile(parent_dir, 'dynamics'));
addpath(fullfile(parent_dir, 'digital_twin'));
addpath(fullfile(parent_dir, 'verification'));
addpath(fullfile(parent_dir, 'utils'));
addpath(root_dir);

urdf_file = fullfile(root_dir, 'universalUR5.urdf');
mesh_dir  = fullfile(root_dir, 'STL_Files');

fprintf('============================================================\n');
fprintf('  Phase 5 Integration Test — Sequential Dynamics Verification\n');
fprintf('  Robot: UR5 (%s)\n', urdf_file);
fprintf('============================================================\n\n');

% -------------------------------------------------------------------------
% 1. Extract robot from URDF
% -------------------------------------------------------------------------
robot   = urdf_parse(urdf_file);
chain   = urdf_build_chain(robot, false);
dh      = urdf_extract_dh(chain, false);
inertia = urdf_extract_inertia(robot, chain, dh, false);

% -------------------------------------------------------------------------
% 2. Run Sequential Dynamic Verification across stages 1..6
% -------------------------------------------------------------------------
opts = struct();
opts.stages    = 1:6;
opts.duration  = 2.0;
opts.tol_tau   = 0.15; % 0.15 N.m tolerance
opts.mesh_dir  = mesh_dir;
opts.verbose   = true;

report = verify_dynamics_sequential(robot, chain, dh, inertia, opts);

% -------------------------------------------------------------------------
% 3. Assertions
% -------------------------------------------------------------------------
assert(report.all_passed, 'Sequential dynamic parameter verification failed.');

fprintf('============================================================\n');
fprintf('  >>> ALL PHASE 5 TESTS PASSED SUCCESSFULLY <<<\n');
fprintf('============================================================\n');
