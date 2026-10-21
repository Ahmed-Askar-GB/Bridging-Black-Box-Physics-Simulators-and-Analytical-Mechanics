%% test_phase4_verification.m
% =========================================================================
% INTEGRATION TEST — Phase 4: Kinematic Ground-Truth Cross-Validation
% =========================================================================
% Validates that the analytical CasADi kinematics (FK and Jacobian) and the
% Simscape Multibody Digital Twin produce identical positions, orientations,
% and spatial velocities under dynamic excitation.
%
% Tests performed:
%   1. verify_fk:       Simscape EE pose vs Analytical CasADi FK
%   2. verify_jacobian: Simscape EE velocity vs Analytical CasADi Jacobian
%
% =========================================================================
% HOW TO RUN:
%   cd to the "End-to-End MATLAB-CasADi Implementation" folder, then:
%     run tests/test_phase4_verification.m
%
% PASS criterion: "ALL PHASE 4 TESTS PASSED" printed, assert passes.
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
fprintf('  Phase 4 Integration Test — Kinematic Ground-Truth Validation\n');
fprintf('  Robot: UR5 (%s)\n', urdf_file);
fprintf('============================================================\n\n');

% -------------------------------------------------------------------------
% 1. Extract robot from URDF & compile analytical CasADi kinematics
% -------------------------------------------------------------------------
fprintf('--- [1/4] Deriving Analytical CasADi Kinematics ---\n');
robot   = urdf_parse(urdf_file);
chain   = urdf_build_chain(robot, false);
dh      = urdf_extract_dh(chain, false);
inertia = urdf_extract_inertia(robot, chain, dh, false);

fk  = fk_casadi(dh, chain);
jac = jacobian_casadi(fk, dh, chain);

model.fk  = fk;
model.jac = jac;
fprintf('  Analytical FK & Jacobian compiled successfully.\n\n');

% -------------------------------------------------------------------------
% 2. Build Simscape Digital Twin with EE frame, sensors, and trajectory
% -------------------------------------------------------------------------
fprintf('--- [2/4] Building Instrumented Digital Twin ---\n');
model_name = 'test_ur5_verify_model';

opts_build = struct();
opts_build.model_name = model_name;
opts_build.output_dir = fullfile(parent_dir, 'digital_twin');
opts_build.stop_time  = '3.0';
opts_build.mesh_dir   = mesh_dir;
opts_build.verbose    = false;

% URDF revolute joint 6 axis is [0, 1, 0], while DH z_5 is [0, -1, 0].
% The joint_signs vector maps between analytical DH coordinates and URDF joint directions.
joint_signs = [1; 1; 1; 1; 1; -1];

dt = dt_build(robot, chain, inertia, opts_build);
dt = dt_add_ee_frame(dt, chain, struct('model', model, 'verbose', false));
dt = dt_add_sensors(dt, struct('verbose', false));
dt = dt_add_trajectory(dt, [], struct('joint_signs', joint_signs, 'verbose', false));

fprintf('  Digital Twin ready for simulation.\n\n');

% -------------------------------------------------------------------------
% 3. Execute Headless Simulation
% -------------------------------------------------------------------------
fprintf('--- [3/4] Running Dynamic Simulation ... ---\n');
simOut = sim(dt.model_name, 'StopTime', '3.0');
fprintf('  Simulation finished.\n\n');

% -------------------------------------------------------------------------
% 4. Cross-Validate Kinematics
% -------------------------------------------------------------------------
fprintf('--- [4/4] Validating Kinematics & Jacobians ---\n');

% 4A. Forward Kinematics Verification
opts_fk = struct('verbose', true, 'tol_pos', 1e-4, 'tol_rot', 1e-4, 'joint_signs', joint_signs);
report_fk = verify_fk(model, simOut, opts_fk);
assert(report_fk.passed, 'Forward Kinematics verification failed.');

% 4B. Jacobian Verification
opts_jac = struct('verbose', true, 'tol_v', 1e-4, 'tol_w', 1e-4, 'joint_signs', joint_signs);
report_jac = verify_jacobian(model, simOut, opts_jac);
assert(report_jac.passed, 'Jacobian verification failed.');

% Cleanup
close_system(dt.model_name, 0);

fprintf('============================================================\n');
fprintf('  >>> ALL PHASE 4 TESTS PASSED SUCCESSFULLY <<<\n');
fprintf('============================================================\n');
