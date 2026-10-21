%% test_phase3_digital_twin.m
% =========================================================================
% INTEGRATION TEST — Phase 3: Digital Twin Layer (Simscape Multibody)
% =========================================================================
% Validates that the programmatic Simscape Multibody Digital Twin pipeline:
%   1. Builds an exact Simscape model without errors (dt_build)
%   2. Anchors the End-Effector reference frame (dt_add_ee_frame)
%   3. Instruments all EE and joint measurement sensors (dt_add_sensors)
%   4. Injects smooth 5th-order excitation trajectory (dt_add_trajectory)
%   5. Executes headless dynamic simulation via sim() successfully
%   6. Logs all states to the MATLAB workspace (ee_pos, ee_vel, ee_quat,
%      ee_omega, q_sim, dq_sim, tau_sim) with correct dimensions.
%
% =========================================================================
% HOW TO RUN:
%   cd to the "End-to-End MATLAB-CasADi Implementation" folder, then:
%     run tests/test_phase3_digital_twin.m
%
% PASS criterion: "ALL PHASE 3 TESTS PASSED" printed, assert passes.
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
addpath(fullfile(parent_dir, 'utils'));
addpath(root_dir);

urdf_file = fullfile(root_dir, 'universalUR5.urdf');
mesh_dir  = fullfile(root_dir, 'STL_Files');

fprintf('============================================================\n');
fprintf('  Phase 3 Integration Test — Simscape Digital Twin\n');
fprintf('  Robot: UR5 (%s)\n', urdf_file);
fprintf('============================================================\n\n');

% -------------------------------------------------------------------------
% 1. Extract robot from URDF
% -------------------------------------------------------------------------
fprintf('--- [1/5] Ingesting URDF & Extracting Kinematics/Dynamics ---\n');
robot   = urdf_parse(urdf_file);
chain   = urdf_build_chain(robot, false);
dh      = urdf_extract_dh(chain, false);
inertia = urdf_extract_inertia(robot, chain, dh, false);
N       = chain.n;
fprintf('  Serial chain: %d DOF extracted successfully.\n\n', N);

% -------------------------------------------------------------------------
% 2. Programmatically construct Simscape Multibody model
% -------------------------------------------------------------------------
fprintf('--- [2/5] Building Simscape Multibody Model (dt_build) ---\n');
model_test_name = 'test_ur5_dt_model';

opts_build = struct();
opts_build.model_name = model_test_name;
opts_build.output_dir = fullfile(parent_dir, 'digital_twin');
opts_build.stop_time  = '3.0';   % 3 second simulation for fast test
opts_build.mesh_dir   = mesh_dir;
opts_build.verbose    = true;

dt = dt_build(robot, chain, inertia, opts_build);

assert(exist(dt.slx_path, 'file') == 4, 'Model .slx file was not created.');
assert(length(dt.joint_blocks) == N, 'Number of joint blocks does not match N.');
fprintf('  Simscape model created: %s\n\n', dt.slx_path);

% -------------------------------------------------------------------------
% 3. Anchor End-Effector Reference Frame
% -------------------------------------------------------------------------
fprintf('--- [3/5] Anchoring EE Reference Frame (dt_add_ee_frame) ---\n');
opts_ee = struct('verbose', true);
dt = dt_add_ee_frame(dt, chain, opts_ee);

assert(isfield(dt, 'ee_port') && ~isempty(dt.ee_port), 'EE frame port missing.');
fprintf('  EE frame anchored successfully.\n\n');

% -------------------------------------------------------------------------
% 4. Add Measurement Sensors
% -------------------------------------------------------------------------
fprintf('--- [4/5] Adding Transform & Joint Sensors (dt_add_sensors) ---\n');
opts_sens = struct('verbose', true);
dt = dt_add_sensors(dt, opts_sens);

assert(dt.sensors_added, 'Sensors flag not set.');
fprintf('  Sensors wired successfully.\n\n');

% -------------------------------------------------------------------------
% 5. Prescribe Excitation Trajectory
% -------------------------------------------------------------------------
fprintf('--- [5/5] Injecting Excitation Trajectory (dt_add_trajectory) ---\n');
opts_traj = struct('verbose', true);
dt = dt_add_trajectory(dt, [], opts_traj);

assert(dt.trajectory_injected, 'Trajectory injection flag not set.');
fprintf('  Trajectory wired successfully.\n\n');

% -------------------------------------------------------------------------
% 6. Run Headless Simulation
% -------------------------------------------------------------------------
fprintf('============================================================\n');
fprintf('  Executing Headless Simscape Simulation ...\n');
fprintf('============================================================\n');

sim_start = tic;
simOut = sim(dt.model_name, 'StopTime', '3.0');
sim_elapsed = toc(sim_start);
fprintf('  Simulation completed in %.2f seconds.\n\n', sim_elapsed);

% -------------------------------------------------------------------------
% 7. Validate Logged Simulation Outputs
% -------------------------------------------------------------------------
fprintf('--- Validating Simulation Output Signals ---\n');

% Check that variables exist in the base workspace or simOut
vars_expected = {'ee_pos', 'ee_vel', 'ee_quat', 'ee_omega', 'q_sim', 'dq_sim', 'tau_sim'};
all_signals_ok = true;

for k = 1:length(vars_expected)
    vname = vars_expected{k};
    try
        data = extract_sig_data(simOut, vname);
        fprintf('  [PASS] Signal "%-8s" : %4d time steps, size [%s]\n', ...
                vname, size(data, 1), num2str(size(data)));
    catch ME
        fprintf('  [FAIL] Signal "%s" extraction error: %s\n', vname, ME.message);
        all_signals_ok = false;
    end
end

assert(all_signals_ok, 'Some expected signals were not logged correctly.');

% Specific signal shape assertions
q_data   = extract_sig_data(simOut, 'q_sim');
ee_p_data= extract_sig_data(simOut, 'ee_pos');
ee_q_data= extract_sig_data(simOut, 'ee_quat');
tau_data = extract_sig_data(simOut, 'tau_sim');

assert(size(q_data,    2) == N, 'q_sim column count must equal N=6.');
assert(size(ee_p_data, 2) == 3, 'ee_pos column count must equal 3 (x,y,z).');
assert(size(ee_q_data, 2) == 4, 'ee_quat column count must equal 4 (qw,qx,qy,qz).');
assert(size(tau_data,  2) == N, 'tau_sim column count must equal N=6.');

% Cleanup: close the test model
close_system(dt.model_name, 0);

fprintf('\n============================================================\n');
fprintf('  >>> ALL PHASE 3 DIGITAL TWIN TESTS PASSED <<<\n');
fprintf('============================================================\n');

% =========================================================================
%  LOCAL HELPER
% =========================================================================
function data = extract_sig_data(simOut, vname)
% Extracts and reshapes signal data to standard [time x dim] format
    if evalin('base', sprintf('exist(''%s'', ''var'')', vname))
        sig = evalin('base', vname);
    else
        sig = simOut.get(vname);
    end
    raw = sig.Data;
    if ndims(raw) == 3
        % [dim x 1 x time] -> transpose to [time x dim]
        data = squeeze(raw)';
    else
        data = raw;
    end
end
