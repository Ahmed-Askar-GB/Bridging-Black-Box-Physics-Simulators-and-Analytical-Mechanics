% scratch/test_stage1_dynamics.m
clear; clc;
scratch_dir = fileparts(mfilename('fullpath'));
this_dir    = fileparts(scratch_dir);
root_dir    = fileparts(this_dir);

addpath(fullfile(this_dir, 'parsing'));
addpath(fullfile(this_dir, 'kinematics'));
addpath(fullfile(this_dir, 'dynamics'));
addpath(fullfile(this_dir, 'digital_twin'));
addpath(fullfile(this_dir, 'verification'));
addpath(fullfile(this_dir, 'utils'));
addpath(root_dir);

urdf_file = fullfile(root_dir, 'universalUR5.urdf');
mesh_dir  = fullfile(root_dir, 'STL_Files');

robot   = urdf_parse(urdf_file);
chain   = urdf_build_chain(robot, false);
dh      = urdf_extract_dh(chain, false);
inertia = urdf_extract_inertia(robot, chain, dh, false);

% Stage 1: links 2..6 zeroed out
inertia_1 = inertia;
for k = 2:6
    inertia_1.m(k) = 0;
    inertia_1.I(:,:,k) = zeros(3,3);
    inertia_1.P_Ci(k,:) = zeros(1,3);
end

% Analytical model for Stage 1
dyn1 = dynamics_casadi(dh, inertia_1, chain);

% Simscape model for Stage 1
model_name = 'test_stage1_model';
opts_build = struct('model_name', model_name, 'output_dir', pwd, 'stop_time', '2.0', ...
                    'mesh_dir', mesh_dir, 'verbose', false);
dt = dt_build(robot, chain, inertia, opts_build);

% Zero out downstream link solids in Simscape
for k = 2:6
    l_name = chain.active_links{k};
    geom_name = sprintf('Geometry_%d_%s', k, regexprep(l_name, '[^a-zA-Z0-9_]', '_'));
    blk = [model_name, '/', geom_name];
    set_param(blk, 'Mass', '0');
    set_param(blk, 'CenterOfMass', '[0, 0, 0]');
    set_param(blk, 'MomentsOfInertia', '[0, 0, 0]');
end

joint_signs = [1; 1; 1; 1; 1; -1];
dt = dt_add_ee_frame(dt, chain, struct('verbose', false));
dt = dt_add_sensors(dt, struct('verbose', false));

% Excitation on Joint 1 only, other joints frozen
t_vec = linspace(0, 2.0, 501);
q0 = [0; 0.2; -0.3; 0.1; -0.2; 0.3];
qf = [0.8; 0.2; -0.3; 0.1; -0.2; 0.3]; % only joint 1 moves
[q_m, dq_m, ddq_m] = traj_poly5(q0, qf, 2.0, t_vec);

traj = struct('time', t_vec, 'q', q_m, 'dq', dq_m, 'ddq', ddq_m);
dt = dt_add_trajectory(dt, traj, struct('joint_signs', joint_signs, 'verbose', false));

simOut = sim(model_name, 'StopTime', '2.0');

% Compare Joint 1 torque:
tau_sim = simOut.get('tau_sim');
t_sim   = tau_sim.Time;
tau_sim_data = tau_sim.Data;
if ndims(tau_sim_data) == 3, tau_sim_data = squeeze(tau_sim_data)'; end

q_sim_data = simOut.get('q_sim').Data;
if ndims(q_sim_data) == 3, q_sim_data = squeeze(q_sim_data)'; end
dq_sim_data = simOut.get('dq_sim').Data;
if ndims(dq_sim_data) == 3, dq_sim_data = squeeze(dq_sim_data)'; end

% Map back joint signs
q_sim_data  = q_sim_data  .* (joint_signs');
dq_sim_data = dq_sim_data .* (joint_signs');

% Compute analytical torque
M_steps = length(t_sim);
tau_ana = zeros(M_steps, 6);
for k = 1:M_steps
    qk  = q_sim_data(k, :)';
    dqk = dq_sim_data(k, :)';
    % Use finite difference or spline for ddq if needed, or trajectory ddq
    % In our trajectory, ddq is known:
    ddqk = [interp1(t_vec, ddq_m(1,:), t_sim(k)); zeros(5,1)];
    tau_k = full(dyn1.tau_fun(qk, dqk, ddqk));
    tau_ana(k, :) = tau_k';
end

tau1_sim = tau_sim_data(:, 1);
tau1_ana = tau_ana(:, 1);

err_tau1 = max(abs(tau1_sim - tau1_ana));
fprintf('Stage 1 (Joint 1 Isolated) Max Torque Error: %.4e N.m\n', err_tau1);
fprintf('  Simscape Torque range: [%.4f, %.4f] N.m\n', min(tau1_sim), max(tau1_sim));
fprintf('  Analytical Torque range: [%.4f, %.4f] N.m\n', min(tau1_ana), max(tau1_ana));

close_system(model_name, 0);
