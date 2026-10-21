% scratch/test_all_6_joints.m
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
fk      = fk_casadi(dh, chain);
jac     = jacobian_casadi(fk, dh, chain);

model.fk  = fk;
model.jac = jac;

model_name = 'test_ur5_all6';
opts_build = struct('model_name', model_name, 'output_dir', pwd, 'stop_time', '3.0', ...
                    'mesh_dir', mesh_dir, 'verbose', false);
dt = dt_build(robot, chain, inertia, opts_build);

% Relative transform from wrist_3_link to DH Frame 6 at q=0:
T_u0 = eye(4);
for i = 1:chain.n
    T_u0 = T_u0 * chain.active_joints{i}.T_effective;
end
T_dh0 = full(fk.fk_ee_fun(zeros(6,1)));
T_rel = T_u0 \ T_dh0;

dt = dt_add_ee_frame(dt, struct('T_ee', T_rel), struct('verbose', false));
dt = dt_add_sensors(dt, struct('verbose', false));

% Excitation trajectory for all 6 joints
t_vec = linspace(0, 3.0, 1001);
q0 = [0; -0.3; 0.2; -0.1; 0.4; 0];
qf = [0.5; 0.2; -0.4; 0.3; -0.2; 0.6];
[q_m, dq_m, ddq_m] = traj_poly5(q0, qf, 3.0, t_vec);

% Simscape trajectory (note: joint 6 is flipped to match DH coordinate direction)
q_sim_traj = q_m;
dq_sim_traj = dq_m;
ddq_sim_traj = ddq_m;
q_sim_traj(6, :) = -q_m(6, :);
dq_sim_traj(6, :) = -dq_m(6, :);
ddq_sim_traj(6, :) = -ddq_m(6, :);

traj = struct('time', t_vec, 'q', q_sim_traj, 'dq', dq_sim_traj, 'ddq', ddq_sim_traj);
dt = dt_add_trajectory(dt, traj, struct('verbose', false));

simOut = sim(model_name, 'StopTime', '3.0');

% Re-align logged q_sim for analytical validation:
q_logged = simOut.get('q_sim');
dq_logged = simOut.get('dq_sim');
q_val = q_logged.Data;
dq_val = dq_logged.Data;
if ndims(q_val) == 3, q_val = squeeze(q_val)'; end
if ndims(dq_val) == 3, dq_val = squeeze(dq_val)'; end

% Joint 6 sign alignment
q_val(:, 6) = -q_val(:, 6);
dq_val(:, 6) = -dq_val(:, 6);

simOut_val = struct();
simOut_val.q_sim    = q_val;
simOut_val.dq_sim   = dq_val;
simOut_val.ee_pos   = simOut.get('ee_pos');
simOut_val.ee_quat  = simOut.get('ee_quat');
simOut_val.ee_vel   = simOut.get('ee_vel');
simOut_val.ee_omega = simOut.get('ee_omega');

fprintf('--- Running verify_fk ---\n');
report_fk = verify_fk(model, simOut_val, struct('verbose', true));

fprintf('--- Running verify_jacobian ---\n');
report_jac = verify_jacobian(model, simOut_val, struct('verbose', true));

close_system(model_name, 0);
