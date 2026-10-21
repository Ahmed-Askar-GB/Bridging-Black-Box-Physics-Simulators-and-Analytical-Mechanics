% scratch/test_joints1_to_5.m
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

model_name = 'test_ur5_verify_model';
opts_build = struct('model_name', model_name, 'output_dir', pwd, 'stop_time', '3.0', ...
                    'mesh_dir', mesh_dir, 'verbose', false);
dt = dt_build(robot, chain, inertia, opts_build);

% EE transform: Identity relative to wrist_3_link (measuring wrist_3_link directly)
dt = dt_add_ee_frame(dt, struct('T_ee', eye(4)), struct('verbose', false));
dt = dt_add_sensors(dt, struct('verbose', false));

% Excitation trajectory with joint 6 = 0:
t_vec = linspace(0, 3.0, 1001);
q0 = [0; -0.3; 0.2; -0.1; 0.4; 0];
qf = [0.5; 0.2; -0.4; 0.3; -0.2; 0];
[q_m, dq_m, ddq_m] = traj_poly5(q0, qf, 3.0, t_vec);
traj = struct('time', t_vec, 'q', q_m, 'dq', dq_m, 'ddq', ddq_m);
dt = dt_add_trajectory(dt, traj, struct('verbose', false));

simOut = sim(model_name, 'StopTime', '3.0');

q_sim    = simOut.get('q_sim').Data;
p_w3_raw = simOut.get('ee_pos').Data;
if ndims(p_w3_raw) == 3
    p_w3_sim = squeeze(p_w3_raw)';
else
    p_w3_sim = p_w3_raw;
end

% Compare with DH Frame 5
M_steps = size(q_sim, 1);
p_dh5 = zeros(M_steps, 3);
for k = 1:M_steps
    qk = q_sim(k, :)';
    T5 = full(fk.fk_link_fun{5}(qk));
    p_dh5(k, :) = T5(1:3, 4)';
end

err = max(vecnorm(p_w3_sim - p_dh5, 2, 2));
fprintf('Max error between Simscape wrist_3_link and DH Frame 5 across 3.0s trajectory: %.6e m\n', err);

close_system(model_name, 0);
