% scratch/test_simscape_dh_relation.m
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

% Build a test model where EE_Transform is IDENTITY (so it measures wrist_3_link directly)
model_name = 'test_check_rel';
opts_build = struct('model_name', model_name, 'output_dir', pwd, 'stop_time', '1.0', ...
                    'mesh_dir', mesh_dir, 'verbose', false);
dt = dt_build(robot, chain, inertia, opts_build);

% Set EE transform to IDENTITY to measure wrist_3_link directly
chain_ident = chain;
chain_ident.T_ee = eye(4);
dt = dt_add_ee_frame(dt, chain_ident, struct('verbose', false));
dt = dt_add_sensors(dt, struct('verbose', false));
dt = dt_add_trajectory(dt, [], struct('verbose', false));

simOut = sim(model_name, 'StopTime', '1.0');

q_sim    = simOut.get('q_sim').Data;
p_w3_raw = simOut.get('ee_pos').Data;
if ndims(p_w3_raw) == 3
    p_w3_sim = squeeze(p_w3_raw)';
else
    p_w3_sim = p_w3_raw;
end

% Let's compare p_w3_sim with DH Frame 5 and DH Frame 6
M_steps = size(q_sim, 1);
p_dh5 = zeros(M_steps, 3);
p_dh6 = zeros(M_steps, 3);
for k = 1:M_steps
    qk = q_sim(k, :)';
    T5 = full(fk.fk_link_fun{5}(qk));
    T6 = full(fk.fk_link_fun{6}(qk));
    p_dh5(k, :) = T5(1:3, 4)';
    p_dh6(k, :) = T6(1:3, 4)';
end

err5 = max(vecnorm(p_w3_sim - p_dh5, 2, 2));
fprintf('Max difference between Simscape wrist_3_link origin and DH Frame 5: %.6e m\n', err5);

% Let's test the relative transform between Simscape wrist_3_link and DH Frame 6
% In Simscape:
q_quat_sim = simOut.get('ee_quat').Data;
if ndims(q_quat_sim) == 3
    q_quat_sim = squeeze(q_quat_sim)';
end

% Reconstruct R_w3_sim at each time step
max_T_rel_variation = 0;
T_rel_first = [];
for k = 1:M_steps
    qk = q_sim(k, :)';
    T6_dh = full(fk.fk_link_fun{6}(qk));
    
    % Simscape wrist_3_link pose
    qw = q_quat_sim(k,1); qx = q_quat_sim(k,2); qy = q_quat_sim(k,3); qz = q_quat_sim(k,4);
    R_w3 = [1 - 2*(qy^2 + qz^2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw);
            2*(qx*qy + qz*qw), 1 - 2*(qx^2 + qz^2),     2*(qy*qz - qx*qw);
            2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx^2 + qy^2)];
    T_w3 = [R_w3, p_w3_sim(k, :)'; 0 0 0 1];
    
    T_rel_k = T_w3 \ T6_dh;
    if k == 1
        T_rel_first = T_rel_k;
    else
        diff_T = max(abs(T_rel_k(:) - T_rel_first(:)));
        if diff_T > max_T_rel_variation
            max_T_rel_variation = diff_T;
        end
    end
end

% Compare joint-by-joint link transforms between Simscape and DH at q = 0 and at perturbed q
fprintf('\n--- Joint direction analysis ---\n');
for j = 1:6
    q_pert = zeros(6, 1);
    q_pert(j) = 0.1; % 0.1 rad perturbation on joint j
    
    % Analytical DH link j transform:
    T_dh_0 = full(fk.fk_link_fun{j}(zeros(6,1)));
    T_dh_p = full(fk.fk_link_fun{j}(q_pert));
    R_dh_rel = T_dh_0(1:3,1:3)' * T_dh_p(1:3,1:3);
    
    fprintf('Joint %d: R_dh_rel trace = %.6f\n', j, trace(R_dh_rel));
end

close_system(model_name, 0);
