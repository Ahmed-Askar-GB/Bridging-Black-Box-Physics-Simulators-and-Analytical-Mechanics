% scratch/test_kinematics_alignment.m
% Investigate kinematic correspondence between URDF Simscape chain and DH chain

clear; clc;
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
root_dir   = fileparts(parent_dir);

addpath(fullfile(parent_dir, 'parsing'));
addpath(fullfile(parent_dir, 'kinematics'));
addpath(fullfile(parent_dir, 'dynamics'));
addpath(fullfile(parent_dir, 'digital_twin'));
addpath(fullfile(parent_dir, 'verification'));
addpath(root_dir);

robot = urdf_parse(fullfile(root_dir, 'universalUR5.urdf'));
chain = urdf_build_chain(robot, false);
dh    = urdf_extract_dh(chain, false);
fk    = fk_casadi(dh, chain);

fprintf('--- DH Table extracted ---\n');
disp(dh.table);

% Check DH parameters against legacy
% Let's compute forward kinematics using both methods at q = 0 and q != 0
q_zero = zeros(6, 1);
[p_dh0, R_dh0, quat_dh0] = fk.fk_fun(q_zero);
p_dh0 = full(p_dh0);

fprintf('DH p_ee at q=0:\n');
disp(p_dh0');

% In URDF, where is ee_link at q=0?
T_urdf = eye(4);
for i = 1:chain.n
    T_urdf = T_urdf * chain.active_joints{i}.T_effective;
end
T_urdf_ee = T_urdf * chain.T_ee;
fprintf('URDF ee_link pos at q=0:\n');
disp(T_urdf_ee(1:3, 4)');

% Notice:
% URDF wrist_3_link origin at q=0:
fprintf('URDF wrist_3_link origin at q=0:\n');
disp(T_urdf(1:3, 4)');

% Now, what about the intermediate DH frames vs URDF frames?
for i = 1:chain.n
    T_dh_i = full(fk.fk_link_fun{i}(q_zero));
    fprintf('Joint %d child %-15s: DH origin = [%.4f, %.4f, %.4f]\n', ...
        i, chain.active_links{i}, T_dh_i(1,4), T_dh_i(2,4), T_dh_i(3,4));
end
