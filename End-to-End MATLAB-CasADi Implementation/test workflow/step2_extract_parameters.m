function [dh, inertia, cfg] = step2_extract_parameters(robot, chain, cfg)
% STEP2_EXTRACT_PARAMETERS  Derive standard DH parameters and inertial tensors.
%
% Performs:
%   1. Geometric foot-point derivation of standard Denavit-Hartenberg parameters.
%   2. Rigid-body inertial transformation into local DH frames.
%   3. Geometric reach computation and refinement of scale-aware tolerances.
%   4. Formatted reporting of kinematic and inertial properties.
%
% =========================================================================
% INPUTS:
%   robot   (struct) Parsed URDF description from step1_parse_chain.
%   chain   (struct) Kinematic chain from step1_parse_chain.
%   cfg     (struct) Workflow configuration from step0_setup_config.
%
% OUTPUTS:
%   dh      (struct) Standard DH parameter table and properties.
%   inertia (struct) Masses, centers of mass, and 3x3xN inertia tensors.
%   cfg     (struct) Updated configuration with reach-refined tolerances.
% =========================================================================

    fprintf('\n>>> STEP 2: Extracting DH Parameters & Dynamic Inertias ...\n');
    n = chain.n;

    dh      = urdf_extract_dh(chain, false);
    inertia = urdf_extract_inertia(robot, chain, dh, false);

    % Print DH Parameters Table
    fprintf('\n  --- Standard DH Parameters Table [a_i, alpha_i, d_i, theta_offset_i] ---\n');
    fprintf('  Joint |      a (m)     |   alpha (rad)  |      d (m)     |  theta_off (rad)\n');
    fprintf('  ------+----------------+----------------+----------------+-----------------\n');
    for i = 1:n
        fprintf('    %d   |  %12.6f  |  %12.6f  |  %12.6f  |  %12.6f\n', ...
                i, dh.table(i, 3), dh.table(i, 4), dh.table(i, 2), dh.table(i, 1));
    end

    % Refine scale-aware tolerances using exact geometric reach from DH table
    L_reach = sum(abs(dh.table(:, 3))) + sum(abs(dh.table(:, 2)));
    cfg.tol = get_verification_tolerances(cfg.opts, L_reach);

    % Print Dynamic Inertias
    fprintf('\n  --- Dynamic Inertia Parameters (Masses & Principal Inertias) ---\n');
    fprintf('  Link  |  Mass (kg)  |     CoM [x, y, z] (m)     |    I_diag [Ixx, Iyy, Izz] (kg*m^2)\n');
    fprintf('  ------+-------------+---------------------------+------------------------------------\n');
    for i = 1:n
        c_i = inertia.P_Ci(i, :);
        I_diag = [inertia.I(1,1,i), inertia.I(2,2,i), inertia.I(3,3,i)];
        fprintf('    %d   |  %9.4f  | [%6.3f, %6.3f, %6.3f] | [%.5e, %.5e, %.5e]\n', ...
                i, inertia.m(i), c_i(1), c_i(2), c_i(3), I_diag(1), I_diag(2), I_diag(3));
    end
    fprintf('  Total Active Arm Mass = %.4f kg (Physical Reach: %.3f m)\n', sum(inertia.m), L_reach);

end
