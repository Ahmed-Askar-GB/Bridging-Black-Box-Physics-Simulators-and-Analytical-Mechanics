function joint_signs = detect_joint_signs(chain, fk)
% DETECT_JOINT_SIGNS  Automated Robot-Agnostic Joint Axis Alignment.
%
% Mathematically reconciles the joint actuation direction defined in the URDF
% (<joint><axis xyz="...">) with the local z-axis direction defined by the
% standard Denavit-Hartenberg (DH) convention.
%
% No hardcoded signs or robot-specific values are used:
%   At the zero configuration (q = 0):
%     - z_{urdf, i} is the world-frame actuation axis of joint i (from chain.Z).
%     - z_{dh, i-1} is the world-frame z-axis of DH frame {i-1} (actuation axis in DH).
%
%   Because both axes lie on the same physical line of rotation:
%     dot(z_{urdf, i}, z_{dh, i-1}) ∈ {+1.0, -1.0}
%
%   Therefore:
%     joint_signs(i) = sign( round( dot(z_{urdf, i}, z_{dh, i-1}) ) )
%
% =========================================================================
% INPUTS:
%   chain (struct) Output of urdf_build_chain(). Required fields:
%                    .n   (int)   Number of active degrees of freedom
%                    .Z   [3xN]   World-frame unit joint axes at q=0
%   fk    (struct) Output of fk_casadi(). Required fields:
%                    .fk_link_fun {cell} Per-link cumulative FK functions
%
% OUTPUT:
%   joint_signs [Nx1 double]  +1 or -1 for each active joint.
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m
%   test_autonomous_verification.m
%   verify_dynamics_sequential.m

    n = chain.n;
    joint_signs = ones(n, 1);

    for i = 1:n
        z_urdf = chain.Z(:, i);

        if i == 1
            % Base frame actuation axis in standard DH is world [0; 0; 1]
            z_dh = [0; 0; 1];
        else
            % Actuation axis of joint i is the z-axis of DH frame {i-1}
            T_prev = full(fk.fk_link_fun{i-1}(zeros(n, 1)));
            z_dh = T_prev(1:3, 3);
        end

        % Unit vector dot product
        d = dot(z_urdf, z_dh);
        s = round(d);
        if abs(s) >= 0.5
            joint_signs(i) = sign(s);
        else
            joint_signs(i) = 1.0;
        end
    end

end
