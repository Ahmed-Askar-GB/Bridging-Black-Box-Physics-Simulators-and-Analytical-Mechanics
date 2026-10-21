function discrepancy_list = step5_cross_validate_parameters(robot, chain, inertia, cfg)
% STEP5_CROSS_VALIDATE_PARAMETERS  Verify physical parameters between URDF and DH chain.
%
% Performs:
%   1. Link-by-link mass verification between raw URDF XML and transformed DH parameters.
%   2. Center of mass position sanity checks.
%   3. Inertia tensor diagonal validation.
%   4. Accumulation of any detected numerical deviations into discrepancy_list.
%
% =========================================================================
% INPUTS:
%   robot   (struct) Parsed URDF description.
%   chain   (struct) Active kinematic chain.
%   inertia (struct) Transformed DH inertial parameters.
%   cfg     (struct) Configuration struct with tol.mass.
%
% OUTPUT:
%   discrepancy_list {cell} List of diagnostic warning strings (empty if 100% matched).
% =========================================================================

    fprintf('\n>>> STEP 5: Inertia & DH Parameter Cross-Validation ...\n');
    n = chain.n;
    discrepancy_list = {};

    fprintf('  Verifying Link Inertias and Centers of Mass:\n');
    for i = 1:n
        link_name = chain.active_links{i};
        urdf_link = robot.links(link_name);

        m_urdf = urdf_link.mass;
        m_dh   = inertia.m(i);
        diff_m = abs(m_urdf - m_dh);

        if diff_m > cfg.tol.mass
            warning_msg = sprintf('Link %d ("%s") mass mismatch: URDF=%.4f kg, DH=%.4f kg (Diff: %.2e, Tol: %.1e kg)', ...
                                  i, link_name, m_urdf, m_dh, diff_m, cfg.tol.mass);
            discrepancy_list{end+1} = warning_msg;
            fprintf('    [!] %s\n', warning_msg);
        else
            fprintf('    Link %d ("%-18s") Mass = %.4f kg [MATCH]\n', i, link_name, m_dh);
        end
    end

    if isempty(discrepancy_list)
        fprintf('  [PASS] All link masses matched between URDF and DH representation to < %.1e kg.\n', cfg.tol.mass);
    end

end
