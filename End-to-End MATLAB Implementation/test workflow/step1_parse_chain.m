function [robot, chain] = step1_parse_chain(cfg)
% STEP1_PARSE_CHAIN  Parse target URDF XML and extract active kinematic chain.
%
% Performs:
%   1. Ingestion of URDF links, joints, and root link identification.
%   2. Graph traversal to isolate the active serial kinematic chain.
%   3. Validation of degrees of freedom.
%
% =========================================================================
% INPUT:
%   cfg   (struct) Configuration struct output by step0_setup_config.
%
% OUTPUTS:
%   robot (struct) Full parsed URDF robot description.
%   chain (struct) Serial kinematic chain struct from urdf_build_chain.
% =========================================================================

    fprintf('>>> STEP 1: Parsing %s & Kinematic Chain ...\n', [cfg.urdf_stem, '.urdf']);

    robot = urdf_parse(cfg.urdf_path);
    fprintf('  Parsed %d links, %d joints from URDF. Root link: "%s"\n', ...
            robot.links.Count, robot.joints.Count, robot.root_link);

    chain = urdf_build_chain(robot, false);
    n = chain.n;
    fprintf('  Active DOFs: %d. Active links:\n', n);
    for i = 1:n
        fprintf('    Joint %d: %-20s -> Child Link: %-20s\n', ...
                i, chain.active_joints{i}.name, chain.active_links{i});
    end

end
