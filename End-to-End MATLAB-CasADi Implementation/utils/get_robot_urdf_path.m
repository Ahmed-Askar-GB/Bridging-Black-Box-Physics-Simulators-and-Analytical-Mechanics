function urdf_path = get_robot_urdf_path(robot_name)
% GET_ROBOT_URDF_PATH  Standardized path resolution for benchmark manipulators.
%
% Supported robot names (case-insensitive):
%   'ur5', 'ur16e', 'abb', 'abb_irb120', 'puma', 'puma560', 'kuka', 'kuka_iiwa'
%
% =========================================================================

    this_dir = fileparts(mfilename('fullpath'));
    % Navigate to workspace root
    cur = this_dir;
    root_dir = '';
    for depth = 1:5
        if exist(fullfile(cur, 'universalUR5.urdf'), 'file')
            root_dir = cur;
            break;
        end
        parent = fileparts(cur);
        if isempty(parent) || strcmp(parent, cur), break; end
        cur = parent;
    end
    if isempty(root_dir)
        root_dir = pwd;
    end

    name_clean = lower(strrep(strrep(robot_name, '_', ''), '-', ''));

    switch name_clean
        case {'ur5', 'universalur5'}
            candidates = {
                fullfile(root_dir, 'universalUR5.urdf');
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'universalUR5.urdf')
            };
        case {'ur16e', 'universalur16e'}
            candidates = {
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'End-to-End MATLAB-CasADi Implementation', 'test_UR16e', 'universalUR16e.urdf');
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'End-to-End MATLAB Implementation', 'test_UR16e', 'universalUR16e.urdf');
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'URDF2DT-Py Full pipeline', 'tests', 'fixtures', 'ur16e.urdf')
            };
        case {'abb', 'abbirb120', 'irb120'}
            candidates = {
                fullfile(root_dir, 'abbIrb120', 'abbIrb120.urdf');
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'abbIrb120', 'abbIrb120.urdf')
            };
        case {'puma', 'puma560'}
            candidates = {
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'End-to-End MATLAB-CasADi Implementation', 'puma560_test', 'puma560_robot_dynamic.urdf');
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'End-to-End MATLAB Implementation', 'puma560_test', 'puma560_robot_dynamic.urdf')
            };
        case {'kuka', 'kukaiiwa', 'kukaiiwa7', 'iiwa', 'iiwa7'}
            candidates = {
                fullfile(root_dir, 'kukaIiwa7.urdf');
                fullfile(root_dir, 'Full pipeline Manuscript revised', 'benchmarks', 'kukaIiwa7.urdf');
                fullfile(root_dir, 'Bridging Black-Box Physics Simulators and Analytical Mechanics', 'URDF2DT-Py Full pipeline', 'tests', 'fixtures', 'kukaIiwa7.urdf')
            };
        otherwise
            if exist(robot_name, 'file')
                urdf_path = robot_name;
                return;
            else
                error('get_robot_urdf_path:UnknownRobot', 'Unknown robot name: %s', robot_name);
            end
    end

    urdf_path = '';
    for i = 1:length(candidates)
        if exist(candidates{i}, 'file')
            urdf_path = candidates{i};
            return;
        end
    end

    error('get_robot_urdf_path:FileNotFound', 'Could not locate URDF for robot %s', robot_name);
end
