function cfg = step0_setup_config(urdf_path, opts)
% STEP0_SETUP_CONFIG  Configure paths, models, naming, and tolerances for verification workflow.
%
% Dynamically establishes:
%   1. Absolute paths and MATLAB path inclusions.
%   2. Model naming and output filenames determined from the URDF file name or opts.
%   3. Scale-aware verification tolerance configuration via get_verification_tolerances.
%   4. Diagnostic banner reporting.
%
% =========================================================================
% INPUTS:
%   urdf_path (char)   [optional] Path to URDF target file.
%                      Default: 'test_UR16e/universalUR16e.urdf'
%   opts      (struct) [optional] Options and tolerance overrides:
%                        .output_dir          Output directory for models and reports
%                        .dh_model_name       Custom name for DH Simulink twin
%                        .smimport_model_name Custom name for smimport Simulink twin
%                        .report_file         Custom path for output .mat report
%                        .mesh_dir            Custom path for mesh assets
%                        .tol_pos, .tol_rot, .tol_vel, .tol_omega, .tol_seq_tau, ...
%
% OUTPUT:
%   cfg       (struct) Fully initialized workflow configuration structure.
% =========================================================================

    this_dir   = fileparts(mfilename('fullpath'));  % 'test workflow' directory
    parent_dir = fileparts(this_dir);               % 'End-to-End MATLAB-CasADi Implementation'
    root_dir   = fileparts(parent_dir);

    % Configure MATLAB paths
    addpath(this_dir);
    addpath(fullfile(parent_dir, 'parsing'));
    addpath(fullfile(parent_dir, 'kinematics'));
    addpath(fullfile(parent_dir, 'dynamics'));
    addpath(fullfile(parent_dir, 'compilation'));
    addpath(fullfile(parent_dir, 'digital_twin'));
    addpath(fullfile(parent_dir, 'control'));
    addpath(fullfile(parent_dir, 'verification'));
    addpath(fullfile(parent_dir, 'visualization'));
    addpath(fullfile(parent_dir, 'utils'));
    addpath(fullfile(parent_dir, 'test_UR16e'));

    if nargin < 1 || isempty(urdf_path)
        urdf_path = fullfile(parent_dir, 'test_UR16e', 'universalUR16e.urdf');
    else
        % Resolve relative paths against parent_dir or current working dir
        if ~exist(urdf_path, 'file') && exist(fullfile(parent_dir, urdf_path), 'file')
            urdf_path = fullfile(parent_dir, urdf_path);
        elseif ~exist(urdf_path, 'file') && exist(fullfile(pwd, urdf_path), 'file')
            urdf_path = fullfile(pwd, urdf_path);
        end
    end

    assert(exist(urdf_path, 'file') == 2, ...
           'step0_setup_config:FileNotFound', 'Target URDF file not found: %s', urdf_path);

    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    % Extract base name of the URDF file
    [urdf_dir, urdf_stem, ~] = fileparts(urdf_path);

    % Determine output directory (default: directory of the URDF file)
    output_dir = urdf_dir;
    if isfield(opts, 'output_dir') && ~isempty(opts.output_dir)
        output_dir = opts.output_dir;
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Determine model names and file paths from URDF file name or opts
    dh_model_name = sprintf('dt_%s_dh', urdf_stem);
    if isfield(opts, 'dh_model_name') && ~isempty(opts.dh_model_name)
        dh_model_name = opts.dh_model_name;
    end

    smimport_model_name = sprintf('dt_%s_smimport', urdf_stem);
    if isfield(opts, 'smimport_model_name') && ~isempty(opts.smimport_model_name)
        smimport_model_name = opts.smimport_model_name;
    end

    smimport_sensed_model_name = sprintf('dt_%s_smimport_sensed', urdf_stem);
    if isfield(opts, 'smimport_sensed_model_name') && ~isempty(opts.smimport_sensed_model_name)
        smimport_sensed_model_name = opts.smimport_sensed_model_name;
    end

    dh_model_file              = fullfile(output_dir, [dh_model_name, '.slx']);
    smimport_model_file        = fullfile(output_dir, [smimport_model_name, '.slx']);
    smimport_sensed_model_file = fullfile(output_dir, [smimport_sensed_model_name, '.slx']);

    report_file = fullfile(output_dir, sprintf('%s_verification_report.mat', urdf_stem));
    if isfield(opts, 'report_file') && ~isempty(opts.report_file)
        report_file = opts.report_file;
    end

    % Resolve mesh directory
    mesh_dir = fullfile(urdf_dir, 'meshes');
    if ~exist(mesh_dir, 'dir')
        mesh_dir = fullfile(parent_dir, 'STL_Files');
    end
    if isfield(opts, 'mesh_dir') && ~isempty(opts.mesh_dir)
        mesh_dir = opts.mesh_dir;
    end

    % Parse initial verification tolerances (nominal reach = 1.0 m)
    tol = get_verification_tolerances(opts, 1.0);

    % Assemble configuration struct
    cfg = struct();
    cfg.urdf_path                  = urdf_path;
    cfg.urdf_stem                  = urdf_stem;
    cfg.output_dir                 = output_dir;
    cfg.dh_model_name              = dh_model_name;
    cfg.smimport_model_name        = smimport_model_name;
    cfg.smimport_sensed_model_name = smimport_sensed_model_name;
    cfg.dh_model_file              = dh_model_file;
    cfg.smimport_model_file        = smimport_model_file;
    cfg.smimport_sensed_model_file = smimport_sensed_model_file;
    cfg.report_file                = report_file;
    cfg.mesh_dir                   = mesh_dir;
    cfg.tol                        = tol;
    cfg.opts                       = opts;
    cfg.t_start                    = tic;

    % Print startup banner
    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('       MODULAR ROBOT VERIFICATION WORKFLOW & TRIPLE DIGITAL TWIN    \n');
    fprintf('====================================================================\n');
    fprintf('  URDF Target           : %s\n', cfg.urdf_path);
    fprintf('  Model 1 (DH Multibody): %s\n', cfg.dh_model_file);
    fprintf('  Model 2 (Raw smimport): %s\n', cfg.smimport_model_file);
    fprintf('  Model 3 (smimport+Sens): %s\n', cfg.smimport_sensed_model_file);
    fprintf('  Report Target         : %s\n', cfg.report_file);
    fprintf('  Tolerances     : Pos=%.2e m | Rot=%.2e rad | Vel=%.2e m/s | SeqTau=%.2f N.m\n', ...
            cfg.tol.pos, cfg.tol.rot, cfg.tol.vel, cfg.tol.seq_tau);
    fprintf('====================================================================\n\n');

end
