function model = matlab_compile(fk, jac, dyn, robot_name, out_dir, opts)
% MATLAB_COMPILE  Bundle and persist pure MATLAB robot models.
%
% Saves compiled function handles and metadata to <robot_name>_model.mat.
% Completely independent of CasADi.
%
% =========================================================================
% INPUTS:
%   fk         (struct) Output from fk_matlab()
%   jac        (struct) Output from jacobian_matlab()
%   dyn        (struct) Output from dynamics_matlab()
%   robot_name (char)   Identifier string (e.g. 'UR16e')
%   out_dir    (char)   Output directory
%   opts       (struct) [optional]
%
% OUTPUT:
%   model      (struct) Compiled model struct
% =========================================================================

    if nargin < 5 || isempty(out_dir)
        out_dir = '.';
    end
    if nargin < 6 || isempty(opts)
        opts = struct();
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    model = struct();
    model.robot_name = robot_name;
    model.timestamp  = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
    model.fk         = fk;
    model.jac        = jac;
    model.dyn        = dyn;

    mat_filepath = fullfile(out_dir, sprintf('%s_model.mat', robot_name));
    save(mat_filepath, 'model');
    model.mat_file = mat_filepath;

    if isfield(opts, 'verbose') && opts.verbose
        fprintf('  [SAVED] Pure MATLAB model persisted to: %s\n', mat_filepath);
    end
end
