function model = casadi_compile(fk, jac, dyn, robot_name, out_dir, opts)
% CASADI_COMPILE  Bundle and persist the CasADi model structs for a robot.
%
% After building the FK, Jacobian, and Dynamics CasADi Function objects via
% fk_casadi, jacobian_casadi, and dynamics_casadi, this function packages them
% into a single 'model' struct and saves it to disk as a .mat file.
%
% This enables:
%   - Reloading the model in subsequent MATLAB sessions without re-deriving
%     the symbolic expressions (which can take 10–60 s for a 6-DOF robot).
%   - Sharing the model across different scripts (verification, simulation, etc.)
%   - Optionally generating C source code for the core dynamics functions.
%
% =========================================================================
% INPUTS:
%   fk          (struct)  Output of fk_casadi().       Contains all FK symbols
%                          and compiled CasADi Function objects.
%   jac         (struct)  Output of jacobian_casadi().  Contains Jacobian symbols
%                          and compiled Function objects.
%   dyn         (struct)  Output of dynamics_casadi().  Contains M/C/G symbols
%                          and compiled Function objects.
%   robot_name  (char)    Short robot identifier string (e.g. 'UR5', 'KUKA_iiwa').
%                          Used as the filename stem: <robot_name>_model.mat
%   out_dir     (char)    Absolute or relative path to the output directory.
%                          The directory is created if it does not exist.
%   opts        (struct)  [optional] Options:
%     .gen_c      (logical) If true, generate C code for dynamics functions.
%                           Requires CasADi's CodeGenerator backend. Default: false.
%     .codegen_dir (char)  Directory for C code output. Default: out_dir/codegen/
%     .verbose    (logical) Print summary to console. Default: false.
%
% =========================================================================
% OUTPUT:
%   model  (struct) containing:
%     .robot_name  (char)    Robot identifier
%     .timestamp   (char)    ISO 8601 creation timestamp
%     .fk          (struct)  Forward kinematics functions and symbols
%     .jac         (struct)  Jacobian functions and symbols
%     .dyn         (struct)  Dynamics functions and symbols
%     .mat_file    (char)    Full path to the saved .mat file
%
% =========================================================================
% USAGE:
%   fk    = fk_casadi(dh, chain);
%   jac   = jacobian_casadi(fk, dh, chain);
%   dyn   = dynamics_casadi(dh, inertia, chain);
%
%   % Save the compiled model for reuse:
%   out   = fullfile(pwd, 'compiled');
%   model = casadi_compile(fk, jac, dyn, 'UR5', out);
%
%   % In a new session, reload:
%   load(model.mat_file, 'model');
%   G_num = full(model.dyn.G_fun(q_test));
%
% =========================================================================
% NOTES:
%   - CasADi Function objects are serializable and survive save/load.
%     They are JIT-compiled C graphs — loading them does NOT require
%     re-deriving symbolic expressions.
%   - Large symbolic expressions (N=7+ DOF) may produce large .mat files.
%     The .mat file stores the compiled binary graph, not the SX expressions.
%   - The SX symbolic fields (fk.q, dyn.M_sym, etc.) are also saved and can
%     be used for further symbolic manipulation after loading.
%
% =========================================================================
% CALLED BY:
%   run_pipeline.m   (Phase 2 → Phase 3 handoff)
%   verify_fk.m      (loads model.fk)
%   verify_jacobian.m (loads model.jac)
%   verify_dynamics_sequential.m (loads model.dyn)

    % -----------------------------------------------------------------------
    % 0. Parse optional arguments
    % -----------------------------------------------------------------------
    if nargin < 6 || isempty(opts)
        opts = struct();
    end
    verbose     = isfield(opts, 'verbose')    && opts.verbose;
    gen_c       = isfield(opts, 'gen_c')      && opts.gen_c;
    codegen_dir = '';
    if isfield(opts, 'codegen_dir') && ~isempty(opts.codegen_dir)
        codegen_dir = opts.codegen_dir;
    end

    % -----------------------------------------------------------------------
    % 1. Validate inputs
    % -----------------------------------------------------------------------
    % Check that the core structs have the expected Function fields.
    assert(isfield(fk,  'fk_ee_fun'),   'fk struct missing fk_ee_fun — run fk_casadi first.');
    assert(isfield(jac, 'J_fun'),       'jac struct missing J_fun — run jacobian_casadi first.');
    assert(isfield(dyn, 'M_fun'),       'dyn struct missing M_fun — run dynamics_casadi first.');
    assert(ischar(robot_name) && ~isempty(robot_name), 'robot_name must be a non-empty string.');
    assert(ischar(out_dir)    && ~isempty(out_dir),    'out_dir must be a non-empty string.');

    % -----------------------------------------------------------------------
    % 2. Create output directory if it does not exist
    % -----------------------------------------------------------------------
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);   % create all intermediate directories as needed
        if verbose
            fprintf('[casadi_compile] Created output directory: %s\n', out_dir);
        end
    end

    % -----------------------------------------------------------------------
    % 3. Assemble the model struct
    % -----------------------------------------------------------------------
    % Timestamp is added to allow tracing which pipeline run produced the model.
    model.robot_name = robot_name;
    model.timestamp  = datestr(now, 'yyyy-mm-ddTHH:MM:SS');  % ISO 8601-like format
    model.fk         = fk;    % full FK struct (symbols + Function objects)
    model.jac        = jac;   % full Jacobian struct
    model.dyn        = dyn;   % full Dynamics struct

    % -----------------------------------------------------------------------
    % 4. Save model to .mat file
    % -----------------------------------------------------------------------
    % The filename encodes the robot name for easy identification.
    mat_filename = fullfile(out_dir, [robot_name '_model.mat']);
    model.mat_file = mat_filename;

    % Save with -v7.3 (HDF5-based) to support large variable sizes.
    % CasADi Function objects serialize correctly with -v7.3.
    % NOTE: Some MATLAB versions may emit a benign SWIG object warning during
    % save; this is cosmetic only and does not impact deserialization.
    save(mat_filename, 'model', '-v7.3');

    if verbose
        mat_info = dir(mat_filename);
        fprintf('[casadi_compile] Model saved: %s  (%.1f MB)\n', ...
                mat_filename, mat_info.bytes / 1e6);
    end

    % -----------------------------------------------------------------------
    % 5. Optional: generate C code for the core dynamics functions
    % -----------------------------------------------------------------------
    % CasADi can export any Function to a self-contained C source file.
    % The generated .c file compiles without any CasADi runtime dependency,
    % making it suitable for embedded deployment (e.g., on a real-time OS).
    %
    % Generated functions (if gen_c = true):
    %   tau_fun.c  — Inverse dynamics: tau = M*ddq + C*dq + G
    %   M_fun.c    — Inertia matrix M(q)
    %   G_fun.c    — Gravity vector G(q)
    % -----------------------------------------------------------------------
    if gen_c
        import casadi.*   % needed for CodeGenerator

        % Default codegen directory: <out_dir>/codegen/
        if isempty(codegen_dir)
            codegen_dir = fullfile(out_dir, 'codegen');
        end
        if ~exist(codegen_dir, 'dir')
            mkdir(codegen_dir);
        end

        % CasADi code generation options
        cg_opts = struct('with_header', true, ...  % generate .h file alongside .c
                         'verbose',     false);     % suppress CasADi internal messages

        robot_id = regexprep(robot_name, '[^a-zA-Z0-9_]', '_');
        orig_dir = pwd;
        cd(codegen_dir);
        try
            % Export inverse dynamics (most commonly needed for control)
            dyn.tau_fun.generate([robot_id '_tau_fun.c'], cg_opts);

            % Export inertia matrix (needed for forward dynamics integrators)
            dyn.M_fun.generate([robot_id '_M_fun.c'], cg_opts);

            % Export gravity vector (needed for gravity-compensation controllers)
            dyn.G_fun.generate([robot_id '_G_fun.c'], cg_opts);
        catch ME
            cd(orig_dir);
            rethrow(ME);
        end
        cd(orig_dir);

        if verbose
            fprintf('[casadi_compile] C code generated in: %s\n', codegen_dir);
            fprintf('  tau_fun.c, M_fun.c, G_fun.c\n');
        end

        % Record codegen output in model struct
        model.codegen_dir = codegen_dir;
    end

    % -----------------------------------------------------------------------
    % 6. Verbose summary
    % -----------------------------------------------------------------------
    if verbose
        fprintf('\n[casadi_compile] Model compilation complete.\n');
        fprintf('  Robot      : %s\n',   model.robot_name);
        fprintf('  Timestamp  : %s\n',   model.timestamp);
        fprintf('  Saved to   : %s\n',   model.mat_file);
        fprintf('  FK funs    : fk_fun, fk_ee_fun, fk_link_fun{1..%d}\n', ...
                length(fk.fk_link_fun));
        fprintf('  Jac funs   : J_fun, Jv_fun, Jw_fun, Jdot_fun\n');
        fprintf('  Dyn funs   : M_fun, C_fun, G_fun, tau_fun, forward_dyn_fun\n');
        if gen_c
            fprintf('  C codegen  : %s\n', model.codegen_dir);
        end
    end

end % function casadi_compile
