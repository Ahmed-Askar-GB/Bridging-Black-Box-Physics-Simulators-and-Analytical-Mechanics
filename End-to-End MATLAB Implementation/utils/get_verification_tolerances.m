function tol = get_verification_tolerances(opts, L_reach)
% GET_VERIFICATION_TOLERANCES  Centralized tolerance management for verification suite.
%
% Extracts, validates, and sets intelligent defaults for all kinematic, dynamic,
% and numerical verification tolerances across the framework.
%
% Allows callers (main scripts, test suites, command line) to flexibly specify
% tolerances via either top-level fields (opts.tol_pos) or nested structs
% (opts.tolerances.pos or opts.tol.pos).
%
% =========================================================================
% TOLERANCES MANAGED:
%   tol.pos       (m)       Position error tolerance (EE translation)
%   tol.rot       (rad)     Orientation error tolerance (geodesic distance)
%   tol.vel       (m/s)     Linear velocity absolute floor tolerance
%   tol.omega     (rad/s)   Angular velocity absolute floor tolerance
%   tol.vel_rel   (-)       Relative velocity scaling coefficient (e.g. 0.05 = 5%)
%   tol.tau       (N*m)     Simscape vs CasADi general torque tolerance
%   tol.seq_tau   (N*m)     Sequential dynamics cross-validation torque tolerance
%   tol.mass      (kg)      Link mass cross-check tolerance
%   tol.sym       (-)       Mass matrix symmetry residual (|M - M^T|)
%   tol.gravity   (N*m)     Static balance invariant residual (|tau(q,0,0) - G(q)|)
%   tol.t_dur     (s)       Simulation duration
%
% =========================================================================
% USAGE:
%   tol = get_verification_tolerances();
%   tol = get_verification_tolerances(opts);
%   tol = get_verification_tolerances(opts, L_reach);
%
% =========================================================================

    if nargin < 1 || isempty(opts)
        opts = struct();
    end

    if nargin < 2 || isempty(L_reach) || L_reach <= 0
        L_reach = 1.0; % standard nominal 1 meter reach if unknown
    end

    % Normalize nested tolerance structs if provided
    t_in = struct();
    if isfield(opts, 'tolerances') && isstruct(opts.tolerances)
        t_in = opts.tolerances;
    elseif isfield(opts, 'tol') && isstruct(opts.tol)
        t_in = opts.tol;
    end

    % Helper for multi-alias lookup
    function val = get_val(alias_cell, default_val)
        val = default_val;
        % Check in opts directly
        for k = 1:length(alias_cell)
            field = alias_cell{k};
            if isfield(opts, field) && ~isempty(opts.(field))
                val = opts.(field);
                return;
            end
        end
        % Check in nested struct
        for k = 1:length(alias_cell)
            field = alias_cell{k};
            % strip 'tol_' prefix if searching nested struct
            short_field = regexprep(field, '^tol_', '');
            if isfield(t_in, short_field) && ~isempty(t_in.(short_field))
                val = t_in.(short_field);
                return;
            elseif isfield(t_in, field) && ~isempty(t_in.(field))
                val = t_in.(field);
                return;
            end
        end
    end

    % Default scale-aware position tolerance: max(5 mm, 2% of total reach)
    default_tol_pos = max(5e-3, 0.02 * L_reach);

    tol = struct();
    tol.pos       = get_val({'tol_pos', 'pos', 'position'}, default_tol_pos);
    tol.rot       = get_val({'tol_rot', 'rot', 'rotation', 'orientation'}, 0.03);      % ~1.7 deg
    tol.vel       = get_val({'tol_vel', 'vel', 'velocity', 'lin_vel'}, 0.25);         % 0.25 m/s
    tol.omega     = get_val({'tol_omega', 'omega', 'ang_vel'}, 0.25);                  % 0.25 rad/s
    tol.vel_rel   = get_val({'tol_vel_rel', 'vel_rel', 'rel_vel'}, 0.05);              % 5% relative
    tol.tau       = get_val({'tol_tau', 'tau', 'torque'}, 0.50);                       % 0.50 N.m
    tol.seq_tau   = get_val({'tol_seq_tau', 'seq_tau', 'sequential_tau'}, 5.0);       % 5.0 N.m
    tol.mass      = get_val({'tol_mass', 'mass'}, 1e-6);                               % 1 mg
    tol.sym       = get_val({'tol_sym', 'sym', 'symmetry'}, 1e-11);                    % matrix symmetry
    tol.gravity   = get_val({'tol_gravity', 'gravity', 'static_balance'}, 1e-11);      % static balance
    tol.t_dur     = get_val({'t_dur', 'duration', 'sim_duration'}, 2.5);              % 2.5 seconds

    % Validate positive numbers
    fnames = fieldnames(tol);
    for idx = 1:length(fnames)
        fn = fnames{idx};
        assert(isnumeric(tol.(fn)) && isscalar(tol.(fn)) && tol.(fn) > 0, ...
               'get_verification_tolerances:InvalidValue', ...
               'Tolerance parameter "%s" must be a positive scalar number (got: %s).', ...
               fn, mat2str(tol.(fn)));
    end

end
