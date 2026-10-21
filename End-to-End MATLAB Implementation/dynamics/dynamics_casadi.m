function dyn = dynamics_casadi(dh, inertia, chain, opts)
% DYNAMICS_CASADI (Pure MATLAB Alias)
% Forwards seamlessly to dynamics_matlab without requiring CasADi.
    if nargin < 4, opts = struct(); end
    dyn = dynamics_matlab(dh, inertia, chain, opts);
end
