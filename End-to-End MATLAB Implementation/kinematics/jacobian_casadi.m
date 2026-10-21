function jac = jacobian_casadi(fk, dh, chain, opts)
% JACOBIAN_CASADI (Pure MATLAB Alias)
% Forwards seamlessly to jacobian_matlab without requiring CasADi.
    if nargin < 4, opts = struct(); end
    jac = jacobian_matlab(fk, dh, chain, opts);
end
