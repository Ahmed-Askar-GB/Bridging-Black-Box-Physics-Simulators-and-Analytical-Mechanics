function fk = fk_casadi(dh, chain, opts)
% FK_CASADI (Pure MATLAB Alias)
% Forwards seamlessly to fk_matlab without requiring CasADi.
    if nargin < 3, opts = struct(); end
    fk = fk_matlab(dh, chain, opts);
end
