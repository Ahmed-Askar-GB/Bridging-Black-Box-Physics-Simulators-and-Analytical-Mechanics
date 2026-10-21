function model = casadi_compile(fk, jac, dyn, robot_name, out_dir, opts)
% CASADI_COMPILE (Pure MATLAB Alias)
% Forwards seamlessly to matlab_compile without requiring CasADi.
    if nargin < 5, out_dir = '.'; end
    if nargin < 6, opts = struct(); end
    model = matlab_compile(fk, jac, dyn, robot_name, out_dir, opts);
end
