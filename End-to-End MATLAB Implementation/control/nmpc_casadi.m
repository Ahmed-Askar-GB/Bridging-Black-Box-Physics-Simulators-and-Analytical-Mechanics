classdef nmpc_casadi < nmpc_matlab
% NMPC_CASADI (Pure MATLAB Alias)
% Inherits seamlessly from nmpc_matlab without requiring CasADi.
    methods
        function obj = nmpc_casadi(dyn, opts)
            if nargin < 2, opts = struct(); end
            obj@nmpc_matlab(dyn, opts);
        end
    end
end
