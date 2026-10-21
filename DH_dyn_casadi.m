function [M, C, G, Ro, fns] = DH_dyn_casadi(DHpram, P_Ci, q, dq, I, m, g, ddq)
    % DH_dyn_casadi: Computes robot dynamic model using standard DH parameters
    % and Lagrangian formulation via CasADi (algorithmic differentiation)
    % instead of MATLAB Symbolic Math Toolbox.
    %
    % Inputs:
    %   - DHpram: Denavit-Hartenberg parameters table (n x 4 or n x 5 matrix):
    %             [a, alpha, theta, d] or [a, alpha, theta, d, sigma]
    %             sigma: 1 for revolute (default), 0 for prismatic
    %             If numeric, joint coordinate q(i) is added automatically to
    %             theta (revolute) or d (prismatic). If CasADi SX expressions
    %             already containing q are passed, they are used directly.
    %   - P_Ci  : Position vectors of link CoMs in local link frames (n x 3 or 3 x n)
    %   - q     : Joint coordinates (n x 1 CasADi SX, optional)
    %   - dq    : Joint velocities (n x 1 CasADi SX, optional)
    %   - I     : Link inertia tensors (3 x 3 x n numeric array or cell array of 3 x 3 matrices)
    %   - m     : Link masses (vector of length n)
    %   - g     : Gravitational acceleration vector (3 x 1, default [0; 0; -9.81])
    %   - ddq   : Joint accelerations (n x 1 CasADi SX, optional, for tau computation)
    %
    % Outputs:
    %   - M     : Inertia / mass matrix (n x n CasADi SX)
    %   - C     : Coriolis and centrifugal matrix (n x n CasADi SX, Christoffel formulation)
    %   - G     : Gravity torque vector (n x 1 CasADi SX)
    %   - Ro    : Structure containing intermediate kinematic transforms and Jacobians
    %   - fns   : Structure containing compiled CasADi Function objects:
    %             fns.M_fun(q)
    %             fns.C_fun(q, dq)
    %             fns.G_fun(q)
    %             fns.tau_fun(q, dq, ddq)
    %             fns.forward_dynamics(q, dq, tau)
    %             fns.generate_c(filename)
    
    import casadi.*

    % Determine degrees of freedom n
    n = size(DHpram, 1);

    % Parse joint types (sigma: 1 = revolute, 0 = prismatic)
    if size(DHpram, 2) >= 5
        sigma = DHpram(:, 5);
    else
        sigma = ones(n, 1);
    end

    % Manage symbolic joint variables q and dq
    if nargin < 3 || isempty(q)
        q = SX.sym('q', n, 1);
    elseif ~isa(q, 'casadi.SX')
        q = SX(q);
    end

    if nargin < 4 || isempty(dq)
        dq = SX.sym('dq', n, 1);
    elseif ~isa(dq, 'casadi.SX')
        dq = SX(dq);
    end

    if nargin < 8 || isempty(ddq)
        ddq = SX.sym('ddq', n, 1);
    elseif ~isa(ddq, 'casadi.SX')
        ddq = SX(ddq);
    end

    % Manage gravity vector g
    if nargin < 7 || isempty(g)
        g = [0; 0; -9.81];
    end
    if ~isa(g, 'casadi.SX')
        g = SX(g);
    end

    % Handle CoM positions P_Ci (ensure 3 x n)
    if size(P_Ci, 1) == n && size(P_Ci, 2) == 3
        P_Ci = transpose(P_Ci);
    end
    if ~isa(P_Ci, 'casadi.SX')
        P_Ci = SX(P_Ci);
    end

    % Build symbolic DH table if q is not already embedded
    DH_sym = SX.zeros(n, 4);
    for i = 1:n
        ai = DHpram(i, 1);
        alphai = DHpram(i, 2);
        theta_val = DHpram(i, 3);
        d_val = DHpram(i, 4);

        if isa(DHpram, 'casadi.SX') && (depends_on(theta_val, q(i)) || depends_on(d_val, q(i)))
            % Coordinate q(i) is already embedded in DH table
            si = theta_val;
            di = d_val;
        else
            % Coordinate q(i) needs to be added to variable parameter
            if sigma(i) == 1
                si = theta_val + q(i);
                di = d_val;
            else
                si = theta_val;
                di = d_val + q(i);
            end
        end
        DH_sym(i, 1) = ai;
        DH_sym(i, 2) = alphai;
        DH_sym(i, 3) = si;
        DH_sym(i, 4) = di;
    end

    % Forward kinematics: individual and cumulative transformations
    Ro.L = struct('A', cell(1, n), 'T0i', cell(1, n), 'jv', cell(1, n), 'jw', cell(1, n));
    Ro.L(1).A = Ai_casadi(DH_sym(1, :));
    Ro.L(1).T0i = Ro.L(1).A;

    for i = 2:n
        Ro.L(i).A = Ai_casadi(DH_sym(i, :));
        Ro.L(i).T0i = Ro.L(i-1).T0i * Ro.L(i).A;
    end

    % Joint actuation axes z (z_{j-1} expressed in base frame 0)
    z = SX.zeros(3, n);
    z(:, 1) = [0; 0; 1];
    for i = 2:n
        z(:, i) = Ro.L(i-1).T0i(1:3, 3);
    end
    Ro.z = z;

    % Center of mass positions relative to preceding origins
    Ro.c = struct('p', cell(1, n));
    Ro.c(1).p = SX.zeros(3, 1);
    Ro.c(1).p(:, 1) = Ro.L(1).A(1:3, 1:3) * P_Ci(:, 1);

    for i = 2:n
        Ro.c(i).p = SX.zeros(3, i);
        Ro.c(i).p(:, 1) = Ro.L(i-1).T0i(1:3, 4) + Ro.L(i).T0i(1:3, 1:3) * P_Ci(:, i);
        for j = 2:i
            Ro.c(i).p(:, j) = Ro.c(i).p(:, 1) - Ro.L(j-1).T0i(1:3, 4);
        end
    end

    % Translational and rotational Jacobians for each link CoM
    for i = 1:n
        Ro.L(i).jv = SX.zeros(3, n);
        Ro.L(i).jw = SX.zeros(3, n);
        for j = 1:n
            if j <= i
                if sigma(j) == 1
                    % Revolute joint: jv = z_{j-1} x r_{j-1, Ci}, jw = z_{j-1}
                    Ro.L(i).jv(:, j) = cross(z(:, j), Ro.c(i).p(:, j));
                    Ro.L(i).jw(:, j) = z(:, j);
                else
                    % Prismatic joint: jv = z_{j-1}, jw = 0
                    Ro.L(i).jv(:, j) = z(:, j);
                    Ro.L(i).jw(:, j) = SX.zeros(3, 1);
                end
            end
        end
    end

    % Gravity torque vector G
    G = SX.zeros(n, 1);
    for i = 1:n
        for j = 1:n
            G(i) = G(i) - m(j) * (g' * Ro.L(j).jv(:, i));
        end
    end

    % Generalized inertia matrix M
    M = SX.zeros(n, n);
    for i = 1:n
        R_i = Ro.L(i).T0i(1:3, 1:3);
        if iscell(I)
            I_i = SX(I{i});
        elseif ndims(I) == 3
            I_i = SX(I(:, :, i));
        else
            I_i = SX(I);
        end
        I0i = R_i * I_i * R_i';
        M = M + m(i) * (Ro.L(i).jv' * Ro.L(i).jv) + Ro.L(i).jw' * I0i * Ro.L(i).jw;
    end

    % Coriolis and centrifugal matrix C via Christoffel symbols of the first kind
    % c_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{kj}/dq_i)
    % C_{ij} = sum_k c_{ijk} * dq_k
    C = SX.zeros(n, n);
    dM = jacobian(M(:), q); % size (n^2) x n
    for i = 1:n
        for j = 1:n
            c_ij = SX(0);
            idx_ij = (j - 1) * n + i;
            for k = 1:n
                idx_ik = (k - 1) * n + i;
                idx_kj = (j - 1) * n + k;
                dM_ijk = dM(idx_ij, k);
                dM_ikj = dM(idx_ik, j);
                dM_kji = dM(idx_kj, i);
                c_ijk = 0.5 * (dM_ijk + dM_ikj - dM_kji);
                c_ij = c_ij + c_ijk * dq(k);
            end
            C(i, j) = c_ij;
        end
    end

    % Total joint torque expression
    tau = M * ddq + C * dq + G;

    % Build compiled CasADi Function objects
    fns = struct();
    fns.M_fun = Function('M_fun', {q}, {M}, {'q'}, {'M'});
    fns.C_fun = Function('C_fun', {q, dq}, {C}, {'q', 'dq'}, {'C'});
    fns.G_fun = Function('G_fun', {q}, {G}, {'q'}, {'G'});
    fns.tau_fun = Function('tau_fun', {q, dq, ddq}, {tau}, {'q', 'dq', 'ddq'}, {'tau'});
    
    tau_input = SX.sym('tau_input', n, 1);
    ddq_forward = solve(M, tau_input - C * dq - G);
    fns.forward_dynamics = Function('forward_dynamics', {q, dq, tau_input}, {ddq_forward}, {'q', 'dq', 'tau'}, {'ddq'});

    % Method to generate C source code for embedded deployment
    fns.generate_c = @(filename) fns.tau_fun.generate(filename);
end

% Local helper to compute DH homogeneous transformation matrix via CasADi SX
function A = Ai_casadi(Dh)
    ai = Dh(1);
    alphai = Dh(2);
    si = Dh(3);
    di = Dh(4);

    A = [cos(si), -sin(si) * cos(alphai),  sin(si) * sin(alphai), ai * cos(si);
         sin(si),  cos(si) * cos(alphai), -cos(si) * sin(alphai), ai * sin(si);
         0,        sin(alphai),            cos(alphai),           di;
         0,        0,                      0,                     1];
end
