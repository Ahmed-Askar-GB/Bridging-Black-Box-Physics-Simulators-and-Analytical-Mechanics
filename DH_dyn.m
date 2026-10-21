function [M,C,G,Ro] = DH_dyn(DHpram,P_Ci,q,dq,I,m,g)
    % DH_dyn: Computes the robot's dynamic model using DH parameters and the Lagrange method
    % Inputs:
    % - DHpram: Denavit-Hartenberg parameters (array of size nx4 or nx5 depending on joint type inclusion)
    % - P_Ci: Position vectors of the center of mass of each link in their respective link frames
    % - q: Generalized joint coordinates (symbolic vector of joint positions)
    % - dq: Generalized joint velocities (symbolic vector)
    % - I: Inertia tensor matrices of the links (3x3xN array)
    % - m: Mass of each link (vector of size N)
    % - g: Gravitational acceleration vector (3x1)
    % Outputs:
    % - M: Inertia matrix
    % - C: Coriolis matrix
    % - G: Gravity vector
    % - Ro: Structure containing intermediate computations for debugging
    
    % Transpose P_Ci for proper handling in matrix operations
    P_Ci = transpose(P_Ci); 
    n = length(q); % Number of degrees of freedom

    % Initialize transformation matrices for link 1
    Ro.L(1).A = Ai(DHpram(1,:)); % Compute the first link's transformation matrix
    Ro.L(1).T0i = Ro.L(1).A; % First link's transformation wrt base frame
    
    % Compute forward kinematics for all links
    for i = 2:n
        Ro.L(i).A = Ai(DHpram(i,:)); % Transformation matrix for link i
        Ro.L(i).T0i = Ro.L(i-1).T0i * Ro.L(i).A; % Overall transformation to link i
    end

    % Initialize z-axis vectors for each joint
    z = sym(zeros(3, n)); 
    z(:,1) = [0;0;1]; % Base z-axis each column of z is z_i-1
    
    for i = 2:n
        z(:,i) = Ro.L(i-1).T0i(1:3,3); % Extract z-axis from previous link's transformation
    end

    % Compute position of the center of mass for each link relative to frame (i-1)
    Ro.c(1).p = Ro.L(1).A(1:3,1:3) * P_Ci(:,1); % Center of mass for the first link
    for i = 2:n
        Ro.c(i).p = sym('p', [3 i]); % Initialize symbolic position for CoM
        Ro.c(i).p(:,1) = Ro.L(i-1).T0i(1:3,4) + Ro.L(i).T0i(1:3,1:3) * P_Ci(:,i); 
        % Additional relative position calculations
        for j = 2:i
            Ro.c(i).p(:,j) = Ro.c(i).p(:,1) - Ro.L(j-1).T0i(1:3,4); 
        end
    end

    % Compute Jacobian matrices for each link
    for i = 1:n
        Ro.L(i).jv = sym(zeros(3, n)); % Linear velocity Jacobian initialization
        Ro.L(i).jw = sym(zeros(3, n)); % Angular velocity Jacobian initialization
        for j = 1:n
            if j <= i
                if DHpram(j,5) == 1 % Revolute joint
                    Ro.L(i).jv(:,j) = cross(z(:,j), Ro.c(i).p(:,j)); % Compute translational Jacobian
                    Ro.L(i).jw(:,j) = z(:,j); % Compute rotational Jacobian
                else % Prismatic joint
                    Ro.L(i).jv(:,j) = z(:,j); % Prismatic contribution to Jacobian
                end
            end
        end
    end

    % Compute the Gravity vector G
    G = sym(zeros(n,1)); % Initialize the gravity vector
    for i = 1:n
        for j = 1:n
            G(i) = G(i) - m(j) * transpose(g) * Ro.L(j).jv(:,i); % Contribution of each link to gravity
        end
    end

    % Compute the Inertia matrix M
    M = sym(zeros(n,n)); % Initialize the inertia matrix
    for i = 1:n
        I0i = Ro.L(i).T0i(1:3,1:3) * I(:,:,i) * transpose(Ro.L(i).T0i(1:3,1:3)); % Transform inertia tensor
        M = M + m(i)*(transpose(Ro.L(i).jv)*Ro.L(i).jv) + ...
            transpose(Ro.L(i).jw)*I0i*Ro.L(i).jw; % Summation for Inertia matrix
    end

    % Compute the Coriolis matrix C
    C = sym(zeros(n,n)); % Initialize the Coriolis matrix
    for i = 1:n
        for j = 1:n
            Cold = 0; % Temporary variable for calculation
            for k = 1:n
                C(i,j) = Cold + (diff(M(i,j),q(k)) + diff(M(i,k),q(j)) - diff(M(k,j),q(i))) * dq(k) / 2;
                Cold = C(i,j); % Store current value
            end
        end
    end
end

% Auxiliary function to calculate individual transformation matrices based on DH parameters
function A = Ai(Dh)
    % Inputs:
    % - Dh: A row vector containing the DH parameters [a, alpha, theta, d]
    ai = Dh(1); alphai = Dh(2); si = Dh(3); di = Dh(4);
    % Compute the transformation matrix
    A = [cos(si), -sin(si)*cos(alphai), sin(si)*sin(alphai), ai*cos(si); 
         sin(si), cos(si)*cos(alphai), -cos(si)*sin(alphai), ai*sin(si); 
         0, sin(alphai), cos(alphai), di; 
         0, 0, 0, 1];
end