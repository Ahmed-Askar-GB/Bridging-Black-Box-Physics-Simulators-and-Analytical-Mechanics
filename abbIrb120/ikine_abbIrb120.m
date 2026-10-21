function q = ikine_abbIrb120(T_target, q_guess)
    % Numerical Inverse Kinematics using Gauss-Newton Method
    max_iter = 100;
    tol = 1e-6;
    q = q_guess(:);

    for i = 1:max_iter
        T_curr = fkine_abbIrb120(q);
        
        % Compute Position Error
        dp = T_target(1:3, 4) - T_curr(1:3, 4);
        
        % Compute Orientation Error using Skew-Symmetric Cross-Product Method
        dR = T_target(1:3, 1:3) * T_curr(1:3, 1:3)';
        do = 0.5 * [dR(3,2)-dR(2,3); dR(1,3)-dR(3,1); dR(2,1)-dR(1,2)];
        
        error_vec = [dp; do];
        if norm(error_vec) < tol
            return;
        end
        
        % Compute Update step via Pseudo-Inverse of the Jacobian
        J = jacobian_abbIrb120(q);
        q = q + pinv(J) * error_vec;
    end
    warning('IK solver reached maximum iterations without fully converging.');
end
