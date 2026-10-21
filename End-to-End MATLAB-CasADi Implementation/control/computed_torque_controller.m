classdef computed_torque_controller < handle
% COMPUTED_TORQUE_CONTROLLER  Inverse dynamics feedback linearization controller.
%
% Implements the classical computed torque law:
%   tau = M(q)*(ddq_d + Kd*(dq_d - dq) + Kp*(q_d - q)) + C(q, dq)*dq + G(q)
%
% Transforms seamlessly between URDF actuation frame and analytical DH frame
% using the autonomous joint_signs vector:
%   q_dh   = joint_signs .* q_urdf
%   dq_dh  = joint_signs .* dq_urdf
%   tau_urdf = joint_signs .* tau_dh
%
% =========================================================================
% USAGE:
%   % 1. Object-Oriented usage:
%   ctrl = computed_torque_controller(dyn, Kp, Kd, joint_signs);
%   tau  = ctrl.compute(q, dq, q_d, dq_d, ddq_d);
%
%   % 2. Static / Functional usage (e.g. inside Simulink MATLAB Function block):
%   tau = computed_torque_controller.compute_torque(dyn, q, dq, q_d, dq_d, ddq_d, Kp, Kd, joint_signs);
% =========================================================================

    properties
        dyn
        Kp
        Kd
        joint_signs
        n
    end

    methods
        function obj = computed_torque_controller(dyn, Kp, Kd, joint_signs)
            % CONSTRUCTOR
            if nargin < 1 || isempty(dyn)
                error('computed_torque_controller:MissingDyn', 'A compiled CasADi dynamics struct is required.');
            end
            obj.dyn = dyn;

            % Determine DOF n from dyn
            if isfield(dyn, 'n')
                obj.n = dyn.n;
            elseif isprop(dyn, 'n')
                obj.n = dyn.n;
            else
                % Infer from tau_fun input dimension
                obj.n = dyn.tau_fun.size1_in(0);
            end

            % Setup Kp matrix
            if nargin < 2 || isempty(Kp)
                obj.Kp = 100.0 * eye(obj.n);
            elseif isscalar(Kp)
                obj.Kp = double(Kp) * eye(obj.n);
            elseif isvector(Kp)
                obj.Kp = diag(double(Kp(:)));
            else
                obj.Kp = double(Kp);
            end

            % Setup Kd matrix
            if nargin < 3 || isempty(Kd)
                obj.Kd = 20.0 * eye(obj.n);
            elseif isscalar(Kd)
                obj.Kd = double(Kd) * eye(obj.n);
            elseif isvector(Kd)
                obj.Kd = diag(double(Kd(:)));
            else
                obj.Kd = double(Kd);
            end

            % Setup joint_signs
            if nargin < 4 || isempty(joint_signs)
                obj.joint_signs = ones(obj.n, 1);
            else
                obj.joint_signs = double(joint_signs(:));
            end
        end

        function tau = compute(obj, q, dq, q_d, dq_d, ddq_d)
            % COMPUTE  Compute control torque command given measured and desired states.
            tau = computed_torque_controller.compute_torque(...
                obj.dyn, q, dq, q_d, dq_d, ddq_d, obj.Kp, obj.Kd, obj.joint_signs);
        end
    end

    methods (Static)
        function tau_urdf = compute_torque(dyn, q, dq, q_d, dq_d, ddq_d, Kp, Kd, joint_signs)
            % COMPUTE_TORQUE  Static function evaluating computed torque control.
            q     = double(q(:));
            dq    = double(dq(:));
            q_d   = double(q_d(:));
            dq_d  = double(dq_d(:));
            ddq_d = double(ddq_d(:));
            n     = length(q);

            if nargin < 7 || isempty(Kp)
                Kp = 100.0 * eye(n);
            elseif isscalar(Kp)
                Kp = double(Kp) * eye(n);
            elseif isvector(Kp) && ~isequal(size(Kp), [n, n])
                Kp = diag(double(Kp(:)));
            end

            if nargin < 8 || isempty(Kd)
                Kd = 20.0 * eye(n);
            elseif isscalar(Kd)
                Kd = double(Kd) * eye(n);
            elseif isvector(Kd) && ~isequal(size(Kd), [n, n])
                Kd = diag(double(Kd(:)));
            end

            if nargin < 9 || isempty(joint_signs)
                signs = ones(n, 1);
            else
                signs = double(joint_signs(:));
            end

            % Tracking errors in URDF coordinates
            e  = q_d - q;
            de = dq_d - dq;

            % Desired virtual acceleration in URDF coordinates
            v_acc = ddq_d + Kp * e + Kd * de;

            % Map states and virtual acceleration to analytical DH frames
            q_dh     = signs .* q;
            dq_dh    = signs .* dq;
            v_acc_dh = signs .* v_acc;

            % Evaluate analytical inverse dynamics in DH coordinates
            if isfield(dyn, 'tau_fun')
                tau_dh = full(dyn.tau_fun(q_dh, dq_dh, v_acc_dh));
            else
                M = full(dyn.M_fun(q_dh));
                C = full(dyn.C_fun(q_dh, dq_dh));
                G = full(dyn.G_fun(q_dh));
                tau_dh = M * v_acc_dh + C * dq_dh + G;
            end

            % Map torque back into URDF actuation coordinates
            tau_urdf = signs .* tau_dh;
        end
    end
end
