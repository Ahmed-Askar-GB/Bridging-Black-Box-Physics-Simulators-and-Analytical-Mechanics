function inertia = urdf_extract_inertia(robot, chain, dh, verbose)
% URDF_EXTRACT_INERTIA  Transform URDF inertial data into standard DH frames.
%
% This is a pure functional refactor of the inertia-mapping logic inside
% GenerateDhParams.calculate_dh_and_dynamics() (My Implementation, lines 557–591).
%
% Each link's inertial properties (mass, CoM, inertia tensor) are defined
% in the URDF link frame.  For use in the Lagrangian dynamic equations
% (DH_dyn.m / DH_dyn_casadi.m), they must be re-expressed in the
% STANDARD DH FRAME of the corresponding link.
%
% =========================================================================
% FRAME CONVENTIONS (critical — see detailed explanation below):
%
%   URDF link frame {L_i}:
%     Defined by T_urdf{i} = T_0_to_L_i at q=0.
%     This is the frame in which the URDF <inertial> properties are given.
%
%   DH frame {F_i}:
%     Defined by T_dh{i} = A_1*A_2*...*A_i at q=0.
%     This is the frame expected by DH_dyn.m for dynamic computation.
%
%   The transformation FROM {L_i} TO {F_i}:
%     R_rel = R_dh_i' * R_urdf_i
%     (where R_dh_i and R_urdf_i are the rotational parts of T_dh{i} and T_urdf{i})
%
% =========================================================================
% CENTER OF MASS (P_Ci) FRAME CONVENTION:
%
%   P_Ci is the CoM of link i expressed as follows:
%     • Measured FROM: the origin of the DH frame {F_{i-1}} (the PREVIOUS DH frame)
%     • Expressed IN:  the orientation of the DH frame {F_i}  (the CURRENT DH frame)
%
%   In code:
%     p_com_global = T_urdf{i} * [com_urdf; 1]   (CoM in world frame)
%     O_dh_prev    = T_dh{i-1}(1:3,4)            (previous DH origin)
%     P_Ci(i,:)    = R_dh_i' * (p_com_global - O_dh_prev)
%
%   This convention EXACTLY matches GenerateDhParams.m lines 579-581:
%     obj.P_Ci(i,:) = (R_curr_dh' * (p_com_global - o_prev_dh))';
%
%   Why this convention?  In the Lagrangian formulation (DH_dyn.m),
%   the CoM Jacobians are computed using p_Ci relative to the PREVIOUS
%   link origin (to build the r_{j-1,Ci} moment arm), projected into the
%   CURRENT frame orientation (where the inertia tensor is diagonal).
%
% =========================================================================
% INERTIA TENSOR (I) FRAME CONVENTION:
%
%   I(:,:,i) is the 3x3 rotational inertia tensor of link i, expressed
%   in the orientation of the DH frame {F_i}.
%
%   In code:
%     R_rel        = R_dh_i' * R_urdf_i   (rotation from URDF frame to DH frame)
%     I(:,:,i)     = R_rel * I_urdf * R_rel'
%
%   where I_urdf is the inertia tensor already rotated to the URDF link
%   frame by urdf_parse() (i.e., I_link = R_inertial * I_raw * R_inertial').
%
%   This convention EXACTLY matches GenerateDhParams.m lines 584-586:
%     R_rel          = R_curr_dh' * R_urdf;
%     obj.I(:,:,i)   = R_rel * link_data.inertia * R_rel';
%
% =========================================================================
% INPUTS:
%   robot    (struct)   Output of urdf_parse().  Must contain:
%                         .links   (containers.Map)  link_name -> link struct
%                                    Each link has: .mass, .com, .inertia
%   chain    (struct)   Output of urdf_build_chain(). Must contain:
%                         .n             (int)
%                         .active_links  (Nx1 cell)  link names in chain order
%                         .T_urdf        (Nx1 cell)  global URDF transforms at q=0
%   dh       (struct)   Output of urdf_extract_dh(). Must contain:
%                         .T_dh          (Nx1 cell)  cumulative DH transforms at q=0
%   verbose  (logical)  [optional, default false]
%
% OUTPUT:
%   inertia  (struct) with fields:
%     .m     [Nx1]    Link masses in kg.
%                     inertia.m(i) = mass of link i (the i-th active link)
%     .P_Ci  [Nx3]   CoM position vectors, one per row.
%                     Row i: CoM of link i FROM DH origin {F_{i-1}}, IN DH frame {F_i}
%                     Units: meters. Shape: [N x 3]
%                     NOTE: dynamics_casadi.m transposes this to [3xN] for CasADi Jacobians.
%     .I     [3x3xN]  Rotational inertia tensors in DH frame orientations.
%                     I(:,:,i): 3x3 symmetric positive-definite matrix in kg·m²
%                               expressed in the orientation of DH frame {F_i}
%
% =========================================================================
% USAGE:
%   robot   = urdf_parse('universalUR5.urdf');
%   chain   = urdf_build_chain(robot);
%   dh      = urdf_extract_dh(chain);
%   inertia = urdf_extract_inertia(robot, chain, dh);
%   inertia = urdf_extract_inertia(robot, chain, dh, true);  % verbose
%
% =========================================================================
% CALLED BY:
%   dynamics_casadi.m (Phase 2)
%   run_pipeline.m    (Master orchestrator)
% =========================================================================
% VALIDATION TARGET:
%   The P_Ci and I values produced by this function MUST match the output
%   of ExtractRobotDynamics.extract_dynamics() from the legacy codebase
%   (GenerateDhParams.My Implementation version).
%   See test_phase1_unit_tests.m for the cross-validation assertions.

    % -----------------------------------------------------------------------
    % 0. Default arguments and input validation
    % -----------------------------------------------------------------------
    if nargin < 4
        verbose = false;
    end

    assert(isstruct(chain) && isfield(chain, 'n') && isfield(chain, 'active_links') && isfield(chain, 'T_urdf'), ...
           'urdf_extract_inertia:invalidChain', 'Input chain struct is missing required fields (.n, .active_links, .T_urdf).');
    assert(isstruct(dh) && isfield(dh, 'T_dh') && ~isempty(dh.T_dh), ...
           'urdf_extract_inertia:invalidDH', 'Input dh struct must contain non-empty .T_dh field from urdf_extract_dh.');
    assert(length(dh.T_dh) >= chain.n, ...
           'urdf_extract_inertia:dimMismatch', 'dh.T_dh length must be at least chain.n.');

    n = chain.n;  % number of DOF / active links

    % -----------------------------------------------------------------------
    % 1. Pre-allocate output arrays
    % -----------------------------------------------------------------------
    m_vec = zeros(n, 1);        % masses [Nx1]
    P_Ci  = zeros(n, 3);        % CoM vectors [Nx3]
    I_ten = zeros(3, 3, n);     % inertia tensors [3x3xN]

    % -----------------------------------------------------------------------
    % 2. Main loop — process each active link
    % -----------------------------------------------------------------------
    for i = 1:n

        % ---- 2a. Retrieve link data from the robot struct -------------------
        link_name = chain.active_links{i};   % name of the i-th active link

        if ~isKey(robot.links, link_name)
            error('urdf_extract_inertia:missingLink', ...
                  'Active link "%s" (index %d) not found in robot.links map.', ...
                  link_name, i);
        end
        link_data = robot.links(link_name);

        % ---- 2b. Extract mass (kg) ------------------------------------------
        m_vec(i) = link_data.mass;

        % ---- 2c. Compute CoM position in the WORLD FRAME at q=0 ------------
        % The URDF <inertial><origin xyz> gives the CoM in the URDF link frame.
        % T_urdf{i} is the global transform of that URDF link frame at q=0.
        % Applying T_urdf{i} as a homogeneous point transform:
        %   p_com_global = R_urdf * p_com_urdf + t_urdf
        %                = T_urdf{i} * [com_urdf; 1]  (in homogeneous coords)
        p_com_hom    = chain.T_urdf{i} * [link_data.com; 1]; % [4x1] homogeneous
        p_com_global = p_com_hom(1:3);                        % [3x1] Cartesian

        % ---- 2d. Compute P_Ci in the DH frame convention -------------------
        % P_Ci(i,:) = CoM of link i measured FROM O_{i-1}^DH, IN orientation of F_i^DH
        %
        % Step 1: Find the origin of the PREVIOUS DH frame in world coordinates.
        %         For i=1 (first link), the previous origin is the world origin [0;0;0].
        if i == 1
            O_prev_dh = [0; 0; 0];          % world frame origin
        else
            O_prev_dh = dh.T_dh{i-1}(1:3, 4);  % DH origin of frame {F_{i-1}}
        end

        % Step 2: Get the rotation matrix of the CURRENT DH frame F_i.
        %         This maps vectors from the world frame to the DH frame {F_i}.
        R_dh_i = dh.T_dh{i}(1:3, 1:3);     % [3x3] rotation of DH frame {F_i}

        % Step 3: Compute P_Ci by:
        %   (a) Taking the vector from the previous DH origin to the world CoM
        %   (b) Rotating that vector into the current DH frame orientation
        %
        % This gives the CoM position as seen BY the i-th DH frame dynamics:
        %   P_Ci(i,:) = R_dh_i' * (p_com_global - O_prev_dh)
        %             = [3x3]  * [3x1]
        %             = [3x1]  -> stored as a row vector in P_Ci
        P_Ci(i, :) = (R_dh_i' * (p_com_global - O_prev_dh))';

        % ---- 2e. Rotate inertia tensor into DH frame orientation ------------
        % The inertia tensor in robot.links is already in the URDF LINK FRAME
        % (rotated from the inertial-origin frame by R_inertial in urdf_parse).
        %
        % To use it in DH_dyn.m, we need it in the DH frame orientation:
        %   I_dh = R_rel * I_urdf * R_rel'
        %   where R_rel = R_dh_i' * R_urdf_i
        %                = transform FROM the URDF link frame TO the DH frame

        % Get the rotation matrix of the URDF link frame at q=0 in world coords
        R_urdf_i = chain.T_urdf{i}(1:3, 1:3);  % [3x3] rotation of URDF frame {L_i}

        % Compute the relative rotation from URDF frame to DH frame:
        %   R_rel = R_dh_i' * R_urdf_i
        % This is the change-of-basis matrix (R_rel * v_urdf = v_dh).
        % Then apply the passive similarity transformation to the inertia tensor:
        %   I_dh = R_rel * I_urdf_link * R_rel'
        % (Note: in dynamics_casadi, the active transformation from DH to world is R * I * R').
        R_rel = R_dh_i' * R_urdf_i;
        I_ten(:, :, i) = R_rel * link_data.inertia * R_rel';

    end % end main loop

    % -----------------------------------------------------------------------
    % 3. Verbose output
    % -----------------------------------------------------------------------
    if verbose
        fprintf('\n[urdf_extract_inertia] === Dynamic Parameters ===\n');
        fprintf('  %-3s | %-8s | %-28s | %-38s\n', ...
                '#', 'Mass(kg)', 'CoM P_Ci (m) [x, y, z]', ...
                'Inertia Diag [Ixx, Iyy, Izz] (kg·m²)');
        fprintf('  %s\n', repmat('=', 1, 85));
        for i = 1:n
            I_diag = diag(I_ten(:, :, i));
            fprintf('  %-3d | %8.4f | [%8.4f, %8.4f, %8.4f] | [%10.4e, %10.4e, %10.4e]\n', ...
                    i, m_vec(i), ...
                    P_Ci(i, 1), P_Ci(i, 2), P_Ci(i, 3), ...
                    I_diag(1), I_diag(2), I_diag(3));
        end
        fprintf('\n');
    end

    % -----------------------------------------------------------------------
    % 4. Assemble the output struct
    % -----------------------------------------------------------------------
    inertia.m    = m_vec;   % [Nx1]    masses in kg
    inertia.P_Ci = P_Ci;   % [Nx3]    CoM vectors in DH frame convention
    inertia.I    = I_ten;  % [3x3xN]  inertia tensors in DH frame orientations

end % function urdf_extract_inertia
