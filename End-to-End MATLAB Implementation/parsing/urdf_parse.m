function robot = urdf_parse(urdf_file)
% URDF_PARSE  Parse a URDF XML file into a clean, self-contained robot struct.
%
% This is a pure functional refactor of the GenerateDhParams.parse_urdf() method.
% It has NO side effects (no persistent state, no class properties), and returns
% ALL extracted data in a single structured container.
%
% The function performs TWO passes over the XML children of <robot>:
%   Pass 1 — collect all <link> nodes with their inertial properties.
%   Pass 2 — collect all top-level <joint> nodes with origin/axis data.
% After both passes, the structural root link is identified via set-difference.
%
% =========================================================================
% INPUT:
%   urdf_file  (char)  Absolute or relative path to the .urdf XML file.
%
% OUTPUT:
%   robot      (struct) with fields:
%     .name          (char)        Robot name attribute from <robot name="...">
%     .urdf_file     (char)        Absolute path to the source .urdf file
%     .root_link     (char)        Name of the structural root link
%                                  (the unique link that is not the child of
%                                   any joint — found via set-difference)
%     .links         (containers.Map)  link_name -> link struct with fields:
%                      .name          (char)      Link name
%                      .mass          (double)    Total mass in kg
%                      .com           (3x1 double) CoM in the inertial origin frame (m)
%                      .R_inertial    (3x3 double) Rotation of the <inertial><origin rpy>
%                                                  (describes how the inertial frame
%                                                   is rotated relative to the link frame)
%                      .inertia       (3x3 double) Symmetric inertia tensor expressed in
%                                                  the URDF link frame (after rotating
%                                                  the raw tensor by R_inertial)
%                      .parent_joint  (char)       Name of the joint whose <child> is this link
%                                                  (empty string '' for the root link)
%     .joints        (containers.Map)  joint_name -> joint struct with fields:
%                      .name    (char)       Joint name
%                      .type    (char)       'revolute'|'prismatic'|'fixed'|'continuous'
%                      .parent  (char)       Parent link name
%                      .child   (char)       Child link name
%                      .xyz     (1x3 double) Translation of joint origin in parent frame (m)
%                      .rpy     (1x3 double) RPY rotation of joint origin in parent frame (rad)
%                                            Convention: R = Rz(yaw)*Ry(pitch)*Rx(roll)
%                      .axis    (1x3 double) Unit rotation/translation axis in joint frame
%                      .T       (4x4 double) Full homogeneous transform T_parent_to_joint
%                                            assembled from xyz and rpy
%
% =========================================================================
% USAGE:
%   robot = urdf_parse('universalUR5.urdf');
%   robot = urdf_parse('/abs/path/to/robot.urdf');
%
% =========================================================================
% CONVENTION NOTES:
%   - All angles are in RADIANS.
%   - The URDF RPY convention is EXTRINSIC XYZ (= intrinsic ZYX):
%       R = Rz(yaw) * Ry(pitch) * Rx(roll)
%   - The inertia tensor stored in robot.links(name).inertia has ALREADY been
%     rotated from the inertial-origin frame to the link frame:
%       I_link = R_inertial * I_raw * R_inertial'
%   - This matches the convention used in GenerateDhParams.parse_urdf() line 177.
%
% =========================================================================
% CALLED BY:
%   urdf_build_chain.m  (Phase 1, Step 2)
%   run_pipeline.m      (Master orchestrator)

    % -----------------------------------------------------------------------
    % 0. Validate the input file path
    % -----------------------------------------------------------------------
    if nargin < 1 || isempty(urdf_file)
        error('urdf_parse:noFile', 'A URDF file path must be provided.');
    end
    if ~exist(urdf_file, 'file')
        error('urdf_parse:fileNotFound', ...
              'URDF file not found: "%s"', urdf_file);
    end

    % -----------------------------------------------------------------------
    % 1. Initialize the output struct and storage maps
    % -----------------------------------------------------------------------
    % Store the absolute canonical path to guard against working directory changes
    file_info = dir(urdf_file);
    if ~isempty(file_info) && ~isempty(file_info(1).folder)
        robot.urdf_file = fullfile(file_info(1).folder, file_info(1).name);
    else
        robot.urdf_file = urdf_file;
    end
    robot.name      = '';
    robot.root_link = '';

    % Use containers.Map for O(1) name-based lookup throughout the pipeline.
    % KeyType 'char' and ValueType 'any' allow struct values.
    robot.links  = containers.Map('KeyType', 'char', 'ValueType', 'any');
    robot.joints = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % -----------------------------------------------------------------------
    % 2. Parse the URDF XML document
    % -----------------------------------------------------------------------
    xDoc     = xmlread(urdf_file);                          % Java DOM parser
    robot_nd = xDoc.getElementsByTagName('robot').item(0); % <robot> root node

    % Extract the robot name attribute (e.g., name="ur5")
    robot_name_attr = char(robot_nd.getAttribute('name'));
    if ~isempty(robot_name_attr)
        robot.name = robot_name_attr;
    end

    % Iterate over DIRECT children of <robot> only.
    % This is critical: getElementsByTagName() searches the ENTIRE subtree,
    % which would capture <joint> tags inside <transmission> blocks.
    % Iterating kids (direct children) avoids this ambiguity.
    kids = robot_nd.getChildNodes();

    % -----------------------------------------------------------------------
    % 3. PASS 1 — Collect all <link> nodes
    % -----------------------------------------------------------------------
    % For each <link>, extract:
    %   - Link name
    %   - Inertial properties: mass, CoM origin (xyz + rpy), inertia tensor
    % Links without an <inertial> block (e.g., virtual/EE links) get zeros.
    % -----------------------------------------------------------------------
    for k = 0:(kids.getLength() - 1)
        nd = kids.item(k);

        % Skip XML nodes that are not <link> elements
        if ~strcmp(char(nd.getNodeName()), 'link')
            continue;
        end

        % Read the link name attribute
        link_name = char(nd.getAttribute('name'));

        % ---- Default inertial values (for links with no <inertial> tag) ----
        mass        = 0.0;                % kg
        com         = [0.0; 0.0; 0.0];   % CoM in inertial-origin frame, (m)
        R_inertial  = eye(3);             % Rotation of inertial origin vs link frame
        inertia_mat = zeros(3, 3);        % Inertia tensor in link frame

        % ---- Check if this link has an <inertial> block --------------------
        ine_nodes = nd.getElementsByTagName('inertial');
        if ine_nodes.getLength() > 0
            ine = ine_nodes.item(0);  % take the first (and only) <inertial>

            % ---- 3a. Extract mass -------------------------------------------
            m_nodes = ine.getElementsByTagName('mass');
            if m_nodes.getLength() > 0
                m_attr = char(m_nodes.item(0).getAttribute('value'));
                if ~isempty(m_attr)
                    mass = str2double(m_attr);  % parse scalar string -> double
                end
            end

            % ---- 3b. Extract inertial origin (xyz and rpy) -----------------
            % The <origin> tag inside <inertial> gives the pose of the inertial
            % reference frame (CoM location + principal-axis orientation) in the
            % link's own coordinate frame.
            o_nodes = ine.getElementsByTagName('origin');
            if o_nodes.getLength() > 0
                o_elem = o_nodes.item(0);

                % xyz: position of CoM in the link frame (meters)
                % NOTE: str2num (not str2double) is used intentionally here because
                % URDF xyz/rpy attributes are space-delimited string arrays (e.g. "0 0.1 -0.2").
                % str2double only handles single scalar tokens, whereas str2num parses vectors.
                if o_elem.hasAttribute('xyz')
                    com_val = str2num(char(o_elem.getAttribute('xyz'))); %#ok<ST2NM>
                    if ~isempty(com_val)
                        com = com_val(:);  % force column vector
                    end
                end

                % rpy: orientation of inertial frame relative to link frame
                % Used to rotate the raw inertia tensor from inertial frame
                % to the link frame.
                if o_elem.hasAttribute('rpy')
                    rpy_val = str2num(char(o_elem.getAttribute('rpy'))); %#ok<ST2NM>
                    if ~isempty(rpy_val)
                        R_inertial = rpy_to_R(rpy_val);
                    end
                end
            end

            % ---- 3c. Extract inertia tensor components ---------------------
            % URDF stores the upper triangle: ixx, ixy, ixz, iyy, iyz, izz
            % (the full tensor is symmetric by definition)
            i_nodes = ine.getElementsByTagName('inertia');
            if i_nodes.getLength() > 0
                ie  = i_nodes.item(0);
                ixx = parse_attr(ie, 'ixx', 0.0);
                ixy = parse_attr(ie, 'ixy', 0.0);
                ixz = parse_attr(ie, 'ixz', 0.0);
                iyy = parse_attr(ie, 'iyy', 0.0);
                iyz = parse_attr(ie, 'iyz', 0.0);
                izz = parse_attr(ie, 'izz', 0.0);

                % Build the symmetric 3x3 inertia tensor from the URDF components.
                % This tensor is initially expressed in the INERTIAL-ORIGIN frame.
                raw_inertia = [ixx, ixy, ixz;
                               ixy, iyy, iyz;
                               ixz, iyz, izz];

                % Rotate the inertia tensor from the inertial-origin frame into
                % the URDF link frame.  By the parallel-axis / rotation rule:
                %   I_link = R_inertial * I_inertial_frame * R_inertial'
                % This matches GenerateDhParams.parse_urdf() line 177.
                inertia_mat = R_inertial * raw_inertia * R_inertial';
            end
        end % end if <inertial> exists

        % ---- Assemble the link struct and store it --------------------------
        link_s.name         = link_name;
        link_s.mass         = mass;
        link_s.com          = com;         % [3x1] in inertial-origin frame (m)
        link_s.R_inertial   = R_inertial;  % [3x3] rotation of inertial frame
        link_s.inertia      = inertia_mat; % [3x3] rotated to URDF link frame
        link_s.parent_joint = '';          % filled in during Pass 2

        robot.links(link_name) = link_s;
    end % end Pass 1

    % -----------------------------------------------------------------------
    % 4. PASS 2 — Collect all top-level <joint> nodes
    % -----------------------------------------------------------------------
    % For each <joint>, extract:
    %   - Joint name, type, parent/child link names
    %   - Origin: xyz translation + rpy rotation (URDF joint frame in parent frame)
    %   - Axis: unit rotation/translation axis in the joint frame
    %   - Full 4x4 homogeneous transform T (assembled from xyz and rpy)
    % Also back-annotates each child link with its .parent_joint name.
    % -----------------------------------------------------------------------
    for k = 0:(kids.getLength() - 1)
        nd = kids.item(k);

        % Skip XML nodes that are not <joint> elements
        if ~strcmp(char(nd.getNodeName()), 'joint')
            continue;
        end

        % Read joint name and type attributes
        jname = char(nd.getAttribute('name'));
        jtype = char(nd.getAttribute('type'));

        % ---- Read parent and child link names ------------------------------
        pnodes = nd.getElementsByTagName('parent');
        cnodes = nd.getElementsByTagName('child');

        % Skip malformed joints missing parent/child tags
        if pnodes.getLength() == 0 || cnodes.getLength() == 0
            warning('urdf_parse:malformedJoint', ...
                    'Joint "%s" is missing <parent> or <child> tag. Skipping.', jname);
            continue;
        end

        pname = char(pnodes.item(0).getAttribute('link'));  % parent link name
        cname = char(cnodes.item(0).getAttribute('link'));  % child link name

        % ---- Defaults (per URDF specification) -----------------------------
        xyz    = [0.0, 0.0, 0.0];  % no translation by default
        rpy    = [0.0, 0.0, 0.0];  % no rotation by default
        axis_v = [1.0, 0.0, 0.0];  % default joint axis is X (URDF spec)

        % ---- Read joint origin (xyz + rpy) ---------------------------------
        % <origin xyz="px py pz" rpy="r p y"/>  describes the transform from
        % the parent link frame to the joint frame at zero configuration.
        % str2num is used to convert space-separated triples to row vectors.
        orig = nd.getElementsByTagName('origin');
        if orig.getLength() > 0
            o = orig.item(0);
            if o.hasAttribute('xyz')
                xyz_val = str2num(char(o.getAttribute('xyz'))); %#ok<ST2NM>
                if ~isempty(xyz_val)
                    xyz = xyz_val;
                end
            end
            if o.hasAttribute('rpy')
                rpy_val = str2num(char(o.getAttribute('rpy'))); %#ok<ST2NM>
                if ~isempty(rpy_val)
                    rpy = rpy_val;
                end
            end
        end

        % ---- Read joint axis -----------------------------------------------
        % <axis xyz="ax ay az"/>  is the rotation axis (revolute/continuous)
        % or translation direction (prismatic) expressed in the JOINT frame.
        % It should be a unit vector; we normalize defensively.
        axnd = nd.getElementsByTagName('axis');
        if axnd.getLength() > 0
            a = axnd.item(0);
            if a.hasAttribute('xyz')
                ax_val = str2num(char(a.getAttribute('xyz'))); %#ok<ST2NM>
                if ~isempty(ax_val)
                    axis_v = ax_val;
                end
            end
        end

        % Normalize axis to unit length (guard against near-zero vectors)
        ax_norm = norm(axis_v);
        if ax_norm > 1e-9
            axis_v = axis_v / ax_norm;
        else
            warning('urdf_parse:zeroAxis', ...
                    'Joint "%s" has a near-zero axis vector. Using [1,0,0].', jname);
            axis_v = [1.0, 0.0, 0.0];
        end

        % ---- Assemble the joint struct --------------------------------------
        joint_s.name   = jname;
        joint_s.type   = jtype;
        joint_s.parent = pname;
        joint_s.child  = cname;
        joint_s.xyz    = xyz(:)';    % ensure row vector [1x3]
        joint_s.rpy    = rpy(:)';    % ensure row vector [1x3]
        joint_s.axis   = axis_v(:)'; % ensure row vector [1x3]

        % Build the 4x4 homogeneous transform for this joint origin.
        % T_parent_to_joint = [R | xyz; 0 0 0 1]
        % where R = Rz(yaw)*Ry(pitch)*Rx(roll)  (URDF extrinsic XYZ convention)
        joint_s.T = rpy_xyz_to_T(rpy, xyz);

        robot.joints(jname) = joint_s;

        % ---- Back-annotate the child link with its parent joint name -------
        % This allows fast "who is my parent joint?" lookups later.
        if isKey(robot.links, cname)
            ls = robot.links(cname);
            ls.parent_joint = jname;
            robot.links(cname) = ls;
        end

    end % end Pass 2

    % -----------------------------------------------------------------------
    % 5. Structural Root Detection via Set-Difference
    % -----------------------------------------------------------------------
    % The root link is the unique link that is NOT the child of ANY joint.
    % This is robust to any link naming convention (no 'base_link' hardcoding).
    %
    % Algorithm:
    %   all_children = {j.child  for all joints j}
    %   root links   = {l.name   for all links l} MINUS all_children
    % -----------------------------------------------------------------------
    jkeys        = keys(robot.joints);
    all_children = cellfun(@(k) robot.joints(k).child, jkeys, ...
                           'UniformOutput', false);

    lkeys = keys(robot.links);
    roots = {};
    for i = 1:length(lkeys)
        % A link is a root if its name does not appear in any joint's child field
        if ~any(strcmp(all_children, lkeys{i}))
            roots{end+1} = lkeys{i}; %#ok<AGROW>
        end
    end

    % Guard: a well-formed URDF must have exactly one root (or a virtual world link)
    if isempty(roots)
        error('urdf_parse:noRoot', ...
              ['No structural root link found. ' ...
               'The kinematic tree may be cyclic or malformed.']);
    end

    % Sort alphabetically for determinism when multiple candidates exist
    % (e.g., when a "world" fixed link is present alongside "base_link")
    roots = sort(roots);
    if length(roots) > 1
        warning('urdf_parse:multipleRoots', ...
                'Multiple structural roots found: {%s}. Using "%s".', ...
                strjoin(roots, ', '), roots{1});
    end

    robot.root_link = roots{1};

end % function urdf_parse


% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  (private to this file — not accessible from outside)
% =========================================================================

function R = rpy_to_R(rpy)
% RPY_TO_R  Convert URDF Roll-Pitch-Yaw angles to a 3x3 rotation matrix.
%
% URDF convention: EXTRINSIC XYZ rotations (equivalent to intrinsic ZYX).
%   R = Rz(yaw) * Ry(pitch) * Rx(roll)
%
% INPUT:
%   rpy  (1x3 or 3x1 double) [roll, pitch, yaw] in radians
%
% OUTPUT:
%   R    (3x3 double) Rotation matrix in SO(3)
    r = rpy(1);   % roll  — rotation about X
    p = rpy(2);   % pitch — rotation about Y
    y = rpy(3);   % yaw   — rotation about Z

    % Elementary rotation matrices
    Rx = [1,      0,       0;
          0,  cos(r), -sin(r);
          0,  sin(r),  cos(r)];

    Ry = [ cos(p), 0, sin(p);
                0, 1,      0;
          -sin(p), 0, cos(p)];

    Rz = [cos(y), -sin(y), 0;
          sin(y),  cos(y), 0;
               0,       0, 1];

    % Extrinsic XYZ: apply Rx first (innermost), then Ry, then Rz (outermost)
    R = Rz * Ry * Rx;
end


function T = rpy_xyz_to_T(rpy, xyz)
% RPY_XYZ_TO_T  Build a 4x4 SE(3) homogeneous transform from RPY and xyz.
%
% INPUTS:
%   rpy  (1x3 or 3x1) [roll, pitch, yaw] in radians (URDF extrinsic XYZ)
%   xyz  (1x3 or 3x1) translation [px, py, pz] in meters
%
% OUTPUT:
%   T    (4x4 double) Homogeneous transform:
%            [ R  | xyz ]
%            [ 0  |  1  ]
    T           = eye(4);
    T(1:3, 1:3) = rpy_to_R(rpy);
    T(1:3, 4)   = xyz(:);    % force column vector
end


function val = parse_attr(element, attr_name, default_val)
% PARSE_ATTR  Safely read a named attribute from an XML element node.
%
% Returns default_val if the attribute is missing or empty.
%
% INPUTS:
%   element      Java DOM Element node
%   attr_name    char  attribute name to read
%   default_val  numeric  value to return if attribute is absent
    if element.hasAttribute(attr_name)
        str = char(element.getAttribute(attr_name));
        if ~isempty(str)
            val = str2double(str);
            return;
        end
    end
    val = default_val;
end
