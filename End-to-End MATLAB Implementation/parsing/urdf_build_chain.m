function chain = urdf_build_chain(robot, verbose)
% URDF_BUILD_CHAIN  Traverse the kinematic tree and resolve the serial active chain.
%
% This is a pure functional refactor of the chain-building logic embedded inside
% GenerateDhParams.calculate_dh_and_dynamics() (My Implementation version).
%
% The function performs:
%   1. Build parent-to-children adjacency maps (active joints and fixed joints)
%   2. Validate that the robot is a serial chain (no kinematic branching)
%   3. BFS from the root through fixed joints to find the first active joint
%   4. Accumulate fixed-joint transforms from the root to that first joint
%   5. Walk the active serial chain, compounding any intermediate fixed joints
%   6. Accumulate the end-effector fixed transform beyond the last active link
%   7. Pre-compute all global URDF frames at the zero configuration (q = 0)
%
% =========================================================================
% INPUTS:
%   robot    (struct)   Output of urdf_parse().  Must contain:
%                         .root_link  (char)
%                         .links      (containers.Map)
%                         .joints     (containers.Map)
%   verbose  (logical)  [optional, default false]
%                       If true, prints a human-readable chain summary.
%
% OUTPUT:
%   chain   (struct) with fields:
%     .n               (int)      Number of active DOF (revolute+prismatic joints)
%     .root_link       (char)     Structural root link name (copied from robot)
%     .active_start    (char)     Name of the first link that has an active child joint
%     .active_joints   (1xN cell) Ordered array of active joint structs.
%                                 Each struct is the raw joint from robot.joints,
%                                 PLUS an extra field:
%                                   .T_effective  (4x4) = accumulated_fixed * T_joint
%                                 This absorbs any fixed joints between this active
%                                 joint and the previous active joint.
%     .active_links    (Nx1 cell) Child link names for each active joint
%                                 (i.e., chain.active_links{i} = chain.active_joints{i}.child)
%     .T_root_to_j1    (4x4)     Accumulated fixed-joint transforms from the
%                                 structural root to the first active joint's parent link.
%                                 At q=0 this is the world pose of the active chain base.
%     .T_ee            (4x4)     Fixed-joint transform from the last active link to the
%                                 true end-effector tip (product of all trailing fixed joints).
%     .d_ee            (double)  Component of T_ee translation ALONG the last joint axis (m).
%                                 Used to correct the last row of the DH table.
%     .a_ee            (double)  Component of T_ee translation PERPENDICULAR to the last
%                                 joint axis (m). Used to correct the last row of the DH table.
%     .T_urdf          (Nx1 cell) Per-link cumulative global URDF transforms at q=0.
%                                 T_urdf{i} = T_0_to_i when all joints are at zero.
%                                 Shape: [4x4].  Used by urdf_extract_inertia().
%     .O               (3xN)    World-frame origin positions of all active joints (1..N) at q=0.
%     .Z               (3xN)    World-frame unit joint axes of all active joints (1..N) at q=0.
%
% =========================================================================
% USAGE:
%   robot = urdf_parse('universalUR5.urdf');
%   chain = urdf_build_chain(robot);
%   chain = urdf_build_chain(robot, true);   % verbose mode
%
% =========================================================================
% ERRORS / WARNINGS:
%   - Error  if the robot has kinematic branching (not a serial chain).
%   - Error  if no active joints are reachable from the root.
%   - Warning if multiple root links are detected (urdf_parse handles this).
%
% =========================================================================
% CALLED BY:
%   urdf_extract_dh.m      (Phase 1, Step 3)
%   urdf_extract_inertia.m (Phase 1, Step 4)
%   run_pipeline.m         (Master orchestrator)

    % -----------------------------------------------------------------------
    % 0. Default arguments
    % -----------------------------------------------------------------------
    if nargin < 2
        verbose = false;
    end

    % Joint types that contribute to the robot's DOF
    ACTIVE_TYPES = {'revolute', 'continuous', 'prismatic'};

    % -----------------------------------------------------------------------
    % 1. Build sorted parent-to-children adjacency maps
    % -----------------------------------------------------------------------
    % Two separate maps:
    %   p2a : parent_link_name -> {active joint names whose parent is this link}
    %   p2f : parent_link_name -> {fixed  joint names whose parent is this link}
    %
    % WHY sorted keys? containers.Map does not guarantee insertion order.
    % Sorting joint names alphabetically before inserting ensures that, for
    % robots with multiple joints at the same parent, the traversal order is
    % DETERMINISTIC regardless of XML layout or OS.
    % -----------------------------------------------------------------------
    p2a = containers.Map('KeyType', 'char', 'ValueType', 'any'); % parent -> active children
    p2f = containers.Map('KeyType', 'char', 'ValueType', 'any'); % parent -> fixed  children

    jkeys = sort(keys(robot.joints));   % sort all joint names alphabetically
    for k = 1:length(jkeys)
        j = robot.joints(jkeys{k});

        if any(strcmp(j.type, ACTIVE_TYPES))
            % This joint contributes a DOF — add to the active map
            p2a = append_to_map(p2a, j.parent, j.name);

        elseif strcmp(j.type, 'fixed')
            % Fixed joint carries geometric info but no motion — add to fixed map
            p2f = append_to_map(p2f, j.parent, j.name);

        % else: 'floating', 'planar' etc. are ignored (not supported for DH)
        end
    end

    % -----------------------------------------------------------------------
    % 2. Serial-chain validation (branching check)
    % -----------------------------------------------------------------------
    % A serial robot has AT MOST one active child joint per link.
    % If any link has 2+ active children, the robot is branched (e.g., a
    % humanoid or parallel mechanism) and standard DH is not applicable.
    % -----------------------------------------------------------------------
    ak = keys(p2a);
    for k = 1:length(ak)
        children = p2a(ak{k});
        if length(children) > 1
            error('urdf_build_chain:notSerialChain', ...
                ['Kinematic branching detected at link "%s": ' ...
                 '%d active child joints found (%s).\n' ...
                 'Standard DH parameter extraction requires a single serial chain.'], ...
                ak{k}, length(children), strjoin(children, ', '));
        end
    end

    % -----------------------------------------------------------------------
    % 3. BFS from root through fixed joints → find the first active joint
    % -----------------------------------------------------------------------
    % Many robots have a fixed "world" -> "base_link" joint before the first
    % revolute joint.  We traverse fixed joints in BFS order to skip these
    % virtual links and find the link that is the TRUE starting parent of the
    % active kinematic chain.
    %
    % Simultaneously, we accumulate the homogeneous transforms of all those
    % fixed joints into T_root_to_j1 so we know the absolute pose of the
    % chain base in the world frame at q = 0.
    % -----------------------------------------------------------------------
    active_start    = '';   % name of the first link that has an active child
    visited         = {};   % BFS visited set (avoid re-visiting)
    queue           = {robot.root_link};

    % parent_trace maps: child_link -> {parent_link, joint_struct}
    % Used to reconstruct the fixed-joint path from root to active_start.
    parent_trace = containers.Map('KeyType', 'char', 'ValueType', 'any');

    while ~isempty(queue)
        node  = queue{1};
        queue = queue(2:end);  % dequeue front element

        % Skip if already visited (cycle guard, though valid URDFs are acyclic)
        if any(strcmp(visited, node))
            continue;
        end
        visited{end+1} = node; %#ok<AGROW>

        if isKey(p2a, node)
            % This node has at least one active child joint → chain starts here
            active_start = node;
            break;
        end

        if isKey(p2f, node)
            % This node has only fixed children → continue BFS through them
            fnames = sort(p2f(node));   % sorted for determinism
            for fn = 1:length(fnames)
                fj = robot.joints(fnames{fn});
                if ~any(strcmp(visited, fj.child))
                    queue{end+1} = fj.child; %#ok<AGROW>
                    % Record the path so we can reconstruct T_root_to_j1
                    parent_trace(fj.child) = struct('parent', node, 'joint', fj);
                end
            end
        end
    end % BFS loop

    if isempty(active_start)
        error('urdf_build_chain:noActiveJoints', ...
              'No active joints reachable from root link "%s".', robot.root_link);
    end

    % -----------------------------------------------------------------------
    % 4. Reconstruct T_root_to_j1 (walk path backwards from active_start)
    % -----------------------------------------------------------------------
    % Walk backwards from active_start through parent_trace to the root,
    % collecting all intermediate fixed-joint transforms.
    % Then multiply them in forward order to build T_root_to_j1.
    %
    % Example path:  world --[fixed]--> base_link --[fixed]--> mount --[revolute]--> ...
    %   active_start = 'mount'
    %   T_root_to_j1 = T(world->base) * T(base->mount)
    % -----------------------------------------------------------------------
    T_root_to_j1 = eye(4);

    % Collect the chain of fixed transforms from root to active_start
    backward_chain = {};    % stores joint transforms in REVERSE order
    curr_back = active_start;
    max_trace_iters = robot.links.Count + 1;
    trace_iters = 0;
    while isKey(parent_trace, curr_back)
        trace_iters = trace_iters + 1;
        if trace_iters > max_trace_iters
            error('urdf_build_chain:cyclicTrace', ...
                  'Cyclic parent trace detected between root and active_start. Malformed URDF.');
        end
        info = parent_trace(curr_back);
        backward_chain{end+1} = info.joint.T;   %#ok<AGROW>  % record T for this fixed joint
        curr_back = info.parent;
    end

    % Multiply in forward order (reverse the backward chain)
    for b = length(backward_chain):-1:1
        T_root_to_j1 = T_root_to_j1 * backward_chain{b};
    end

    if verbose
        fprintf('[urdf_build_chain] Active chain starts at: "%s"\n', active_start);
        fprintf('[urdf_build_chain] Fixed-joint pre-transform from root:\n');
        disp(T_root_to_j1);
    end

    % -----------------------------------------------------------------------
    % 5. Walk the active serial chain, compounding intermediate fixed joints
    % -----------------------------------------------------------------------
    % Starting from active_start, follow p2a to collect active joints in order.
    % Between any two consecutive active joints there may be fixed joints
    % (e.g., a fixed offset frame before a wrist joint).
    % We compound those fixed transforms into T_effective on the active joint.
    %
    % Result: chain.active_joints{i}.T_effective encodes
    %   T_{i-1 active link} --> {i active joint frame}
    % as a single transform, absorbing all intermediate fixed offsets.
    % -----------------------------------------------------------------------
    active_joints_list = {};  % ordered list of active joint structs
    curr_link          = active_start;
    T_fixed_accum      = eye(4);  % accumulator for intermediate fixed transforms
    max_chain_iters    = robot.joints.Count + 1;
    chain_iters        = 0;

    while true
        chain_iters = chain_iters + 1;
        if chain_iters > max_chain_iters
            error('urdf_build_chain:cyclicChain', ...
                  'Cyclic active/fixed joint chain detected. Malformed kinematic tree.');
        end
        if isKey(p2a, curr_link)
            % Found an active joint from this link
            jnames = sort(p2a(curr_link));  % sorted -> deterministic
            j      = robot.joints(jnames{1});

            % Compute the effective transform for this active joint, which
            % absorbs any preceding fixed-joint offsets:
            %   T_effective = (accumulated_fixed) * T_this_joint
            j.T_effective    = T_fixed_accum * j.T;

            active_joints_list{end+1} = j; %#ok<AGROW>
            T_fixed_accum             = eye(4);   % reset accumulator
            curr_link                 = j.child;  % advance to next link

        elseif isKey(p2f, curr_link) && has_active_descendant(p2a, p2f, curr_link, robot)
            % Fixed joint(s) before the next active joint — accumulate their transforms
            fnames = sort(p2f(curr_link));  % sorted -> deterministic
            fj     = robot.joints(fnames{1});
            T_fixed_accum = T_fixed_accum * fj.T;  % compound fixed transforms
            curr_link     = fj.child;

        else
            % No more active joints reachable from curr_link → end of chain
            break;
        end
    end

    n = length(active_joints_list);  % number of DOF

    if n == 0
        error('urdf_build_chain:emptyChain', ...
              'Active kinematic chain is empty (no active joints found after active_start).');
    end

    % Extract the ordered child link names for convenience
    active_links = cell(n, 1);
    for i = 1:n
        active_links{i} = active_joints_list{i}.child;
    end

    if verbose
        fprintf('[urdf_build_chain] Active serial chain (%d DOF):\n', n);
        for i = 1:n
            fprintf('  %2d. %-40s [%s]  ->  Child: %s\n', ...
                    i, active_joints_list{i}.name, ...
                    active_joints_list{i}.type, active_links{i});
        end
    end

    % -----------------------------------------------------------------------
    % 6. Accumulate the end-effector (EE) fixed transform
    % -----------------------------------------------------------------------
    % After the last active link (curr_link at this point), there may be
    % additional fixed joints that position the EE tool frame.
    % We walk all of them and multiply their transforms into T_ee.
    %
    % Example:  wrist3_link --[fixed]--> ee_link --[fixed]--> tool0
    %   T_ee = T(wrist3->ee) * T(ee->tool0)
    % -----------------------------------------------------------------------
    T_ee = eye(4);
    tip  = curr_link;  % link at the end of the active chain

    while isKey(p2f, tip)
        fnames = sort(p2f(tip));    % sorted -> deterministic
        fj     = robot.joints(fnames{1});
        T_ee   = T_ee * fj.T;      % accumulate fixed EE offsets
        tip    = fj.child;

        if verbose
            fprintf('[urdf_build_chain] EE fixed joint: "%s"  ->  "%s"\n', ...
                    fj.name, fj.child);
        end
    end

    % Decompose T_ee into d_ee and a_ee relative to the last joint axis.
    % This is needed to override the last row of the DH table:
    %   d_ee = projection of EE translation onto the last joint axis (z_{n-1})
    %   a_ee = magnitude of EE translation perpendicular to the last joint axis
    last_axis = active_joints_list{end}.axis(:);    % [3x1] from joint struct
    last_axis = last_axis / norm(last_axis);         % ensure unit length
    t_ee_loc  = T_ee(1:3, 4);                       % EE translation vector [3x1]
    d_ee      = dot(t_ee_loc, last_axis);            % signed component along axis
    a_ee      = norm(t_ee_loc - d_ee * last_axis);  % perpendicular magnitude

    if verbose
        fprintf('[urdf_build_chain] EE offset: d_ee = %.6f m,  a_ee = %.6f m\n', ...
                d_ee, a_ee);
    end

    % -----------------------------------------------------------------------
    % 7. Pre-compute global URDF joint frames at the zero configuration
    % -----------------------------------------------------------------------
    % Propagate homogeneous transforms from the world frame (via T_root_to_j1)
    % through each active joint using its T_effective field.
    % At q = 0, each joint's transform is fixed (no joint variable contribution).
    %
    % O(:,i) = world-frame origin of active joint i at q=0  [3x1]
    % Z(:,i) = world-frame unit axis  of active joint i at q=0  [3x1]
    %
    % T_urdf{i} = cumulative global transform up to active link i at q=0
    %             (this is the pose of the URDF child link frame in world coords)
    % -----------------------------------------------------------------------
    O      = zeros(3, n);    % joint origin positions  [3xN]
    Z      = zeros(3, n);    % joint unit axes          [3xN]
    T_urdf = cell(n, 1);     % per-link global transforms

    T_global = T_root_to_j1; % start from the world-to-chain-base transform

    for i = 1:n
        % Apply this active joint's effective transform (including any
        % intermediate fixed offsets absorbed by T_effective)
        T_global = T_global * active_joints_list{i}.T_effective;

        T_urdf{i} = T_global;                   % global pose of link i

        O(:, i)   = T_global(1:3, 4);           % joint i origin in world frame

        % Rotate the joint axis from joint frame to world frame using the
        % rotational part of T_global.  Then normalize (should already be unit).
        zv       = T_global(1:3, 1:3) * active_joints_list{i}.axis(:);
        Z(:, i)  = zv / norm(zv);
    end

    % -----------------------------------------------------------------------
    % 8. Assemble the output struct
    % -----------------------------------------------------------------------
    chain.n               = n;
    chain.root_link       = robot.root_link;
    chain.active_start    = active_start;
    chain.active_joints   = active_joints_list;   % {1xN} cell of joint structs
    chain.active_links    = active_links;         % {Nx1} cell of child link names
    chain.T_root_to_j1    = T_root_to_j1;        % [4x4] world->chain-base
    chain.T_ee            = T_ee;                 % [4x4] last-link->EE-tip
    chain.d_ee            = d_ee;                 % scalar (m)
    chain.a_ee            = a_ee;                 % scalar (m)
    chain.T_urdf          = T_urdf;              % {Nx1} cell of [4x4]
    chain.O               = O;                   % [3xN] world origins
    chain.Z               = Z;                   % [3xN] world axes

end % function urdf_build_chain


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function map = append_to_map(map, key, value)
% APPEND_TO_MAP  Append a string value to a list stored in a containers.Map.
%
% If the key does not exist, creates a new entry {value}.
% If the key exists, appends value to the existing cell array.
%
% This is used to build the parent->children adjacency lists.
    if isKey(map, key)
        lst = map(key);
        lst{end+1} = value;
        map(key) = lst;
    else
        map(key) = {value};
    end
end


function tf = has_active_descendant(p2a, p2f, link_name, robot)
% HAS_ACTIVE_DESCENDANT  Check if a link has any active joint in its subtree.
%
% Performs a DFS/BFS through fixed joints to determine whether the subtree
% rooted at link_name eventually reaches an active joint.
%
% This is used in the chain-walking loop (Step 5) to decide whether a
% fixed joint should be compounded into the chain or ignored.
%
% INPUTS:
%   p2a       containers.Map  parent -> {active joint names}
%   p2f       containers.Map  parent -> {fixed  joint names}
%   link_name char            starting link
%   robot     struct          full robot struct (for joint lookup)
%
% OUTPUT:
%   tf        logical         true if an active descendant exists

    visited = {link_name};
    queue   = {link_name};
    tf      = false;

    while ~isempty(queue)
        nd    = queue{1};
        queue = queue(2:end);

        % If this link has active children, we found an active descendant
        if isKey(p2a, nd)
            tf = true;
            return;
        end

        % Follow fixed joints to look deeper in the tree
        if isKey(p2f, nd)
            fnames = p2f(nd);
            for fn = 1:length(fnames)
                fj = robot.joints(fnames{fn});
                if ~any(strcmp(visited, fj.child))
                    visited{end+1} = fj.child; %#ok<AGROW>
                    queue{end+1}   = fj.child; %#ok<AGROW>
                end
            end
        end
    end
end
