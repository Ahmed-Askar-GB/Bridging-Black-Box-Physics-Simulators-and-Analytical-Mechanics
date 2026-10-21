"""BFS serial kinematic chain traversal and active joint resolution."""

from __future__ import annotations
import numpy as np
from collections import deque
from urdf2dt.containers import Robot, Joint, Chain


def urdf_build_chain(robot: Robot, verbose: bool = False) -> Chain:
    """Traverse the robot link-joint graph using BFS to extract the single serial active kinematic chain.

    Accumulates fixed transforms into T_root_to_j1 and T_ee.
    Raises ValueError if kinematic branching among active joints is detected.
    """
    if not robot.root_link:
        raise ValueError("Robot root link is empty.")

    # Build adjacency mapping: parent_link -> list of joints
    adj: dict[str, list[Joint]] = {}
    for joint in robot.joints.values():
        adj.setdefault(joint.parent, []).append(joint)

    # Classify joint types
    active_types = {"revolute", "continuous", "prismatic"}

    # BFS from root_link to discover ordered serial path
    queue: deque[tuple[str, np.ndarray]] = deque([(robot.root_link, np.eye(4))])
    visited_links: set[str] = {robot.root_link}

    active_joints: list[Joint] = []
    active_links: list[str] = []
    T_root_to_j1 = np.eye(4)
    found_first_active = False

    # Track transforms along the serial chain
    current_link = robot.root_link
    accumulated_fixed_prefix = np.eye(4)

    # Step-by-step traversal along the dominant chain
    while True:
        outgoing_joints = adj.get(current_link, [])
        if not outgoing_joints:
            break

        # Check for active branching
        active_outgoing = [j for j in outgoing_joints if j.joint_type in active_types]
        if len(active_outgoing) > 1:
            raise ValueError(f"Kinematic branching detected at link '{current_link}'. URDF2DT requires a serial chain.")

        # If there's an active joint, follow it
        if len(active_outgoing) == 1:
            joint = active_outgoing[0]
            if not found_first_active:
                found_first_active = True
                T_root_to_j1 = accumulated_fixed_prefix.copy()
            active_joints.append(joint)
            active_links.append(joint.child)
            current_link = joint.child
        else:
            # Only fixed joints
            joint = outgoing_joints[0]
            if not found_first_active:
                accumulated_fixed_prefix = accumulated_fixed_prefix @ joint.T_parent_child
            current_link = joint.child

    n = len(active_joints)
    if n == 0:
        raise ValueError(f"No active joints (revolute/prismatic) found in robot '{robot.name}'.")

    # Now find tip transforms (T_ee) following the last active link
    T_ee = np.eye(4)
    tip_link = active_links[-1]
    while True:
        tip_outgoing = adj.get(tip_link, [])
        if not tip_outgoing:
            break
        # Take the fixed joint connecting the tip
        fixed_j = tip_outgoing[0]
        T_ee = T_ee @ fixed_j.T_parent_child
        tip_link = fixed_j.child

    # Compute global transforms T_urdf at q=0, joint origins O, and unit axes Z
    T_urdf: list[np.ndarray] = []
    O = np.zeros((3, n), dtype=np.float64)
    Z = np.zeros((3, n), dtype=np.float64)

    T_current = T_root_to_j1.copy()

    for i, joint in enumerate(active_joints):
        # Transform to joint i origin
        T_joint_origin = T_current @ joint.T_parent_child
        O[:, i] = T_joint_origin[:3, 3]
        # World-frame axis
        z_world = T_joint_origin[:3, :3] @ joint.axis
        norm_z = np.linalg.norm(z_world)
        Z[:, i] = z_world / norm_z if norm_z > 1e-12 else np.array([0.0, 0.0, 1.0])

        T_current = T_joint_origin.copy()
        T_urdf.append(T_current.copy())

    # EE offsets
    d_ee = float(T_ee[2, 3])
    a_ee = float(np.sqrt(T_ee[0, 3]**2 + T_ee[1, 3]**2))

    if verbose:
        print(f"Extracted active chain with {n} DOF:")
        for idx, (j, l) in enumerate(zip(active_joints, active_links)):
            print(f"  [{idx+1}] Joint: {j.name} ({j.joint_type}) -> Link: {l}")
        print(f"  T_ee translation: {T_ee[:3, 3]}, d_ee: {d_ee:.4f}, a_ee: {a_ee:.4f}")

    return Chain(
        n=n,
        active_joints=active_joints,
        active_links=active_links,
        active_start=robot.root_link,
        O=O,
        Z=Z,
        T_urdf=T_urdf,
        T_root_to_j1=T_root_to_j1,
        T_ee=T_ee,
        d_ee=d_ee,
        a_ee=a_ee,
    )
