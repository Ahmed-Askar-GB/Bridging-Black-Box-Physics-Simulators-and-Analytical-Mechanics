"""XML ingestion of links and joints from URDF files."""

from __future__ import annotations
import os
import xml.etree.ElementTree as ET
import numpy as np
from urdf2dt.containers import Link, Joint, Robot
from urdf2dt.utils.math_utils import rpy_to_R, xyz_rpy_to_T


def _parse_vec3(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=np.float64)
    vals = [float(x) for x in text.strip().split()]
    return np.array(vals, dtype=np.float64)


def urdf_parse(urdf_path: str) -> Robot:
    """Parse a URDF file into a structured Robot dataclass.

    Rotates raw link inertia tensors into the local URDF link frame immediately:
      I_link = R_inertial @ I_raw @ R_inertial.T
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    robot_name = root.attrib.get("name", "robot")
    links: dict[str, Link] = {}
    joints: dict[str, Joint] = {}

    # Parse Links
    for link_elem in root.findall("link"):
        name = link_elem.attrib["name"]
        link = Link(name=name)

        inertial_elem = link_elem.find("inertial")
        if inertial_elem is not None:
            mass_elem = inertial_elem.find("mass")
            if mass_elem is not None:
                link.mass = float(mass_elem.attrib.get("value", 0.0))

            origin_elem = inertial_elem.find("origin")
            if origin_elem is not None:
                link.origin_xyz = _parse_vec3(origin_elem.attrib.get("xyz"))
                link.origin_rpy = _parse_vec3(origin_elem.attrib.get("rpy"))
                link.com = link.origin_xyz.copy()

            inertia_elem = inertial_elem.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.attrib.get("ixx", 0.0))
                ixy = float(inertia_elem.attrib.get("ixy", 0.0))
                ixz = float(inertia_elem.attrib.get("ixz", 0.0))
                iyy = float(inertia_elem.attrib.get("iyy", 0.0))
                iyz = float(inertia_elem.attrib.get("iyz", 0.0))
                izz = float(inertia_elem.attrib.get("izz", 0.0))

                I_raw = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ], dtype=np.float64)

                # Rotate raw inertia from inertial origin frame to link frame
                R_inertial = rpy_to_R(link.origin_rpy)
                link.inertia = R_inertial @ I_raw @ R_inertial.T

        links[name] = link

    # Parse Joints
    parent_links: set[str] = set()
    child_links: set[str] = set()

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "revolute")

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue

        parent_name = parent_elem.attrib["link"]
        child_name = child_elem.attrib["link"]

        parent_links.add(parent_name)
        child_links.add(child_name)

        joint = Joint(
            name=name,
            parent=parent_name,
            child=child_name,
            joint_type=joint_type,
        )

        origin_elem = joint_elem.find("origin")
        if origin_elem is not None:
            joint.origin_xyz = _parse_vec3(origin_elem.attrib.get("xyz"))
            joint.origin_rpy = _parse_vec3(origin_elem.attrib.get("rpy"))

        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            axis_vec = _parse_vec3(axis_elem.attrib.get("xyz"), default=(0.0, 0.0, 1.0))
            norm = np.linalg.norm(axis_vec)
            if norm > 1e-12:
                joint.axis = axis_vec / norm
            else:
                joint.axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            joint.lower = float(limit_elem.attrib.get("lower", -np.pi))
            joint.upper = float(limit_elem.attrib.get("upper", np.pi))
            joint.effort = float(limit_elem.attrib.get("effort", 100.0))
            joint.velocity = float(limit_elem.attrib.get("velocity", 10.0))

        joint.T_parent_child = xyz_rpy_to_T(joint.origin_xyz, joint.origin_rpy)
        joint.T_effective = joint.T_parent_child.copy()

        joints[name] = joint

    # Determine structural root link (parent links that are never children)
    candidate_roots = parent_links - child_links
    root_link = next(iter(candidate_roots)) if candidate_roots else (next(iter(links.keys())) if links else "")

    return Robot(
        name=robot_name,
        links=links,
        joints=joints,
        root_link=root_link,
    )
