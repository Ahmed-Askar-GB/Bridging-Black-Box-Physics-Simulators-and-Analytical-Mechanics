"""Render publication-quality 3D MuJoCo snapshots of the four benchmark robots for Figure 5.

Generates:
  - Full pipeline Manuscript revised/figures/robot_ur5.png
  - Full pipeline Manuscript revised/figures/robot_ur16e.png
  - Full pipeline Manuscript revised/figures/robot_abb.png
  - Full pipeline Manuscript revised/figures/robot_puma560.png
"""

from __future__ import annotations

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent.parent
MANUSCRIPT_FIGURES = WORKSPACE_ROOT / "Full pipeline Manuscript revised" / "figures"
MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)


def build_mjcf_for_robot(
    urdf_path: Path,
    mesh_dirs: list[Path],
    color_map: dict[str, str],
    floor_size: float = 3.0,
) -> str:
    """Parse URDF, resolve STL meshes, inject studio lighting, floor, and manufacturer colors."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    # Resolve all mesh filenames
    for m_elem in root.iter("mesh"):
        fn = m_elem.attrib.get("filename", "")
        base = os.path.basename(fn)
        base_no_ext = os.path.splitext(base)[0]
        for md in mesh_dirs:
            for ext in [".stl", ".obj"]:
                cand = os.path.join(str(md), base_no_ext + ext)
                if os.path.exists(cand):
                    m_elem.set("filename", cand.replace("\\", "/"))
                    break

    # Add compiler discardvisual='false' and balanceinertia='true'
    mj_elem = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mj_elem,
        "compiler",
        attrib={"discardvisual": "false", "balanceinertia": "true"},
    )

    s_urdf = ET.tostring(root, encoding="unicode")
    m_init = mujoco.MjModel.from_xml_string(s_urdf)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        tmp_path = f.name
    f.close()

    try:
        mujoco.mj_saveLastXML(tmp_path, m_init)
        mjcf_tree = ET.parse(tmp_path)
        mjcf_root = mjcf_tree.getroot()

        # Visual quality & offscreen framebuffer
        vis = mjcf_root.find("visual")
        if vis is None:
            vis = ET.SubElement(mjcf_root, "visual")
        ET.SubElement(vis, "global", attrib={"offwidth": "2048", "offheight": "1536"})
        ET.SubElement(vis, "quality", attrib={"shadowsize": "4096"})
        ET.SubElement(
            vis,
            "headlight",
            attrib={
                "ambient": "0.55 0.55 0.56",
                "diffuse": "0.80 0.80 0.82",
                "specular": "0.22 0.22 0.22",
            },
        )
        ET.SubElement(vis, "map", attrib={"shadowscale": "0.45", "znear": "0.02", "zfar": "25.0"})

        # Assets: Seamless Pure White Studio Sky and Subtle Shadow-Catcher Floor
        asset = mjcf_root.find("asset")
        if asset is None:
            asset = ET.SubElement(mjcf_root, "asset")

        ET.SubElement(
            asset,
            "texture",
            attrib={
                "name": "studio_sky",
                "type": "skybox",
                "builtin": "flat",
                "rgb1": "1.0 1.0 1.0",
                "rgb2": "1.0 1.0 1.0",
                "width": "512",
                "height": "512",
            },
        )
        ET.SubElement(
            asset,
            "texture",
            attrib={
                "name": "floor_texture",
                "type": "2d",
                "builtin": "flat",
                "rgb1": "0.99 0.99 0.99",
                "rgb2": "0.99 0.99 0.99",
                "width": "512",
                "height": "512",
            },
        )
        ET.SubElement(
            asset,
            "material",
            attrib={
                "name": "floor_mat",
                "texture": "floor_texture",
                "reflectance": "0.01",
                "roughness": "0.9",
            },
        )

        wb = mjcf_root.find("worldbody")

        # Studio lighting
        # Key light (directional, casts soft ground contact shadow)
        ET.SubElement(
            wb,
            "light",
            attrib={
                "name": "key_light",
                "directional": "true",
                "pos": "2.2 -2.2 3.8",
                "dir": "-0.5 0.5 -0.75",
                "castshadow": "true",
                "diffuse": "0.88 0.88 0.90",
                "specular": "0.30 0.30 0.30",
            },
        )
        # Fill light (soft ambient fill from opposite side)
        ET.SubElement(
            wb,
            "light",
            attrib={
                "name": "fill_light",
                "directional": "true",
                "pos": "-2.5 1.5 2.5",
                "dir": "0.7 -0.4 -0.6",
                "castshadow": "false",
                "diffuse": "0.40 0.40 0.42",
                "specular": "0.08 0.08 0.08",
            },
        )
        # Rim light (subtle edge highlight)
        ET.SubElement(
            wb,
            "light",
            attrib={
                "name": "rim_light",
                "directional": "true",
                "pos": "0.0 3.2 2.4",
                "dir": "0.0 -0.8 -0.6",
                "castshadow": "false",
                "diffuse": "0.32 0.32 0.35",
                "specular": "0.18 0.18 0.18",
            },
        )

        # Ground plane
        ET.SubElement(
            wb,
            "geom",
            attrib={
                "name": "studio_floor",
                "type": "plane",
                "size": f"{floor_size} {floor_size} 0.05",
                "material": "floor_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )

        # Apply coloring and eliminate duplicate collision geoms
        for b in mjcf_root.iter("body"):
            b_name = b.attrib.get("name", "")
            for g in b.findall("geom"):
                g_grp = g.attrib.get("group")
                m_name = g.attrib.get("mesh", "")
                if g_grp == "0" or (g_grp is None and any(x.attrib.get("group") == "1" for x in b.findall("geom"))):
                    g.attrib["rgba"] = "0 0 0 0"
                    g.attrib["group"] = "3"
                    g.attrib["contype"] = "0"
                    g.attrib["conaffinity"] = "0"
                else:
                    g.attrib["contype"] = "0"
                    g.attrib["conaffinity"] = "0"
                    for key, rgba_val in color_map.items():
                        if key in b_name.lower() or (m_name and key in m_name.lower()):
                            g.attrib["rgba"] = rgba_val
                            break

        for g in wb.findall("geom"):
            if g.attrib.get("name") != "studio_floor":
                m_name = g.attrib.get("mesh", "")
                for key, rgba_val in color_map.items():
                    if key in m_name.lower() or key in ["base", "link1"]:
                        g.attrib["rgba"] = rgba_val
                        break

        return ET.tostring(mjcf_root, encoding="unicode")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def render_robot_snapshot(
    name: str,
    mjcf_str: str,
    qpos: np.ndarray,
    cam_lookat: list[float],
    cam_dist: float,
    cam_azimuth: float,
    cam_elevation: float,
    out_path: Path,
    img_width: int = 1600,
    img_height: int = 1200,
    target_out_w: int = 1200,
    target_out_h: int = 900,
):
    """Compile MJCF, advance forward kinematics to showcase pose, render offscreen snapshot, and autocrop with consistent framing."""
    m = mujoco.MjModel.from_xml_string(mjcf_str)
    d = mujoco.MjData(m)

    if len(qpos) <= m.nq:
        d.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, img_height, img_width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_lookat
    cam.distance = cam_dist
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation

    renderer.update_scene(d, camera=cam)
    raw_img = renderer.render()

    # Intelligent bounding box crop centered on robot + shadow with 4:3 aspect ratio
    arr = np.array(raw_img)
    diff = np.abs(arr.astype(float) - [255, 255, 255]).sum(axis=-1)
    mask = diff > 15
    ys, xs = np.where(mask)
    if len(ys) > 0:
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        w_box = xmax - xmin
        h_box = ymax - ymin
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0

        target_ar = target_out_w / float(target_out_h)  # 4:3
        pad = 1.15
        req_w = w_box * pad
        req_h = h_box * pad
        if req_w / req_h < target_ar:
            req_w = req_h * target_ar
        else:
            req_h = req_w / target_ar

        left = int(round(xc - req_w / 2.0))
        right = int(round(xc + req_w / 2.0))
        top = int(round(yc - req_h / 2.0))
        bottom = int(round(yc + req_h / 2.0))

        # Pad canvas if crop extends beyond raw image boundaries
        pad_top = max(0, -top)
        pad_left = max(0, -left)
        pad_bottom = max(0, bottom - img_height)
        pad_right = max(0, right - img_width)

        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            padded_arr = np.pad(
                arr,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=255,
            )
            top += pad_top
            bottom += pad_top
            left += pad_left
            right += pad_left
            cropped = padded_arr[top:bottom, left:right]
        else:
            cropped = arr[top:bottom, left:right]

        pil_img = Image.fromarray(cropped).resize((target_out_w, target_out_h), Image.Resampling.LANCZOS)
    else:
        pil_img = Image.fromarray(raw_img).resize((target_out_w, target_out_h), Image.Resampling.LANCZOS)

    pil_img.save(str(out_path), quality=95)
    print(f"[{name.upper()}] Perfectly framed snapshot -> {out_path} ({target_out_w}x{target_out_h})")


def main():
    print("=" * 80)
    print("  GENERATING MUJOCO ROBOT GALLERY SNAPSHOTS FOR FIGURE 5")
    print("=" * 80)

    # Robot specifications, accurate geometry framing, and manufacturer color palettes
    configs = {
        "robot_ur5": {
            "title": "Universal Robots UR5",
            "urdf": CURRENT_DIR / "tests" / "fixtures" / "ur5.urdf",
            "mesh_dirs": [
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "STL_Files",
            ],
            "color_map": {
                "base": "0.32 0.35 0.38 1.0",
                "shoulder": "0.10 0.46 0.73 1.0",
                "upper": "0.86 0.88 0.90 1.0",
                "forearm": "0.86 0.88 0.90 1.0",
                "wrist_1": "0.10 0.46 0.73 1.0",
                "wrist1": "0.10 0.46 0.73 1.0",
                "wrist_2": "0.86 0.88 0.90 1.0",
                "wrist2": "0.86 0.88 0.90 1.0",
                "wrist_3": "0.10 0.46 0.73 1.0",
                "wrist3": "0.10 0.46 0.73 1.0",
            },
            # Showcase pose displaying kinematics: [pan, lift, elbow, wrist1, wrist2, wrist3]
            "qpos": np.array([0.25, -1.05, 1.45, -1.95, -1.57, 0.0]),
            # Centered on active 3D volume
            "cam_lookat": [0.30, 0.12, 0.25],
            "cam_dist": 1.25,
            "cam_azimuth": 140.0,
            "cam_elevation": -20.0,
            "floor_size": 2.5,
        },
        "robot_ur16e": {
            "title": "Universal Robots UR16e",
            "urdf": CURRENT_DIR / "tests" / "fixtures" / "ur16e.urdf",
            "mesh_dirs": [
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "meshes" / "ur16e" / "visual",
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "meshes" / "ur10e" / "visual",
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "meshes" / "ur16e" / "collision",
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "End-to-End MATLAB-CasADi Implementation" / "test_UR16e" / "meshes" / "ur10e" / "collision",
            ],
            "color_map": {
                "base": "0.18 0.20 0.23 1.0",
                "shoulder": "0.06 0.35 0.62 1.0",
                "upper": "0.28 0.31 0.35 1.0",
                "forearm": "0.28 0.31 0.35 1.0",
                "wrist_1": "0.06 0.35 0.62 1.0",
                "wrist1": "0.06 0.35 0.62 1.0",
                "wrist_2": "0.28 0.31 0.35 1.0",
                "wrist2": "0.28 0.31 0.35 1.0",
                "wrist_3": "0.06 0.35 0.62 1.0",
                "wrist3": "0.06 0.35 0.62 1.0",
            },
            "qpos": np.array([0.20, -1.00, 1.40, -1.95, -1.57, 0.25]),
            "cam_lookat": [0.32, 0.15, 0.32],
            "cam_dist": 1.40,
            "cam_azimuth": 140.0,
            "cam_elevation": -20.0,
            "floor_size": 2.5,
        },
        "robot_abb": {
            "title": "ABB IRB 120",
            "urdf": WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "abbIrb120" / "abbIrb120.urdf",
            "mesh_dirs": [
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "abbIrb120" / "meshes" / "irb120_3_58" / "visual",
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "abbIrb120" / "meshes" / "irb120_3_58" / "collision",
            ],
            "color_map": {
                "base": "0.93 0.42 0.12 1.0",
                "link_1": "0.93 0.42 0.12 1.0",
                "link_2": "0.93 0.42 0.12 1.0",
                "link_3": "0.93 0.42 0.12 1.0",
                "link_4": "0.93 0.42 0.12 1.0",
                "link_5": "0.93 0.42 0.12 1.0",
                "link_6": "0.18 0.19 0.21 1.0",
            },
            # Articulated pose displaying spherical wrist: [j1, j2, j3, j4, j5, j6]
            "qpos": np.array([0.25, -0.40, 0.60, 0.0, 0.70, 0.0]),
            "cam_lookat": [0.08, 0.03, 0.28],
            "cam_dist": 1.18,
            "cam_azimuth": 142.0,
            "cam_elevation": -18.0,
            "floor_size": 2.0,
        },
        "robot_puma560": {
            "title": "Unimation Puma 560",
            "urdf": WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "End-to-End MATLAB Implementation" / "puma560_test" / "puma560_robot_dynamic.urdf",
            "mesh_dirs": [
                WORKSPACE_ROOT / "Bridging Black-Box Physics Simulators and Analytical Mechanics" / "End-to-End MATLAB Implementation" / "puma560_test" / "meshes",
            ],
            "color_map": {
                "link1": "0.26 0.28 0.30 1.0",
                "link2": "0.92 0.90 0.85 1.0",
                "link3": "0.92 0.90 0.85 1.0",
                "link4": "0.24 0.25 0.27 1.0",
                "link5": "0.92 0.90 0.85 1.0",
                "link6": "0.24 0.25 0.27 1.0",
                "link7": "0.72 0.74 0.76 1.0",
                "puma_link1": "0.26 0.28 0.30 1.0",
                "puma_link2": "0.92 0.90 0.85 1.0",
                "puma_link3": "0.92 0.90 0.85 1.0",
                "puma_link4": "0.24 0.25 0.27 1.0",
                "puma_link5": "0.92 0.90 0.85 1.0",
                "puma_link6": "0.24 0.25 0.27 1.0",
                "puma_link7": "0.72 0.74 0.76 1.0",
            },
            # Articulated pose displaying spatial offset: [j1, j2, j3, j4, j5, j6]
            "qpos": np.array([0.0, -0.55, 1.15, 0.0, 0.70, 0.0]),
            "cam_lookat": [0.28, -0.10, 0.36],
            "cam_dist": 1.45,
            "cam_azimuth": 142.0,
            "cam_elevation": -18.0,
            "floor_size": 2.5,
        },
    }

    for fig_id, cfg in configs.items():
        print(f"\nProcessing {cfg['title']} ({fig_id})...")
        mjcf_str = build_mjcf_for_robot(
            urdf_path=cfg["urdf"],
            mesh_dirs=cfg["mesh_dirs"],
            color_map=cfg["color_map"],
            floor_size=cfg["floor_size"],
        )

        out_file = MANUSCRIPT_FIGURES / f"{fig_id}.png"
        render_robot_snapshot(
            name=fig_id,
            mjcf_str=mjcf_str,
            qpos=cfg["qpos"],
            cam_lookat=cfg["cam_lookat"],
            cam_dist=cfg["cam_dist"],
            cam_azimuth=cfg["cam_azimuth"],
            cam_elevation=cfg["cam_elevation"],
            out_path=out_file,
            img_width=1600,
            img_height=1000,
        )

    print("\n" + "=" * 80)
    print("  ALL 4 ROBOT GALLERY SNAPSHOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
