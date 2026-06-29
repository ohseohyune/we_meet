#!/usr/bin/env python3
"""Build a pipe/flange inspection scene using the Modified-DH robot model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "robot_model_modified_dh.xml"
BASE_SCENE_XML = ROOT / "scene.xml"
OUT_XML = ROOT / "scene_modified_dh.xml"


def find_required(parent: ET.Element, tag: str, name: str | None = None) -> ET.Element:
    for child in parent:
        if child.tag != tag:
            continue
        if name is None or child.get("name") == name:
            return child
    label = tag if name is None else f"{tag} name={name!r}"
    raise ValueError(f"Could not find {label}.")


def indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print ElementTree output in-place."""
    pad = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = pad


def ensure_d405_camera(robot_root: ET.Element) -> None:
    """Add a fixed render camera at the MDH TCP frame if it is not present."""
    if robot_root.find(".//camera[@name='d405_camera']") is not None:
        return
    tcp_frame = robot_root.find(".//body[@name='tcp_frame']")
    if tcp_frame is None:
        raise ValueError("Modified-DH robot is missing body name='tcp_frame'.")
    camera = ET.Element("camera", {"name": "d405_camera", "pos": "0 0 0", "fovy": "65"})
    # Keep it before sites/geoms so the TCP body remains easy to inspect.
    tcp_frame.insert(0, camera)


def remove_robot_only_ground(worldbody: ET.Element) -> None:
    for child in list(worldbody):
        if child.tag == "geom" and child.get("name") == "ground":
            worldbody.remove(child)


def mount_robot_on_pedestal(worldbody: ET.Element) -> None:
    robot_base = find_required(worldbody, "body", "robot_base")
    robot_base.set("pos", "0 0 0.12")


def build_scene() -> ET.ElementTree:
    robot_tree = ET.parse(ROBOT_XML)
    scene_tree = ET.parse(BASE_SCENE_XML)
    robot_root = robot_tree.getroot()
    scene_root = scene_tree.getroot()

    ensure_d405_camera(robot_root)
    robot_root.set("model", "arm6dof_modified_dh_pipe_flange")

    # Reuse visual and asset settings from the working pipe/flange scene.
    for tag in ("visual", "asset"):
        existing = robot_root.find(tag)
        if existing is not None:
            robot_root.remove(existing)
        copied = scene_root.find(tag)
        if copied is not None:
            robot_root.insert(2 if tag == "visual" else 3, deepcopy(copied))

    robot_world = find_required(robot_root, "worldbody")
    scene_world = find_required(scene_root, "worldbody")
    remove_robot_only_ground(robot_world)
    mount_robot_on_pedestal(robot_world)

    for tag, name in (
        ("light", "sun"),
        ("light", "fill"),
        ("geom", "floor"),
        ("geom", "pedestal"),
        ("body", "pipe_flange_assembly"),
        ("body", "pipe_support"),
        ("body", "traj_markers"),
    ):
        try:
            copied = deepcopy(find_required(scene_world, tag, name))
            if tag == "geom" and name == "pedestal":
                copied.set("contype", "0")
                copied.set("conaffinity", "0")
            robot_world.append(copied)
        except ValueError:
            if name in {"sun", "fill"}:
                continue
            raise

    # The MDH robot has different body names than the original Standard-DH
    # scene, so the old contact exclusions would refer to missing bodies.
    old_contact = robot_root.find("contact")
    if old_contact is not None:
        robot_root.remove(old_contact)

    indent(robot_root)
    return ET.ElementTree(robot_root)


def main() -> None:
    if not ROBOT_XML.exists():
        raise FileNotFoundError(f"Missing {ROBOT_XML}")
    tree = build_scene()
    tree.write(OUT_XML, encoding="utf-8", xml_declaration=True)
    print(OUT_XML)


if __name__ == "__main__":
    main()
