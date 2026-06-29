"""Configuration dataclasses for dh2mujoco."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple


@dataclass
class DHColumnOrder:
    """Maps each DH parameter name to its column index in the raw (N×4) array.

    Default matches the problem statement convention: (a, alpha, d, theta).
    Set indices to match whatever column layout your source data uses.
    """

    a: int = 0      # a_{i-1} for Modified DH, a_i for Standard DH
    alpha: int = 1  # alpha_{i-1} for Modified DH, alpha_i for Standard DH
    d: int = 2      # d_i
    theta: int = 3  # theta_i  (offset or current angle, depending on mode)


@dataclass
class JointConfig:
    """Per-joint MJCF attributes applied to every revolute joint."""

    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    range: Tuple[float, float] = (-3.14159265358979, 3.14159265358979)
    damping: float = 2.0
    frictionloss: float = 0.1
    armature: float = 0.05
    gear: float = 1.0


@dataclass
class Config:
    """Top-level configuration for the DH → MJCF conversion pipeline.

    Modes
    -----
    1 : Modified DH (Craig). ``theta`` column is the joint *offset*;
        the actual joint angle is added on top.
        Transform: Rx(alpha) * Tx(a) * Rz(theta_off + q) * Tz(d)

    2 : Modified DH, ``theta`` treated as the current joint angle (ignored
        for XML generation; set to zero in the body hierarchy).
        Transform: Rx(alpha) * Tx(a) * Rz(q) * Tz(d)

    3 : Standard DH. ``theta`` is the joint offset.
        Transform: Rz(theta_off + q) * Tz(d) * Tx(a) * Rx(alpha)

    4 : Modified DH with offsets supplied separately via ``joint_offsets``.
        Transform: same as Mode 1 but uses ``joint_offsets[i]`` instead of
        the table's theta column.
    """

    # DH convention mode (1–4, see docstring)
    MODE: Literal[1, 2, 3, 4] = 1

    # Column mapping from raw DH array
    column_order: DHColumnOrder = field(default_factory=DHColumnOrder)

    # Geometry
    GEOM_TYPE: Literal["capsule", "box"] = "capsule"
    LINK_RADIUS: float = 0.014

    # Debug output
    SHOW_FRAME: bool = True
    SHOW_MATRIX: bool = False

    # XML output
    EXPORT_XML: bool = True
    OUTPUT_PATH: str = "dh2mujoco/output/generated_robot.xml"

    # Orientation representation in the XML
    USE_QUATERNION: bool = True   # if False, tries USE_EULER
    USE_EULER: bool = False

    # Joint parameters
    joint: JointConfig = field(default_factory=JointConfig)

    # Used only by Mode 4: explicit theta offsets, one per joint
    joint_offsets: List[float] = field(default_factory=list)

    # Roll-to-bend correction: set of 1-indexed joint numbers whose rotation
    # axis is parallel to the link (roll joint) and should be redirected to
    # the perpendicular direction so that the joint visibly bends the arm.
    # Applied by post-multiplying the pre-body quat by Rx(π/2) and adjusting
    # the post-body pos/quat to preserve FK at q=0.
    roll_fix_joints: set = field(default_factory=set)

    # Joint axis alignment overrides.  Keys are 1-indexed source joint
    # numbers, values are 1-indexed target joint numbers.  The source joint
    # keeps its own body pose, but its local MJCF axis is chosen so its world
    # rotation axis at q=0 matches the target joint's world rotation axis.
    same_axis_as: Dict[int, int] = field(default_factory=dict)

    # Home (keyframe) joint angles in radians, one per joint.
    # Empty list → all zeros.
    home_qpos: List[float] = field(default_factory=list)
