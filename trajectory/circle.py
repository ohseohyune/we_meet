"""
Segmented circular Cartesian trajectory for flange inspection.
"""

from __future__ import annotations

import numpy as np

from reference.dream_math import T_from_Rp, quintic_spline


BOTTOM_FORBIDDEN_CENTER = -np.pi / 2.0
BOTTOM_FORBIDDEN_HALF_ANGLE = np.deg2rad(30.0)
BOTTOM_LEFT_LIMIT = 3.0 * np.pi / 2.0 - BOTTOM_FORBIDDEN_HALF_ANGLE
BOTTOM_RIGHT_LIMIT = -np.pi / 2.0 + BOTTOM_FORBIDDEN_HALF_ANGLE


SEGMENTS = (
    {
        "name": "top_to_lower_left_ccw",
        "description": "Segment 1: 12시(top) -> lower-left safe limit, CCW",
        "phi0": np.pi / 2.0,
        "phif": BOTTOM_LEFT_LIMIT,
    },
    {
        "name": "lower_left_to_top_cw",
        "description": "Segment 2: lower-left safe limit -> 12시(top), CW",
        "phi0": BOTTOM_LEFT_LIMIT,
        "phif": np.pi / 2.0,
    },
    {
        "name": "top_to_lower_right_cw",
        "description": "Segment 3: 12시(top) -> lower-right safe limit, CW",
        "phi0": np.pi / 2.0,
        "phif": BOTTOM_RIGHT_LIMIT,
    },
    {
        "name": "lower_right_to_top_ccw_final",
        "description": "Segment 4: lower-right safe limit -> 12시(top), CCW",
        "phi0": BOTTOM_RIGHT_LIMIT,
        "phif": np.pi / 2.0,
    },
)


FEASIBLE_CAPTURE_SEGMENTS = (
    {
        "name": "top_left_capture",
        "description": "Feasible capture arc near upper-left seam.",
        "base_segment_id": 1,
        "phi0": np.deg2rad(90.0),
        "phif": np.deg2rad(106.0),
    },
    {
        "name": "lower_left_capture",
        "description": "Feasible capture arc near lower-left seam.",
        "base_segment_id": 1,
        "phi0": np.deg2rad(228.0),
        "phif": np.deg2rad(239.0),
    },
    {
        "name": "lower_left_return_capture",
        "description": "Feasible return capture arc near lower-left seam.",
        "base_segment_id": 2,
        "phi0": np.deg2rad(239.0),
        "phif": np.deg2rad(235.0),
    },
    {
        "name": "top_left_return_capture",
        "description": "Feasible return capture arc near upper-left seam.",
        "base_segment_id": 2,
        "phi0": np.deg2rad(119.0),
        "phif": np.deg2rad(90.0),
    },
    {
        "name": "top_right_capture",
        "description": "Feasible capture arc near upper-right seam.",
        "base_segment_id": 3,
        "phi0": np.deg2rad(90.0),
        "phif": np.deg2rad(61.0),
    },
    {
        "name": "lower_right_capture",
        "description": "Feasible capture arc near lower-right seam.",
        "base_segment_id": 3,
        "phi0": np.deg2rad(-37.0),
        "phif": np.deg2rad(-59.0),
    },
    {
        "name": "lower_right_return_capture",
        "description": "Feasible return capture arc near lower-right seam.",
        "base_segment_id": 4,
        "phi0": np.deg2rad(-59.0),
        "phif": np.deg2rad(-48.0),
    },
    {
        "name": "top_right_return_capture",
        "description": "Feasible return capture arc near upper-right seam.",
        "base_segment_id": 4,
        "phi0": np.deg2rad(73.0),
        "phif": np.deg2rad(90.0),
    },
)


DEFAULT_MULTI_RING_SPECS = (
    {
        "name": "near_oblique",
        "description": "closer oblique view of the seam",
        "x_offset": -0.1533,
        "radius": 0.120,
    },
    {
        "name": "nominal",
        "description": "nominal inspection view",
        "x_offset": -0.1533,
        "radius": 0.120,
    },
    {
        "name": "far_oblique",
        "description": "farther oblique view for depth fusion overlap",
        "x_offset": -0.1533,
        "radius": 0.120,
    },
)


def inside_orientation(position: np.ndarray, target: np.ndarray, phi: float | None = None) -> np.ndarray:
    """Physical camera frame with optical -z aimed at the matching seam point.

    Roll is fixed so camera +y follows the seam tangent for the current
    circumferential angle.  That keeps the captured image frame aligned with
    the weld seam while camera -z looks at the seam point.
    """
    look_axis = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    look_axis /= np.linalg.norm(look_axis) + 1e-12

    z_axis = -look_axis

    if phi is None:
        y_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        y_ref = np.array([0.0, -np.sin(phi), np.cos(phi)], dtype=float)

    y_axis = y_ref - np.dot(y_ref, z_axis) * z_axis
    if np.linalg.norm(y_axis) < 1e-9:
        y_ref = np.array([1.0, 0.0, 0.0], dtype=float)
        y_axis = y_ref - np.dot(y_ref, z_axis) * z_axis
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    x_axis = np.cross(y_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        x_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        x_axis = x_ref - np.dot(x_ref, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    return np.column_stack([x_axis, y_axis, z_axis])


def pose_from_position(position: np.ndarray, R: np.ndarray) -> np.ndarray:
    return T_from_Rp(R, np.asarray(position, dtype=float).reshape(3))


def total_segment_angle() -> float:
    """Total absolute angular travel over the requested inspection segments."""
    return float(sum(abs(seg["phif"] - seg["phi0"]) for seg in SEGMENTS))


def signed_angle_delta(phi: float, center: float) -> float:
    """Shortest signed angular delta from center to phi, in radians."""
    return float(np.arctan2(np.sin(phi - center), np.cos(phi - center)))


def is_in_bottom_forbidden_sector(
    phi: float,
    center: float = BOTTOM_FORBIDDEN_CENTER,
    half_angle: float = BOTTOM_FORBIDDEN_HALF_ANGLE,
) -> bool:
    """Return True when phi is inside the lower support-bar exclusion sector."""
    return abs(signed_angle_delta(phi, center)) < float(half_angle) - 1e-12


def horizontal_fov_from_vertical(width: int, height: int, fovy_deg: float) -> float:
    aspect = float(width) / float(height)
    fovy = np.deg2rad(fovy_deg)
    return float(2.0 * np.arctan(aspect * np.tan(0.5 * fovy)))


def estimate_frames_for_overlap(
    camera_x_offset: float,
    camera_radius: float,
    target_radius: float,
    fovy_deg: float,
    width: int,
    height: int,
    overlap: float = 0.75,
    min_frames: int = 20,
) -> int:
    """
    Estimate capture count for one segmented seam pass from desired image overlap.

    This approximates visible seam arc length using the horizontal FOV projected
    at the camera-to-seam distance.  A minimum count is kept because the D405 FOV
    is wide relative to the synthetic DN100 seam, which otherwise yields too few
    frames for robust stitching or depth fusion.
    """
    overlap = float(np.clip(overlap, 0.0, 0.95))
    distance_to_seam = np.linalg.norm(
        [camera_x_offset, max(camera_radius - target_radius, 1e-6)]
    )
    fovx = horizontal_fov_from_vertical(width, height, fovy_deg)
    visible_width = 2.0 * distance_to_seam * np.tan(0.5 * fovx)
    seam_arc = target_radius * total_segment_angle()
    step_arc = max(visible_width * (1.0 - overlap), 1e-6)
    return int(max(min_frames, np.ceil(seam_arc / step_arc) + 1))


def estimate_multi_ring_frames(
    ring_specs=DEFAULT_MULTI_RING_SPECS,
    target_radius: float = 0.03025,
    fovy_deg: float = 58.0,
    width: int = 1280,
    height: int = 800,
    overlap: float = 0.75,
    min_frames_per_ring: int = 20,
) -> tuple[int, list[int]]:
    """Return total frame count and per-ring frame counts for overlap-driven capture."""
    per_ring = [
        estimate_frames_for_overlap(
            camera_x_offset=float(ring["x_offset"]),
            camera_radius=float(ring["radius"]),
            target_radius=target_radius,
            fovy_deg=fovy_deg,
            width=width,
            height=height,
            overlap=overlap,
            min_frames=min_frames_per_ring,
        )
        for ring in ring_specs
    ]
    return int(sum(per_ring)), per_ring


def segmented_circle_trajectory(
    center: tuple[float, float, float] | np.ndarray = (0.30250, 0.0, 0.57),
    radius: float = 0.100,
    segment_duration: float = 9.0,
    dt: float = 0.01,
    yaw: float = 0.0,
    orientation_target: tuple[float, float, float] | np.ndarray = (0.50250, 0.0, 0.57),
    target_radius: float = 0.03025,
    feasible_only: bool = False,
) -> dict:
    """
    Generate independent quintic-spline circle segments in the YZ plane.

    The camera path is still a larger orbit around the pipe/flange assembly,
    but each pose looks at the pipe/flange contact seam point for the same
    angle phi instead of always looking at the flange center.

    This is the original XY circle rotated 90 degrees about the y-axis, then
    reparameterized so 12시(top) is +z and 6시(bottom) is -z.

    The lower support bar is treated as a forbidden sector around 6시(bottom).
    With the default 30 degree half-angle, the generated path never enters
    240..300 degrees.  Each side of the circle is scanned as an open arc:
      1. top -> lower-left safe limit
      2. lower-left safe limit -> top
      3. top -> lower-right safe limit
      4. lower-right safe limit -> top
    """
    center = np.asarray(center, dtype=float).reshape(3)
    orientation_target = np.asarray(orientation_target, dtype=float).reshape(3)
    time_values: list[float] = []
    angle_values: list[float] = []
    positions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    segment_ids: list[int] = []
    base_segment_ids: list[int] = []
    segment_names: list[str] = []

    active_segments = FEASIBLE_CAPTURE_SEGMENTS if feasible_only else SEGMENTS
    for seg_id, segment in enumerate(active_segments):
        phi0 = segment["phi0"]
        phif = segment["phif"]
        local_t = np.arange(0.0, segment_duration + dt * 0.5, dt)
        if seg_id > 0:
            local_t = local_t[1:]

        for tau in local_t:
            s, _, _ = quintic_spline(
                tau,
                0.0,
                segment_duration,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            phi = phi0 + s * (phif - phi0)
            p = center + np.array(
                [0.0, radius * np.cos(phi), radius * np.sin(phi)],
                dtype=float,
            )
            target = orientation_target + np.array(
                [0.0, target_radius * np.cos(phi), target_radius * np.sin(phi)],
                dtype=float,
            )

            time_values.append(seg_id * segment_duration + tau)
            angle_values.append(phi)
            positions.append(p)
            targets.append(target)
            R = inside_orientation(p, target, phi)
            poses.append(pose_from_position(p, R))
            segment_ids.append(seg_id + 1)
            base_segment_ids.append(int(segment.get("base_segment_id", seg_id + 1)))
            segment_names.append(segment["name"])

    return {
        "time": np.asarray(time_values),
        "angles": np.asarray(angle_values),
        "positions": np.asarray(positions),
        "targets": np.asarray(targets),
        "poses": np.asarray(poses),
        "segment_id": np.asarray(segment_ids, dtype=int),
        "base_segment_id": np.asarray(base_segment_ids, dtype=int),
        "segment_name": np.asarray(segment_names),
        "segment_definitions": active_segments,
        "feasible_only": bool(feasible_only),
        "bottom_forbidden_center": float(BOTTOM_FORBIDDEN_CENTER),
        "bottom_forbidden_half_angle": float(BOTTOM_FORBIDDEN_HALF_ANGLE),
        "bottom_forbidden_width": float(2.0 * BOTTOM_FORBIDDEN_HALF_ANGLE),
        "center": center,
        "orientation_target": orientation_target,
        "target_radius": float(target_radius),
        "radius": float(radius),
    }


def multi_ring_segmented_trajectory(
    seam_center: tuple[float, float, float] | np.ndarray = (0.50250, 0.0, 0.57),
    ring_specs=DEFAULT_MULTI_RING_SPECS,
    segment_duration: float = 9.0,
    dt: float = 0.01,
    target_radius: float = 0.03025,
    feasible_only: bool = False,
) -> dict:
    """Generate multiple segmented inspection rings around the same seam circle."""
    seam_center = np.asarray(seam_center, dtype=float).reshape(3)
    merged: dict[str, list] = {
        "time": [],
        "angles": [],
        "positions": [],
        "targets": [],
        "poses": [],
        "segment_id": [],
        "base_segment_id": [],
        "segment_name": [],
        "ring_id": [],
        "ring_name": [],
        "ring_radius": [],
        "ring_x_offset": [],
    }

    time_offset = 0.0
    for ring_index, ring in enumerate(ring_specs, start=1):
        x_offset = float(ring["x_offset"])
        radius = float(ring["radius"])
        center = seam_center + np.array([x_offset, 0.0, 0.0])
        traj = segmented_circle_trajectory(
            center=center,
            radius=radius,
            segment_duration=segment_duration,
            dt=dt,
            orientation_target=seam_center,
            target_radius=target_radius,
            feasible_only=feasible_only,
        )

        ring_time = traj["time"] + time_offset
        if ring_index > 1:
            # Avoid duplicate timestamps at ring boundaries.
            ring_time = ring_time + dt

        count = len(ring_time)
        merged["time"].extend(ring_time)
        merged["angles"].extend(traj["angles"])
        merged["positions"].extend(traj["positions"])
        merged["targets"].extend(traj["targets"])
        merged["poses"].extend(traj["poses"])
        merged["segment_id"].extend(traj["segment_id"])
        merged["base_segment_id"].extend(traj["base_segment_id"])
        merged["segment_name"].extend(traj["segment_name"])
        merged["ring_id"].extend([ring_index] * count)
        merged["ring_name"].extend([ring["name"]] * count)
        merged["ring_radius"].extend([radius] * count)
        merged["ring_x_offset"].extend([x_offset] * count)
        time_offset = float(ring_time[-1])

    return {
        "time": np.asarray(merged["time"], dtype=float),
        "angles": np.asarray(merged["angles"], dtype=float),
        "positions": np.asarray(merged["positions"], dtype=float),
        "targets": np.asarray(merged["targets"], dtype=float),
        "poses": np.asarray(merged["poses"], dtype=float),
        "segment_id": np.asarray(merged["segment_id"], dtype=int),
        "base_segment_id": np.asarray(merged["base_segment_id"], dtype=int),
        "segment_name": np.asarray(merged["segment_name"]),
        "ring_id": np.asarray(merged["ring_id"], dtype=int),
        "ring_name": np.asarray(merged["ring_name"]),
        "ring_radius": np.asarray(merged["ring_radius"], dtype=float),
        "ring_x_offset": np.asarray(merged["ring_x_offset"], dtype=float),
        "ring_specs": tuple(dict(ring) for ring in ring_specs),
        "feasible_only": bool(feasible_only),
        "bottom_forbidden_center": float(BOTTOM_FORBIDDEN_CENTER),
        "bottom_forbidden_half_angle": float(BOTTOM_FORBIDDEN_HALF_ANGLE),
        "bottom_forbidden_width": float(2.0 * BOTTOM_FORBIDDEN_HALF_ANGLE),
        "center": seam_center,
        "orientation_target": seam_center,
        "target_radius": float(target_radius),
        "radius": np.asarray(merged["ring_radius"], dtype=float),
    }
