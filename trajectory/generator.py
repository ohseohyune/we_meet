"""
trajectory_generator.py — Circular 360° inspection trajectory around a pipe flange.

Coordinate Frames (all in robot base frame)
--------------------------------------------
  World/Base  : origin at robot mounting point, z-up, x-forward.
  Pipe        : cylinder axis along world +x; pipe end (flange face) at x = PIPE_OFFSET_X + PIPE_LENGTH.
  Flange      : origin at pipe-end center; face normal along -x (facing robot).
  Flange_face : same as Flange frame.
  Seam        : circular pipe/flange contact line at pipe OD radius.
  Camera      : optical center follows the larger inspection circle.
  TCP         : located 100 mm behind camera along the same look-at axis in mujoco_viewer.py.

Trajectory Assumptions
-----------------------
- Inspection plane is perpendicular to the pipe axis (world +x).
- Robot base at world origin; pipe center placed at (PIPE_OFFSET_X, 0, PIPE_HEIGHT).
- Flange face at x = PIPE_OFFSET_X + PIPE_LENGTH, at height z = PIPE_HEIGHT.
- Seam target is the pipe/flange contact circle at pipe outer radius.
- Trajectory is a larger circle in the y-z plane, offset toward the robot.
- Camera always looks from trajectory point toward the current seam point.
- mujoco_viewer.py converts camera waypoints to TCP waypoints using:
  p_tcp = p_camera - 0.10 * z_camera.
- World +z used as up-hint for computing right-hand camera frame.
- N waypoints evenly spaced from 0° to 360° (start==end for closed loop).
- Smooth SLERP orientation interpolation provided as a utility.
"""

import numpy as np
from typing import List, Tuple


# ── Scene configuration ──────────────────────────────────────────────────────
# Robot workspace centroid is near (0, 0, -0.59) based on DH sampling.
# The arm reaches ±0.8m in x/y and -1.1 to 0 in z.
# Pipe/flange placed within robot workspace:
#   - Pipe along world +x, starting at x=0.30, ending at x=0.55
#   - Flange face at x=0.55, pipe axis height at z=0.57
# Inspection circle in y-z plane (perpendicular to pipe x-axis):
#   - seam center at (0.55, 0, 0.57)
#   - radius = 0.1652 m
PIPE_OFFSET_X  = 0.30   # x-distance from robot base to pipe start (m)
PIPE_LENGTH    = 0.25   # pipe length (m)
PIPE_HEIGHT    = 0.57   # z-height of pipe center axis (m) in z-up world
PIPE_OD        = 0.0605  # pipe outer diameter matches the flange bore (m)
STANDOFF       = 0.1652173913  # camera orbit radius in the y-z plane (m)
TRAJECTORY_X_OFFSET = -0.18  # move camera orbit toward robot along world -x (m)
N_WAYPOINTS    = 36     # waypoints around 360° (every 10°)

# Derived
FLANGE_CENTER = np.array([PIPE_OFFSET_X + PIPE_LENGTH, 0.0, PIPE_HEIGHT])
FLANGE_NORMAL = np.array([-1.0, 0.0, 0.0])   # face normal points toward robot
SEAM_RADIUS = PIPE_OD / 2.0
SEAM_CENTER = FLANGE_CENTER.copy()


# ── Frame-building utilities ─────────────────────────────────────────────────

def look_at_rotation(pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
    """
    Compute 3×3 rotation matrix such that:
      +z axis points from `pos` toward `target`
      +y axis is as close as possible to `up_hint` (default: world +z)
      +x = +y × +z (right-hand)

    This defines the camera frame: camera looks along its +z.
    """
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])

    z_cam = target - pos
    norm = np.linalg.norm(z_cam)
    if norm < 1e-9:
        raise ValueError("Camera position coincides with target.")
    z_cam /= norm

    # Degenerate: camera looking straight up/down
    if abs(np.dot(z_cam, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])

    x_cam = np.cross(up_hint, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    R = np.column_stack([x_cam, y_cam, z_cam])  # columns are axes
    return R


def make_pose(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform from pos and 3×3 rotation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos
    return T


# ── Trajectory equations ─────────────────────────────────────────────────────
#
# Let φ ∈ [0, 2π) be the inspection angle.
#
# Seam target (pipe/flange contact line):
#   s(φ) = flange_center + r_pipe * [0, cos(φ), sin(φ)]
#
# Camera position (in base frame):
#   p(φ) = flange_center + [x_offset, R * cos(φ), R * sin(φ)]
#   where R = STANDOFF, and the circle is in the y-z plane.
#
# Camera orientation:
#   Computed by look_at(p(φ), s(φ)), so +z always points at the
#   current pipe/flange seam point.
#
# This gives a smooth orbit where the camera always faces the inspection seam,
# moving in a full 360° circle in the plane perpendicular to the pipe axis.

def trajectory_position(phi: float) -> np.ndarray:
    """Camera optical-center position for inspection angle φ (radians)."""
    offset = np.array([TRAJECTORY_X_OFFSET, STANDOFF * np.cos(phi), STANDOFF * np.sin(phi)])
    return FLANGE_CENTER + offset


def seam_target_position(phi: float) -> np.ndarray:
    """Pipe/flange seam target point for inspection angle φ (radians)."""
    offset = np.array([0.0, SEAM_RADIUS * np.cos(phi), SEAM_RADIUS * np.sin(phi)])
    return SEAM_CENTER + offset


def trajectory_orientation(phi: float) -> np.ndarray:
    """3×3 rotation (look-at) for inspection angle φ (radians)."""
    pos = trajectory_position(phi)
    return look_at_rotation(pos, seam_target_position(phi))


def trajectory_pose(phi: float) -> np.ndarray:
    """4×4 TCP pose (base frame) for inspection angle φ (radians)."""
    pos = trajectory_position(phi)
    R   = trajectory_orientation(phi)
    return make_pose(pos, R)


# ── Waypoint generation ──────────────────────────────────────────────────────

def generate_waypoints(
    n: int = N_WAYPOINTS,
    start_deg: float = 0.0,
    full_circle: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate N evenly spaced inspection waypoints.

    Returns
    -------
    angles   : (N,) array of phi values [rad]
    positions: list of N position vectors [m]
    rotations: list of N 3×3 rotation matrices
    poses    : list of N 4×4 homogeneous transforms
    """
    start_rad = np.deg2rad(start_deg)
    if full_circle:
        # Include endpoint = startpoint for closed loop visualization
        angles = np.linspace(start_rad, start_rad + 2 * np.pi, n, endpoint=False)
    else:
        angles = np.linspace(start_rad, start_rad + 2 * np.pi, n)

    positions = []
    rotations = []
    poses     = []

    for phi in angles:
        pos = trajectory_position(phi)
        R   = trajectory_orientation(phi)
        T   = make_pose(pos, R)
        positions.append(pos)
        rotations.append(R)
        poses.append(T)

    return angles, positions, rotations, poses


# ── SLERP interpolation for smooth path densification ───────────────────────

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions q0 and q1, t∈[0,1]."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = min(1.0, dot)
    theta = np.arccos(dot) * t
    q_perp = q1 - q0 * dot
    n = np.linalg.norm(q_perp)
    if n < 1e-9:
        return q0
    q_perp /= n
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [w, x, y, z]."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def densify_trajectory(poses: list, steps_between: int = 5) -> list:
    """Insert `steps_between` interpolated poses between each waypoint pair."""
    dense = []
    for i in range(len(poses) - 1):
        T0, T1 = poses[i], poses[i+1]
        q0 = rot_to_quat(T0[:3,:3])
        q1 = rot_to_quat(T1[:3,:3])
        for k in range(steps_between):
            t = k / steps_between
            pos = (1-t) * T0[:3,3] + t * T1[:3,3]
            R   = quat_to_rot(slerp(q0, q1, t))
            dense.append(make_pose(pos, R))
    dense.append(poses[-1])
    return dense


# ── Waypoint report ──────────────────────────────────────────────────────────

def print_waypoints(angles, positions, rotations):
    print(f"\n{'─'*70}")
    print(f"  Inspection Trajectory  |  Seam center: {SEAM_CENTER}")
    print(f"  Pipe/flange seam radius: {SEAM_RADIUS*1000:.1f} mm")
    print(f"  Standoff: {STANDOFF} m  |  N waypoints: {len(angles)}")
    print(f"{'─'*70}")
    print(f"  {'#':>3}  {'φ [°]':>7}  {'x':>7}  {'y':>7}  {'z':>7}  quaternion [w x y z]")
    print(f"{'─'*70}")
    for i, (phi, pos, R) in enumerate(zip(angles, positions, rotations)):
        q = rot_to_quat(R)
        print(f"  {i+1:3d}  {np.rad2deg(phi):7.1f}  "
              f"{pos[0]:7.3f}  {pos[1]:7.3f}  {pos[2]:7.3f}  "
              f"[{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {q[3]:6.3f}]")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    angles, positions, rotations, poses = generate_waypoints()
    print_waypoints(angles, positions, rotations)

    # Verify: distance from each camera point to seam center.
    dists = [np.linalg.norm(p - SEAM_CENTER) for p in positions]
    print(f"Camera orbit distances from seam center: min={min(dists):.4f}  max={max(dists):.4f}")

    # Verify: camera +z points toward the matching seam point.
    for i, (phi, pos, R) in enumerate(zip(angles, positions, rotations)):
        cam_z = R[:, 2]
        to_seam = seam_target_position(phi) - pos
        to_seam /= np.linalg.norm(to_seam)
        alignment = np.dot(cam_z, to_seam)
        assert alignment > 0.999, f"Waypoint {i}: look-at misaligned (dot={alignment:.4f})"
    print("All seam look-at constraints verified OK.")
