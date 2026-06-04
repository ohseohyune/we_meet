"""
flange_generator.py — Procedural synthetic flange geometry for MuJoCo.

Assumptions
-----------
Flange type  : raised-face weld-neck flange (simplified flat disc + bolt bosses).
Dimensions   : scaled relative to pipe OD (PIPE_OD).  Chosen to be realistic for a
               DN100 / 4-inch nominal pipe, which is common in industrial inspection.

Geometry summary (all meters)
------------------------------
  PIPE_OD           = 0.114  m  (DN100 pipe OD, 4-inch schedule 40)
  PIPE_ID           = 0.102  m
  PIPE_LENGTH       = 0.500  m

  FLANGE_OD         = 0.230  m  (≈ 2× pipe OD, ASME B16.5 Class 150 approx)
  FLANGE_ID         = 0.102  m  (matches pipe ID)
  FLANGE_THICKNESS  = 0.023  m  (≈ 0.2× pipe OD)
  FLANGE_FACE_OD    = 0.152  m  (raised face OD)
  FLANGE_FACE_T     = 0.002  m  (raised face height / ring height)

  BOLT_CIRCLE_D     = 0.190  m  (bolt circle diameter, ≈ 0.83× flange OD)
  BOLT_HOLE_D       = 0.019  m  (bolt hole diameter, M16 class)
  N_BOLTS           = 8          (8-bolt pattern)

MuJoCo representation
---------------------
MuJoCo cannot subtract volumes (Boolean ops), so the flange is assembled from
additive geoms.  The center bore is approximated by omitting geometry in that
region and adding a thin-walled inner cylinder.

Geom layout (all positions relative to flange frame origin = pipe-end center):
  1. flange_disc      : cylinder (flange body, full disc from OD to ID approximated)
  2. flange_inner_ring: thin cylinder at bore to suggest hollow (visual only)
  3. bolt_N           : small cylinders at bolt hole positions (N of them)
  4. raised_face      : thin disc representing raised face on flange

Pipe representation:
  5. pipe_body : cylinder from x=0 to x=-PIPE_LENGTH (behind flange)

The flange frame x-axis is the pipe axis; x=0 is the flange face (front).
The pipe extends in the -x direction.
"""

import numpy as np

# ── Dimensions ────────────────────────────────────────────────────────────────
PIPE_OD          = 0.1140
PIPE_ID          = 0.1020
PIPE_LENGTH      = 0.500

FLANGE_OD        = 0.230
FLANGE_ID        = PIPE_ID
FLANGE_T         = 0.023
FLANGE_FACE_OD   = 0.152
FLANGE_FACE_T    = 0.002

BOLT_CIRCLE_R    = 0.095   # radius = 0.190 / 2
BOLT_HOLE_R      = 0.0095  # radius = 0.019 / 2
BOLT_BOSS_R      = 0.0140  # slightly larger for visible boss
BOLT_BOSS_H      = 0.005   # boss protrusion
N_BOLTS          = 8


def bolt_positions() -> list:
    """Return list of (y, z) positions for bolt holes on bolt circle."""
    angles = np.linspace(0, 2*np.pi, N_BOLTS, endpoint=False)
    return [(BOLT_CIRCLE_R * np.cos(a), BOLT_CIRCLE_R * np.sin(a)) for a in angles]


def generate_flange_xml(flange_pos: tuple = (1.05, 0.0, 0.30)) -> str:
    """
    Generate MuJoCo XML snippet for the pipe-flange assembly.

    Parameters
    ----------
    flange_pos : (x, y, z) position of the flange face center in world frame.
                 Pipe extends in the -x direction from this point.
                 Default: (1.05, 0, 0.30) puts flange ~0.55m from robot base
                 at the end of a 0.5m pipe starting at x=0.55.

    Returns
    -------
    XML string to embed inside <worldbody>.
    """
    fx, fy, fz = flange_pos
    # Pipe body center is behind flange by PIPE_LENGTH/2
    pipe_cx = fx - PIPE_LENGTH / 2.0

    bolts_xml = ""
    for i, (by, bz) in enumerate(bolt_positions()):
        # Bolt boss: small cylinder at flange front face, centered on bolt circle
        bolts_xml += f"""
        <geom name="bolt_boss_{i}" type="cylinder"
              pos="{BOLT_BOSS_H/2:.4f} {by:.4f} {bz:.4f}"
              size="{BOLT_BOSS_R:.4f} {BOLT_BOSS_H/2:.4f}"
              euler="0 1.5708 0"
              rgba="0.25 0.25 0.25 1"/>"""

    xml = f"""
    <!-- ═══════════════════════════════════════════════════════════
         Pipe-Flange Assembly
         Pipe axis: world +x.  Flange face at world pos {flange_pos}.
         Pipe extends in -x from flange face.
         ═══════════════════════════════════════════════════════════ -->
    <body name="pipe_flange_assembly" pos="{fx:.4f} {fy:.4f} {fz:.4f}" euler="0 1.5708 0">

      <!-- ── Pipe body ─────────────────────────────────────────── -->
      <!-- Cylinder axis is local +z after euler rotation; placed so
           pipe end (z=0) coincides with flange face.              -->
      <geom name="pipe_outer" type="cylinder"
            pos="0 0 {-PIPE_LENGTH/2:.4f}"
            size="{PIPE_OD/2:.4f} {PIPE_LENGTH/2:.4f}"
            rgba="0.55 0.55 0.60 1"/>
      <!-- Inner bore (hollow illusion via darker inner cylinder)  -->
      <geom name="pipe_inner" type="cylinder"
            pos="0 0 {-PIPE_LENGTH/2:.4f}"
            size="{PIPE_ID/2:.4f} {PIPE_LENGTH/2:.4f}"
            rgba="0.15 0.15 0.15 1"/>

      <!-- ── Flange disc ───────────────────────────────────────── -->
      <geom name="flange_disc" type="cylinder"
            pos="0 0 {FLANGE_T/2:.4f}"
            size="{FLANGE_OD/2:.4f} {FLANGE_T/2:.4f}"
            rgba="0.45 0.50 0.55 1"/>
      <!-- Inner bore of flange (darker to look hollow)           -->
      <geom name="flange_bore" type="cylinder"
            pos="0 0 {FLANGE_T/2:.4f}"
            size="{FLANGE_ID/2:.4f} {FLANGE_T/2+0.001:.4f}"
            rgba="0.12 0.12 0.12 1"/>

      <!-- ── Raised face ───────────────────────────────────────── -->
      <geom name="raised_face" type="cylinder"
            pos="0 0 {FLANGE_T + FLANGE_FACE_T/2:.4f}"
            size="{FLANGE_FACE_OD/2:.4f} {FLANGE_FACE_T/2:.4f}"
            rgba="0.60 0.62 0.65 1"/>
      <!-- Raised face bore -->
      <geom name="raised_face_bore" type="cylinder"
            pos="0 0 {FLANGE_T + FLANGE_FACE_T/2:.4f}"
            size="{FLANGE_ID/2:.4f} {FLANGE_FACE_T/2+0.001:.4f}"
            rgba="0.12 0.12 0.12 1"/>

      <!-- ── Bolt bosses (N={N_BOLTS}) ─────────────────────────── -->
      <!-- Represented as small cylinders protruding from face.
           Actual holes cannot be subtracted in MuJoCo; holes are
           implied by dark spot geoms below.                       -->{bolts_xml}

      <!-- Bolt hole indicators (dark discs at hole positions)     -->
{"".join(
    f"""
      <geom name="bolt_hole_{i}" type="cylinder"
            pos="{FLANGE_T + FLANGE_FACE_T + 0.001:.4f} {by:.4f} {bz:.4f}"
            size="{BOLT_HOLE_R:.4f} 0.0015"
            euler="0 1.5708 0"
            rgba="0.05 0.05 0.05 1"/>"""
    for i, (by, bz) in enumerate(bolt_positions())
)}

      <!-- ── Flange frame marker (site) ───────────────────────── -->
      <site name="flange_center" pos="0 0 {FLANGE_T + FLANGE_FACE_T:.4f}"
            size="0.012" rgba="1 0 0 1" type="sphere"/>
      <site name="flange_frame_x" pos="0.05 0 {FLANGE_T + FLANGE_FACE_T:.4f}"
            size="0.005 0.05" euler="0 1.5708 0" rgba="1 0 0 0.8" type="cylinder"/>
      <site name="flange_frame_y" pos="0 0.05 {FLANGE_T + FLANGE_FACE_T:.4f}"
            size="0.005 0.05" euler="1.5708 0 0" rgba="0 1 0 0.8" type="cylinder"/>
      <site name="flange_frame_z" pos="0 0 {FLANGE_T + FLANGE_FACE_T + 0.05:.4f}"
            size="0.005 0.05" rgba="0 0 1 0.8" type="cylinder"/>

    </body>
    """
    return xml


def print_flange_dimensions():
    print("\n═══════════════════════════════════════════")
    print("  Synthetic Flange Geometry (DN100 / 4\")")
    print("═══════════════════════════════════════════")
    print(f"  Pipe OD              : {PIPE_OD*1000:.1f} mm")
    print(f"  Pipe ID              : {PIPE_ID*1000:.1f} mm")
    print(f"  Pipe length          : {PIPE_LENGTH*1000:.0f} mm")
    print(f"  Flange OD            : {FLANGE_OD*1000:.1f} mm")
    print(f"  Flange ID (bore)     : {FLANGE_ID*1000:.1f} mm")
    print(f"  Flange thickness     : {FLANGE_T*1000:.1f} mm")
    print(f"  Raised face OD       : {FLANGE_FACE_OD*1000:.1f} mm")
    print(f"  Raised face height   : {FLANGE_FACE_T*1000:.1f} mm")
    print(f"  Bolt circle diameter : {BOLT_CIRCLE_R*2000:.1f} mm")
    print(f"  Bolt hole diameter   : {BOLT_HOLE_R*2000:.1f} mm")
    print(f"  Number of bolts      : {N_BOLTS}")
    print("═══════════════════════════════════════════\n")


if __name__ == "__main__":
    print_flange_dimensions()
    xml = generate_flange_xml()
    print("Generated XML snippet:")
    print(xml[:500], "...")
