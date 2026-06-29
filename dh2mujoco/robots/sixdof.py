"""6-DOF serial manipulator – DH parameter table.

Source data
-----------
The raw C-style array (28 values, 7 rows × 4 columns) is the same data that
the existing ``model/franka.py`` module uses.  After calibration the row
format was determined to be:

    (d_i,  theta_i_offset,  a_{i-1},  alpha_{i-1})

i.e. the column order is  [d, theta, a, alpha]  *not*  [a, alpha, d, theta].

The first 6 rows are revolute joints; the 7th row is the fixed EE/TCP frame.

To use the problem-statement's default column order [a, alpha, d, theta]
the user would instead set ``COLUMN_ORDER_PROBLEM`` below and re-arrange the
table accordingly.
"""

from __future__ import annotations

import numpy as np

from ..config import DHColumnOrder, Config
from ..dh_parser import DHTable, parse_dh_array


# ---------------------------------------------------------------------------
# Raw DH array  (column layout: d | theta_offset | a | alpha)
# ---------------------------------------------------------------------------
# This matches the existing working model in model/franka.py.
_RAW_DH = np.array(
    [
        #  d            theta_off     a            alpha
        [0.132184,    3.146991,    -0.061099,   -1.568837],  # joint 1
        [0.091502,    0.002344,     0.345932,   -0.004401],  # joint 2
        [0.005000,   -1.577745,     0.103901,    1.574759],  # joint 3
        [0.344628,   -0.009289,    -0.000588,   -1.581564],  # joint 4
        [-0.024818,  -3.104470,     0.074472,   -1.553936],  # joint 5
        [0.042524,    1.581094,     0.070882,   -1.520796],  # joint 6
        [-0.24450,   -1.570796,     0.0,          0.0],      # EE / TCP
    ],
    dtype=float,
)

# Column order that matches the raw array above
COLUMN_ORDER_EXISTING: DHColumnOrder = DHColumnOrder(a=2, alpha=3, d=0, theta=1)

# Column order as stated in the problem description: (a, alpha, d, theta)
# Use this if your source data is arranged that way.
COLUMN_ORDER_PROBLEM: DHColumnOrder = DHColumnOrder(a=0, alpha=1, d=2, theta=3)

N_JOINTS: int = 6

# Joint offsets extracted from the theta column (column index 1).
# Used by Mode 4 so the offsets are stored separately from the DH table,
# making qpos=0 correspond to the robot's zero-angle configuration.
JOINT_OFFSETS: list[float] = [
    3.146991,   # joint 1
    0.002344,   # joint 2
   -1.577745,   # joint 3
   -0.009289,   # joint 4
   -3.104470,   # joint 5
    1.581094,   # joint 6
]


def make_table(column_order: DHColumnOrder = COLUMN_ORDER_EXISTING) -> DHTable:
    """Return the :class:`DHTable` for this robot.

    Parameters
    ----------
    column_order :
        Which column layout the raw data uses.  Defaults to the layout
        already validated against the existing MuJoCo model.
    """
    return parse_dh_array(_RAW_DH, column_order, n_joints=N_JOINTS)


def make_config(mode: int = 4) -> Config:
    """Return a default :class:`Config` for this robot.

    Parameters
    ----------
    mode :
        DH convention mode.  Defaults to 4 (Modified DH, offsets explicit).
        When mode == 4, ``joint_offsets`` is auto-populated from
        :data:`JOINT_OFFSETS` so the DH table theta column is not used.
    """
    cfg = Config(
        MODE=mode,
        column_order=COLUMN_ORDER_EXISTING,
        GEOM_TYPE="capsule",
        LINK_RADIUS=0.014,
        SHOW_FRAME=True,
        SHOW_MATRIX=False,
        EXPORT_XML=True,
        OUTPUT_PATH="dh2mujoco/output/generated_robot.xml",
        USE_QUATERNION=True,
    )
    if mode == 4:
        cfg.joint_offsets = list(JOINT_OFFSETS)
    if mode == 3:
        # Joint 4's DH rotation axis is parallel to the link (roll joint).
        # Apply roll→bend correction so q4 visibly swings the forearm.
        cfg.roll_fix_joints = {4}
        # Make joints 4 and 5 rotate about the same world axis as joint 6
        # in the generated MJCF model.
        cfg.same_axis_as = {4: 6, 5: 6}
        # Home pose: joints 4 and 5 at 90°.
        import math
        cfg.home_qpos = [0.0, 0.0, 0.0, math.pi / 2, math.pi / 2, 0.0]
    return cfg
