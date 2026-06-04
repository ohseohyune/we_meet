"""
Deprecated.

The simulation now uses the real Franka Panda MuJoCo model from
`franka_sim-master` instead of generating a robot from the provided DH
parameters.

Use:
  python3 mujoco_viewer.py --ik

Robot/scene source:
  scene.xml
  control/franka_ik_solver.py
"""


if __name__ == "__main__":
    print(__doc__)
