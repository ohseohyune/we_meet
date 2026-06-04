# Pipe Flange Inspection MuJoCo Simulation

This project builds a MuJoCo simulation for a Franka Panda robot arm that inspects a synthetic pipe flange with a TCP-mounted D405-style camera.

## 1. System Architecture

- `legacy/dh/dh_to_xml.py`: deprecated; DH-generated robot is no longer used.
- `robot_model.xml`: standalone robot model generated from the DH data.
- `tools/flange_generator.py`: documents and generates a synthetic flange model.
- `scene.xml`: complete MuJoCo scene with the Franka Panda mesh model, pipe, flange, camera, trajectory markers, actuators, and sensors.
- `trajectory/generator.py`: creates a seam-focused inspection trajectory around the pipe/flange contact circle.
- `control/franka_ik_solver.py`: MuJoCo Jacobian IK for the Franka Panda `tcp` site.
- `mujoco_viewer.py`: loads the scene, computes IK, animates the trajectory, and optionally renders camera frames.

## 2. Main Assumptions

- DH convention is standard DH: `T_i = Rz(q_i + theta_offset_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i)`.
- All 7 joints are revolute and rotate about local `z`.
- `franka_sim-master/franka_panda.xml` / `assets/chain0_nogripper.xml` provide the robot kinematics and meshes.
- The provided DH parameters are intentionally not used in the current Franka version.
- Joint limits are conservative symmetric limits: `[-2.8973, 2.8973] rad`.
- The pipe is represented procedurally unless a converted CAD mesh is added.
- The flange is synthetic because no flange CAD is available.
- Camera is rigidly attached 100 mm in front of the TCP along TCP `-z`.
- TCP `+z` is the inspection look axis used by IK and points toward the current pipe/flange seam point.
- MuJoCo camera forward is local `-z`, so the fixed camera has a 180-degree x-axis rotation.
- The pipe axis is world `+x`; the flange face center is at `(0.55, 0.0, 0.57)`.
- The seam target circle uses pipe OD radius `0.057 m`.
- Inspection waypoints form a circle in the world `y-z` plane with radius `0.25 m`.

## 3. Franka Model

The robot now uses the Franka Panda MuJoCo model from `franka_sim-master`.

- Kinematics come from `assets/chain0_nogripper.xml`.
- Visual/collision meshes come from `franka_sim-master/meshes`.
- Joint names are `panda0_joint1` through `panda0_joint7`.
- DH-generated robot geometry is not used.

## 4. Robot XML Structure

`robot_model.xml` and the robot section of `scene.xml` include:

- `panda0_link0` through `panda0_link7`.
- `panda0_joint1` through `panda0_joint7` hinge joints.
- Franka visual mesh geoms.
- `tool` body, TCP site, and colored TCP frame sites.
- `inspection_camera` fixed camera mounted 100 mm in front of TCP.
- Position actuators and joint/TCP sensors.

## 5. Synthetic Flange Generation

The flange is modeled as a simplified raised-face weld-neck style flange:

- Pipe OD: `0.114 m`
- Pipe ID / flange bore: `0.102 m`
- Flange OD: `0.230 m`
- Flange thickness: `0.023 m`
- Raised face OD: `0.152 m`
- Bolt circle diameter: `0.190 m`
- Bolt holes: `8 x 0.019 m`

MuJoCo does not support Boolean subtraction in MJCF. Bore and bolt holes are represented visually using dark cylinders/discs and visible bolt bosses.

## 6. Pipe-Flange Assembly

In `scene.xml`:

- Pipe body origin is at `(0.425, 0.0, 0.57)`.
- Pipe axis is aligned to world `+x` by `euler="0 1.5708 0"`.
- Pipe extends from `x=0.30` to `x=0.55`.
- Flange is placed at the pipe end near `x=0.55`.
- `flange_center` site marks the pipe/flange center; the inspection target is the circular seam at pipe OD radius.

If you have `pipe.SLDPRT`, convert it before MuJoCo import:

1. In SolidWorks, save as binary STL with known units.
2. Place it under `meshes/pipe.stl`.
3. Add this to the `<asset>` section:

```xml
<mesh name="pipe_mesh" file="pipe.stl" scale="0.001 0.001 0.001"/>
```

4. Replace the procedural pipe cylinder with:

```xml
<geom name="pipe_mesh_geom" type="mesh" mesh="pipe_mesh" rgba="0.55 0.55 0.60 1"/>
```

OBJ conversion workflow:

```bash
meshlabserver -i pipe.stl -o meshes/pipe.obj
```

or use Blender: import STL, confirm scale/origin, export OBJ.

## 7. Coordinate Frames

- Robot Base Frame: world frame, origin `(0, 0, 0)`, `z` up.
- Pipe Frame: pipe axis along world `+x`.
- Flange Frame: origin at flange face center, face normal toward robot along world `-x`.
- TCP Frame: Franka end-effector tool frame at `panda0_link7`.
- Camera Frame: optical center is `p_camera = p_tcp - 0.10 * z_tcp`; TCP `+z` is aligned toward the current seam point.

## 8. Inspection Trajectory Equations

Let `c = [0.55, 0.0, 0.57]^T`, camera orbit radius `r = 0.25`, seam radius `r_s = 0.057`, camera x-offset `x_o = -0.18`, and inspection angle `phi in [0, 2*pi)`.

Seam target:

```text
s(phi) = c + r_s * [0, cos(phi), sin(phi)]^T
```

Position:

```text
p(phi) = c + [x_o, r * cos(phi), r * sin(phi)]^T
```

Viewing direction:

```text
z_cam(phi) = normalize(s(phi) - p(phi))
```

Orientation:

```text
x_cam = normalize(cross(up_hint, z_cam))
y_cam = cross(z_cam, x_cam)
R = [x_cam y_cam z_cam]
```

The default `up_hint` is world `+z`, with a fallback to world `+y` near singular cases.

## 9. IK Implementation

`control/franka_ik_solver.py` provides MuJoCo model-based Jacobian IK:

- Uses the loaded `scene.xml` Franka model.
- Solves the `tcp` site position.
- Aligns TCP `+z` with the current seam look-at direction.
- Uses random-restart fallback for trajectory continuity.

## 10. Usage

Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

Print desired waypoint poses:

```bash
python3 -m trajectory.generator
```

Verify full 360-degree IK trajectory without opening a viewer:

```bash
python3 mujoco_viewer.py --ik --verify --no-viewer
```

Export desired poses and joint trajectory:

```bash
python3 mujoco_viewer.py --ik --export-csv --no-viewer
```

Open interactive MuJoCo viewer and animate:

```bash
python3 mujoco_viewer.py --ik
```

Render inspection camera frames:

```bash
python3 mujoco_viewer.py --camera --no-viewer
```

Export a dataset with black separator frames inserted at the trajectory turn
points (`12 -> 6`, `6 -> 12`, `12 -> 6`, `6 -> 12` segment boundaries):

```bash
python3 tools/export_inspection_dataset.py \
  --frames 60 \
  --start-index 1 \
  --out inspection_dataset_with_separators \
  --insert-separators \
  --retries 8
```

Separator frames are saved as black RGB/depth PNG images and `NaN` raw depth
arrays.  In `metadata.csv`, separator rows have `is_separator=1` and a
`separator_reason` such as `segment_1_to_2`.

Export a seam-focused multi-ring dataset with overlap-based frame planning:

```bash
python3 tools/export_inspection_dataset.py \
  --multi-ring \
  --overlap 0.75 \
  --min-frames-per-ring 20 \
  --start-index 1 \
  --out inspection_dataset_multiring_001_060 \
  --insert-separators \
  --retries 8
```

This generates three synchronized inspection passes:

- `near_oblique`: 20 frames
- `nominal`: 20 frames
- `far_oblique`: 20 frames

Each frame stores `seam_target_x/y/z`, `ring_id`, `ring_name`, robot joints, TCP pose, and the actual render camera pose `T_world_render_camera`.

## Project Folder Structure

```text
wemeet/
  README.md
  requirements.txt
  robot_model.xml
  scene.xml
  main.py
  mujoco_viewer.py
  control/
    clik.py
    franka_ik_solver.py
    jacobian.py
  trajectory/
    circle.py
    generator.py
  tools/
    d405_depth_capture.py
    export_inspection_dataset.py
    flange_generator.py
    trajectory_diagnostics.py
  legacy/
    dh/
      dh_to_xml.py
      ik_solver.py
  outputs/
    archives/
    depth/
    diagnostics/
    plots/
    trajectories/
  meshes/
    README.md
```
