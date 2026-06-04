# Mesh Assets

Put converted CAD meshes here.

Recommended workflow for `pipe.SLDPRT`:

1. Export `pipe.SLDPRT` from SolidWorks as binary STL.
2. Save it as `meshes/pipe.stl`.
3. If the STL is in millimeters, use `scale="0.001 0.001 0.001"` in MJCF.
4. Confirm the mesh origin and pipe axis before replacing the procedural pipe in `scene.xml`.

