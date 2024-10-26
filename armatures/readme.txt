n_joints: number of joints
name: armature name
parent: parent joint index (must satisfying parent[i] < i)
bone: joint local position expressed in the parent frame
gravity: gravitational acceleration in the world frame
com: body center of mass in the joint frame
mass: body mass
inertia: body inertia in the com frame
           if not compiled with USE_DIAGONAL_INERTIA, each row contains 6 floats Ixx, Iyy, Izz, Ixy, Iyz, Ixz.
           if compiled with USE_DIAGONAL_INERTIA, each row contains 3 floats Ixx, Iyy, Izz. This is the case that the com frame is aligned with principal axis of inertia.
