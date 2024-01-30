# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
import sys

from vispy import app, visuals, scene, geometry,color
from vispy.scene.cameras import ArcballCamera, MagnifyCamera, perspective, turntable

# from vispy_grid3D import GridLinesVisual
h36m_limb = [
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [0, 7],
    [7, 8],
    [8, 9],
    [8, 11],
    [8, 14],
    [9, 10],
    [11, 12],
    [12, 13],
    [14, 15],
    [15, 16],
]
part_plot = [
    [10, 9, 8],
    [8, 7, 0],
    [16, 15, 14, 8],
    [13, 12, 11, 8],
    [3, 2, 1, 0],
    [6, 5, 4, 0],
]
fast_plot = [[3, 2, 1, 0, 4, 5, 6], [16, 15, 14, 8, 11, 12, 13], [10, 9, 8, 7, 0]]


def get_3D_data(file):
    """
    load numpy estimation files of 17 3D joint Point
    Dataset Format => [Frames,17,3]
    """
    data = np.load(file)  # [Frames,17,3]

    return data


# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys="interactive", show=True,size=(1080,720))
frame = 0

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()

view.camera =scene.TurntableCamera(elevation=30, azimuth=0, up='+z',distance=15)
# view.camera = turntable.TurntableCamera(fov=60, parent=view.scene, scale_factor=3,distance=10)
# view.camera = perspective.PerspectiveCamera(
#         fov=60,parent=view.scene ,scale_factor=10
# )
# view.camera = "turntable"
# view.camera = ArcballCamera(
#         fov=45, distance=10, interactive=True, parent=view.scene
# )
# # view.camera = "turntable"
# view.camera.fov = 72.5
# view.camera.distance = 2
# grid =  scene.visuals.GridLines()
# view.add(grid)
# grid = canvas.central_widget.add_grid()
# b2 = grid.add_view(row=0, col=0)

# grid_lines = scene.visuals.create_visual_node(GridLinesVisual)()
# b2.add(grid_lines)

# data
n = 17
pts = get_3D_data(
    rf"C:\Users\65904\Desktop\Train_VIT\cam_record\Data\N0001\N0001_WALK_R05\subject_3D\3D_N0001_WALK_R05.npy"
)

# axis = scene.visuals.XYZAxis(parent=view.scene)  # x=red, y=green, z=blue
pts[:, :, [2, 1]] = pts[:, :, [1, 2]]  # swap y,z
pts[:, :, -1] = -(
    pts[:, :, -1]  - np.max(pts[:, :, -1])
)  # upside down of z and set foot point to xy plane
pts*=5  #scale person factor
XL, YL, WS, HS, D = 8, 4, 4, 2, "+z"

## create xy plane
vertices, faces, outline = geometry.create_plane(
    width=XL, height=YL, width_segments=WS, height_segments=HS, direction=D
)
colors = []
for _ in range(faces.shape[0]):
    colors.append(np.array([0.2, 0.2, 1.0, 0.4]))

# print(vertices['color'])
# exit()
plane = scene.visuals.Plane(
    width=XL,
    height=YL,
    width_segments=WS,
    height_segments=HS,
    direction=D,
    # vertex_colors=vertices['color'],
    face_colors=np.array(colors),
    edge_color=color.color_array.Color(color='white', alpha=None, clip=False),
    # edge_color="k",
    parent=view.scene,
)

rot1 = scene.transforms.MatrixTransform()
rot1.rotate(45,(0,1,0))
rot1.translate([1,1,1])
plane.transform = rot1
lines_plot = []


for idx, line in enumerate(fast_plot):
    limb_pos = []
    plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
    lines_plot.append(plot3D(parent=view.scene))


def update(ev):
    global pts, frame, timer
    pos = pts[frame]
    for idx, line in enumerate(fast_plot):
        limb_pos = []
        # plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        # lines_plot.append(plot3D(parent=view.scene))
        ## set pos
        for point in line:
            limb_pos.append(pos[point])
        lines_plot[idx].set_data(
            limb_pos,
            width=2.0,
            color="red",
            edge_color="y",
            symbol="x",
            face_color=(0.2, 0.2, 1, 0.8),
        )
    frame += 1

    if frame >= len(pts):
        frame=0
        # timer.stop()


#     # pos[:, 1] = np.random.normal(size=n)
#     for i in range(n):
#         colors[i] = (np.random.random(), np.random.random(), 0, 0.8)
#     p1.set_data(
#     pos, face_color=colors, symbol="o", size=10, edge_width=0.5, edge_color="blue"
#     )

timer = app.Timer()
timer.connect(update)
timer.start(0.02)


# run
if __name__ == "__main__":
    if sys.flags.interactive != 1:
        app.run()
