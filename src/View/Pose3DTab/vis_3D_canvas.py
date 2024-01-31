import numpy as np
from PyQt5 import QtWidgets,QtCore

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
import numpy as np
import sys

from vispy import app as vis_app, visuals, scene, geometry, color
from vispy.scene.cameras import ArcballCamera, MagnifyCamera, perspective, turntable


## custom
# from util.utils import get_3D_skeletons

XL, YL, WS, HS, D = 8, 4, 4, 2, "+z"
IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (800, 600)  # (width, height)
NUM_LINE_POINTS = 200
ori_limb = [
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
body_limb = [
    [10, 9, 8],
    [8, 7, 0],
    [16, 15, 14, 8],
    [13, 12, 11, 8],
    [3, 2, 1, 0],
    [6, 5, 4, 0],
]
fast_limb = [[3, 2, 1, 0, 4, 5, 6], [16, 15, 14, 8, 11, 12, 13], [10, 9, 8, 7, 0]]


# class MyMainWindow(QtWidgets.QMainWindow):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         central_widget = QtWidgets.QWidget()
#         main_layout = QtWidgets.QHBoxLayout()

#         # self._controls = Controls()
#         # main_layout.addWidget(self._controls)
#         self._canvas_wrapper = Canvas3DWrapper()
#         main_layout.addWidget(self._canvas_wrapper.canvas.native)

#         central_widget.setLayout(main_layout)
#         self.setCentralWidget(central_widget)




class Canvas3DWrapper:
    def __init__(self):
        self.lines_plot = []
        self.canvas = scene.SceneCanvas(keys="interactive", show=True, size=CANVAS_SIZE)
        self.view = self.canvas.central_widget.add_view()
        self.lines_mode = fast_limb
        self.frame=0
        ## setting camera
        self.view.camera = scene.TurntableCamera(
            elevation=30, azimuth=0, up="+z", distance=15
        )
        self.datas=[]
        self.plane = self.set_floor_plane()
        self.set_lines_plot()
    def test_plot(self):
        pass
    def set_floor_plane(self):
        vertices, faces, outline = geometry.create_plane(
            width=XL, height=YL, width_segments=WS, height_segments=HS, direction=D
        )
        colors = []
        for _ in range(faces.shape[0]):
            colors.append(np.array([0.2, 0.2, 1.0, 0.4]))

        plane = scene.visuals.Plane(
            width=XL,
            height=YL,
            width_segments=WS,
            height_segments=HS,
            direction=D,
            # vertex_colors=vertices['color'],
            face_colors=np.array(colors),
            edge_color=color.color_array.Color(color="white", alpha=None, clip=False),
            # edge_color="k",
            parent=self.view.scene,
        )
        # rotate and translate
        # rot1 = scene.transforms.MatrixTransform()
        # rot1.rotate(45, (0, 1, 0))
        # rot1.translate([1, 1, 1])
        # plane.transform = rot1

        return plane

    def resample_data_to_canvas(self, datas):
        datas[:, :, [2, 1]] = datas[:, :, [1, 2]]  # swap y,z
        datas[:, :, -1] = -(
            datas[:, :, -1] - np.max(datas[:, :, -1])
        )  # upside down of z and set foot point to xy plane
        datas *= 5  # scale person factor
        return datas

    def set_lines_plot(self, plotline="fast_plot"):
        self.lines_mode = fast_limb
        if plotline == "ori_limb":
            self.lines_mode = ori_limb
        elif plotline == "body_limb":
            self.lines_mode = body_limb

        for _ in range(len(self.lines_mode)):
            plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
            self.lines_plot.append(plot3D(parent=self.view.scene))

    def update_points(self, id):
        if len(self.datas)<=id:
            return
        pos=self.datas[id]
        for idx, line in enumerate(self.lines_mode):
            limb_pos = []
            # plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
            # lines_plot.append(plot3D(parent=view.scene))
            ## set pos
            for point in line:
                limb_pos.append(pos[point])
            self.lines_plot[idx].set_data(
                limb_pos,
                width=2.0,
                color="red",
                edge_color="y",
                symbol="x",
                face_color=(0.2, 0.2, 1, 0.8),
            )
        self.frame += 1

        if self.frame >= len(self.datas):
            self.frame = 0

    # def test_timer(self):
    #     self.timer = QtCore.QTimer()        
    #     self.timer .timeout.connect(self.update)   
    #     self.timer .start(10)               

    def set_datas(self,data):
        # data = get_3D_data(
        #     rf"C:\Users\65904\Desktop\Train_VIT\cam_record\Data\N0001\N0001_WALK_L01\subject_3D\3D_N0001_WALK_L01.npy"
        # )
        data=data.copy()
        self.datas=self.resample_data_to_canvas(data)





def get_3D_data(file):
    """
    load numpy estimation files of 17 3D joint Point
    Dataset Format => [Frames,17,3]
    """
    data = np.load(file)  # [Frames,17,3]

    return data


# if __name__ == "__main__":
#     app = use_app("pyqt5")
#     app.create()
#     win = MyMainWindow()
#     win.show()
#     app.run()
