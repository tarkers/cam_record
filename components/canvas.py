from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from util.utils import load_3d_angles, gen_2d_angles, generate_2D_skeletons


class Canvas(FigureCanvas):
    def __init__(self, widget, cfg, fps=100):
        self.fps = fps
        self.cfg = cfg
        self.fig, self.ax = plt.subplots(figsize=(1, 1), dpi=90)
        self.fig.set_size_inches(4, 5)
        super().__init__(self.fig)
        self.setParent(widget)

        self.clear_chart()

    def cv_to_pyplot(self, color):
        """
        change cv2 color to pytplot color
        """
        color = np.asanyarray(color)
        color[[0, 2]] = color[[2, 0]]
        return tuple(color / 255)

    def plot_data(self, data):
        # dr, dl, d2r, d2l, gr, gl = data
        
        #3D
        if len(data[0]) > 0:
            self.vis_lines[0] = plt.plot(
                data[0],
                linestyle="solid",
                color=self.cv_to_pyplot(self.cfg["Pose3D"]["Right"]["Color"]),
                linewidth=1,
                label=self.cfg["Pose3D"]["Right"]["Name"],
            )
        if len(data[1]) > 0:
             self.vis_lines[1] = plt.plot(
                data[1],
                linestyle="solid",
                color=self.cv_to_pyplot(self.cfg["Pose3D"]["Left"]["Color"]),
                linewidth=1,
                label=self.cfg["Pose3D"]["Left"]["Name"],
            )

        
        #2D
        if len(data[2]) > 0:
            self.vis_lines[2]  = plt.plot(
                data[2],
                linestyle="solid",
                color=self.cv_to_pyplot(self.cfg["Pose2D"]["Right"]["Color"]),
                linewidth=1,
                label=self.cfg["Pose2D"]["Right"]["Name"],
            )
            
        if len(data[3]) > 0:
            self.vis_lines[3] = plt.plot(
                data[3],
                linestyle="solid",
                color=self.cv_to_pyplot(self.cfg["Pose2D"]["Left"]["Color"]),
                linewidth=1,
                label=self.cfg["Pose2D"]["Left"]["Name"],
            )
            
        if len(data[4]) > 0:
            self.vis_lines[4]  = plt.plot(
                data[4],
                linestyle="solid",
                color=self.cv_to_pyplot(self.cfg["VICON"]["Right"]["Color"]),
                linewidth=1,
                label=self.cfg["VICON"]["Right"]["Name"],
            )
        if len(data[5]) > 0:
            self.vis_lines[5]  = plt.plot(
                data[5],
                linestyle="solid",
                color=self.cv_to_pyplot(self.cfg["VICON"]["Left"]["Color"]),
                linewidth=1,
                label=self.cfg["VICON"]["Left"]["Name"],
            )

    def clear_chart(self):
        # clear line

        self.ax.clear()
        self.frame_num = 0
        self.label_vls = {}
        self.datas = []
        self.check_list = [False]*6
        self.knee_keypoints = np.array([])

        # chart vis variable
        self.bar = self.ax.axvline(x=0, color="purple", picker=1)
        self.ax.yaxis.set_ticks(np.arange(-20, 90, 10))
        self.ax.tick_params(axis="y", labelsize=12)
        self.ax.tick_params(axis="x", labelsize=12)
        self.ax.set_ylabel("Angle", fontsize=14)
        self.ax.set_title("Knee Angle (frame)", fontsize=14)
        self.ax.set_ylim(-20, 90)
        self.ax.grid()
        self.vis_lines=[[]]*6   #dr, dl, d2r, d2l ,vr, vl
        
        # self.vl, self.vr, self.dl, self.dr = [], [], [], []
        self.fig.canvas.draw()

    def show_lines(self, sn, vn=None, start_index=0, person_id=1, is_2D=False):
        """
        預設一開始只先顯示3D 及Groundtruth angle
        """
        # generate 2D skeleton
        self.knee_keypoints = generate_2D_skeletons(sn, vn, person_id)

        if not is_2D:  # generate 3D angle
            # generate 2D angle
            d2r, d2l, _, _ = gen_2d_angles(self.knee_keypoints,None,start_index)
            
            if vn:
                dr, dl, gr, gl = load_3d_angles(
                    rf"Data/{sn}/{vn}/subject_3D/3D_{vn}.npy",
                    rf"Data/{sn}/{vn}/Calibrate_Video/{vn}.csv",
                    start_index,
                )
                self.datas = np.array([dr, dl, d2r, d2l, gr, gl],dtype=object)

            else:
                dr, dl, _, _ = load_3d_angles(rf"Data/{sn}/{vn}/subject_3D/3D_{vn}.npy")
                self.datas = np.array([dr, dl, d2r, d2l, [], []],dtype=object)
        else:
            if vn:
                dr, dl, gr, gl = gen_2d_angles(
                    self.knee_keypoints,
                    rf"Data/{sn}/{vn}/Calibrate_Video/{vn}.csv",
                    start_index,
                )

                self.datas = np.array([[], [], d2r, d2l, gr, gl],dtype=object)
            else:
                dr, dl, _, _ = gen_2d_angles(self.knee_keypoints,None,start_index)
                self.datas = np.array([[], [], d2r, d2l, [], []],dtype=object)

        # plot on chart
        self.plot_data(
            [
                dr,
                dl,
                [],
                [],
                [],
                [],
            ]
        )
        if vn:
            self.plot_data([[], [], [], [], gr, gl])

        self.fig.canvas.draw()
        self.ax.legend(loc="upper right")

    def move_bar(self, frame_num):
        if self.bar:
            self.bar.remove()
        self.bar = self.ax.axvline(x=frame_num, color="purple", picker=1)
        self.fig.canvas.draw()
        tmp = []
        assert len(self.datas) == 6 or len(self.datas) == 4
        if frame_num > len(self.datas[0]) - 1:
            tmp = ["Nan"] * len(self.datas)
        else:
            # print(self.datas.shape)
            for angles in self.datas:
                if len(angles)>frame_num:
                    tmp.append("Nan" if np.isnan(angles[frame_num]) else str(int(angles[frame_num])))
                else:
                    tmp.append("Nan")
            # for angle in self.datas[:, frame_num]:
            #     tmp.append("Nan" if np.isnan(angle) else str(int(angle)))

        return tmp

    def draw_image(self, img, frame_idx):
        if frame_idx > len(self.knee_keypoints) - 1:  # no found data
            return img
        img = img.copy()

        ## draw point
        left_points = self.knee_keypoints[frame_idx, 3:, :2]
        right_points = self.knee_keypoints[frame_idx, :3, :2]
        for lp, rp in zip(left_points, right_points):
            cv2.circle(img, lp, 3, self.cfg["Pose3D"]["Left"]["Color"], 1)
            cv2.circle(img, rp, 3, self.cfg["Pose3D"]["Right"]["Color"], 1)

        ## draw line
        for i in range(2):
            cv2.line(
                img,
                left_points[i],
                left_points[i + 1],
                self.cfg["Pose3D"]["Left"]["Color"],
                2,
            )  # left line
            cv2.line(
                img,
                right_points[i],
                right_points[i + 1],
                self.cfg["Pose3D"]["Right"]["Color"],
                2,
            )  # right line

        return img

    def change_visible(self, check_list):
        """
        線圖依照checkbox顯示
        """
        if (
            check_list == self.check_list
        ):  # if same then no need to waste time for refresh
            return
        
        for idx in range(len(self.vis_lines)):
            if check_list[idx] == False:
                if len(self.vis_lines[idx]) != 0:  # need remove
                    l = self.vis_lines[idx].pop(0)
                    l.remove()
                    self.vis_lines[idx] = []
            elif check_list[idx] == True and len(self.vis_lines[idx]) == 0:
                tmp = [[]]*6
                tmp[idx] = self.datas[idx]
                self.plot_data(tmp)

        self.fig.canvas.draw()
        self.ax.legend(loc="upper right")
        self.check_list = check_list
