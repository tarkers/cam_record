
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import util

# # test code
RIGHT_KNEE = 12
LEFT_KNEE = 13
MARK = 14
AUTO_MARK = 15


class Canvas(FigureCanvas):

    def __init__(self, parent, mainUi=None):
        self.mainUI = mainUi
        self.fig, self.ax = plt.subplots(figsize=(1, 1), dpi=80)
        self.fig.set_size_inches(4, 5)
        super().__init__(self.fig)
        self.setParent(parent)
        self.init_var()
    
    def init_var(self):
        self.FRAME_PER_SECOND = 50
        self.frame_num = 0
        self.label_vls = {}
        self.vl = self.ax.axvline(x=0, color='purple', picker=1)
        self.ax.yaxis.set_ticks(np.arange(-10, 90, 10))
        self.ax.tick_params(axis='y', labelsize=12)
        self.ax.tick_params(axis='x', labelsize=12)
        self.ax.set_ylabel('angle', fontsize = 14)
        self.ax.set_title('Knees Angle (frame)', fontsize = 14)
        self.ax.set_ylim(-20, 90)
        # self.ax.set_xlim(xmin=0,xmax=100)
        self.op_data = []
        self.vicon_data = []
        self.lines = [None, None, None, None]
        self.vl_line, self.vr_line, self.ol_line, self.or_line = None, None, None, None
        self.ax.grid()

    def get_vicon_data(self):
        return self.vicon_data.copy()
    
    def get_openpose_data(self):
        return self.op_data.copy()

    def load_openpose_file(self, filename):
        file = pd.read_csv(
            filename, low_memory=False, index_col=[0], header=[0, 1])
        self.clear_vls()
        self.op_data = np.array(file.values)
        
        # MARK
        for idx, item in enumerate(self.op_data[:, MARK]):
            if not pd.isna(item):
                self.mainUI.listwidget_add(idx, item)
        # AUTO_MARK
        for idx, item in enumerate(self.op_data[:, 15]):
            if not pd.isna(item):
                self.mainUI.listwidget_add(idx, item)
        self.ax.axis(xmin=0, xmax=self.op_data.shape[0])
        # check data
        if self.op_data[:, MARK].dtype=="float64":
            self.op_data=self.op_data.astype('object')
            self.op_data[:,MARK]=['' if x is np.nan else x for x in self.op_data[:,MARK]]
            self.op_data[:,AUTO_MARK]=['' if x is np.nan else x for x in self.op_data[:,AUTO_MARK]]
        return True, self.op_data

    # 0:rknee 1:lknee
    def load_vicon_file(self, filename,is_extract=False):
        if is_extract:
            try:
                df = pd.read_csv(filename,index_col=[0])
                self.vicon_data = np.array(df.values)
                return True, self.vicon_data
            except pd.errors.ParserError:
                print("Ground truth csv錯誤\n(請確認是否為正確的csv檔案!)")
                return False, None
            
        try:
            df = pd.read_csv(filename, header=None, low_memory=False)
        except pd.errors.ParserError:
            print("Ground truth csv錯誤\n(請確認是否為正確的csv檔案!)")
            return False, None
        try:
            first_column = df.iloc[:, 0]

            indices = np.where(first_column == "TRAJECTORIES")[0][0]
            start_index = indices + 2
            columns = [df.iloc[start_index], df.iloc[start_index + 1]]
            df = df.iloc[start_index + 2:, ]
            df.columns = columns
            first_col = df.isnull().iloc[:, 0].tolist()
            # find end of data
            df = df.iloc[:first_col.index(True), ]

            col_list = [ "RKneeAngles", "LKneeAngles"]

            file = pd.DataFrame(
                df[[ ("RKneeAngles", "X"), ("LKneeAngles", "X")]]).copy()
            file.columns = col_list
            # 100hz to 50hz
            # self.files['vicon']=self.files['vicon'].drop(index=df.index[1::2])
            file = file.reset_index(drop=True)
            self.vicon_data = np.array(file.values).astype(float)
            if len(self.op_data) == 0:
                self.ax.axis(xmin=0, xmax=self.vicon_data.shape[0])
        except Exception as e:
            print("csv 內容格式不正確")

            return False, None
        return True, self.vicon_data

    def update_frame(self, frame_num=1):
        self.update_vl_line(self.frame_num)
        # self.fig.canvas.draw()
        self.frame_num = frame_num

    def update_vl_line(self, x_num):
        if self.vl:
            self.vl.remove()
        self.vl = self.ax.axvline(x=x_num, color='purple', picker=1)
        self.fig.canvas.draw()

    def remove_lines(self, remove_list):

        for item in remove_list:
            if item != None and len(item) != 0:
                line = item.pop(0)
                line.remove()
                item = None

    def clear_lines(self,start=0,finish=-1):
        for line in self.lines:
            if line != None and len(line) != 0:
                l = line.pop(0)
                l.remove()

        self.fig.canvas.draw()

    def find_indices(self, list_to_check, item_to_find):
        return [idx for idx, value in enumerate(list_to_check) if str(value) == str(item_to_find)]

    def select_vls_mode(self, data, key, is_checked):
        mark_list = self.find_indices(
            data[:, MARK], POINTMAP[key][OPENPOSE_COL.Mark.value[0]])
        auto_list = self.find_indices(
            data[:,AUTO_MARK], POINTMAP[key][OPENPOSE_COL.Auto_Mark.value[0]])
        if is_checked:
            for index in mark_list:
                self.label_vls_changed(
                    index, POINTMAP[key][OPENPOSE_COL.Mark.value[0]], 'red', draw_now=False)
            for index in auto_list:
                self.label_vls_changed(
                    index, POINTMAP[key][OPENPOSE_COL.Auto_Mark.value[0]], 'red', draw_now=False)
        else:
            for index in mark_list:
                self.label_vls_changed(
                    f"{index}_{POINTMAP[key][OPENPOSE_COL.Mark.value[0]]}", None, label_color=None, draw_now=False)
            for index in auto_list:
                self.label_vls_changed(
                    f"{index}_{POINTMAP[key][OPENPOSE_COL.Auto_Mark.value[0]]}", None, label_color=None, draw_now=False)
        self.fig.canvas.draw()

    def select_knee_mode(self, data, key, is_checked):
        item = LINEMAP[key]
        if len(data) == 0:
            print(f"no {LINEMAP[key]['label']} data")
            return
        if self.lines[key] != None and len(self.lines[key]) != 0:
            l = self.lines[key][0]
            l.remove()
            self.lines[key] = None
        if is_checked:
            tmp = self.ax.plot(data, color=item['color'], label=item['label'])
            self.lines[key] = tmp
            
            self.ax.legend(loc='upper right')
        # for mark_value

        self.fig.canvas.draw()

    def clear_legend(self):
        if self.ax.get_legend() != None:
            self.ax.get_legend().remove()

    def clear_vls(self):
        for key in self.label_vls:
            self.label_vls[key].remove()
        self.label_vls.clear()

    def clear_chart(self):
        self.clear_legend()
        self.clear_lines()
        self.clear_vls()
        self.op_data = []
        self.vicon_data = []
        return
    # test

    def set_x_slim(self, min, max):
        self.ax.axis(xmin=min, xmax=max)
        # max is for len count
        if self.mainUI.is_on_detect:
            self.lines[2] = self.ax.plot(
                np.zeros(max), color=LINEMAP[2]['color'], label=LINEMAP[2]['label'])
            self.lines[3] = self.ax.plot(
                np.zeros(max), color=LINEMAP[3]['color'], label=LINEMAP[3]['label'])
            self.ax.legend(loc='upper right')
        
    # update detect lines
    # rknee: 0 lknee:1
    def update_knee_lines(self, knee_angles, frame_num=None,kind=[2,3]):
        max_x = int(self.ax.get_xlim()[1])
       
        if len(knee_angles[:, 0]) < max_x:
            # r_knee = np.zeros(max_x)
            for i in range(2):
                if self.lines[kind[i]]==None:
                    continue
                knee = np.zeros(max_x)
                knee[:len(knee_angles[:, i])] = knee_angles[:, i]
                self.lines[kind[i]][0].set_ydata(knee)
            # l_knee = np.zeros(max_x)
            # r_knee[:len(knee_angles[:, 0])
            #        ] = knee_angles[:, 0]
            # l_knee[:len(knee_angles[:, 1])
            #        ] = knee_angles[:, 1]
            # try:
            #     self.lines[kind[0]][0].set_ydata(r_knee)
            #     self.lines[kind[1]][0].set_ydata(l_knee)
            # except TypeError:
            #     pass
        else:
            for i in range(2):
                if self.lines[kind[i]]==None:
                    continue
                self.lines[kind[i]][0].set_ydata(knee_angles[:max_x, i])
        if frame_num != None and frame_num <= max_x:
            self.frame_num = frame_num
            self.update_vl_line(self.frame_num)
        else:
            self.fig.canvas.draw()
            
    def label_vls_changed(self, key, label, label_color=None, draw_now=True):
        # remove item
        if label == None:
            if key not in self.label_vls:
                return
            if self.label_vls[key]:
                self.label_vls[key].remove()
                self.label_vls[str(key) + "_t"].remove()
            del self.label_vls[key]
            del self.label_vls[str(key) + "_t"]
        # add item
        else:
            key_name = f"{key}_{label}"
            if label_color != "black":
                label_color = "green" if "AUTO" in label else "red"
            if key_name not in self.label_vls:
                self.label_vls[key_name] = self.ax.axvline(
                    x=key, color=label_color, picker=1)
                self.label_vls[str(
                    key_name) + "_t"] = self.ax.text(key + 1, -15, key_name, rotation=90)
        if draw_now:
            self.fig.canvas.draw()
        
        
