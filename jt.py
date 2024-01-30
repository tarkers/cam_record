import numpy as np 
import pandas as pd
POSE2D={"Nose":"鼻子",
    "LEye":"左眼",
    "REye":"右眼",
    "LEar":"左耳",
    "REar":"右耳",
    "LShoulder":"左肩",
    "RShoulder":"右肩",
    "LElbow":"左手肘",
    "RElbow":"右手肘",
    "LWrist":"左手腕",
    "RWrist":"右手腕",
    "LHip":"左臀",
    "RHip":"右臀",
    "LKnee":"左膝",
    "Rknee":"右膝",
    "LAnkle":"左腳踝",
    "RAnkle":"右腳踝",
    "Head":"頭部",
    "Neck":"脖子",
    "Hip":"臀部中心",
    "LBigToe":"左大拇指",
    "RBigToe":"右大拇指",
    "LSmallToe":"左腳小指",
    "RSmallToe":"右腳小指",
    "LHeel":"左腳跟",
    "RHeel":"右腳跟"}

def create_2D_csv(data=None):
    first_header = (
        ["ImageID"] + ["ID"] + ["BBox"] * 4 + list(np.repeat(np.array([list(POSE2D.keys())]), 3))
    )
    second_header = (
        ["ID"] + ["ID"] + ["X", "Y", "W", "H"] + ["X", "Y", "Confidence"] * len(list(POSE2D.keys()))
    )

    array = [first_header] + [second_header]
    header = pd.MultiIndex.from_arrays(array, names=("Names", "Points"))
    df = pd.DataFrame(data, columns=header)
    return df

print(POSE2D.keys())
create_2D_csv()