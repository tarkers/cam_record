import numpy as np
import json
import cv2
import os
from libs.vis import read_gt
from scipy.signal import savgol_filter
import yaml
import pandas as pd
from easydict import EasyDict as edict

POSE2D = {
    "Nose": "鼻子",
    "LEye": "左眼",
    "REye": "右眼",
    "LEar": "左耳",
    "REar": "右耳",
    "LShoulder": "左肩",
    "RShoulder": "右肩",
    "LElbow": "左手肘",
    "RElbow": "右手肘",
    "LWrist": "左手腕",
    "RWrist": "右手腕",
    "LHip": "左臀",
    "RHip": "右臀",
    "LKnee": "左膝",
    "Rknee": "右膝",
    "LAnkle": "左腳踝",
    "RAnkle": "右腳踝",
    "Head": "頭部",
    "Neck": "脖子",
    "Hip": "臀部中心",
    "LBigToe": "左大拇指",
    "RBigToe": "右大拇指",
    "LSmallToe": "左腳小指",
    "RSmallToe": "右腳小指",
    "LHeel": "左腳跟",
    "RHeel": "右腳跟",
}

POSE3D = {
    "Root": "臀部中心",
    "RHip": "右臀",
    "RKnee": "右膝",
    "RFoot": "右腳踝",
    "LHip": "左臀",
    "LKnee": "左膝",
    "LFoot": "左腳踝",
    "Spine": "脊椎中心",
    "Thorax": "胸膛頂",
    "Neck": "頸部",
    "Head": "頭頂",
    "LShoulder": "左肩",
    "LElbow": "左手肘",
    "LWrist": "左手腕",
    "RShoulder": "右肩",
    "RElbow": "右手肘",
    "RWrist": "右手腕",
}


def camera_alignment(image, square_w=30):
    h, w, ch = image.shape
    for i in range(0, w, square_w):
        image = cv2.line(image, (i, 0), (i, h), (0, 255, 0), 1)
    for y in range(0, h, square_w):
        image = cv2.line(image, (0, y), (w, y), (0, 255, 0), 1)
    return image


def video_to_frame(input_video_path, start_frame=0, end_frame=2000):
    """Return video_image,fps,count"""
    video_images = []
    vidcap = cv2.VideoCapture(input_video_path)
    if vidcap.isOpened() == False:
        print("Error opening the video file")
        exit()
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    count = 0
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == True:
            if start_frame <= count:
                video_images.append(frame)
            count += 1
            if count > end_frame:
                break

        else:
            break
    vidcap.release()

    # set image count labels
    return video_images, fps, count


def set_video_capture(input_video_path):
    """Return vidcap,fps,count,fourcc,(w,h)"""
    vidcap = cv2.VideoCapture(input_video_path)
    fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
    if vidcap.isOpened() == False:
        print("Error opening the video file")
        return None
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return vidcap, fps, count, fourcc, (w, h)


def load_json(json_path, video_name=None):
    """
    if give video_names
    return id,start_frame
    else return video name list
    """
    json_path = os.path.join(json_path, "index.json")
    with open(json_path) as f:
        d = json.load(f)
    if not video_name:
        return list(d.keys())
    else:
        try:
            Id = d[video_name]["id"]
            sf = d[video_name]["start_frame"]
            return Id, sf
        except Exception:
            return None


def load_2d_json(json_path, person_id=1):
    """
    load 2d keypoints from json file
    format:xxx/xxx/results.json
    """
    json_path = os.path.join(json_path, "results.json")
    with open(json_path) as f:
        d = json.load(f)

    keypoints = []
    for data in d:
        if data["idx"] == person_id:
            keypoints.append(np.array(data["keypoints"]).reshape(-1, 3))

    return np.array(keypoints)


def filter_smooth(right, left, smooth=19):
    for _ in range(2):
        left = savgol_filter(left, smooth, 3)
        right = savgol_filter(right, smooth, 3)
    return right, left


def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return min(np.degrees(angle), 180 - np.degrees(angle))


def evaluate_angle(data: np.ndarray):
    """
    data:[F,J,3]
    // keypoint_info: https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/_base_/datasets/h36m.py
    \nkeypoint_info={
        0:root
        1:right_hip
        2:right_knee
        3:right_foot
        4:left_hip
        5:left_knee
        6:left_foot
        7:spine
        8:thorax
        9:neck_base
        10:head
        11:left_shoulder
        12:left_elbow
        13:left_wrist
        14:right_shoulder
        15:right_elbow
        16:right_wrist
    """
    right, left = [], []
    for i in range(len(data)):
        a, b, c = data[i, :3]  # right
        d, e, f = data[i, 3:]  # left
        r, l = round(cal_angle(a, b, c), 2), round(cal_angle(d, e, f), 2)
        right.append(r)
        left.append(l)
    return right, left


def load_3d_angles(npy_path, gt_csv=None, start_index=0):
    """
    load 3d keypoints from npy file
    format:xxx/xxx/3D_xxx.npy
    gt_csv: VICON groundtruth csv
    """
    keypoints = np.load(npy_path)
    keypoints = np.concatenate(
        ([keypoints[:, (1, 2, 3), :], keypoints[:, (4, 5, 6), :]]), axis=1
    )
    return load_angles(keypoints, gt_csv, start_index)


def load_angles(data, gt_csv, start_index):
    right, left = evaluate_angle(data)
    right, left = filter_smooth(right, left)
    right = np.concatenate((np.full(start_index, np.nan), right), axis=0)
    left = np.concatenate((np.full(start_index, np.nan), left), axis=0)

    if gt_csv:
        gt, _ = read_gt(gt_csv)

        min_right = min(len(right), len(gt[:, 0]))
        min_left = min(len(left), len(gt[:, 1]))
        tr, tl = right[:min_right], left[:min_left]
        tgr, tgl = gt[:min_right, 0], gt[:min_left, 1]
        return tr, tl, tgr, tgl

    return right, left, None, None


def gen_2d_angles(keypoints, gt_csv=None, start_index=0):
    return load_angles(keypoints, gt_csv, start_index)


def generate_2D_skeletons(sn, vn, person_id=1):
    """
    get skleton from 2D
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},-->3
    {12, "RHip"},-->0
    {13, "LKnee"},-->4
    {14, "Rknee"},-->1
    {15, "LAnkle"},-->5
    {16, "RAnkle"},-->2
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},

    """

    keypoints = load_2d_json(rf"Data/{sn}/{vn}/subject_2D", person_id)

    keypoints = np.concatenate(
        ([keypoints[:, (12, 14, 16), :], keypoints[:, (11, 13, 15), :]]), axis=1
    )

    return keypoints.astype(int)


def get_3D_skeletons(np_files):
    """
    load numpy estimation files of 17 3D joint Point
    Dataset Format => [Frames,17,3]
    """
    data = np.load(np_files)  # [Frames,17,3]

    return data


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def create_2D_csv(data=None):
    first_header = (
        ["ImageID"]
        + ["ID"]
        + ["BBox"] * 4
        + list(np.repeat(np.array([list(POSE2D.keys())]), 3))
    )
    second_header = (
        ["ID"]
        + ["ID"]
        + ["X", "Y", "W", "H"]
        + ["X", "Y", "Confidence"] * len(list(POSE2D.keys()))
    )

    array = [first_header] + [second_header]
    header = pd.MultiIndex.from_arrays(array, names=("Names", "Points"))
    df = pd.DataFrame(data, columns=header)
    convert_dict = {("ImageID", "ID"): str, ("ID", "ID"): int}
    df = df.astype(convert_dict)
    return df


def create_3D_csv(data=None):
    first_header = ["ImageID"] + list(np.repeat(np.array([list(POSE3D.keys())]), 3))
    second_header = ["ID"] + ["X", "Y", "Z"] * len(list(POSE3D.keys()))

    array = [first_header] + [second_header]
    header = pd.MultiIndex.from_arrays(array, names=("Names", "Points"))
    df = pd.DataFrame(data, columns=header)
    convert_dict = {("ImageID", "ID"): str}
    df = df.astype(convert_dict)
    return df


def create_Bbox_csv():
    first_header = ["ImageID"] + ["ID"] + ["ClassID"] + ["BBox"] * 5
    second_header = ["ID"] + ["ID"] + ["X", "Y", "W", "H", "Confidence"]

    array = [first_header] + [second_header]
    header = pd.MultiIndex.from_arrays(array, names=("Names", "Points"))
    df = pd.DataFrame(None, columns=header)
    convert_dict = {("ImageID", "ID"): str, ("ID", "ID"): int}
    df = df.astype(convert_dict)
    return df


def save_json(data, file_path):
    # Serializing json
    json_object = json.dumps(data)

    # Writing to sample.json
    with open(file_path, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    pass
