from enum import Enum,IntEnum
import numpy as np
import json
import cv2
import os
from libs.vis import read_gt
from scipy.signal import savgol_filter
import yaml
from easydict import EasyDict as edict

class DataType(Enum):
    DEFAULT = {"name": "default", "tips": "", "filter": ""}
    IMAGE = {"name": "image", "tips": "",
             "filter": "Image files (*.jpeg *.png *.tiff *.psd *.pdf *.eps *.gif)"}
    VIDEO = {"name": "video", "tips": "",
             "filter": "Video files ( *.WEBM *.MPG *.MP2 *.MPEG *.MPE *.MPV *.OGG *.MP4 *.M4P *.M4V *.AVI *.WMV *.MOV *.QT *.FLV *.SWF)"}
    CSV = {"name": "csv", "tips": "",
           "filter": "Video files (*.csv)"}
    FOLDER = {"name": "folder", "tips": "", "filter": ""}

class MessageType(IntEnum):
    INFORMATION=1,
    QUESTION=2,
    WARNING=3,
    CRITICAL=4,

class MessageButtonType(IntEnum):
    YES=0,
    No=1,
    Cancel=2,
    OK=3,
def video_to_frame(input_video_path):
    '''
    return video_image,fps,count
    '''
    video_images = []
    vidcap = cv2.VideoCapture(input_video_path)
    if (vidcap.isOpened() == False):
        print("Error opening the video file")
        exit()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    count = 0
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == True:
            video_images.append(frame)
            count += 1
        else:
            break
    vidcap.release()

    # set image count labels
    return video_images, fps, count

def load_json(json_path,video_name=None):
    '''
    if give video_names
    return id,start_frame
    else return video name list
    '''
    json_path=os.path.join(json_path,"index.json")
    with open(json_path) as f:
        d = json.load(f)
    if not video_name:
        return list(d.keys())
    else:
        try:
            Id=d[video_name]['id']
            sf=d[video_name]['start_frame']
            return Id,sf
        except Exception :
            return None
        


def load_2d_json(json_path,person_id=1):
    '''
    load 2d keypoints from json file 
    format:xxx/xxx/results.json
    '''
    json_path=os.path.join(json_path,"results.json")
    with open(json_path) as f:
        d = json.load(f)
    
    keypoints=[]
    for data in d:
        if data["idx"]==person_id:
            keypoints.append(np.array(data['keypoints']).reshape(-1,3))
            
    return np.array(keypoints)


def filter_smooth(right,left,smooth=19):
    for _ in range(2):
        left = savgol_filter(left,smooth, 3)
        right = savgol_filter( right,smooth, 3)
    return right,left

def cal_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return min(np.degrees(angle),180-np.degrees(angle))

def evaluate_angle(data:np.ndarray):
    '''
    data:[F,J,3]
    evaluate the angle of our estimate data
    '''
    right,left=[],[]
    for i in range(len(data)):
        a,b,c=data[i,:3] #right
        d,e,f=data[i,3:] #left
        r,l=round(cal_angle(a,b,c),2),round(cal_angle(d,e,f),2)
        right.append(r)
        left.append(l)
    return  right,left

def load_3d_angles(npy_path,gt_csv=None,start_index=0):
    '''
    load 3d keypoints from npy file 
    format:xxx/xxx/3D_xxx.npy
    gt_csv: VICON groundtruth csv
    '''
    keypoints=np.load(npy_path)
    keypoints=np.concatenate(([keypoints[:, (1,2,3), :],keypoints[:, (4,5,6), :]]),axis=1)
    return load_angles(keypoints,gt_csv,start_index)

def load_angles(data,gt_csv,start_index):
    right,left=evaluate_angle(data)
    right,left=filter_smooth(right,left)
    right=np.concatenate((np.full(start_index,np.nan), right), axis=0)
    left=np.concatenate((np.full(start_index,np.nan), left), axis=0)
    
    if gt_csv:
        gt,_=read_gt(gt_csv)

        min_right=min(len(right),len(gt[:,0]))
        min_left=min(len(left),len(gt[:,1]))
        tr,tl=right[:min_right],left[:min_left]
        tgr,tgl=gt[:min_right,0],gt[:min_left,1]
        return tr,tl,tgr,tgl
        
    return right,left,None,None

def gen_2d_angles(keypoints,gt_csv=None,start_index=0):
    
    return load_angles(keypoints,gt_csv,start_index)




def generate_2D_skeletons( sn, vn, person_id=1):
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

    keypoints=np.concatenate(([keypoints[:, (12,14,16), :],keypoints[:, (11, 13, 15), :]]),axis=1)
    
    return keypoints.astype(int)

    




def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config