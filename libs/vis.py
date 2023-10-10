import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
from scipy.signal import savgol_filter
import pathlib
import shutil
import random
import torch
from libs.analyze import cal_RMSE, cal_MAE, cal_SD, gradient_analysis

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def filter_smooth(right, left, smooth=19):
    for _ in range(2):
        left = savgol_filter(left, smooth, 3)
        right = savgol_filter(right, smooth, 3)
    return right, left


def read_gt(filename):
    try:
        df = pd.read_csv(filename, header=None, low_memory=False)
       
    except pd.errors.ParserError:
        print("Ground truth csv錯誤\n(請確認是否為正確的csv檔案!)")
        return None
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
        
        file = file.reset_index(drop=True)
        gt = np.array(file.values).astype(float)
    except Exception as e:
        print("csv 內容格式不正確")

        return None
    return gt, df


def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return min(np.degrees(angle), 180 - np.degrees(angle))


def evaluate_angle(data:np.ndarray):
    '''
    data:[F,J,3]
    evaluate the angle of our estimate data
    '''
    right, left = [], []
    for i in range(len(data)):
        a, b, c = data[i, [1, 2, 3]]  # right
        d, e, f = data[i, [4, 5, 6]]  # left
        r, l = round(cal_angle(a, b, c), 2), round(cal_angle(d, e, f), 2)
        right.append(r)
        left.append(l)
    return  right, left


def vis_angle(basename, data, out_folder='out_mask', show=True, linewidth=2):
    '''
    # N:kind of data , F:number of frames 
    data:{
        'right':[[],[],...],
        'left':[[],[],...],
        'names':['data','ground truth',...],
        'color':['b','r','g',....]
    }   
    show: if false then save fig only
    '''
    data_nums = len(data['names'])
    plt.figure(figsize=(13, 4))
    plt.ylim([-20, 100])

    for i in range(data_nums):
        right = data['right'][i]
        name = data['names'][i]
        color = data['color'][i]
        plt.plot(right, linestyle="solid" , color=color, linewidth=linewidth, label=f'{name}_R')

    plt.title(f'{basename} Right Knee')
    plt.legend(loc="upper right")
    plt.ylabel("Angle")  # y label
    plt.xlabel("Frame")  # x label
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_folder, f"{basename}_right.png"))
    plt.close()
    plt.figure(figsize=(13, 4))
    plt.ylim([-20, 100])
    for i in range(data_nums):
        left = data['left'][i]
        name = data['names'][i]
        color = data['color'][i]
        plt.plot(left, linestyle="solid" , color=color, linewidth=linewidth, label=f'{name}_L')

    plt.title(f'{basename} Left Knee')
    plt.legend(loc="upper right")
    plt.ylabel("Angle")  # y label
    plt.xlabel("Frame")  # x label
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_folder, f"{basename}_left.png"))
    plt.close()


def get_start_end(data, gt):
    start_not_nan = 0
    end_not_nan = min(len(data), len(gt))
    is_start = False
    for x in range(arr[i], end_not_nan): 
        if is_start and  np.isnan(gt[x]):
            end_not_nan = min(x, end_not_nan)
            break
        elif not is_start and not np.isnan(gt[x]):
            start_not_nan = x
            is_start = True
    return start_not_nan, end_not_nan


def save_all_analyze(file_place, data):
    columns = ["name", "RMSE/angle(Right Leg)", "RMSE/angle(Left Leg)", "MAE/angle(Right Leg)", "MAE/angle(Left Leg)", "SD/angle(Right Leg)", "SD/angle(Left Leg)", "mu/%(Right Leg)", "mu/%(Left Leg)"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_place, sep=',', index=True, encoding='utf-8')
    pass


def renew_folder(folder_path):
    if os .path.exists(folder_path):
        shutil.rmtree(folder_path)
    p = pathlib.Path(folder_path)
    p.mkdir(parents=True, exist_ok=True)


def plot_boxes(ori_image, boxes, ids=None, classes=None , line_thickness=3):
    '''
    plot detect bboxes with ids and classes
    '''
    # cv2 show
    img = ori_image.copy()
    #bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for idx, box in enumerate(boxes):
        # # draw boxes
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        if ids is not None:
            tmp = ids[idx]
            while isinstance(ids[idx], list):
                ids[idx].sort()
                tmp = ids[idx][0]
            color = get_color_fast(int(abs(tmp)))
        else:
            color = BLUE
            
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        
        # # draw ids
        if ids is not None:
            tf = max(tl - 1, 1)  # font thickness
            _id = str(int(ids[idx]))
            cv2.putText(img, _id, (c1[0], c1[1] - 2), 0, tl / 3, [255, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
        
        # # draw classes
        if  classes is not None:
            label = classes[idx]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def plot_2d_skeleton(frame, im_res, showbox=False, tracking=False,fast=True, format='coco'):
    '''
    from alphapose database import
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num=17
    
    frame = frame.copy()
    #bgr to rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if len(im_res['result']) > 0:
        # if fast:
        #     kp_num=17
        # else:
        #     kp_num = len(im_res['result'][0]['keypoints'])
        kp_num = len(im_res['result'][0]['keypoints'])
        vis_thres = [0.4] * kp_num
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)]  # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    else:
        raise NotImplementedError
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        if kp_num == 17: 
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            vis_thres.append(vis_thres[-1])
        if tracking:
            unique_id = human['idx']
            
            if isinstance(human['idx'], list):
                human['idx'].sort()
                unique_id = human['idx'][0]
            color = get_color_fast(int(abs(unique_id)))
        else:
            color = BLUE

        # Draw bboxes
        if 'box' in human.keys():
            bbox = human['box']
            bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]  # xmin,xmax,ymin,ymax

            if showbox:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            if tracking:
                cv2.putText(img, str(int(unique_id)), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
        
        # Draw keypoints
        for n in range(len(vis_thres)):
            if kp_scores[n] <= vis_thres[n]:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                if tracking:
                    cv2.circle(img, (cor_x, cor_y), 3, color, -1)
                else:
                    cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
                # cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    if tracking:
                        cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    else:
                        cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    # cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255, 255, 255), 1)

    return img


if __name__ == '__main__':
    import glob
    tests = sorted(glob.glob(rf"D:\Chen\transform_bag\Bert3D\N0002\*.npy"))
    gts = sorted(glob.glob(rf"D:\Chen\transform_bag\Calibrate_Video\N0002\*.csv"))
    arr = np.zeros(len(tests)).astype(int)
    arr = [0, 173, 0, 45, 90, 35, 0, 22, 0, 51, 0, 48, 29, 15, 29, 30, 0, 2]
    analyze_arr = []
    names = []
    
    # fig folder
    fig_path = r"D:\Chen\transform_bag\Bert3D\N0002\FIG"
    renew_folder(fig_path)
    for i, test in enumerate(tests):
        name = os.path.basename(test).split(".")[0]
        names.append(name)
        data = np.load(test)
        gt, df = read_gt(gts[i])
       
        right, left = evaluate_angle(data)
        right, left = filter_smooth(right, left)
        right = np.concatenate((np.full(arr[i], np.nan), right), axis=0)
        left = np.concatenate((np.full(arr[i], np.nan), left), axis=0)
        # rsnn,renn=get_start_end(right,gt[:,0])
        # lsnn,lenn=get_start_end(left,gt[:,1])
        min_right = min(len(right), len(gt[:, 0]), 450)
        min_left = min(len(left), len(gt[:, 1]), 450)
        tr, tl = right[:min_right], left[:min_left]
        tgr, tgl = gt[:min_right, 0], gt[:min_left, 1]
        # tr,tl=right[rsnn:min(rsnn+400,renn)],left[lsnn:min(lsnn+400,lenn)]
        # tgr,tgl=gt[rsnn:min(rsnn+400,renn),0],gt[lsnn:min(lsnn+400,lenn),1]

        l_mean, r_mean = cal_MAE(tl, tgl), cal_MAE(tr, tgr)  # to cal sd
        tmp = np.array([cal_RMSE(tr, tgr, 1), cal_RMSE(tl, tgl, 1), \
             cal_MAE(tr, tgr, 1), cal_MAE(tl, tgl, 1), \
             cal_SD(tr, tgr, r_mean, 1), cal_SD(tl, tgl, l_mean, 1), \
            gradient_analysis(tr, tgr, 1), gradient_analysis(tl, tgl, 1)])
        analyze_arr.append(tmp)
        test = {
            'right':[tr, tgr],
            'left':[tl, tgl],
            'names':['data', 'vicon'],
            'color':['b', 'r']
        }
        print(name)
        
        vis_angle(name, test, fig_path, False)
    analyze_arr = np.array(analyze_arr)
    # create average
    names.append("Average")
    avg = np.round(np.average(analyze_arr, axis=0), decimals=1) 
    analyze_arr = np.concatenate((analyze_arr, [avg]), axis=0)
    analyze_arr = np.concatenate((np.array([names]).T, analyze_arr), axis=1)
    print(analyze_arr)
    # print(avg)
    save_all_analyze(r"D:\Chen\transform_bag\Bert3D\N0002\N0002.csv", analyze_arr)

