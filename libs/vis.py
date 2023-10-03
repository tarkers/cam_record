import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
from scipy.signal import savgol_filter
import pathlib
import shutil
import random

from libs.analyze import cal_RMSE,cal_MAE,cal_SD,gradient_analysis

def filter_smooth(right,left,smooth=19):
    for _ in range(2):
        left = savgol_filter(left,smooth, 3)
        right = savgol_filter( right,smooth, 3)
    return right,left

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
    return gt,df

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
        a,b,c=data[i,[1,2,3]] #right
        d,e,f=data[i,[4,5,6]] #left
        r,l=round(cal_angle(a,b,c),2),round(cal_angle(d,e,f),2)
        right.append(r)
        left.append(l)
    return  right,left

def vis_angle(basename,data,out_folder='out_mask',show=True, linewidth=2):
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
    data_nums=len(data['names'])
    plt.figure(figsize=(13,4))
    plt.ylim([-20, 100])

    for i in range(data_nums):
        right=data['right'][i]
        name=data['names'][i]
        color=data['color'][i]
        plt.plot(right, linestyle ="solid" ,color=color,linewidth=linewidth,label=f'{name}_R')

    plt.title(f'{basename} Right Knee')
    plt.legend(loc="upper right")
    plt.ylabel("Angle") # y label
    plt.xlabel("Frame") # x label
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_folder,f"{basename}_right.png"))
    plt.close()
    plt.figure(figsize=(13,4))
    plt.ylim([-20, 100])
    for i in range(data_nums):
        left=data['left'][i]
        name=data['names'][i]
        color=data['color'][i]
        plt.plot(left, linestyle ="solid" ,color=color,linewidth=linewidth,label=f'{name}_L')

    plt.title(f'{basename} Left Knee')
    plt.legend(loc="upper right")
    plt.ylabel("Angle") # y label
    plt.xlabel("Frame") # x label
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_folder,f"{basename}_left.png"))
    plt.close()

def get_start_end(data,gt):
    start_not_nan=0
    end_not_nan=min(len(data),len(gt))
    is_start=False
    for x in range(arr[i],end_not_nan): 
        if is_start and  np.isnan(gt[x]):
            end_not_nan=min(x,end_not_nan)
            break
        elif not is_start and not np.isnan(gt[x]):
            start_not_nan=x
            is_start=True
    return start_not_nan,end_not_nan

def save_all_analyze(file_place,data):
    columns=["name","RMSE/angle(Right Leg)","RMSE/angle(Left Leg)","MAE/angle(Right Leg)","MAE/angle(Left Leg)","SD/angle(Right Leg)","SD/angle(Left Leg)","mu/%(Right Leg)","mu/%(Left Leg)"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_place, sep=',', index=True, encoding='utf-8')
    pass

def renew_folder(folder_path):
    if os .path.exists(folder_path):
        shutil.rmtree(folder_path)
    p = pathlib.Path(folder_path)
    p.mkdir(parents=True, exist_ok=True)

def plot_boxes(ori_image,boxes,color=None, labels=None ,line_thickness=3):
    '''
    plot detect bboxes with labels
    '''
    # cv2 show
    img =ori_image.copy()
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for box,label in zip(boxes,labels):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img



if __name__ == '__main__':
    import glob
    tests=sorted(glob.glob(rf"D:\Chen\transform_bag\Bert3D\N0002\*.npy"))
    gts=sorted(glob.glob(rf"D:\Chen\transform_bag\Calibrate_Video\N0002\*.csv"))
    arr=np.zeros(len(tests)).astype(int)
    arr=[0,173,0,45,90,35,0,22,0,51,0,48,29,15,29,30,0,2]
    analyze_arr=[]
    names=[]
    
    #fig folder
    fig_path=r"D:\Chen\transform_bag\Bert3D\N0002\FIG"
    renew_folder(fig_path)
    for i,test in enumerate(tests):
        name=os.path.basename(test).split(".")[0]
        names.append(name)
        data=np.load(test)
        gt,df=read_gt(gts[i])
       
        right,left=evaluate_angle(data)
        right,left=filter_smooth(right,left)
        right=np.concatenate((np.full(arr[i],np.nan), right), axis=0)
        left=np.concatenate((np.full(arr[i],np.nan), left), axis=0)
        # rsnn,renn=get_start_end(right,gt[:,0])
        # lsnn,lenn=get_start_end(left,gt[:,1])
        min_right=min(len(right),len(gt[:,0]),450)
        min_left=min(len(left),len(gt[:,1]),450)
        tr,tl=right[:min_right],left[:min_left]
        tgr,tgl=gt[:min_right,0],gt[:min_left,1]
        # tr,tl=right[rsnn:min(rsnn+400,renn)],left[lsnn:min(lsnn+400,lenn)]
        # tgr,tgl=gt[rsnn:min(rsnn+400,renn),0],gt[lsnn:min(lsnn+400,lenn),1]

        l_mean,r_mean=cal_MAE(tl,tgl),cal_MAE(tr,tgr)   #to cal sd
        tmp=np.array([cal_RMSE(tr,tgr,1),cal_RMSE(tl,tgl,1),\
             cal_MAE(tr,tgr,1),cal_MAE(tl,tgl,1),\
             cal_SD(tr,tgr,r_mean,1),cal_SD(tl,tgl,l_mean,1),\
            gradient_analysis(tr,tgr,1),gradient_analysis(tl,tgl,1)])
        analyze_arr.append(tmp)
        test={
            'right':[tr,tgr],
            'left':[tl,tgl],
            'names':['data','vicon'],
            'color':['b','r']
        }
        print(name)
        
        
        vis_angle(name,test,fig_path,False)
    analyze_arr=np.array(analyze_arr)
    #create average
    names.append("Average")
    avg= np.round(np.average(analyze_arr, axis=0), decimals=1) 
    analyze_arr=np.concatenate((analyze_arr,[avg]),axis=0)
    analyze_arr=np.concatenate((np.array([names]).T,analyze_arr),axis=1)
    print(analyze_arr)
    # print(avg)
    save_all_analyze(r"D:\Chen\transform_bag\Bert3D\N0002\N0002.csv",analyze_arr)
    
    

