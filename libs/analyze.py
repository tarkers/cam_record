import  numpy as np
import math
import os
from glob import glob
from sklearn import preprocessing as pp
# angle_files=glob('./out_mask/*.npy')
# angle_op_2d_file=glob(r'./op_11_angle/*.npy')

def cal_RMSE(d,g,digit=2):
    '''
    d: ur estimate data
    g: ur groundtruth data
    digit: rmse decimal place
    '''
    datas=d.copy()
    gts=g.copy()
    total=0
    miss_data=0
    # datas,gts=skip_data(datas,gts)
    for data,gt in zip(datas,gts):
        if np.isnan(data) or np.isnan(gt):
            miss_data+=1
            continue
        total +=  (data-gt)*(data-gt)

    rmse=math.sqrt(total/max(len(datas)-miss_data,1))
    return round(rmse,digit)

def cal_MAE(d,g,digit=1):
    '''
    d: ur estimate data
    g: ur groundtruth data
    digit: mae decimal place
    '''
    datas=d.copy()
    gts=g.copy()
    total=0
    miss_data=0

    for data,gt in zip(datas,gts):
        if np.isnan(data) or np.isnan(gt):
            miss_data+=1
            continue
        total +=  abs(data-gt)

    mae=total/max(len(datas)-miss_data,1)
   
    return round(mae,digit)

def cal_SD(d,g,means,digit=2):
    datas=d.copy()
    gts=g.copy()
    total=0
    miss_data=0
    # datas,gts=skip_data(datas,gts)
    for data,gt in zip(datas,gts):
        if np.isnan(data) or np.isnan(gt):
            miss_data+=1
            continue
        total += (means-abs(data-gt))**2

    std=math.sqrt(total/max(len(datas)-miss_data,1))
    
    return round(std,digit)

def gradient_analysis(d,g,digit=2,tho=0.15):
    '''
    d: ur estimate data
    g: ur groundtruth data
    digit:  decimal place
    tho:error factor
    '''
    datas=d.copy()
    gts=g.copy()
    ##convert np.nan to zero
    datas[np.isnan(datas)] = 0
    gts[np.isnan(gts)] = 0
    
    omega=0

    datas,gts=datas/np.linalg.norm(datas),gts/np.linalg.norm(gts)
    for i in range(1,len(datas)):
        data_gr,gt_gr=round(datas[i]-datas[i-1],5),round((gts[i]-gts[i-1]),5)

        if data_gr*gt_gr>=0 and abs(data_gr-gt_gr)<tho:
            omega+=1    

    return round(100*(omega/len(datas)),digit)

def skip_data(datas,gts,d_range=75):
    skip_line=max(np.count_nonzero(np.isnan(datas)),np.count_nonzero(np.isnan(gts)))
    datas,gts=datas[skip_line:skip_line+d_range],gts[skip_line:skip_line+d_range]
    return datas,gts

