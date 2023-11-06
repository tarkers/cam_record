import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import cv2
from pathlib import Path
import shutil
import math
import torch
from util import update_config
import numpy as np
from  numpy.linalg import norm
import collections
from util.utils import video_to_frame
from util import Timer
skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
from mpl_toolkits.mplot3d import Axes3D # must need for plot 3d
from libs.detector.apis import get_detector
from ball_nodes import TrackBall
import skimage.measure

def color_extract_mask(img):
    # bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv , (0,0,0) , (180,60,255))  # white
    blue_mask=cv2.inRange(hsv , (100,110,0) , (125,225,180))
    mask=cv2.bitwise_or(blue_mask,mask)   # blue black color
    # mask=cv2.bitwise_not(y_mask)
    image=cv2.bitwise_and(img, img,mask=mask)
   
    return mask,image

def do_morphologyEx(mask):
    for _ in range(1,2):
        mask=cv2.dilate(mask, (3,3),iterations=2)
        # kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
        # mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
        # mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask[mask>50]=255
    # cv2.imshow("tt",mask)
    # cv2.waitKey(0)
    return mask

def find_ball_shape(img,mask):
    mask=mask.copy()
    _,mask = cv2.threshold(mask,85,255,cv2.THRESH_BINARY)
    ##testt##
    rgb=img.copy()

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 40
    params.maxThreshold = 95
    # params.blobColor = 254
    params.minArea = 80
    params.maxArea = 600
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.minConvexity = 0.35
    params.minCircularity =.35
    params.maxCircularity = 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
    params.maxInertiaRatio = 1
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints=detector.detect(mask)
    key_point_xys=[]
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        r=int(keyPoint.size)
        if mask[y,x]>0 :
            ## ckeck color of ball test ##
            # crop=img[y-r:y+r,x-r:x+r,:]
            # c_y,c_x=crop.shape[0]//2,crop.shape[1]//2
            # e_mask,e_img=color_extract_mask(crop)
            # # if e_mask[c_y,c_x]>50 :
            # #     key_point_xys.append([x,y,r])
                
            # cv2.namedWindow("e_mask",cv2.WINDOW_NORMAL)  
            # cv2.imshow("e_mask",e_img)
            # print("cropt",y,x)
            # key=cv2.waitKey(0)
            # if key==27:
            #     exit()
            key_point_xys.append([x,y,r])
            cv2.circle(rgb, (x,y), r//2, (0,0,255), 3) 
    
             
    # cv2.imshow("test",rgb)
    # key=cv2.waitKey(0)
    # if key==27:
    #     exit()
    return rgb,key_point_xys

def find_ball_contour(img,mask):
    
    # mask=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)

    object_rects=[]
    img=img.copy()
    contours,_ = cv2.findContours(mask, 1, 2)
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        c_y,c_x=  y+h//2 ,x+w//2
        if w>8 and h>8 :
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
            object_rects.append([x,y,w,h])
    cv2.imshow("crop",img)
    cv2.waitKey(0)
    return img,object_rects

def extract_ball_from_person(img,mask):
    e_mask,e_img=color_extract_mask(img)
    mask=cv2.bitwise_and(mask,e_mask) 
    cv2.imshow("tt",mask)
    cv2.waitKey(0)
    find_ball_contour(e_img,mask)
    find_ball_shape(e_img,mask)
    
    
def test_video(path):
    cap=cv2.VideoCapture(path)
    cv2.namedWindow("tests",cv2.WINDOW_NORMAL)  
    tracker=TrackBall()
    knn=cv2.createBackgroundSubtractorKNN(history=450, dist2Threshold=650.0,detectShadows=True)    
    idx=0
    while True:
        ret,frame=cap.read()
        frame=frame[450:1050,:1850,:]
        img=frame.copy()
        
        img = cv2.GaussianBlur(img,(5,5),0)  # 去除球的軌跡Noise
        
        mask=knn.apply(img)
        mask = cv2.medianBlur(mask,5)
        mask = cv2.medianBlur(mask,3)
        # mask=do_morphologyEx(mask)
        
        img=cv2.bitwise_and(frame, frame,mask=mask)
        
        ## this do test ##
        # img=do_morphologyEx(img)
        # img = cv2.medianBlur(img,5)
        # img = cv2.GaussianBlur(img,(5,5),0)  
        # img = cv2.medianBlur(img,3)
        # img=do_morphologyEx(img)
        
        test_image,ball_points=find_ball_shape(img,mask)
        _,obj_rects=find_ball_contour(test_image,mask)
        if len(ball_points) >0:
            tracker.match_keypoints(ball_points,obj_rects,idx,img,mask)
        new_test=tracker.extract_path
            # rect = cv2.minAreaRect(cnt)
            # box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
        if new_test is not None and len(new_test) > 0:
            search_center=new_test[0].point
            for data in new_test:
                # if img[y,x,0]>0:
                # if tracker.ball_found:
                x,y=data.point
                cv2.circle(img, (x,y), 5, (0,255,255), -1) 
        
        
        ## do this when we find a ball path ##
        # find_ball_contour(img,mask)
        if ret :
            # cv2.imshow("mask",mask)
            cv2.imshow("tests",img)
            # cv2.imshow("tests",color_extract(frame))
            key=cv2.waitKey(0)
            if key==27:
                break

        else:
            break
        idx+=1
    cv2.destroyAllWindows()
if __name__ == "__main__":
   
    # test_video(rf"Test\D2\Cz03_649_2023-10-15_15-00-26-108.mkv")
    # test_video(rf"Test\D2\Cz03_585_2023-10-15_13-15-23-539.mkv")
    # test_video(rf"Test\D1\Cz03_251_2023-10-14_14-36-37-620.mkv")
    test_video(rf"Test\D1\Cz03_845_2023-10-14_19-14-49-225.mkv")
    exit()