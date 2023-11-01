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
from track_ball import TrackBall
import skimage.measure

def color_extract(img):
    # bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv , (0,0,0) , (180,60,255))  # white
    mask=cv2.bitwise_or(cv2.inRange(hsv , (100,80,0) , (125,225,255)),mask)  #blue black color
    # mask=cv2.bitwise_not(y_mask)
    image=cv2.bitwise_and(img, img,mask=mask)
    # cv2.imshow("ee",img)
    # cv2.waitKey(0)
    return image

def do_morphologyEx(mask):
    
    for i in range(1,2):
        mask=cv2.dilate(mask, (3,3),iterations=2)
        # kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
        # mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
        # mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask[mask>50]=255
    # cv2.imshow("tt",mask)
    # cv2.waitKey(0)
    return mask

def find_ball_shape(img,mask):
    # mask=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(img,5)
    # img=cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
    # 
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 35
    params.maxThreshold = 95
    # params.blobColor = 254
    params.minArea = 80
    params.maxArea = 600
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.minConvexity = 0.35
    params.minCircularity =.25
    params.maxCircularity = 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.45
    params.maxInertiaRatio = 1
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints=detector.detect(mask)
    key_point_xys=[]
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        
        # if np.count_nonzero(img[x-2:x+2,y-2:y+2] > 0) >15:
        if mask[y,x]>50 :
            key_point_xys.append([x,y,int(keyPoint.size)])
        
    # print("================================================")
    for data in key_point_xys:
        x,y,size=data
        if img[y,x,0]>0:
            cv2.circle(img, (x,y), size//2, (0,0,255), 2) 
    # cv2.drawKeypoints(img,keypoints,img,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("test",img)
    key=cv2.waitKey(0)
    if key==27:
        exit()
    return img,key_point_xys

def find_ball_contour(img,mask):
    mask=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img=img.copy()
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        c_x,c_y= (x + w)//2, (y + h)//2
        if w>10 and h>10 :
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("crop",img)
    cv2.waitKey(0)

def test_video(path):
    cap=cv2.VideoCapture(path)
    cv2.namedWindow("tests",cv2.WINDOW_NORMAL)  
    knn=cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=750.0,detectShadows=True)    
    while True:
        ret,frame=cap.read()
        frame=frame[450:1050,:1850,:]
        img=frame.copy()
        
        img = cv2.GaussianBlur(img,(5,5),0)  # 去除球的軌跡Noise
        
        mask=knn.apply(img)
        mask = cv2.medianBlur(mask,5)
        mask=do_morphologyEx(mask)
        
        img=cv2.bitwise_and(frame, frame,mask=mask)
        
        ## this do test ##
        # img=do_morphologyEx(img)
        # img = cv2.medianBlur(img,5)
        # img = cv2.GaussianBlur(img,(5,5),0)  
        # img = cv2.medianBlur(img,3)
        # img=do_morphologyEx(img)
        
        cv2.imshow("tt",mask)
        cv2.waitKey(0)
        find_ball_contour(img,mask)
        find_ball_shape(img,mask)
        if ret :
            cv2.imshow("tests",img)
            # cv2.imshow("tests",color_extract(frame))
            key=cv2.waitKey(0)
            if key==27:
                break

        else:
            break
    cv2.destroyAllWindows()
if __name__ == "__main__":
   
    # test_video(rf"Test\D2\Cz03_649_2023-10-15_15-00-26-108.mkv")
    # test_video(rf"Test\D2\Cz03_585_2023-10-15_13-15-23-539.mkv")
    test_video(rf"Test\D2\Cz03_750_2023-10-15_13-57-12-838.mkv")
    exit()