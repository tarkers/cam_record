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
candidates=[]
candidates_path={}
def knn_video(path):
    cap=cv2.VideoCapture(path)
    knn=cv2.createBackgroundSubtractorKNN(detectShadows=True)
    timer=Timer()
    i=0
    while True:
        ret,frame=cap.read()
        if ret:
            frame=frame[450:1050,:1850,:]
            
            # img=frame[450:1050,:1850,:]
            timer.tic()
            fgmask=knn.apply(frame)
            fgmask =cv2.threshold(fgmask.copy(),254,255,cv2.THRESH_BINARY)[1]
            # dilated=cv2.dilate(fgmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
            cv2.imwrite(rf"Test\image\{i:05}.png",fgmask)
            # out=give_box(fgmask,frame)
            out=fgmask
            timer.toc()
            cv2.putText(out, str(timer.fps), (10, 60), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("frame",out)
            key=cv2.waitKey(1)
            if key==27:
                break
            i+=1
        else:
            break
    cv2.destroyAllWindows()

def give_box(fgmask,img):
    fgmask =cv2.threshold(fgmask.copy(),254,255,cv2.THRESH_BINARY)[1]
    # dilated=cv2.dilate(fgmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
    circles = cv2.HoughCircles(fgmask,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)       
    contours,hier=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.full((img.shape[0], img.shape[1],3), 0, dtype=np.uint8)
   
    # img=cv2.drawContours(img,contours,-1,(0,255,0),2)
    for c in contours:
        area=cv2.contourArea(c)
        # print(area)
        if area>1600 :
            (x,y,w,h)=cv2.boundingRect(c)
            mask[y:y+h,x:x+w,:]=img[y:y+h,x:x+w,:]
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    
    return img

def get_nearlist_path():
    pass

def knn_images(images):
    knn=cv2.createBackgroundSubtractorKNN(detectShadows=False)
    for img in images:
        img=cv2.imread(img)
        fgmask=knn.apply(img)
        cv2.imshow("frame",fgmask)
        key=cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()

def calculate_length(p1,p2):
    return round(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def point_to_line(p1,p2,p3):
    return norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)

def cal_match_path(path:dict):
    '''
    match  3 candidate path
    '''
    for k,v in path.items():
        print(k,v)

def match_keypoints(key_points,frame_idx):

    global candidates,candidates_path
    fps=60
    ball_found=False
    if len(key_points)>3: # start to match candidate
        pass
    else:
       
        for ball_candidate in key_points:
            if len(candidates)==0: #first frame
                candidates.append(ball_candidate)
            else:
                first_roots=len(candidates)
                for idx in range(first_roots):
                    root=candidates[idx]
                    # x,y,size=ball_candidate
                  
                    two_point_length=calculate_length(ball_candidate[:-1],root[:-1])
                    if two_point_length<150:
                        if frame_idx not in candidates_path:
                            candidates_path[frame_idx]=[]
                        candidates_path[frame_idx].append([root,ball_candidate])
                        if len(candidates_path.keys())>3:
                            cal_match_path(candidates_path)
                    else:
                        if ball_candidate not in candidates:
                            candidates.append(ball_candidate)
                    # print("len: ",str(calculate_length(ball_candidate[:-1],root[:-1])))
                    
                pass
    candidates=candidates[]            
    
def find_ball_per_frame(img):
    img = cv2.medianBlur(img,5)
    img=cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
    # 
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 55
    params.maxThreshold = 95
    # params.blobColor = 254
    params.minArea = 80
    params.maxArea = 450
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.minConvexity = 0.45
    params.minCircularity =.25
    params.maxCircularity = 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
    params.maxInertiaRatio = 1
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints=detector.detect(img)
    key_point_rect=[]
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        
        # if np.count_nonzero(img[x-2:x+2,y-2:y+2] > 0) >15:
        if img[y,x]>0 and int(keyPoint.size)>10:
            key_point_rect.append([x,y,int(keyPoint.size)])
        
    # print("================================================")
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for data in key_point_rect:
        x,y,size=data
        cv2.circle(img, (x,y), size//2, (0,0,255), 2) 
    cv2.drawKeypoints(img,keypoints,img,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("test",img)
    # key=cv2.waitKey(1)
    # if key==27:
    #     exit()
    return img,key_point_rect

def track_ball(img):
    img,key_point=find_ball_per_frame(img)
    # print(len(key_point))
    
    return img,key_point
if __name__ == "__main__":
   
    # knn_video(rf"D:\Chen\cam_record\Test\JAPAN vs USA _ Highlights _ Men s OQT 2023_3_1.mp4")
    # folder=rf"Test\image"
    # folder2=rf"Test\image_new"
    # folder = Path(folder) 
    # folder2 = Path(folder2) 
    # if folder.exists() and folder.is_dir():
    #     shutil.rmtree(folder)
    #     folder.mkdir(parents=True, exist_ok=True)
        
    # if folder2.exists() and folder2.is_dir():
    #     shutil.rmtree(folder2)
    #     folder2.mkdir(parents=True, exist_ok=True)
        
    # knn_video(rf"Test\D1\Cz03_310_2023-10-14_15-11-22-365.mkv")
    or_images,fps,count=video_to_frame(rf"Test\D1\Cz03_310_2023-10-14_15-11-22-365.mkv")
    test=cv2.imread(rf"D:\Chen\cam_record\test.png",cv2.IMREAD_GRAYSCALE)
    test = update_config(r"libs\configs\configs.yaml").Detector
    test.gpus = "0"
    test.gpus = (
        [int(i) for i in test.gpus.split(",")]
        if torch.cuda.device_count() >= 1
        else [-1]
    )
    test.device="0"
    # test.device = 
    test.detbatch = test.detbatch * len(test.gpus)
    yolo=get_detector(test)
    cv2.namedWindow("crop",cv2.WINDOW_NORMAL)
    ball_found=False
    images=sorted(glob.glob(rf"Test\image\*"))
    i=0
    key_points=[]
    test=cv2.imread(images[0],cv2.IMREAD_GRAYSCALE)
    
    mask = np.full((test.shape[0], test.shape[1],3), 0, dtype=np.uint8)
    for test in images:
        test=cv2.imread(test,cv2.IMREAD_GRAYSCALE)
        img,key_point_rect=track_ball(test)
        
        if len(key_point_rect)>0:
            # key_points.append(key_point_rect)
            match_keypoints(key_point_rect,i)

        # cv2.imwrite(rf"Test\image_new\{i:05}.png",img)
        see = or_images[i].copy()[450:1050,:1850,:]
        for data in key_point_rect:
            x,y,size=data
            r=size+5
            crop=see[max(0,y-r):y+r,max(0,x-r):x+r,:]
            cv2.imshow("crop",crop)
            # img_k=yolo.image_preprocess(crop) 
            # if isinstance(img_k, np.ndarray):
            #     img_k = torch.from_numpy(img_k)
            # # # add one dimension at the front for batch if image shape (3,h,w)
            # if img_k.dim() == 3:
            #     img_k = img_k.unsqueeze(0)
            
            # with torch.no_grad(): # Record original image resolution
            #     im_dim_list_k = crop.shape[1], crop.shape[0]
            #     im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
            #     dets = yolo.images_detection(img_k, im_dim_list_k)
            #     if dets is not None:
            #         print(dets )
            key=cv2.waitKey(0)
            if key==27:
                exit()
            cv2.circle(see, (x,y), size//2, (0,0,255), -1) 
        cv2.imshow("test",see)
        key=cv2.waitKey(0)
        if key==27:
            exit()
        i+=1