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


def non_max_suppression(boxes,scores,threshold=0.45):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    :param boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    :param scores: a list of corresponding scores
    :param threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    :return: a list of indices of the boxes to keep after non-max suppression
    """
    # Sort the boxes by score in descending order
    if scores is None:
        order=list(range(len(boxes)))
    else:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep


def knn_video(path):
    cap=cv2.VideoCapture(path)
    knn=cv2.createBackgroundSubtractorKNN(history=550, dist2Threshold=650.0,detectShadows=True)
    timer=Timer()
    i=0
    mask = None
    search_center=[]
    tracker=TrackBall()
    while True:
        ret,frame=cap.read()
        if ret:
            img=frame[450:1050,:1850,:]

            rgb=img.copy()


            timer.tic()
            img = cv2.GaussianBlur(img,(3,3),0)  
            img=knn.apply(img)
            test=img.copy()
            # contours,hierarchy = cv2.findContours(img, 1, 2)
            # for cnt in contours:
            #     x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
            #     if w>10 and h >10:
            #         cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # if tracker.ball_found:  # limit search range
            #     x,y=search_center
            #     # print(max(y-100,0),min(1050,y+100),max(x-100,0),min(1850,x+100))
            #     test=img[max(y-100,0):min(1050,y+100),max(x-100,0):min(1850,x+100)]
            #     test_rgb=rgb[max(y-70,0):min(1050,y+70),max(x-70,0):min(1850,x+70)]
            #     test = cv2.medianBlur(test,5)
            #     test = cv2.GaussianBlur(test,(3,3),0)
            #     contours,hierarchy = cv2.findContours(test, 1, 2)
            #     for cnt in contours:
            #         x,y,w,h=cv2.boundingRect(cnt)
            #         if w*h>60 and w*h<800:
            #             cv2.rectangle(test_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
            #     cv2.imshow("crop",test_rgb)
            #     cv2.waitKey(0)
                
            # img = cv2.medianBlur(img,3)
            
            # se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
            # img=cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
            timer.toc()
            timer.tic()
            
            # contours,hierarchy = cv2.findContours(img, 1, 2)
            # for cnt in contours:
            #     x,y,w,h=cv2.boundingRect(cnt)
            #     if w*h>80 and w*h<800:
            #         cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #     elif w*h<80:
            #         img[x:x+w,y:y+h]=0
                    
                    
            # cv2.imshow("blur",img)
            # cv2.waitKey(1)
       
            img = cv2.medianBlur(img,5)
            img = cv2.GaussianBlur(img,(3,3),0)
            img,key_point_xys=find_ball_per_frame(img)
            # if key_point_xys is not None:
            #     for data in key_point_xys:
            #         x,y,size=data
            #         # if img[y,x,0]>0:
            #         cv2.circle(rgb, (x,y), size//2, (255,0,0), 3) 
            
            
            if len(key_point_xys) >0:
                tracker.match_keypoints(key_point_xys,i,rgb,test)
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
                    cv2.circle(rgb, (x,y), 5, (0,255,255), -1) 
                        # else:
                        #     cv2.circle(rgb, (x,y), 5, (255,0,0), -1) 

            # fgmask =cv2.threshold(img.copy(),100,255,cv2.THRESH_BINARY)[1]
            # dilated=cv2.dilate(fgmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
            # cv2.imwrite(rf"Test\image\{i:05}.png",fgmask)
            # out=give_box(fgmask,frame)
            timer.toc()
            # contours,hierarchy = cv2.findContours(img, 1, 2)
            # for cnt in contours:
            #     x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
            #     if w>10 and h >10:
            #         cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(rgb, str(timer.fps), (10, 60), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("frame",rgb)
            key=cv2.waitKey(0)
            if key==27:
                break
            i+=1
        else:
            break
    cv2.destroyAllWindows()



def get_nearlist_path():
    pass

def knn_images(images):
    knn=cv2.createBackgroundSubtractorKNN(detectShadows=True)
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


    
def find_ball_per_frame(img):
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
    keypoints=detector.detect(img)
    key_point_xys=[]
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        
        # if np.count_nonzero(img[x-2:x+2,y-2:y+2] > 0) >15:
        if img[y,x]>50 :
            key_point_xys.append([x,y,int(keyPoint.size)])
        
    # print("================================================")
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # for data in key_point_xys:
    #     x,y,size=data
    #     if img[y,x,0]>0:
    #         cv2.circle(img, (x,y), size//2, (0,0,255), 2) 
    # cv2.drawKeypoints(img,keypoints,img,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("test",img)
    # key=cv2.waitKey(1)
    # if key==27:
    #     exit()
    return img,key_point_xys



if __name__ == "__main__":
   
    # knn_video(rf"Test\D2\Cz03_649_2023-10-15_15-00-26-108.mkv")
    knn_video(rf"Test\D2\Cz03_750_2023-10-15_13-57-12-838.mkv")
    exit()
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
        
    # knn_video(rf"Test\D2\Cz03_585_2023-10-15_13-15-23-539.mkv")
    # exit()
    or_images,fps,count=video_to_frame(rf"Test\D1\Cz03_310_2023-10-14_15-11-22-365.mkv")
    # test = update_config(r"libs\configs\configs.yaml").Detector
    # test.gpus = "0"
    # test.gpus = (
    #     [int(i) for i in test.gpus.split(",")]
    #     if torch.cuda.device_count() >= 1
    #     else [-1]
    # )
    # test.device="0"
    # # test.device = 
    # test.detbatch = test.detbatch * len(test.gpus)
    # yolo=get_detector(test)
    cv2.namedWindow("crop",cv2.WINDOW_NORMAL)
    ball_found=False
    images=sorted(glob.glob(rf"Test\image\*"))
    i=0
    key_points=[]
    test=cv2.imread(images[0],cv2.IMREAD_GRAYSCALE)
    
    mask = np.full((test.shape[0], test.shape[1],3), 0, dtype=np.uint8)
    for test in images:
        test=cv2.imread(test,cv2.IMREAD_GRAYSCALE)
        img,key_point_xys=find_ball_per_frame(test)
        
        if len(key_point_xys)>0:
            # key_points.append(key_point_xys)
            tracker.match_keypoints(key_point_xys,i)

        # cv2.imwrite(rf"Test\image_new\{i:05}.png",img)
        see = or_images[i].copy()[450:1050,:1850,:]
        for data in key_point_xys:
            x,y,size=data
            r=size+5
        #     crop=see[max(0,y-r):y+r,max(0,x-r):x+r,:]
        #     cv2.imshow("crop",crop)
        #     # img_k=yolo.image_preprocess(crop) 
        #     # if isinstance(img_k, np.ndarray):
        #     #     img_k = torch.from_numpy(img_k)
        #     # # # add one dimension at the front for batch if image shape (3,h,w)
        #     # if img_k.dim() == 3:
        #     #     img_k = img_k.unsqueeze(0)
            
        #     # with torch.no_grad(): # Record original image resolution
        #     #     im_dim_list_k = crop.shape[1], crop.shape[0]
        #     #     im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
        #     #     dets = yolo.images_detection(img_k, im_dim_list_k)
        #     #     if dets is not None:
        #     #         print(dets )
        #     key=cv2.waitKey(0)
        #     if key==27:
        #         exit()
            cv2.circle(see, (x,y), size//2, (0,0,255), -1) 
        cv2.imshow("test",see)
        cv2.imshow("test1",img)
        key=cv2.waitKey(0)
        if key==27:
            exit()
        i+=1