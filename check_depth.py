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

a = np.array([
     [20,20],
     [20,20]
])
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

def nms(bboxes, scores, iou_thresh=0.35):
    """

    :param bboxes: 检测框列表
    :param scores: 置信度列表
    :param iou_thresh: IOU阈值
    :return:
    """

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0]+bboxes[:, 2]
    y2 =  bboxes[:, 1]+bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # 结果列表
    result = []
    if scores is None:
        scores=np.ones(len(bboxes))
    index = scores.argsort()[::-1]  

    while index.size > 0:
        i = index[0]
        result.append(i) 

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores

def knn_video(path):
    cap=cv2.VideoCapture(path)
    knn=cv2.createBackgroundSubtractorKNN(detectShadows=True)
    timer=Timer()
    i=0
    mask = None
    tracker=TrackBall()
    while True:
        ret,frame=cap.read()
        if ret:
            img=frame[450:1050,:1850,:]
            rgb=img.copy()
            # if mask is None:
            #     mask = np.full((img.shape[0], img.shape[1],3), 0, dtype=np.uint8) 
            #     rgb=mask
            # img=frame[450:1050,:1850,:]
            timer.tic()
            
           
            img = cv2.GaussianBlur(img,(3,3),0)
            
            img[img<70]=0   
            img=knn.apply(img)
            img = cv2.medianBlur(img,3)
            img = cv2.medianBlur(img,5)
            # contours,hierarchy = cv2.findContours(img, 1, 2)
            # bboxes=[]
            # for cnt in contours:
            #     x,y,w,h=cv2.boundingRect(cnt)
            #     if w*h>150 :
            #         cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #     if w*h>150 and w*h<1000:
            #         bboxes.append(np.array([x,y,x + w,y+h]))
            # if len(bboxes)>0:
            #     print(len(bboxes))
            #     keep_id=non_max_suppression(np.array(bboxes),None)
            #     print("keep_id",len(keep_id))
            #     for i in keep_id:
            #         x1,y1,x2,y2=bboxes[i]
            #         cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # img = cv2.medianBlur(img,3)
            # img = cv2.medianBlur(img,5)
            # img = cv2.GaussianBlur(img,(3,3),0)
            # se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
            # img=cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
            # img = cv2.medianBlur(img,5)
            # img[img<70]=0   
            # # img[img>150]=255   
            # se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
            # img=cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
           
            # # se=cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
            # # img=cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
            # # img = cv2.medianBlur(img,5)
            # # img=cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)
            # # # img[img<80]=0   
            # # img=cv2.dilate(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
            # # # img[img>200]=255   
            # contours,hierarchy = cv2.findContours(img, 1, 2)
            img,key_point_xys=find_ball_per_frame(img,)
            if len(key_point_xys) >0:
                new_test=tracker.match_keypoints(key_point_xys,i,rgb)
            # for cnt in contours:
            #     x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
            #     if w>10 and h >10:
            #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     # rect = cv2.minAreaRect(cnt)
            #     # box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            #     # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
                if new_test is not None:
                    for data in new_test:
                        x,y,size,idx=data
                        # if img[y,x,0]>0:
                        cv2.circle(rgb, (x,y), size, (0,255,255), -1) 
            if key_point_xys is not None:
                for data in key_point_xys:
                    x,y,size=data
                    # if img[y,x,0]>0:
                    cv2.circle(rgb, (x,y), size//2, (0,255,0), -1) 
            # fgmask =cv2.threshold(img.copy(),100,255,cv2.THRESH_BINARY)[1]
            # dilated=cv2.dilate(fgmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
            # cv2.imwrite(rf"Test\image\{i:05}.png",fgmask)
            # out=give_box(fgmask,frame)
            timer.toc()
            # img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
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

    
    fps=60
    ball_found=False
    if len(candidates)==0:
        candidates=[key_points]
        return 
    
    for start_idx in range(len(candidates)-1):
        for end_idx in range(start_idx,len(candidates)):
            two_point_length=calculate_length(candidates[start_idx][:-1],candidates[end_idx][:-1])
            if (end_idx-start_idx)*150<two_point_length:
                cal_match_path(candidates_path)
        
    candidates.append(key_points)
   
    
def find_ball_per_frame(img):
    # img = cv2.medianBlur(img,5)
    # img=cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
    # 
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 45
    params.maxThreshold = 95
    # params.blobColor = 254
    params.minArea = 75
    params.maxArea = 600
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.minConvexity = 0.25
    params.minCircularity =.35
    params.maxCircularity = 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
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


    # print(len(key_point))
    
    return img,key_point
if __name__ == "__main__":
   
    knn_video(rf"Test\D1\Cz03_252_2023-10-14_13-08-49-295.mkv")
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