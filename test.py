import sys
from tqdm import tqdm
import torch
import cv2 
import time
import json
import os
## folder ##
from util import update_config
from util.webcam_queue import webCamDetectQueue
from util.pose_queue import Pose2DQueue
from libs.detector.apis import get_detector
from util import Timer
from libs.vis import plot_boxes

def loop():
    n = 0
    while True:
        yield n
        n += 1
        

        # print(json_results)
        
if __name__ == "__main__":
    cfg = update_config(r"D:\Chen\transform_bag\libs\configs\configs.yaml")
    pose_cfg = update_config(r"D:\Chen\transform_bag\libs\configs\configs.yaml").Pose2D
    pose_cfg.mode="webcam"
    pose_cfg.device="0"
    pose_cfg.gpus=[0]
    pose_cfg.min_box_area=0
    # cfg.enable_tracking = False
    # cfg.detect_classes=[0]
    test = update_config(r"D:\Chen\transform_bag\libs\configs\configs.yaml").Detector
    test.gpus = "0"
    test.gpus = (
        [int(i) for i in test.gpus.split(",")]
        if torch.cuda.device_count() >= 1
        else [-1]
    )
    test.device="0"
    # test.device = 
    test.detbatch = test.detbatch * len(test.gpus)
    # test.posebatch = test.posebatch * len(test.gpus)
    test.tracking = None
    # test.tracking = test.pose_track or test.pose_flow or test.detector=='tracker'

    print(test)
    q = webCamDetectQueue(0, test, get_detector(test))
    poseq=Pose2DQueue(pose_cfg)
    # det_worker=q.start()
    print("Starting webcam demo, press Ctrl + C to terminate...")
    sys.stdout.flush()
    im_names_desc = loop()
    timer = Timer()
    q.start()
    
    yn=q.yolo_classes
    time.sleep(5)
    poseq.start()
    time.sleep(5)
    
    ## test  write  datas
    tmp=[]
    
    for i in im_names_desc:
        if q._stopped:
            print("exiting...")
            exit()

        start_time = timer.tic()
        
        with torch.no_grad():
            item= q.read()
            
            poseq.box_put(item)
            (
                inps,
                orig_img,   
                im_name,
                object_ids,
                boxes,
                scores,
                ids,
                cropped_boxes,
            ) =item
            
            # pose2D=poseq.pose_read()
            
            
            # if inps is None:
                
            #     continue
            # poseq.pose_estimation(inps, boxes,scores,im_name, ids,cropped_boxes)
            # if inps is not None:
            #     poseq.test_pose(inps)
            
            # test=inps[0]
            # test=test.permute(1, 2, 0)
            # test =test.cpu().numpy()
            # import numpy as np
            # np.save("test.npy",inps.cpu().numpy())
            # print(test.shape)
            
            # test=cv2.normalize(test,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
            # print(test.shape)
            # q.terminate()
            # poseq.terminate()
            # exit()
            labels=[]
            if cropped_boxes is not None :
                for o_ids in object_ids:
                    labels.append(yn[int(o_ids)])
                new_iamge=plot_boxes(orig_img,cropped_boxes,(0,0,255),labels)
                cv2.imshow("test",new_iamge)
                # cv2.imshow("test",cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
                # cv2.imwrite("test.jpg",cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(50)
                if key == ord('q') or key == 27: # Esc
                    q.terminate()
                    poseq.terminate()
                    cv2.destroyAllWindows()
                    break
            else:
                cv2.imshow("test",cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(50)
                if key == ord('q') or key == 27: # Esc
                    q.terminate()
                    # poseq.terminate()
                    cv2.destroyAllWindows()
                    break
            # print(scores)
        if i > 80:
            q.terminate()
            poseq.terminate()
            cv2.destroyAllWindows()
    
    # if len(tmp):
    #     with open('data.json', 'w') as f:
    #         json.dump(data, f)
    # x.join()
    # y.join()