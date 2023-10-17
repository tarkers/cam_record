import sys
from tqdm import tqdm
import torch
import cv2 
import time
import json
import os

## folder ##
from queue import Queue
import numpy as np

import torch
import torch.multiprocessing as mp


from tracker.mc_bot_sort import BoTSORT
from util import update_config
from util.webcam_queue import webCamDetectQueue
from util.pose_queue import Pose2DQueue
from util.writer_queue import WriterDQueue
from libs.detector.apis import get_detector
from util import Timer
from libs.vis import plot_boxes,plot_2d_skeleton



        
if __name__ == "__main__":

    cfg = update_config(r"libs\configs\configs.yaml")
    pose_cfg = update_config(r"libs\configs\configs.yaml").Pose2D
    pose3d_cfg = update_config(r"libs\configs\configs.yaml").Pose3D
    tracker_cfg = update_config(r"libs\configs\configs.yaml").TRACKER
    writer_cfg = update_config(r"libs\configs\configs.yaml").Writer
    
    pose_cfg.mode="webcam"
    pose_cfg.device="0"
    pose_cfg.gpus=[0]
    pose_cfg.min_box_area=0
    # cfg.tracking = False
    # cfg.detect_classes=[0]
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
    # test.posebatch = test.posebatch * len(test.gpus)
    # test.tracking = test.pose_track or test.pose_flow or test.detector=='tracker'
    ## test code ##
    cfg.sp=False
    
    # Create tracker
    tracker = BoTSORT(tracker_cfg, frame_rate=60.0)
    # video=rf'test_video.mp4'
    video=rf'Test\Cz03_715_2023-10-14_19-35-05-578.mkv'
    q = webCamDetectQueue(video, test, get_detector(test),tracker)
    video_info=q.videoinfo
    video_info['name']=os.path.basename(video)

    
    writer=WriterDQueue(writer_cfg,video_info,"Test",True)
    print("Starting webcam demo, press Ctrl + C to terminate...")
    sys.stdout.flush()

    if writer_cfg.write_skeleton:
        if cfg.sp:
            _stopped = False
            box_queue = Queue(maxsize=150)
            pose_queue = Queue(maxsize=150)
        else:
            _stopped = mp.Value("b", False)
            box_queue = mp.Queue(maxsize=150)
            pose_queue = mp.Queue(maxsize=150)
        
        poseq=Pose2DQueue(pose_cfg)
        q_process=q.start(box_queue)
        time.sleep(2)
        pose_process=poseq.start(box_queue,pose_queue)
        time.sleep(2)
        writer_process=writer.start(pose_queue)

        q_process[0].join()
        pose_process[0].join()
        writer_process[0].join()
        print("finish 2d estimate")
    else:
        if cfg.sp:
            _stopped = False
            box_queue = Queue(maxsize=150)
        else:
            _stopped = mp.Value("b", False)
            box_queue = mp.Queue(maxsize=150)
        q_process=q.start(box_queue)
        time.sleep(2)
        writer_process=writer.start(box_queue)
        q_process[0].join()
        writer_process[0].join()
        print("finish bbox  estimate")

    
    
    ##################################
    # for i in im_names_desc:
    #     start_time = timer.tic()
        
    #     with torch.no_grad():

    #         item= q.read()  # read bbox estimation
    #         poseq.box_put(item)
    #         (
    #             inps,
    #             orig_img,   
    #             im_name,
    #             class_ids,
    #             boxes,
    #             scores,
    #             ids,
    #             cropped_boxes,
    #         ) =item
            
    #         items=poseq.pose_read()
    #         if items is not None:
    #             (pose2D,t_orig_img)=items
    #         else:
    #             pose2D=None
    #             t_orig_img=None
    #         ## test run 3D POSE###
    #         # writer.read(pose2D)
            

    #         if pose2D is not None :
    #             new_iamge=plot_2d_skeleton(cv2.cvtColor(t_orig_img,cv2.COLOR_BGR2RGB),pose2D)
    #             # new_iamge=plot_boxes(orig_img,cropped_boxes,(0,0,255),labels)
    #             cv2.imshow("skeleton",new_iamge)
    #             # cv2.imshow("test",cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
    #             # cv2.imwrite("test.jpg",cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
    #             key = cv2.waitKey(50)
    #             if key == ord('q') or key == 27: # Esc
    #                 q.terminate()
    #                 # poseq.terminate()
    #                 # writer.terminate()
    #                 cv2.destroyAllWindows()
    #                 # break
    #         elif t_orig_img is not None:
    #             cv2.imshow("skeleton",cv2.cvtColor(t_orig_img,cv2.COLOR_BGR2RGB) )
    #             # cv2.imshow("test",cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
    #             # cv2.imwrite("test.jpg",cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
    #             key = cv2.waitKey(50)
    #             if key == ord('q') or key == 27: # Esc
    #                 q.terminate()
    #                 # poseq.terminate()
    #                 # writer.terminate()
    #                 cv2.destroyAllWindows()
    #                 # break
    #         if cropped_boxes is not None:
    #             new_iamge=plot_boxes(orig_img,cropped_boxes,ids,None,(0,0,255))
    #             cv2.imshow("box",cv2.cvtColor(new_iamge,cv2.COLOR_BGR2RGB))
    #             key = cv2.waitKey(30)
    #             if key == ord('q') or key == 27: # Esc
    #                 q.terminate()
    #                 # poseq.terminate()
    #                 # writer.terminate()
    #                 cv2.destroyAllWindows()
    #                 # break
    #         else:
    #             cv2.imshow("box",cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB))
    #             key = cv2.waitKey(30)
    #             if key == ord('q') or key == 27: # Esc
    #                 q.terminate()
    #                 # poseq.terminate()
    #                 # writer.terminate()
    #                 cv2.destroyAllWindows()
    #                 # break
    #         # print(scores)
    #     if i > 30:
    #         is_finish=True
    #         print("out of bound!")
    #         q.terminate()
    #         poseq.terminate()
    #         # writer.terminate()
    #         cv2.destroyAllWindows()
    #         break
    # if not is_finish:
    #     print("stop thread")
    #     q.terminate()
    #     poseq.terminate()
    #     # writer.terminate()
    #     cv2.destroyAllWindows()
        
    ###############################
    # q_thread[0].terminate()
    # q_thread[0].join()
    # pose_thread[0].terminate()
    # pose_thread[0].join()
    # writer_thread[0].terminate()
    # writer_thread[0].join()
    # if len(tmp):
    #     with open('data.json', 'w') as f:
    #         json.dump(data, f)
    # x.join()
    # y.join()