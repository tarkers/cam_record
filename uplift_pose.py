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
from util import webCamDetectQueue,Pose2DQueue,WriterDQueue

from libs.detector.apis import get_detector
from util import Timer



        
if __name__ == "__main__":

    cfg = update_config(r"libs\configs\configs.yaml")
    pose_cfg = cfg.Pose2D
    pose3d_cfg = cfg.Pose3D
    tracker_cfg = cfg.TRACKER
    writer_cfg = cfg.Writer
    
    pose_cfg.mode="webcam"
    pose_cfg.device="0"
    pose_cfg.gpus=[0]
    pose_cfg.min_box_area=0
    test = cfg.Detector
    
    test.gpus = "0"
    test.device="0"
    # test.device = 
    test.detbatch = test.detbatch * len(test.gpus)
    # test.posebatch = test.posebatch * len(test.gpus)
    # test.tracking = test.pose_track or test.pose_flow or test.detector=='tracker'
    ## test code ##
    cfg.sp=False
    
    # Create tracker
    tracker = BoTSORT(tracker_cfg, frame_rate=100.0)
    # video=rf'test_video.mp4'
    video=rf'Data\N0002\N0002_WALK_L01\Calibrate_Video\N0002_WALK_L01.avi'
    q = webCamDetectQueue(video, test, get_detector(test),tracker)
    video_info=q.videoinfo
    video_info['name']=os.path.basename(video)

    
    writer=WriterDQueue(writer_cfg,video_info,"Test",True)
    print("Starting webcam demo, press Ctrl + C to terminate...")
    sys.stdout.flush()

    if writer_cfg.write_skeleton:
        if cfg.sp:
            # _stopped = False
            box_queue = Queue(maxsize=150)
            pose_queue = Queue(maxsize=150)
        else:
            # _stopped = mp.Value("b", False)
            box_queue = mp.Queue(maxsize=150)
            pose_queue = mp.Queue(maxsize=150)
        
        poseq=Pose2DQueue(pose_cfg)
        q_process=q.start(box_queue)
        time.sleep(3)
        pose_process=poseq.start(box_queue,pose_queue)
        time.sleep(3)
        writer_process=writer.start(pose_queue)

        q_process[0].join()
        pose_process[0].join()
        writer_process[0].join()
        print("finish 2d estimate")
    else:
        if cfg.sp:
            # _stopped = False
            box_queue = Queue(maxsize=150)
        else:
            # _stopped = mp.Value("b", False)
            box_queue = mp.Queue(maxsize=150)
        q_process=q.start(box_queue)
        time.sleep(2)
        writer_process=writer.start(box_queue)
        q_process[0].join()
        writer_process[0].join()
        print("finish bbox  estimate")

