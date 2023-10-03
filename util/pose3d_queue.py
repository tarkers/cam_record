import cv2
from threading import Thread
from itertools import count
from queue import Queue
import numpy as np
import os
import json
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
## our folder##
from libs.Pose3D.lib.utils.tools import *
from libs.Pose3D.lib.utils.learning import load_backbone
from libs.Pose3D.lib.utils.utils_data import flip_data
from libs.Pose3D.lib.data.dataset_wild import PoseDataset
class Pose3DQueue:
    """
    generate pose queue
    """

    def __init__(self, cfg=None,queueSize=500):
        self.cfg = cfg
        self.all_results=[]
        self.pause_stream=False
        
        
        if cfg.sp:
            self._stopped = False
            self.queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value("b", False)
            self.queue = mp.Queue(maxsize=queueSize)

        self.load_model()
    
    def load_model(self):
        model_backbone = load_backbone(self.cfg)
   
        if torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()

        print('Loading checkpoint', self.cfg.evaluate)
        checkpoint = torch.load(self.cfg.evaluate, map_location=lambda storage, loc: storage)
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        self.model = model_backbone
        self.model.eval()
        self.testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 2,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
        }
        
        
        
    def start_worker(self, target):
       
        if self.cfg.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        """
        start a thread to  process  pose estimation
        """
        self.write_json_worker = self.start_worker(self.joints_process)
        return [ self.write_json_worker ]

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.write_json_worker.join()

    def stop(self):
        # dump datas queues
        self.clear_queues()

    def terminate(self):
        if self.cfg.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        print("write termination")
        

    def clear_queues(self):
        self.clear(self.queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def read(self,  item):
        self.wait_and_put(self.queue, item)
        
    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()

    def joints_process(self):
        
        
        pose_queue = []
        while True:
            if self.stopped:
                break
            if self.queue.empty():
                continue
            
            if not self.queue.full() and not self.pause_stream: 
                # inps=self.wait_and_get(self.queue)
                data=self.wait_and_get(self.queue)
                # all_results.append(data)
                
                # wild_dataset = PoseDataset(json, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)
                # test_loader = DataLoader(wild_dataset, **testloader_params)
        
                self.run_3D_pose()
    
    def extract_specific_id(self):
        pass
        
    def run_3D_pose(self,test_loader,for_eval=True):
        wild_dataset = PoseDataset(json, clip_len=self.cfg.clip_len, scale_range=[1,1], focus=opts.focus)
        test_loader = DataLoader(wild_dataset, **self.testloader_params)
        results_all=[]
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                N, T = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                if self.cfg.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if self.cfg.flip:    
                    batch_input_flip = flip_data(batch_input)
                    predicted_3d_pos_1 = self.model(batch_input)
                    predicted_3d_pos_flip = self.model(batch_input_flip)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
                else:
                    predicted_3d_pos = self.model(batch_input)
                if self.cfg.rootrel:
                    predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
                else:
                    predicted_3d_pos[:,0,0,2]=0
                    pass
                if self.cfg.gt_2d:
                    predicted_3d_pos[...,:2] = batch_input[...,:2]
                results_all.append(predicted_3d_pos.cpu().numpy())
        results_all = np.hstack(results_all)
        results_all = np.concatenate(results_all)     
        print(results_all.shape)  
    @property
    def stopped(self):
        if self.cfg.sp:
            return self._stopped
        else:
            return self._stopped.value

