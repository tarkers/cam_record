import cv2
from threading import Thread
from itertools import count
from queue import Queue
import numpy as np
import os
import json
import torch
import torch.multiprocessing as mp
import time
## our folder##
from libs.pose2D.models import builder

class WriterQueue:
    """
    generate pose queue
    """

    def __init__(self, cfg=None,queueSize=500):
        self.cfg = cfg
        self.all_results=[]
        self.pause_stream=False
        # initialize the queue used to store data
        """
        pose_queue: the buffer storing  cropped human image for pose estimation
        
        """
        if cfg.sp:
            self._stopped = False
            self.queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value("b", False)
            self.queue = mp.Queue(maxsize=queueSize)

        
        
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
        self.write_json_worker = self.start_worker(self.write_json)
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

    def write_json(self):
        all_results=[]
        while True:
            if self.stopped:
                break
            if self.queue.empty():
                continue
            
            if not self.queue.full() and not self.pause_stream: 
                # inps=self.wait_and_get(self.queue)
                data=self.wait_and_get(self.queue)
                all_results.append(data)
                
        self.write_pose_json(all_results)
        

    

        
    def write_pose_json(self,all_results,for_eval=True):
        json_results = []
        for im_res in all_results:
            im_name = im_res['imgname']
            for human in im_res['result']:
                keypoints = []
                result = {}
                if for_eval:
                    result['image_id'] = int(os.path.basename(im_name).split('.')[0].split('_')[-1])
                else:
                    result['image_id'] = os.path.basename(im_name)
                result['category_id'] = 1

                kp_preds = human['keypoints']
                kp_scores = human['kp_score']
                pro_scores = human['proposal_score']
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                result['keypoints'] = keypoints
                result['score'] = float(pro_scores)
                if 'box' in human.keys():
                    result['box'] = human['box']
                    
                json_results.append(result)
                    
        with open("test.json", 'w') as json_file:
            json_file.write(json.dumps(json_results))
        self.stop()
    

    @property
    def stopped(self):
        if self.cfg.sp:
            return self._stopped
        else:
            return self._stopped.value

