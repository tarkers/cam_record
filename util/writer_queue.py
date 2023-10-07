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
from Pose3D.lib.utils.tools import *
from Pose3D.lib.utils.learning import load_backbone
from Pose3D.lib.utils.utils_data import flip_data

class WriterDQueue:
    """
    generate writer queue
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
        
    def write_2D_id_pose_json(self,results,folder_path,for_eval=True):
        focus_ids={}
        for item in results:
            new_data={"image_id":item['image_id'],
                                        'keypoints':item['keypoints'],
                                        'box':item['box'],
                                        'score':item['score']
                                        }
            if item['idx'] not in focus_ids:
                focus_ids[item['idx']] =[new_data]
            else:
                focus_ids[item['idx']].append(new_data)
        #save json by id    
        for id in focus_ids:
            with open(rf"{folder_path}\subject_2D\{id}.json", 'w') as json_file:
                json_file.write(json.dumps(focus_ids[id]))
                
    def write_2D_pose_json(self,all_results,for_eval=True):
        '''
        write who ids in a result files
        '''
        json_results = []
        tracklet={'tracklet':[]}
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
                try:
                    result['idx'] = int(human['idx'])
                except TypeError:
                    result['idx'] = int(human['idx'][0])
                    tracklet['tracklet'].append([int(x) for x in human['idx']])              

                result['class_id'] = int(human['class_id'])
                result['score'] = float(pro_scores)
                if 'box' in human.keys():
                    result['box'] = human['box']
                    
                json_results.append(result)
                    
        with open("results.json", 'w') as json_file:
            json_file.write(json.dumps(json_results))
            
        with open("tracklet.json", 'w') as json_file:
            json_file.write(json.dumps(tracklet))
        self.stop()            
        
    
    
        
        
        
    def start_worker(self, target,pose_queue):
       
        if self.cfg.sp:
            p = Thread(target=target, args=(pose_queue,))
        else:
            p = mp.Process(target=target, args=(pose_queue,))
        # p.daemon = True
        p.start()
        return p

    def start(self,pose_queue):
        """
        start a thread to  process  pose estimation
        """
        self.writer_worker = self.start_worker(self.joints_process,pose_queue)
        return [ self.writer_worker ]

    def stop(self):
        # indicate that the thread should be stopped
        # self.save(None, None, None, None, None, None, None)
        # self.writer_worker.join()
        pass


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

    def joints_process(self,pose_queue):

        data_store=[]
        while True:
            if self.stopped:
                break
            if pose_queue.empty():
                continue
            
            if not pose_queue.full() and not self.pause_stream: 
                # inps=self.wait_and_get(self.queue)
                item=pose_queue.get()
                if item is not None:
                    (im_name,result)=item
                    if im_name==None and result==None: #queue reach the end
                        print("pose queue reach end!")
                        break
                    else:
                        data_store.append(result)
                        
                        # print("POSE GET:",item[0],flush=True)

                # if len(pose_queue)>15:
                #     datas=pose_queue.copy()
                #     # print(datas.keys())
                #     pose_queue=[]
                #     self.run_3D_pose(datas)
                # all_results.append(data)
                

        print("start writing json data")
        self.write_2D_id_pose_json(data_store)
        
        
    
        
    def run_3D_pose(self,wild_dataset,for_eval=True):
        # print(wild_dataset)
        return
        # if not wild_dataset:
        #     return
        # wild_dataset = PoseDataset(json, clip_len=self.cfg.clip_len, scale_range=[1,1], focus=opts.focus)
        # test_loader = DataLoader(wild_dataset, **self.testloader_params)
        # results_all=[]
        # with torch.no_grad():
        #     for batch_input in tqdm(test_loader):
        #         N, T = batch_input.shape[:2]
        #         if torch.cuda.is_available():
        #             batch_input = batch_input.cuda()
        #         if self.cfg.no_conf:
        #             batch_input = batch_input[:, :, :, :2]
        #         if self.cfg.flip:    
        #             batch_input_flip = flip_data(batch_input)
        #             predicted_3d_pos_1 = self.model(batch_input)
        #             predicted_3d_pos_flip = self.model(batch_input_flip)
        #             predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
        #             predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        #         else:
        #             predicted_3d_pos = self.model(batch_input)
        #         if self.cfg.rootrel:
        #             predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        #         else:
        #             predicted_3d_pos[:,0,0,2]=0
        #             pass
        #         if self.cfg.gt_2d:
        #             predicted_3d_pos[...,:2] = batch_input[...,:2]
        #         results_all.append(predicted_3d_pos.cpu().numpy())
        # results_all = np.hstack(results_all)
        # results_all = np.concatenate(results_all)     
        # print(results_all.shape)  
    @property
    def stopped(self):
        if self.cfg.sp:
            return self._stopped
        else:
            return self._stopped.value
