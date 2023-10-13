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
import pathlib

# # our folder##
from Pose3D.lib.utils.tools import *
from Pose3D.lib.utils.learning import load_backbone
from Pose3D.lib.utils.utils_data import flip_data
from libs.vis import plot_boxes, plot_2d_skeleton
from util import Timer

class WriterDQueue:
    """
    generate writer queue
    """

    def __init__(self, cfg=None, video_info=None, store_path='./', store_video=True, queueSize=500):
        self.cfg = cfg
        self.all_results = []
        self.pause_stream = False
        self.store_video = store_video
        self.video_info = video_info
        self.video_base_name = {
            'base':self.video_info['name'].rsplit('.')[0],
            'ext':self.video_info['name'].rsplit('.')[1]
        }
        store_folder = rf"{store_path}\{self.video_base_name['base']}"
        
        if self.cfg.write_skeleton:
            self.folder_path = rf"{store_folder}\subject_2D"
        else:
            self.folder_path = rf"{store_folder}\bbox"
            
        # create data folder
        p = pathlib.Path(self.folder_path)
        p.mkdir(parents=True, exist_ok=True)  

        if cfg.sp:
            self._stopped = False
            self.queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value("b", False)
            self.queue = mp.Queue(maxsize=queueSize)

        
    def write_2D_id_pose_json(self, for_eval=True):
       
        focus_ids = {}
        # get results json 
        f = open(rf"{self.folder_path}\results.json")
        results = json.load(f)
        
        for item in results:
            new_data = {"image_id":item['image_id'],
                                        'keypoints':item['keypoints'],
                                        'box':item['box'],
                                        'score':item['score']}
            if item['idx'] not in focus_ids:
                focus_ids[item['idx']] = [new_data]
            else:
                focus_ids[item['idx']].append(new_data)
        
        # create data folder
        id_path=self.folder_path+"\ids"
        p = pathlib.Path(id_path)
        p.mkdir(parents=True, exist_ok=True)
        # save json by id    
        for _id in focus_ids:
            test = [ t['image_id'] for t in  focus_ids[_id]]
            results = {
                'results':focus_ids[_id],
                'frame_list':test
            }
            with open(rf"{id_path}\{_id}.json", 'w') as json_file:
                json_file.write(json.dumps(results))
                    
    def write_2D_pose_json(self, all_results, for_eval=True):
        '''
        write who ids in a result files
        '''
        json_results = []
        tracklet = {'tracklet':[]}
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
                
                if 'idx'  in human:
                    if isinstance(human['idx'],list) or isinstance(human['idx'],np.ndarray):
                        human['idx'].sort()
                        result['idx']=human['idx'][0]
                    else:
                        try:
                            result['idx'] = int(human['idx'])
                        except TypeError:
                            print(type(human['idx']))
                else:
                    result['idx'] = None        

                result['class_id'] = int(human['class_id'])
                result['score'] = float(pro_scores)
                if 'box' in human.keys():
                    result['box'] = human['box']
                    
                json_results.append(result)
 
        with open(rf"{self.folder_path}\results.json", 'w') as json_file:
            json_file.write(json.dumps(json_results))
            
        with open(rf"{self.folder_path}\tracklet.json", 'w') as json_file:
            json_file.write(json.dumps(tracklet))
        self.stop()            
    
    def write_bbox_json(self, data):
        for item in data:
            tmp_bbox = []
            tmp_cropped = []
            boxes = item['boxes']
            cropped_boxes = item['cropped_boxes']
            for i in range(len(boxes)):
                tmp_bbox.append(boxes[i].tolist())
                tmp_cropped.append(cropped_boxes[i].tolist())
            if not self.cfg.write_tracking:
                del item['ids']
            else:
                item['ids'] = item['ids'].tolist()
            item['boxes'] = tmp_bbox
            item['cropped_boxes'] = tmp_cropped
            item['scores'] = torch.flatten(item['scores']).tolist()
            item['class_ids'] = torch.flatten(item['class_ids']).tolist()
            item['boxes'] = tmp_bbox

        # # create data folder
        p = pathlib.Path(self.folder_path)
        p.mkdir(parents=True, exist_ok=True)   
        print(rf"Save bbox json to :{self.folder_path}\bbox_results.json")
        with open(rf"{self.folder_path}\bbox_results.json", 'w') as json_file:
            json_file.write(json.dumps(data))
        pass
        
    def start_worker(self, target, data_queue):
       
        if self.cfg.sp:
            p = Thread(target=target, args=(data_queue,))
        else:
            p = mp.Process(target=target, args=(data_queue,))
        # p.daemon = True
        p.start()
        return p

    def start(self, data_queue):
        """
        start a thread to  process  pose estimation
        """
        if self.cfg.write_skeleton:
            self.writer_worker = self.start_worker(self.write_skeletons_data, data_queue)
        else:
            self.writer_worker = self.start_worker(self.write_bbox_data, data_queue)
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

    def read(self, item):
        self.wait_and_put(self.queue, item)
        
    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()
    
    def set_up_video(self):
        fourcc, ext = self.recognize_video_ext(self.video_base_name["ext"])
        if self.video_info is not None:
            video_base = rf'{self.folder_path}/b_{self.video_base_name["base"]}'
            if self.cfg.write_skeleton:
                video_base = rf'{self.folder_path}/s_{self.video_base_name["base"]}'   
            if self.cfg.write_tracking:
                video_base = video_base + "_i"
            video_path = video_base + ext
                
        video_writer = cv2.VideoWriter(video_path, fourcc, self.video_info['fps'], self.video_info['frameSize'])
        print("Video Stored: ", video_path)
        return video_writer
    
    def write_bbox_data(self, data_queue):
        if self.store_video:
            video_writer = self.set_up_video()
        ## generate vis video window
        if self.cfg.vis_video:
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL) 
        data_store = []
        while True:
            if self.stopped:
                break
            if data_queue.empty():
                continue
            
            if not self.pause_stream: 
                # inps=self.wait_and_get(self.queue)
                item = data_queue.get()
                if item is  None:
                        continue
                if item[0] == None:  # bbox queue is empty
                    print("bbox queue is finish")
                    break
                (
                    _,  # inps
                    orig_img,
                    im_name,
                    class_ids,
                    bbox,  # bbox
                    scores,
                    ids,
                    cropped_boxes,  # cropped boxeds for pose estimation
                ) = item
                data_store.append({
                'image_id':im_name,
                'boxes':bbox,
                'cropped_boxes':cropped_boxes,
                'scores':scores,
                'class_ids':class_ids,
                'ids':ids })
                print("BOX GET:",im_name,flush=True)
                if self.store_video: 
                    
                        # writ to video frame
                    if self.cfg.write_tracking:
                        img = plot_boxes(orig_img, bbox, ids, None)
                        
                    else:
                        img = plot_boxes(orig_img, bbox, None, None)
                    
                    video_writer.write(img)
                    if self.cfg.vis_video:
                        cv2.imshow("Video",img)
                        cv2.waitKey(1)
                          
        if self.store_video:
            video_writer.release()        

        print("start writing bbox json data")
        if self.cfg.write_box:
            self.write_bbox_json(data_store)
            # if self.cfg.write_tracking:
            #     self.write_bbox_ids_json()
    
    def write_skeletons_data(self, data_queue):
        if self.store_video:
            video_writer = self.set_up_video()
        ## generate vis video window
        if self.cfg.vis_video:
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL) 
            
        data_store = []
        timer=Timer()   
        
        while True:
            if self.stopped:
                break
            if data_queue.empty():
                continue
            
            if  not self.pause_stream: 
                # inps=self.wait_and_get(self.queue)
                item = data_queue.get()
                if item is not None:
                    (orig_img, result) = item
                    if orig_img is None and result is None:  # queue reach the end
                        print("pose queue reach end!")
                        break
                    else:
                        data_store.append(result)
                        timer.toc()
                        meta=rf"POSE GET:{result['imgname']},FPS: {timer.fps}"
                        print(meta,flush=True)
                       

                        # write to video frame
                        if self.store_video:
                            img = plot_2d_skeleton(orig_img, result, self.cfg.write_box, self.cfg.write_tracking)
                            video_writer.write(img)
                            if self.cfg.vis_video:
                                cv2.putText(img, meta, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                cv2.imshow("Video",img)
                                cv2.waitKey(1)
                            # print("frame count:", result['imgname'])  
                        timer.tic()
        if self.store_video:
            video_writer.release()        

        print("start writing pose json data")
        if self.cfg.write_skeleton:
            self.write_2D_pose_json(data_store)
            if self.cfg.write_tracking:
                self.write_2D_id_pose_json()

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'
    
    @property
    def stopped(self):
        if self.cfg.sp:
            return self._stopped
        else:
            return self._stopped.value

