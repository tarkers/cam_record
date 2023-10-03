import cv2
from threading import Thread
from itertools import count
from queue import Queue
import numpy as np

import torch
import torch.multiprocessing as mp
import time
## our folder##

from libs.pose2D.models import builder
from libs.transforms import flip
from libs.pose2D.utils.pPose_nms import pose_nms
from libs.pose2D.utils.transforms import get_func_heatmap_to_coord

EVAL_JOINTS = range(17)

class Pose2DQueue:
    """
    generate pose queue
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.all_results=[]
        self.pause_stream = False
        self.eval_joints = EVAL_JOINTS
        self.queueSize = 150 if self.cfg.mode == "webcam" else self.cfg.qsize
        self.pose_queueSize = 243
        self.batchSize = self.cfg.posebatch
        self.norm_type = None   #for evaluate set None
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        if self.cfg.flip:
            self.batchSize = int(self.batchSize / 2)

        # self._sigma = cfg.DATA_PRESET.SIGMA
        self.device='cuda' if self.cfg.device!="cpu" else "cpu"
        self.model=None
        # initialize the queue used to store data
        """
        pose_queue: the buffer storing  cropped human image for pose estimation
        
        """
        if cfg.sp:
            self._stopped = False
            self.box_queue = Queue(maxsize=self.queueSize)
            self.pose_queue = Queue(maxsize=self.pose_queueSize)
        else:
            self._stopped = mp.Value("b", False)
            self.box_queue = mp.Queue(maxsize=self.queueSize)
            self.pose_queue = mp.Queue(maxsize=self.pose_queueSize)

    def load_model(self):
        self.model = builder.build_sppe(
            self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET
        )
        print("Loading pose model from %s..." % (self.cfg.checkpoint,))
        self.model.load_state_dict(
            torch.load(self.cfg.checkpoint, map_location=self.device)
        )
        # pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        # if self.cfg.pose_track:       #init pose track
        #     tracker = Tracker(tcfg, args)
        if len(self.cfg.gpus) > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.cfg.gpus
            ).to(self.device)
        else:
            self.model.to(self.device)
        self.model.eval()
        
        
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
        image_preprocess_worker = self.start_worker(self.image_process)
        return [image_preprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        
        if self.cfg.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        print("pose termination")
        self.stop()

    def clear_queues(self):
        self.clear(self.box_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def box_put(self,item):
        self.wait_and_put(self.box_queue,item)
    
    def pose_read(self):
        return self.wait_and_get(self.pose_queue)
          
    
    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped and not queue.empty():
            return queue.get()

    def image_process(self):
        if not self.model:
            self.load_model()
        
        # keep looping infinitely
        with torch.no_grad():
            while True:
                if self.stopped:
                    print("pose queue is stopped!!")
                    return
                if self.box_queue.empty():
                    continue
                
                if not self.box_queue.full() and not self.pause_stream: 
                    
                    # inps=self.wait_and_get(self.box_queue)
                    item=self.wait_and_get(self.box_queue)
                    if item[0] is not None:
                        self.pose_estimation(item)
                    else:
                        self.wait_and_put(self.pose_queue,(None,item[1]))
                    # (
                    #     inps,
                    #     orig_img,   
                    #     im_name,
                    #     object_ids,
                    #     boxes,
                    #     scores,
                    #     ids,
                    #     cropped_boxes,
                    # )=self.wait_and_get(self.box_queue)
                    # if inps is not None:
                    #     self.pose_estimation(inps,
                    #     orig_img,   
                    #     im_name,
                    #     object_ids,
                    #     boxes,
                    #     scores,
                    #     ids,
                    #     cropped_boxes,)
                    # else:
                    #     pass
                else:
                    print("pose is full")
    def pose_estimation(self,item):
        (inps,orig_img, im_name,object_ids,
                        boxes,
                        scores,
                        ids,
                        cropped_boxes)=item
        inps = inps.to(self.device)
       
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % self.batchSize:
            leftover = 1
        num_batches = datalen // self.batchSize + leftover
        hm = []
        

        for j in range(num_batches):
            inps_j = inps[j * self.batchSize : min((j + 1) * self.batchSize, datalen)]
            # if self.cfg.flip:
            #     inps_j = torch.cat((inps_j, flip(inps_j)))
            # pose model execute

            hm_j = self.model(inps_j)

            # if self.cfg.flip:
            #     hm_j_flip = flip_heatmap(
            #         hm_j[int(len(hm_j) / 2) :], self.joint_pairs, shift=True
            #     )
            #     hm_j = (hm_j[0 : int(len(hm_j) / 2)] + hm_j_flip) / 2
            
            hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()
        
        
        assert hm.dim() == 4

        ## check keypoints ##
        face_hand_num = 110
        if hm.size()[1] == 136:
            self.eval_joints = [*range(0,136)]
        elif hm.size()[1] == 26:
            self.eval_joints = [*range(0,26)]
        elif hm.size()[1] == 133:
            self.eval_joints = [*range(0,133)]
        elif hm.size()[1] == 68:
            face_hand_num = 42
            self.eval_joints = [*range(0,68)]
        elif hm.size()[1] == 21:
            self.eval_joints = [*range(0,21)]
        

        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            if isinstance(self.heatmap_to_coord, list):
                pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                    hm[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
                pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                    hm[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self.heatmap_to_coord(hm[i][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            pose_nms(boxes, scores, ids, preds_img, preds_scores, self.cfg.min_box_area, use_heatmap_loss=self.use_heatmap_loss)
        
        
        ## perpare resultes
        _result = []
        for k in range(len(scores)):
            _result.append(
                {
                    'keypoints':preds_img[k],
                    'kp_score':preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                    'idx':ids[k],
                    'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                }
            )

        result = {
            'imgname': im_name,
            'result': _result
        }
        ### this is for tracking ####
        # if self.opt.pose_flow:
        #     poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
        #     for i in range(len(poseflow_result)):
        #         result['result'][i]['idx'] = poseflow_result[i]['idx']
        ######
        
        #put result in pose_queue
        self.wait_and_put(self.pose_queue,(result,orig_img))
        
        return

    

        
    # def write_temp_files(self, inputs):
    #     '''
    #     writer save to temp_files 
    #     '''
    #     _result = []
    #     for k in range(len(scores)):
    #         _result.append(
    #             {
    #                 'keypoints':preds_img[k],
    #                 'kp_score':preds_scores[k],
    #                 'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
    #                 'idx':ids[k],
    #                 'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
    #             }
    #         )

    #     result = {
    #         'imgname': im_name,
    #         'result': _result
    #     }
    
    

    @property
    def stopped(self):
        if self.cfg.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    @property
    def joint_26_pairs(self):
        return [
            (15, 12),
            (12, 9),
            (9, 13),
            (13, 16),
            (16, 18),
            (18, 20),
            (20, 22),
            (9, 14),
            (14, 17),
            (17, 19),
            (19, 21),
            (21, 23),
            (9, 6),
            (6, 3),
            (3, 0),
            (0, 1),
            (1, 4),
            (4, 7),
            (7, 10),
            (0, 2),
            (2, 5),
            (5, 8),
            (8, 11),
        ]
