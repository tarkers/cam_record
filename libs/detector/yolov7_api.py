# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of yolo detector"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from abc import ABC, abstractmethod
import platform
import argparse
import time
from pathlib import Path
import torch
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from yolo.preprocess import prep_image, prep_frame
from yolo.darknet import Darknet
from yolo.util import unique
from yolo.bbox import bbox_iou
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages,LoadImageTensor
from yolov7.utils.general import check_img_size,xyxy2xywh ,non_max_suppression,scale_coords
# from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from libs.detector.apis import BaseDetector
# from . import  yolov7_cfg  as self.detector_opt
#only windows visual studio 2013 ~2017 support compile c/cuda extensions
#If you force to compile extension on Windows and ensure appropriate visual studio
#is intalled, you can try to use these ext_modules.
if platform.system() != 'Windows':
    from detector.nms import nms_wrapper

class YOLODetector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(YOLODetector, self).__init__()

        self.detector_cfg = cfg
        self.detector_opt = opt
        
        self.model_cfg = opt.CONFIG
        # self.model_weights = opt.get('WEIGHTS', 'detector/yolo/data/yolov3-spp.weights')
        self.inp_dim = opt.INP_DIM
        self.nms_thres = opt.NMS_THRES
        self.confidence = 0.3 if (False if not hasattr(opt, 'tracking') else opt.tracking) else opt.CONFIDENCE
        self.num_classes = opt.NUM_CLASSES
        self.model = None
        self.detector_opt.device = select_device(self.detector_opt.device)
        self.half = self.detector_opt.device != 'cpu'  # half precision only supported on CUDA
    
    def load_model(self):
        args = self.detector_opt
        
        # print('Loading YOLO model..')
        # self.model = Darknet(self.model_cfg)
        # self.model.load_weights(self.model_weights)
        # self.model.net_info['height'] = self.inp_dim
        self.model = attempt_load(self.detector_opt.weights, map_location=self.detector_opt.device)  # load FP32 model
        # self.model = attempt_load(self.detector_opt.weights, map_location=self.detector_opt.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.inp_dim, s=self.stride )  # check img_size
        if self.detector_opt.trace:
           self.model= TracedModel(self.model, self.detector_opt.device, self.inp_dim)

        if self.half:
            self.model.half()  # to FP16

        # classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        # if args:
        #     if len(args.gpus) > 1:
        #         self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus).to(args.device)
        #     else:
        #         self.model.to(args.device)
        # else:
        #     self.model.cuda()
        # self.model.eval()

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        if isinstance(img_source, str):
            img, orig_img, im_dim_list = prep_image(img_source, self.inp_dim)
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(img_source, self.inp_dim)
        else:
            raise IOError('Unknown image source type: {}'.format(type(img_source)))
        if self.model   is None:
            self.load_model()

        return img

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and 
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        """
        global tt
        
        args = self.detector_opt
        
        datasets = LoadImageTensor(imgs, img_size=self.imgsz, stride=self.stride)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()

        if self.detector_opt.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.detector_opt.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        t0 = time.time()
        preds = []
        write=False
        for path, img, im0, vid_cap in datasets:
            img = torch.from_numpy(img).to(self.detector_opt.device)
            
            img = img.half() if self.half else img.float()  # uint8 to fp16/32

          
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            
            # Warmup
            if self.detector_opt.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.detector_opt.augment)[0]
            
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                
                pred = self.model(img, augment=self.detector_opt.augment)[0]
                
                
                # exit()
                if not write:
                    preds=pred
                    write=True
                else:
                    preds = torch.cat((preds, pred), 0)

                '''
                test view detect image
                
                # test = non_max_suppression(pred, self.detector_opt.conf_thres, self.detector_opt.iou_thres, classes=self.detector_opt.classes, agnostic=self.detector_opt.agnostic_nms)
                # for i, det in enumerate(test):
                #     if len(det) and i==0:
                #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
                #         test=det.clone()
                #         test[:, :4] = scale_coords(img.shape[2:], test[:, :4], im0.shape).round()
                        # for *xyxy, conf, cls in reversed(test):
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                            # plot_one_box(xyxy, im0, label="test", color=(0,255,0), line_thickness=1)
                        # cv2.imshow("test", im0)
                        # cv2.waitKey(1)  # 1 millisecond
                '''
        dets = self.dynamic_write_results(preds, self.confidence, 
                                            self.num_classes, nms=True, 
                                            nms_conf=self.nms_thres)
        
        if isinstance(dets, int) or dets.shape[0] == 0:
            return 0
        dets = dets.cpu()
        
        orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)
        dets[:, [1, 3]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 1].view(-1, 1)) / 2
        dets[:, 1:5] /= scaling_factor
        for i in range(dets.shape[0]):
            dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
            dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])
        return dets


    
    def dynamic_write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        prediction_bak = prediction.clone()
        
        dets = self.write_results(prediction.clone(), confidence, num_classes, nms, nms_conf)
        if isinstance(dets, int):
            return dets

        if dets.shape[0] > 100:
            nms_conf -= 0.05
            dets = self.write_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)

        return dets

    def write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        args = self.detector_opt
        #prediction: (batchsize, num of objects, (xc,yc,w,h,box confidence, 80 class scores))
        conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)
        prediction = prediction * conf_mask

        try:
            ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
        except:
            return 0

        #the 3rd channel of prediction: (xc,yc,w,h)->(x1,y1,x2,y2)
        box_a = prediction.new(prediction.shape)
        
        box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
        box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
        box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
        box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
        prediction[:,:,:4] = box_a[:,:,:4]

        batch_size = prediction.size(0)
        
        output = prediction.new(1, prediction.size(2) + 1)
        write = False
        num = 0
        for ind in range(batch_size):
            #select the image from the batch
            image_pred = prediction[ind]

            #Get the class having maximum score, and the index of that class
            #Get rid of num_classes softmax scores 
            #Add the class index and the class score of class having maximum score
            max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:,:5], max_conf, max_conf_score)
            #image_pred:(n,(x1,y1,x2,y2,c,s,idx of cls))
            
            image_pred = torch.cat(seq, 1)

            #Get rid of the zero entries
            non_zero_ind =  (torch.nonzero(image_pred[:,4]))

            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)

            #Get the various classes detected in the image
            try:
                img_classes = unique(image_pred_[:,-1])
            except:
                continue

            #WE will do NMS classwise
            #print(img_classes)
            for cls in img_classes:
                #get the detections with one particular class
                cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()

                image_pred_class = image_pred_[class_mask_ind].view(-1,7)

                #sort the detections such that the entry with the maximum objectness
                #confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)
            
                #if nms has to be done
                if nms:
                    if platform.system() != 'Windows':
                        #We use faster rcnn implementation of nms (soft nms is optional)
                        nms_op = getattr(nms_wrapper, 'nms')
                        #nms_op input:(n,(x1,y1,x2,y2,c))
                        #nms_op output: input[inds,:], inds
                        _, inds = nms_op(image_pred_class[:,:5], nms_conf)

                        image_pred_class = image_pred_class[inds]
                    else:
                        # Perform non-maximum suppression
                        max_detections = []
                        while image_pred_class.size(0):
                            # Get detection with highest confidence and save as max detection
                            max_detections.append(image_pred_class[0].unsqueeze(0))
                            # Stop if we're at the last detection
                            if len(image_pred_class) == 1:
                                break
                            # Get the IOUs for all boxes with lower confidence
                            ious = bbox_iou(max_detections[-1], image_pred_class[1:], args)
                            # Remove detections with IoU >= NMS threshold
                            image_pred_class = image_pred_class[1:][ious < nms_conf]

                        image_pred_class = torch.cat(max_detections).data

                #Concatenate the batch_id of the image to the detection
                #this helps us identify which image does the detection correspond to 
                #We use a linear straucture to hold ALL the detections from the batch
                #the batch_dim is flattened
                #batch is identified by extra batch column

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq,1)
                    write = True
                else:
                    out = torch.cat(seq,1)
                    output = torch.cat((output,out))
                num += 1
    
        if not num:
            return 0
        #output:(n,(batch_ind,x1,y1,x2,y2,c,s,idx of cls))
        
        return output

    def detect_one_img(self, img_name):
        """
        Detect bboxs in one image
        Input: 'str', full path of image
        Output: '[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]',
        The output results are similar with coco results type, except that image_id uses full path str
        instead of coco %012d id for generalization. 
        """
        args = self.detector_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        if not self.model:
            self.load_model()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        dets_results = []
        #pre-process(scale, normalize, ...) the image
        img, orig_img, img_dim_list = prep_image(img_name, self.inp_dim)
        with torch.no_grad():
            img_dim_list = torch.FloatTensor([img_dim_list]).repeat(1, 2)
            img = img.to(args.device) if args else img.cuda()
            prediction = self.model(img, args=args)
            #do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(prediction, self.confidence,
                                              self.num_classes, nms=True,
                                              nms_conf=self.nms_thres)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return None
            dets = dets.cpu()

            img_dim_list = torch.index_select(img_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.inp_dim / img_dim_list, 1)[0].view(-1, 1)
            dets[:, [1, 3]] -= (self.inp_dim - scaling_factor * img_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.inp_dim - scaling_factor * img_dim_list[:, 1].view(-1, 1)) / 2
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, img_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, img_dim_list[i, 1])

                #write results
                det_dict = {}
                x = float(dets[i, 1])
                y = float(dets[i, 2])
                w = float(dets[i, 3] - dets[i, 1])
                h = float(dets[i, 4] - dets[i, 2])
                det_dict["category_id"] = 1
                det_dict["score"] = float(dets[i, 5])
                det_dict["bbox"] = [x, y, w, h]
                det_dict["image_id"] = int(os.path.basename(img_name).split('.')[0])
                dets_results.append(det_dict)

            return dets_results
