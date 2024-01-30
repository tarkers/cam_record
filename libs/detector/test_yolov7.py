# from libs.detector.apis import BaseDetector
import math
import time
import cv2
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
import os


import sys

sys.path.insert(0, os.path.dirname(__file__))
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def bbox_iou(box1, box2, args=None):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if not args:
        inter_area = torch.max(
            inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()
        ) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda()
        )
    else:
        inter_area = torch.max(
            inter_rect_x2 - inter_rect_x1 + 1,
            torch.zeros(inter_rect_x2.shape).to(args.device),
        ) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1,
            torch.zeros(inter_rect_x2.shape).to(args.device),
        )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[
                :, 4:5
            ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)

            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = bbox_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def letterbox_image(img, inp_dim):
    """resize image with unchanged aspect ratio using padding"""
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[
        (h - new_h) // 2 : (h - new_h) // 2 + new_h,
        (w - new_w) // 2 : (w - new_w) // 2 + new_w,
        :,
    ] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = cv2.imread(img) if isinstance(img, str) else img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = letterbox_image(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


class YOLODetector:
    def __init__(self, cfg):
        super(YOLODetector, self).__init__()

        self.detector_cfg = cfg
        self.model_cfg = cfg.CONFIG
        self.model_weights = cfg.WEIGHTS
        self.inp_dim = cfg.INP_DIM
        self.nms_thres = cfg.NMS_THRES
        self.iouthres = cfg.IOUTHRES
        self.trace = cfg.trace
        self.confidence = (
            0.3
            if (False if not hasattr(cfg, "tracking") else cfg.tracking)
            else cfg.CONFIDENCE
        )
        self.num_classes = cfg.NUM_CLASSES
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = False if torch.cuda.is_available() else True
        self.load_model()

    def load_model(self):
        print("Start Loading Detector  model..")
        self.model = attempt_load(
            self.model_weights, map_location=self.device
        )  # load FP32 model
        # self.model = attempt_load(self.detector_opt.weights, map_location=self.detector_opt.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.inp_dim, s=self.stride)  # check img_size

        
            
        if self.trace:
           self.model= TracedModel(self.model, self.device, self.inp_dim)
           
        if not torch.cuda.is_available():
            self.model.half()  # to FP16


    def inference_image(self, img_source):
        # image_preprocess
        img, orig_img, im_dim_list = prep_image(img_source, self.inp_dim)
        print(img.shape)
        t1 = time_synchronized()
        with torch.no_grad():
            img = img.to(self.device)
            print(img)
            pred = self.model(img)[0]

        pred = non_max_suppression(
            pred, self.confidence, self.iouthres, classes=self.num_classes
        )
        t2 = time_synchronized()

        print(t2 - t1)
        print(pred)
        exit()
