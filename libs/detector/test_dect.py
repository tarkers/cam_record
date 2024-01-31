import cv2
from threading import Thread
from itertools import count
from queue import Queue
import numpy as np

import torch
import torch.multiprocessing as mp

# # our folder##
from libs.presets import BoxTransform
from libs.bbox import bbox_xywh_to_xyxy

# from utils import update_config
# from libs.detector.apis import get_detector
from libs.detector.detector_api import YOLODetector


class YOLODetectortest:
    """
    use webcam stream to detect object
    """

    def __init__(
        self, cfg, input_source=None, detector=None, tracker=None, queueSize=150
    ):
        self.path = input_source
        self.cfg = cfg
        self.pause_stream = False
        self.detector = YOLODetector(cfg)
        self.tracker = tracker

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == "simple":
            self.transformation = BoxTransform(
                self,
                scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0,
            )

        # initialize the queue used to store data
        """
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if self.cfg.sp:
            self._stopped = False
            # self.queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value("b", False)
            # self.queue = mp.Queue(maxsize=queueSize)

    def start_worker(self, target, queue):
        if self.cfg.sp:
            p = Thread(target=target, args=(queue,))

        else:
            p = mp.Process(target=target, args=(queue,))

        # p.daemon = True
        p.start()

        return p

    def start(self, queue):
        """
        start a thread to pre process images for object detection
        """
        image_preprocess_worker = self.start_worker(self.detector_process, queue)
        return [image_preprocess_worker]

    def terminate(self):
        """
        set the stopped to true and left None to image
        """
        print("bounding box termination")
        if self.cfg.sp:
            self._stopped = True
        else:
            self._stopped.value = True

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def detect_for_one_frame(self, data):
        frame_idx = 0
        # expected frame shape like (1,3,h,w) or (3,h,w)
        img_k ,orig_img ,im_dim_list_k = self.detector.image_preprocess(data)  # preprocess image to gray

        if isinstance(img_k, np.ndarray):
            img_k = torch.from_numpy(img_k)
        # # add one dimension at the front for batch if image shape (3,h,w)
        if img_k.dim() == 3:
            img_k = img_k.unsqueeze(0)

        orig_img = orig_img[:, :, ::-1]    #rgb to bgr
        im_name = str(frame_idx) + ".jpg"
        frame_idx += 1

        with torch.no_grad():  # Record original image resolution
            im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
        img_det = self.image_detection((img_k, orig_img, im_name, im_dim_list_k))
        item = self.image_postprocess(img_det)
        return item

    def detector_process(self, queue):
        # self.queue =queue

        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), "Cannot capture source"
        frame_idx = 0
        self.tracker.init_fastreid()  # load for thread

        # keep looping infinitely
        for _ in count():
            if self.stopped:
                print("detect queue is stopped!!")
                stream.release()
                break
            if (
                not queue.full() and not self.pause_stream
            ):  # make sure queue has empty place
                (grabbed, frame) = stream.read()

                if not grabbed:
                    print("Finish streaming images!")
                    stream.release()
                    break

                item = self.detect_for_one_frame(frame)

                queue.put(item)
                # # test terminate code
                # if frame_idx >10:
                #     print("Finish streaming images!")
                #     stream.release()
                #     break
                # cv2.waitKey(30)
            else:
                print("webcam is full")

        ## put terminate data ###
        queue.put((None, None, None, None, None, None, None, None))
        print("BOX PUT:", "terminate")
        self.terminate()

    def image_detection(self, inputs):
        """
        start to detect images object
        dets(n,(batch_ind,x1,y1,x2,y2,box_confidence,cls_score,idx of cls))
        bbox:dets [:,1:5]
        box_confidence:dets [:,5]
        class_ids:dets [:,6]
        cls_score:dets [:,7]
        """
        img, orig_img, im_name, im_dim_list = inputs
        if img is None or self.stopped:
            return (None, None, None, None, None, None, None, None)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:  # detect is None
                return (orig_img, im_name, None, None, None, None, None, None)

            dets = dets.cpu()
            if isinstance(dets, torch.FloatTensor) and self.cfg.tracking:
                dets = torch.Tensor.numpy(dets)

            if self.cfg.tracking:
                (boxes, ids, scores, class_ids) = self.tracker_udpate(
                    dets[:, 1:], orig_img
                )
            else:
                class_ids = dets[:, 6]
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]
                ids = torch.zeros(scores.shape)

        if isinstance(boxes, int) or boxes.size(0) == 0:
            return (orig_img, im_name, None, None, None, None, None, None)
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        return (
            orig_img,
            im_name,
            class_ids,
            boxes,
            scores,  # enforce class
            ids,  # enforce class
            inps,
            cropped_boxes,
        )

    def tracker_udpate(self, dets, orig_img):
        online_targets = self.tracker.update(dets, orig_img)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []
        for t in online_targets:
            if t.tlwh[2] * t.tlwh[3] > self.cfg.min_box_area:
                online_tlwhs.append(t.tlwh)
                online_ids.append(t.track_id)
                online_scores.append(t.score)
                online_cls.append(int(t.cls))

        # if have bbox then change to tesor
        if len(online_tlwhs):
            online_tlwhs = torch.from_numpy(bbox_xywh_to_xyxy(np.array(online_tlwhs)))
        else:
            online_tlwhs = torch.FloatTensor([])
        online_ids = torch.FloatTensor(online_ids)
        online_scores = torch.FloatTensor(online_scores)
        online_cls = torch.FloatTensor(online_cls)
        return (online_tlwhs, online_ids, online_scores, online_cls)

    def image_postprocess(self, inputs):
        with torch.no_grad():
            (
                orig_img,
                im_name,
                class_ids,
                boxes,
                scores,
                ids,
                inps,
                cropped_boxes,
            ) = inputs

            if orig_img is None or self.stopped:  # no  image
                return (None, None, None, None, None, None, None, None)

            if boxes is None or boxes.nelement() == 0:  # no detect object
                return (None, orig_img, im_name, class_ids, boxes, scores, ids, None)

            # print("detect!")
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            return (
                inps,
                orig_img,
                im_name,
                class_ids,
                boxes,
                scores,
                ids,
                cropped_boxes,
            )

    def read(self):
        """
        Returns:
            inps: cropped images
            orig_img: original frame (is RGB)
            im_name: image name
            class_ids: detect object class IDs
            boxes: list of boxes detected
            scores: bbox detect scores
            ids: ids for tracker
            cropped_boxes: list of new box coordinates
        """
        return self.wait_and_get(self.queue)

    @property
    def stopped(self):
        if self.cfg.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def yolo_classes(self):
        """
        return 80 object names base on idx
        """
        return [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]