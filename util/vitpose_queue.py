import torch
import numpy as np
from queue import Queue
import numpy as np
import torch.multiprocessing as mp
from torchvision.transforms import transforms
from libs.pose2D.utils.pPose_nms import pose_nms
from libs.vit.models.model import ViTPose
from libs.vit.utils.top_down_eval import keypoints_from_heatmaps
from libs.pose2D.utils.transforms import get_func_heatmap_to_coord
from libs.vit.configs.ViTPose_base_halpe_256x192 import (
    data_cfg,
    model as model_cfg,
)
from util.timer import time_synchronized


class VitPoseQueue:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.batchSize = self.cfg.posebatch
        self.queueSize = 150 if self.cfg.mode == "webcam" else self.cfg.qsize
        self.pose_queueSize = 243
        self.img_size = data_cfg["image_size"]
        self.norm_type = None  # for evaluate set None
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.use_heatmap_loss = (
            self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss") == "MSELoss"
        )
        # self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.pose_fps = 0
        self.model = None
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
        self.model = ViTPose(self.model_cfg)
        checkpoint = torch.load(self.cfg.checkpoint)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def image_preprocess(self, ori_image):
        ori_h, ori_w = ori_image[0], ori_image[1]
        img_tensor = (
            transforms.Compose(
                [
                    transforms.Resize((self.img_size[1], self.img_size[0])),
                    transforms.ToTensor(),
                ]
            )(ori_image)
            .unsqueeze(0)
            .to(self.device)
        )
        return img_tensor, ori_w, ori_h

    def pose_estimation(self, item):
        (inps, orig_img, im_name, class_ids, boxes, scores, ids, cropped_boxes) = item
        inps = inps.to(self.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % self.batchSize:
            leftover = 1
        num_batches = datalen // self.batchSize + leftover
        hm = []
        t1 = time_synchronized()
        for j in range(num_batches):
            inps_j = inps[j * self.batchSize : min((j + 1) * self.batchSize, datalen)]
            # pose model execute
            hm_j = self.model(inps_j)
            hm.append(hm_j)

        hm = torch.cat(hm)
        hm = hm.cpu()
        assert hm.dim() == 4
        ## check keypoints ##
        face_hand_num = 110
        if hm.size()[1] == 136:
            self.eval_joints = [*range(0, 136)]
        elif hm.size()[1] == 26:
            self.eval_joints = [*range(0, 26)]
        elif hm.size()[1] == 133:
            self.eval_joints = [*range(0, 133)]
        elif hm.size()[1] == 68:
            face_hand_num = 42
            self.eval_joints = [*range(0, 68)]
        elif hm.size()[1] == 21:
            self.eval_joints = [*range(0, 21)]

        pose_coords = []
        pose_scores = []
        hm = hm.detach().numpy()
        for i in range(hm.shape[0]):
            xmin, ymin, xmax, ymax = cropped_boxes[i].tolist()  # x,y,w,h
            w = xmax - xmin
            h = ymax - ymin
            heatmap = np.expand_dims(hm[i][self.eval_joints], axis=0)

            pose_coord, pose_score = keypoints_from_heatmaps(
                heatmaps=heatmap,
                center=np.array([[xmin + w // 2, ymin + h // 2]]),
                scale=np.array([[w, h]]),
                unbiased=True,
                use_udp=True,
            )

            ## add x, y
            # pose_coord+=[xmin,ymin]
            pose_coords.append(torch.from_numpy(pose_coord))
            pose_scores.append(torch.from_numpy(pose_score))
            
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        # print(preds_img.size(),preds_scores)
        # exit()
        # exit()
        boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(
            boxes,
            scores,
            ids,
            preds_img,
            preds_scores,
            self.cfg.min_box_area,
            use_heatmap_loss=self.use_heatmap_loss,
        )

        ## perpare resultes
        _result = []
        for k in range(len(scores)):
            _result.append(
                {
                    "keypoints": preds_img[k],
                    "kp_score": preds_scores[k],
                    "proposal_score": torch.mean(preds_scores[k])
                    + scores[k]
                    + 1.25 * max(preds_scores[k]),
                    "idx": ids[k],
                    "class_id": class_ids[k],
                    "box": [
                        boxes[k][0],
                        boxes[k][1],
                        boxes[k][2] - boxes[k][0],
                        boxes[k][3] - boxes[k][1],
                    ],
                }
            )

        result = {"imgname": im_name, "result": _result}
        t2 = time_synchronized()
        self.pose_fps = round((1 / max((t2 - t1), 0.000001)), 2)
        if scores:
            return (orig_img, result), self.pose_fps
        else:
            return (orig_img, None), self.pose_fps  # not have pose for the bounding box

    def get_skeleton_from_single_image(self, image):
        img_tensor, ori_w, ori_h = self.image_preprocess(image)

        heatmaps = self.model(img_tensor).detach().cpu().numpy()  # N, 26, h/4, w/4
        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[ori_w // 2, ori_h // 2]]),
            scale=np.array([[ori_w, ori_h]]),
            unbiased=True,
            use_udp=True,
        )
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)
        return points
