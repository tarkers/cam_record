

from easydict import EasyDict as edict

cfg = edict()
cfg.INP_DIM =  640
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.1
cfg.NUM_CLASSES = 80
weights=r'.\libs\detector\yolov7\data\yolov7.pt'
source=""
imgsz=640
conf_thres=0.25
iou_thres=0.45
device=''
view_img=False
save_txt=False
save_conf=False
nosave=False
classes=None
agnostic_nms=False
augment=False
update=False
# project=False
# name=False
exist_ok=False
trace=True