

from ..bbox import (_box_to_center_scale, _center_scale_to_box,
                    _clip_aspect_ratio)
from ..transforms import (addDPG, affine_transform, flip_joints_3d,
                          get_affine_transform, im_to_torch)
import cv2
class BoxTransform(object):
    def __init__(self,dataset, input_size, output_size,scale_factor=0,
                 rot=0,train=False) -> None:
        self._input_size = input_size
        self._heatmap_size = output_size
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        pass
    
    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)
        # print(bbox)
        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox