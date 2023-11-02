import json
import pathlib
import glob
from util.pose3d_generator import Pose3DGenerator
from util import update_config
if __name__ == "__main__":
    videoinfo = {"fourcc": 2, "fps": 60, "frameSize": [1920,1080]}
    pose3d_cfg = update_config(r"libs\configs\configs.yaml").Pose3D
    test=Pose3DGenerator(videoinfo,pose3d_cfg)
    json_paths=glob.glob(rf"Test\test2_serve_ball\subject_2D\ids\*.json")
    # file_basename='Ptz02_Fps060_20230926_191002_C001_T0332_0345'
    # test.preprocess_skeletons(rf"C:\Users\CHEN_KU\Desktop\github\cam_record\Test\subject_2D\14.json",file_basename,'./Test')
    for js_path in json_paths:
        file_basename='test2_serve_ball'
        test.preprocess_skeletons(js_path,file_basename,rf'Test\test2_serve_ball')