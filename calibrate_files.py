import cv2
import numpy as np
import glob
import os
import pathlib
from utils.utils import video_to_frame
import shutil
import json

def create_video(folder="u_mask", video_str=None):
    save_place = os.path.join(folder, video_str)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(save_place, fourcc, 100, (1280, 720))
    return writer


def calibrate_folder(folder_path, out_folder):
    camera_matrix, dist = np.load(r"good_matrics.npy", allow_pickle=True)
    videos = glob.glob(rf"{folder_path}\*.avi")
    # create folder
    p = pathlib.Path(out_folder)
    p.mkdir(parents=True, exist_ok=True)
    for video in videos:
        basename = os.path.basename(video)
        writer = create_video(out_folder, f"{basename}")
        images, fps, _ = video_to_frame(video)
        h, w = images[0].shape[:2]
        for idx, image in enumerate(images):
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist, (w, h), 1, (w, h)
            )
            # undistort
            undistort_image = cv2.undistort(
                image, camera_matrix, dist, None, newcameramtx
            )
            cv2.imshow("image", undistort_image)
            writer.write(undistort_image)
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break
        writer.release()


def rename_files(folder_path, is_patient=False):
    folder_name = os.path.basename(folder_path)
    videos = (
        glob.glob(rf"{folder_path}\*.mp4")
        + glob.glob(rf"{folder_path}\*.npy")
        + glob.glob(rf"{folder_path}\*.avi")
    )
    for video_name in videos:
        basename = os.path.basename(video_name)
        folder = video_name.replace(basename, "")
        data = basename.split(".")
        base = data[0][-14:]

        ext = data[-1]
        print(rf"{folder}{folder_name}_{base}.{ext}")
        os.rename(video_name, rf"{folder}{folder_name}_{base}.{ext}")


if __name__ == "__main__":
    pass
    data=np.load(rf"Data\P1188\P1188_WALK_L03\subject_3D\3D_P1188_WALK_L03.npy")
           
        
    # rename_files(r"Calibrate_Video\P1189")
    # calibrate_folder(r"D:\Chen\transform_bag\VICON\P1189",r"D:\Chen\transform_bag\Calibrate_Video\P1189")
