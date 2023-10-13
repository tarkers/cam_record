
from util.utils import DataType ,video_to_frame
import cv2
video_path=rf"D:\Chen\cam_record\Test\JAPAN vs USA _ Highlights _ Men s OQT 2023.mp4"
images,fps,_=video_to_frame(video_path,0,300)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cap = cv2.VideoWriter(rf"D:\Chen\cam_record\Test\JAPAN vs USA _ Highlights _ Men s OQT 2023_3_1.mp4", fourcc, 60.0, (1920,1080))

for img in images:
    cap.write(img)
   

# Release everything if job is finished
cap.release()
# out.release()
# cv2.destroyAllWindows()

    