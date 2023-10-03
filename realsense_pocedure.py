import cv2
import os
import numpy as np
def see_alignment(video_path=0):
    def camera_alignment(image, square_w=30):
        h, w, ch = image.shape
        for i in range(0, w, square_w):
            image = cv2.line(image, (i, 0),  (i, h), (0, 255, 0), 1)
        for y in range(0, h, square_w):
            image = cv2.line(image, (0, y), (w, y), (0, 255, 0), 1)
        return image
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():  
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame=camera_alignment(frame,20)
        cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
        cv2.imshow("frame",frame)
        if cv2.waitKey(150) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def extract_images(video_path,save_folder="images"):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cap = cv2.VideoCapture(video_path)
    count=0
    while cap.isOpened():
        
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if count %25==0:
            cv2.imwrite(os.path.join(save_folder,f"{count}.jpg"),frame)
        if cv2.waitKey(1) == ord('q') or count//25>20:
            break
        count+=1
    cap.release()
    cv2.destroyAllWindows()

def edge_detection(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 10
    high_threshold = 80
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    
    cv2.imshow("image",edges)
    cv2.waitKey(0)
if __name__ =="__main__":
    see_alignment(r"C:\Users\bigsh\Desktop\camera_calibration\realsense\75.jpg")