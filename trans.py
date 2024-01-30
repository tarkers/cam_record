import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name


 


def read_frame(name,start=0):
    arr=[]
    i=0
    cap = cv2.VideoCapture(name)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if i >= start:
            # Display the resulting frame
                if name[:2]=="3D":
                    arr.append(frame[200:1000-200,210:1000-150,:])
                    # cv2.imshow("test",frame[200:1000-200,210:1000-150,:])
                    # cv2.waitKey(0)
                else:
                    arr.append(frame)
        else: 
            break
        i+=1
    # When everything done, release the video capture object
    cap.release()
    return arr
# Closes all the frames
arr1,arr2=read_frame(rf"3D_N0001_WALK_R06.mp4"),read_frame(rf"N0001_WALK_R06.avi")
test=[]
if len(arr2)>len(arr1):
    arr2=arr2[:len(arr1)]
for a1,a2 in zip(arr1,arr2):
    a1 = cv2.resize(a1, (1280,720), interpolation = cv2.INTER_AREA)

    test.append(cv2.vconcat([a2,a1]))

out = cv2.VideoWriter('C_N0001_WALK_R06.mp4',cv2.VideoWriter_fourcc('M','P','V','4'), 100, (1280,1440))
for t in test:
    out.write(t)
out.release()

print(len(arr1),len(arr2))