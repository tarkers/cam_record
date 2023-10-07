import cv2
stream = cv2.VideoCapture(rf"D:\Chen\cam_record\person_walking.mp4")    #test
# stream = cv2.VideoCapture(int(input_source))
assert stream.isOpened(), "Cannot capture source"