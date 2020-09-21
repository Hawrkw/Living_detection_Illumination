# coding=utf-8
# 导入相应的python包
import numpy as np
import dlib
import cv2
cap = cv2.VideoCapture(2)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
i = 53
while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow('frame',frame)
    c = cv2.waitKey(1)
    if c == ord('q'):
        rects = detector(frame_gray, 0)
        if len(rects) > 0:
            cv2.imwrite('flash_live/' + str(i) + '.jpg',frame)
            print(i)
            i += 1
    if c == 27:#esc键值为27
        break
