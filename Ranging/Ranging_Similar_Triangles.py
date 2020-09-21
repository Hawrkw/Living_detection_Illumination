# coding=utf-8
# 导入相应的python包
import numpy as np
import dlib
import cv2
cap = cv2.VideoCapture(2)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#单位像素/w/H
pixel_w = 0.00075
pixel_h = 0.00075
#焦距
f = 0.4
#待测物体的宽/高
thing_w = 20
thing_h = 20
while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    rects = detector(frame_gray, 0)
    if len(rects) > 0:
        x_left = rects[0].left()
        y_left = rects[0].top()
        x_right = rects[0].right()
        y_right = rects[0].bottom()
        rect_w = x_right - x_left
        rect_h = y_right - y_left

        #计算成像宽/高
        width = rect_w * pixel_w
        height = rect_h * pixel_h

        #分别以成像宽高为标准计算距离
        distance_w = thing_w * f / width
        distance_h = thing_h * f / height
        cv2.rectangle(frame,(x_left,y_left),(x_right,y_right),(0,0,255),3)
        cv2.putText(frame,'distance:' + str(round(distance_h,2)),(x_left,y_left),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0))
        cv2.putText(frame,'rect_w:'+str(rect_w) +' rect_h:' + str(rect_h),(x_left,y_left - 15),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        print(distance_h)


