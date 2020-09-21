import cv2
from threading import Thread
import util
import json
import time
import random
import math
from inference import inference,inference_112_112,liveDetect_crop, liveDetect
import numpy as np
from Ultra_Light_Fast.run_video_face_detect import detector_Ultra,detect_Ultra
from collections import OrderedDict
import tvm
import onnx
import tvm.relay as relay
from tvm.contrib import graph_runtime
from PIL import Image, ImageDraw, ImageFont
import dlib
from imutils import face_utils
from math import *

def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a

def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return angle_trunc(atan2(deltaY, deltaX))

def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#arraytoimage
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "/home/firefly/Desktop/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def rotate_about_center(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    

def captureFrame():
#初始值为[-10,-10,-10,-10]，其值>0表示检测到人脸
    global g_box
#目标是否匹配--是否在数据库中--为什么是两个值？这两个值同时变化，变化的还一样-匹配上为[1,1]，否则为[-1,-1]
    global g_coor
    global picture
    global frameID
    global name
    global live_flag
    global frame_flag
#这个好像也没什么用
    global control  
	#这个好像没什么用
    global fuck_flag
    #global fuck2_flag
	#这个flag用来控制0-10帧中都展示最初结果
    global fuck3_flag

    while(True):
        control = 1
        frameID = frameID+1
	#每10帧检测1次
        if frameID%10 == 0:#per n frame
            frame_flag = 1#detect and classify
        #只检测到人脸未匹配到--这里g_box列表的四个值与g_coor列表中的两个值都用来判断多增加了判断的次数有点多余
        elif g_box[0]>0 and g_box[1]>0 and g_box[2]>0 and g_box[3]>0 and g_coor[0]<0 and g_coor[1]<0: #and fuck_flag !=1:
            _, picture = capture.read()
            if fuck3_flag == 1: 
                if live_flag == 1:                    
                    picture = cv2ImgAddText(picture, '活体通过',g_box[0]+25,g_box[3]+30, (0,255,0), 40)
                elif live_flag == 0:                
                    picture = cv2ImgAddText(picture, '活体未通过',g_box[0]+25,g_box[3]+30, (255,0,0), 40)                   
            if fuck3_flag == 1:
                cv2.rectangle(picture,(g_box[0],g_box[1]),(g_box[2],g_box[3]),(0,0,255),2)           
                picture = cv2ImgAddText(picture, '匹配失败',g_box[0]+10,g_box[1]-50, (255,0,0), 40)                                  
                cv2.imshow('Face_System',picture)
                cv2.waitKey(1)
        #检测且匹配到
        elif g_box[0]>0 and g_box[1]>0 and g_box[2]>0 and g_box[3]>0 and g_coor[0]>0 and g_coor[1]>0:

            _, picture = capture.read()
            if live_flag == 1:
                 picture = cv2ImgAddText(picture, '活体通过',g_box[0]+25,g_box[3]+30, (0,255,0), 40)
            #picture = cv2.resize(picture, (640,480))
            x,y = g_coor
            cv2.rectangle(picture,(g_box[0],g_box[1]),(g_box[2],g_box[3]),(0,255,0),2)
            #cv2.putText(picture, '匹配成功 : {}'.format(name),(g_box[0],g_box[1]-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (000, 255, 000), 2)
            picture = cv2ImgAddText(picture, '匹配成功 : {}'.format(name),g_box[0]-20,g_box[1]-50, (0,255,0), 40)
            cv2.imshow('Face_System',picture) 
            #time.sleep(1)
            cv2.waitKey(1000)
            #fuck_flag = 1
            #cv2.waitKey(1)
        else:
            _, picture = capture.read()
            #picture = cv2.resize(picture, (640,480))
            cv2.imshow('Face_System',picture)
            cv2.waitKey(1)   
        
def getResult():
    global g_box
#目标是否匹配--是否在数据库中
    global g_coor
    global picture
    global frameID
    global name
    global live_flag
    global frame_flag  
    global control  
    global fuck_flag   
    global fuck2_flag
    global fuck3_flag

    #detect model load
    Ultra_predictor, Ultra_candidate_size, Ultra_threshold = detector_Ultra()
    threshold = 0.5
 
    #classify model load
    ctx = tvm.cpu()
    loaded_json = open('/home/firefly/FACE_System/MobileFaceNet/base_base4_two_60_2.json').read()
    loaded_lib = tvm.runtime.load_module('/home/firefly/FACE_System/MobileFaceNet/base_base4_two_60_2.so')
    loaded_params = bytearray(open('/home/firefly/FACE_System/MobileFaceNet/base_base4_two_60_2.params', 'rb').read())
    model = graph_runtime.create(loaded_json, loaded_lib, ctx)
    model.load_params(loaded_params)
     
    #live_detect model laoad
    loaded_json = open('/home/firefly/FACE_System/LiveModel/base11_data_sum_2_crop_optim_add_90.json').read()
    loaded_lib = tvm.runtime.load_module('/home/firefly/FACE_System/LiveModel/base11_data_sum_2_crop_optim_add_90.so')
    loaded_params = bytearray(open('/home/firefly/FACE_System/LiveModel/base11_data_sum_2_crop_optim_add_90.params', 'rb').read())
    live_model = graph_runtime.create(loaded_json, loaded_lib, ctx)
    live_model.load_params(loaded_params)        

    #数据库数据初始化
    tmpname = 'None'    
    f = open('massage.json',encoding='UTF-8')
    user_dic=json.load(f)
    
    v3 = []
    for item in user_dic.keys():
        v2 = user_dic[item]['value']
        v3.append(v2)
    v3 = np.array(v3)
    
    negs = 0
    pos = 0

    #关键点检测模型，用于人脸矫正
    predictor_path = 'dlib/shape_predictor_5_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    while True:    
        if frame_flag != 1: 
            continue
        g_box = [-10, -10, -10, -10]
        #fuck_flag = 0  
        live_flag = -1

        if picture is not None:
            # detect  h = 680 w = 960
            h, w, _ = picture.shape

            t = time.time()
            
	#Ultra_predictor是检测类的实例化对象，detect_Ultra调用Ultra_predictor对象中的检测方法来返回检测结果
            t1, box= util.get_face3(detect_Ultra, Ultra_predictor, Ultra_candidate_size, Ultra_threshold, picture, 0)
            #print('detect spend time: ',time.time()-t)
	    #如果检测到
            if box is not None:
                
                # living
		#这里是因为人脸检测网络的图片输入resize为320×240,所以要获得人脸在原图上的坐标则需要进行相应的转换
                t = time.time()
                y1 = box[1]*(h/240)
                y2 = box[3]*(h/240)
                x1 = box[0]*(w/320)
                x2 = box[2]*(w/320)
                w_ = x2-x1
                h_ = y2-y1
#--对于检测到的人脸边框将其大小进行扩大--进行边框检测
                x1_n = x1-w_*1
                x2_n = x2+w_*1
                y1_n = y1-h_*1
                y2_n = y2+h_*1
                x1_n = int(np.maximum(x1_n, 0)) 
                y1_n = int(np.maximum(y1_n, 0))   
                x2_n = int(np.minimum(x2_n, w-1))   
                y2_n = int(np.minimum(y2_n, h-1)) 
                img_n = picture[y1_n:y2_n, x1_n:x2_n]
                #print(img_n.shape)
                #cv2.imshow('r',img_n)
                #cv2.waitKey(1)
                out = liveDetect(img_n, live_model)
		#0-flash_live_fake 1-flash_live 分数
                f_pro = out[0]
                t_pro = out[1]
                print('假脸概率：',np.exp(f_pro)/(np.exp(t_pro)+np.exp(f_pro)))
                print('真脸概率：',1-np.exp(f_pro)/(np.exp(t_pro)+np.exp(f_pro)))
                if np.exp(t_pro)+np.exp(f_pro)!=0:
                    if np.exp(f_pro)/(np.exp(t_pro)+np.exp(f_pro)) > 0.5:
                    #if f_pro > t_pro:
                        live_flag = 0#fake num per n frame
                    else:
                        #print('pos num: {}'.format(pos))
                        live_flag = 1#flash_live num per n frame
                print('live spend time: ',time.time()-t)

            if box is not None:
                #fuck2_flag = -1 
                fuck3_flag = -1
   
                g_box[0] = int(box[0]*(w/320))
                g_box[1] = int(box[1]*(h/240))
                g_box[2] = int(box[2]*(w/320))
                g_box[3] = int(box[3]*(h/240))
                area = (g_box[2]-box[0])*(g_box[3]-box[1])
                center_x = (g_box[2]+g_box[0])/2
                center_y = (g_box[3]+g_box[1])/2
 		#人脸只能位于中间否则认为是非活体
                if center_x<w*1/4 or center_y<h*1/4 or center_x>w*3/4 or center_y>h*3/4:
                    print('边界')
                    live_flag = 0
                print('area: ',area)
		#人脸的box不能太小，否则也认为是非活体
                if area<30000: 
                    print('面积')
                    live_flag = 0
                
                #for correct
                t = time.time()
                rect = dlib.rectangle(int(g_box[0]),int(g_box[1]),int(g_box[2]),int(g_box[3]))
                gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                left = shape[2]
                right = shape[0]

                x=np.array([0,0])
                y=np.array([right[0]-left[0], -(right[1]-left[1])])
                angle = math.atan(y[1]/y[0])*180 / np.pi
                
                print('landmark spend time: ',time.time()-t)
                #lefteyehull = util.eye_aspect_ratio(lefteye)
                #cv2.drawContours(picture, [lefteyehull], -1, (0,255,0), 1)
                #活体检测通过-进行人脸匹配
                #live_flag = 1
                if live_flag == 1:#face is flash_live
                    #classify
                    t = time.time()
                    h, w, _ = t1.shape
                    
                    center = (w//2, h//2)
                    M = cv2.getRotationMatrix2D(center, -angle, 1.0) 
                    rotated_t1 = cv2.warpAffine(t1, M, (w, h)) 
                    #cv2.imshow("Rotated", rotated_t1)
                    
                    t1 = (rotated_t1 -127.5)/128
                    v1 = inference(t1,model)
                    v1 = v1[np.newaxis]
                    similarityList_numpy = util.cos_similarity(v1, v3)
                    max =similarityList_numpy.max()
                    #print('classify spend time:',time.time()-t)
                    maxindex=similarityList_numpy.argmax()
                    name = list(user_dic.keys())[maxindex]
                    print('similarity:',max)
#这个阈值选择记得在训练的时候选好
                    if max > threshold:
                        print('目标匹配')
                        g_coor[0], g_coor[1] = 1, 1 #box[0], box[1]-10
                    else:
                        print('目标不匹配')
                        g_coor = [-1,-1]
                        #fuck_flag2 = 1
                        
                #这里将线程暂停一下，使另一个线程能够得到g_coor，g_box的值后再进行改变      
                time.sleep(0.1)#result show time
                #fuck2_flag = 1  
                fuck3_flag = 1
                if g_coor[0]>0:
                    g_box = [-1,-1,-1,-1]
                    g_coor = [-1 ,-1]
                frame_flag = 0

if __name__ =='__main__':
	#opencv调用外接摄像头，一般是传入摄像头编号，如本机自带摄像头cv2.VideoCapture(0)
    capture = cv2.VideoCapture("/dev/video10")
#设置视频流的帧的宽度
    capture.set(3,960)
#设置视频流的帧的高度
    capture.set(4,680)
    global g_box
    global g_coor
    global picture
    global frameID
    global name
    global live_flag
    global frame_flag
    global control
    #global fuck_flag
    #global fuck2_flag
    global fuck3_flag
    g_box = [-10, -10, -10, -10]#for detect result
    g_coor = [-1,-1]#for classify result
    picture = None
    frameID = 0
    name = None
    live_flag = -1
    frame_flag = 0
    control = -1
    #fuck_flag = -1
    #fuck2_flag = -1
    fuck3_flag = -1
    t1 = Thread(target = captureFrame)
    t2 = Thread(target = getResult)
    t1.start()
    t2.start()


        
'''这里表示裁剪后人脸活体检测
t = time.time()
if img_for_live_detect is not None:
    img_for_live_detect = (img_for_live_detect - 127.5) / 128.0
    img_for_live_detect = img_for_live_detect.transpose(2, 0, 1)
    img_for_live_detect = torch.from_numpy(img_for_live_detect).unsqueeze(0).float()
    out = live_net(img_for_live_detect)
    if out[0,0] > out[0,1]:#0号index大为假脸
        live_flag = 0
    else:
        live_flag = 1
print('Living detect spend time: ',time.time()-t)
'''   
#如果检测到人脸,就进行人脸识别
        
'''
picture_flive = picture.copy()
picture_flive = cv2.resize(picture_flive, (32,32))
picture_flive = (picture_flive - 127.5) / 128.0
picture_flive = picture_flive.transpose(2, 0, 1)
picture_flive = torch.from_numpy(picture_flive).unsqueeze(0).float()
out = live_net(picture_flive)
if out[0,0] > out[0,1]:#0号index大为假脸
live_flag = 0
else:
live_flag = 1
'''      
          
'''
elif fuck_flag == 1 and g_coor[0]==-10 and g_coor[1]==-10:
    _, picture = capture.read()
    #picture = cv2.resize(picture, (640,480))
    cv2.putText(picture, 'Unrecognized',(g_box[0],g_box[1]-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (000, 255, 000), 2)
    cv2.rectangle(picture,(g_box[0],g_box[1]),(g_box[2],g_box[3]),(0,0,255),2) 
    cv2.imshow('Face_System',picture)
    cv2.waitKey(1)
'''

'''
#live_flag = 1
#live detect for crop face
t = time.time()
if img_for_live_detect is not None:
    score = liveDetect_crop(img_for_live_detect, live_model)
    #print(score)
    if score > 0.90:
        live_flag = 1
    else:
        live_flag = 0
print('Living detect spend time: ',time.time()-t)
'''
'''
if negs+pos != 0:
    if negs+pos <5:
        continue
'''