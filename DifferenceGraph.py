import dlib
import cv2
import numpy as np
import math
from imutils import face_utils
# predictor_path = 'dlib/shape_predictor_5_face_landmarks.dat'
# predictor = dlib.shape_predictor(predictor_path)
#rect = dlib.rectangle(int(g_box[0]),int(g_box[1]),int(g_box[2]),int(g_box[3]))
# gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
# shape = predictor(gray, rect)
# shape = face_utils.shape_to_np(shape)
# left = shape[2]
# right = shape[0]
#
# x=np.array([0,0])
# y=np.array([right[0]-left[0], -(right[1]-left[1])])
# angle = math.atan(y[1]/y[0])*180 / np.pi
# h, w, _ = t1.shape
#
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, -angle, 1.0)
# rotated_t1 = cv2.warpAffine(t1, M, (w, h))

cap = cv2.VideoCapture(0)
#detector检测图片中出现的人脸
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
####视频关键点
def key_point_video():
    while 1:
        ret, img = cap.read()
        # 取灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 人脸数rects
        rects = detector(img_gray, 0)
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])

                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(img, pos, 2, color=(0, 255, 0))
                # 利用cv2.putText输出1-68
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(idx + 1), None, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
#图片关键点--仿射矩阵法--不行
def key_point_img():
    # cv2读取图像
    #自然光图像
    natural_light_img = cv2.imread("flash_live/flash_live/亮/场景1/1/18.jpg")
    #左光源图像是说光源的光打在左脸上
    left_light_img = cv2.imread("flash_live/flash_live/亮/场景1/1/19.jpg")
    #left_light_img = cv2.imread("0.jpg")
    natural_light_img_gray = cv2.cvtColor(natural_light_img,cv2.COLOR_RGB2GRAY)

    left_light_img_gray = cv2.cvtColor(left_light_img,cv2.COLOR_RGB2GRAY)


    # 自然光图片中人脸的位置-二维列表
    natural_light_rects = detector(natural_light_img_gray,0)
    #我们的数据都是一人的所以这里直接用rects[0]来表示
    natural_light_landmarks = np.matrix([[p.x, p.y] for p in predictor(natural_light_img, natural_light_rects[0]).parts()])
    natural_light_landmarks = np.array(natural_light_landmarks,np.float32)
    # 光源照在人脸左边图片中人脸的位置-二维列表
    left_light_rects = detector(natural_light_img_gray, 0)
    left_light_landmarks = np.matrix(
        [[p.x, p.y] for p in predictor(left_light_img, left_light_rects[0]).parts()])
    left_light_landmarks = np.array(left_light_landmarks,np.float32)
    order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序
    image_shape = natural_light_img_gray.shape
    #左光源图的四个坐标点
    left_dst = np.array([
        [left_light_landmarks[36][0], left_light_landmarks[36][1]],
        [left_light_landmarks[45][0], left_light_landmarks[45][1]],
        [left_light_landmarks[48][0], left_light_landmarks[48][1]],
        [left_light_landmarks[54][0], left_light_landmarks[54][1]]
    ], dtype="float32")
    #新图的四个坐标点
    dst = np.array([
        [natural_light_landmarks[36][0],natural_light_landmarks[36][1]],
        [natural_light_landmarks[45][0],natural_light_landmarks[45][1]],
        [natural_light_landmarks[48][0],natural_light_landmarks[48][1]],
        [natural_light_landmarks[54][0], natural_light_landmarks[54][1]]
    ],dtype="float32")
    M = cv2.getPerspectiveTransform(left_dst,dst)
    warped = cv2.warpPerspective(left_light_img_gray,M,(640,480))

    cv2.imshow('natural',natural_light_img_gray)
    cv2.imshow('left', left_light_img_gray)
    cv2.imshow("wa",warped)
    cv2.waitKey(0)

    # 取灰度
    # img_gray = cv2.cvtColor(natural_light_img, cv2.COLOR_RGB2GRAY)
    # # 图片中人脸的位置-二维列表
    # rects = detector(img_gray, 0)
    # for i in range(len(rects)):
    #     landmarks = np.matrix([[p.x, p.y] for p in predictor(natural_light_img, rects[i]).parts()])
    #     for idx, point in enumerate(landmarks):
    #         # 68点的坐标
    #         pos = (point[0, 0], point[0, 1])
    #
    #         # 利用cv2.circle给每个特征点画一个圈，共68个
    #         cv2.circle(natural_light_img, pos, 2, color=(0, 255, 0))
    #         # 利用cv2.putText输出1-68
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(natural_light_img, str(idx + 1), None, font, 10, (0, 0, 255), 10, cv2.LINE_AA)
    #
    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", natural_light_img)
    # cv2.waitKey(0)
#获取截取的人脸区域坐标：左上-右下
def get_rect(rect,image_shape):
    x_left = rect[0].left()
    y_left = rect[0].top()
    x_right = rect[0].right()
    y_right = rect[0].bottom()
    w = x_right - x_left
    h = y_right - y_left
    x_left_expand = x_left - w * 1 / 2
    y_left_expand = y_left - h * 1 / 2
    x_right_expand = x_right + w * 1 / 2
    y_right_expand = y_right + h * 1 / 2
    x_left_expand = int(np.maximum(x_left_expand, 0))
    y_left_expand = int(np.maximum(y_left_expand, 0))
    x_right_expand = int(np.minimum(x_right_expand, image_shape[1] - 1))
    y_right_expand = int(np.minimum(y_right_expand, image_shape[0] - 1))
    return [x_left_expand,y_left_expand,x_right_expand,y_right_expand]
#本实验中每个像素点在实际距离中是0.035cm
#方法传入鼻尖的光照方向，以及一张灰度图像
#返回所有像素点的光照方向
def get_light_direction_of_each_pixel(A,img,nose:[]):
    # opencv读出的图像（宽，高）
    img_shape = img.shape
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 自然光原图片中人脸的位置-二维列表
    rects = detector(img_gray, 0)
    #face_area = get_rect(rects,img_shape)
    if len(rects) != 0:

        # 我们的数据都是一人的所以这里直接用rects[0]来表示
        landmarks = np.matrix(
            [[p.x, p.y] for p in predictor(img, rects[0]).parts()])
        landmarks = np.array(landmarks, np.float32)
        #鼻子的关键点是30
        nose_x,nose_y = landmarks[30][0],landmarks[30][1]
    elif len(nose) != 0:
        nose_x, nose_y = nose[0], nose[1]
    #成像与真实情况相反，所以先算右脸
    #i是行--width--image_shape[0] j是列--length---image_shape[1]
    #nose_x--列 nose_y--行
    A_all = []
    for i in range(img_shape[0]):
        A_row = []
        for j in range(img_shape[1]):
            #三维坐标轴-y
            y = (nose_x - j) * 0.035
            # 三维坐标轴-z
            z = (nose_y - i) * 0.035

            change_length = [0,y,z]
            change_length = np.array(change_length)
            change_length = np.expand_dims(change_length,axis=0).repeat(3,axis=0)
            #根据鼻尖光照方向，获取当前像素点光照方向
            A_new = np.subtract(A,change_length)
            #对当前像素点光照方向进行正则化，将其转为单位向量
            A_norm = np.linalg.norm(A_new, axis=1, keepdims=True)
            A_unit = A_new / A_norm
            A_new = np.matrix(A_unit)
            A_reverse = A_new.I
            A_row.append(A_reverse)
        A_row = np.array(A_row)
        A_all.append(A_row)
    A_all = np.array(A_all)
    return A_all


#单映矩阵法
#这里传了两个路径，其实可以传图片进来--python重载比较复杂，需要使用默认参数--这里我们对参数类型进行判断来达到重载的目的
#返回值：截取后的人脸差分图像
def get_difference_graph(natural_light_img_path,light_source_img_path):
    # cv2读取图像
    #如果参数是str类型，则为图片路径，否则则是传入图片--这里应该做bug处理
    if type(natural_light_img_path) == str and type(light_source_img_path) == str:
        # 自然光图像
        natural_light_img = cv2.imread(natural_light_img_path)
        # 左光源图像是说光源的光打在左脸上
        light_source_img = cv2.imread(light_source_img_path)
    else:
        natural_light_img = natural_light_img_path
        # 左光源图像是说光源的光打在左脸上
        light_source_img = light_source_img_path

    #图片转换为灰度图像
    natural_light_img_gray = cv2.cvtColor(natural_light_img, cv2.COLOR_RGB2GRAY)

    light_source_img_gray = cv2.cvtColor(light_source_img, cv2.COLOR_RGB2GRAY)

    # 我们的图像大小以自然光为准，带光源图像也要resize成自然光图像大小---暂时这么确定
    #opencv读出的图像（宽，高）
    natural_light_img_shape = natural_light_img_gray.shape

    # 自然光原图片中人脸的位置-二维列表
    natural_light_rects = detector(natural_light_img_gray, 0)


    # 我们的数据都是一人的所以这里直接用rects[0]来表示
    natural_light_landmarks = np.matrix(
        [[p.x, p.y] for p in predictor(natural_light_img, natural_light_rects[0]).parts()])
    natural_light_landmarks = np.array(natural_light_landmarks, np.float32)
    # 光源照在人脸图片中人脸的位置-二维列表
    light_source_rects = detector(light_source_img_gray, 0)



    light_source_landmarks = np.matrix(
        [[p.x, p.y] for p in predictor(light_source_img, light_source_rects[0]).parts()])
    light_source_landmarks = np.array(light_source_landmarks, np.float32)
    #order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序


    #通过人脸68个关键点计算单应矩阵
    h,mask = cv2.findHomography(light_source_landmarks,natural_light_landmarks,cv2.RANSAC)
    #warped是对齐后的图像
    warped = cv2.warpPerspective(light_source_img_gray,h,(natural_light_img_shape[1],natural_light_img_shape[0]))
    #warped = np.array(warped,dtype='int32')
    #natural_light_img_gray = np.array(natural_light_img_gray,dtype='int32')
    M = np.ones(natural_light_img_shape, dtype="uint8") * 255
    res = cv2.subtract(warped,natural_light_img_gray)

    # res = cv2.subtract(M,res)
    res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)

    res = np.maximum(res,0)

    x_left = natural_light_rects[0].left()
    y_left = natural_light_rects[0].top()
    x_right = natural_light_rects[0].right()
    y_right = natural_light_rects[0].bottom()
    w = x_right - x_left
    h = y_right - y_left
    x_left_expand = x_left - w * 1 / 2
    y_left_expand = y_left - h * 1 / 2
    x_right_expand = x_right + w * 1 / 2
    y_right_expand = y_right + h * 1 / 2
    x_left_expand = int(np.maximum(x_left_expand, 0))
    y_left_expand = int(np.maximum(y_left_expand, 0))
    x_right_expand = int(np.minimum(x_right_expand, natural_light_img_shape[1] - 1))
    y_right_expand = int(np.minimum(y_right_expand, natural_light_img_shape[0] - 1))
    face_area = [x_left_expand,y_left_expand,x_right_expand,y_right_expand]
    res = res[y_left_expand:y_right_expand, x_left_expand:x_right_expand,:]

    #box = [x_left_expand,y_left_expand,x_right_expand,y_right_expand]
    #裁减人脸后人脸上关键点的坐标也会进行改变
    nose = []
    if x_left_expand > 0 and y_left_expand > 0:
        nose.append(natural_light_landmarks[30][0] - x_left_expand)
        nose.append(natural_light_landmarks[30][1] - y_left_expand)
    elif x_left_expand > 0 and y_left_expand == 0:
        nose.append(natural_light_landmarks[30][0] - x_left_expand)
        nose.append(natural_light_landmarks[30][1])
    elif x_left_expand == 0 and y_left_expand > 0:
        nose.append(natural_light_landmarks[30][0])
        nose.append(natural_light_landmarks[30][1] - y_left_expand)
    else:
        nose.append(natural_light_landmarks[30][0])
        nose.append(natural_light_landmarks[30][1])
    return res,natural_light_img[y_left_expand:y_right_expand, x_left_expand:x_right_expand,:],nose,face_area

    #cv2.destroyWindow(window_name)
#这里传了两个路径，其实可以传图片进来--python重载比较复杂，需要使用默认参数--这里我们对参数类型进行判断来达到重载的目的
def alignImage(natural_light_img_path,light_source_img_path,result_img_path):
    # cv2读取图像
    #如果参数是str类型，则为图片路径，否则则是传入图片--这里应该做bug处理
    if type(natural_light_img_path) == str and type(light_source_img_path) == str:
        # 自然光图像
        natural_light_img = cv2.imread(natural_light_img_path)
        # 左光源图像是说光源的光打在左脸上
        light_source_img = cv2.imread(light_source_img_path)
    else:
        natural_light_img = natural_light_img_path
        # 左光源图像是说光源的光打在左脸上
        light_source_img = light_source_img_path

    #图片转换为灰度图像
    natural_light_img_gray = cv2.cvtColor(natural_light_img, cv2.COLOR_RGB2GRAY)

    light_source_img_gray = cv2.cvtColor(light_source_img, cv2.COLOR_RGB2GRAY)

    # 我们的图像大小以自然光为准，带光源图像也要resize成自然光图像大小---暂时这么确定
    #opencv读出的图像（宽，高）
    natural_light_img_shape = natural_light_img_gray.shape
    # 自然光原图片中人脸的位置-二维列表
    natural_light_rects = detector(natural_light_img_gray, 0)
    #通过人脸检测获得的人脸位置将其扩展
    x_left = natural_light_rects[0].left()
    y_left = natural_light_rects[0].top()
    x_right = natural_light_rects[0].right()
    y_right = natural_light_rects[0].bottom()
    w = x_right - x_left
    h = y_right - y_left
    x_left_expand = x_left - w * 1 / 2
    y_left_expand = y_left - h * 1 / 2
    x_right_expand = x_right + w * 1 / 2
    y_right_expand = y_right + h * 1 / 2
    x_left_expand = int(np.maximum(x_left_expand, 0))
    y_left_expand = int(np.maximum(y_left_expand, 0))
    x_right_expand = int(np.minimum(x_right_expand, natural_light_img_shape[1] - 1))
    y_right_expand = int(np.minimum(y_right_expand, natural_light_img_shape[0] - 1))
    natural_light_img_gray = cv2.resize(natural_light_img_gray[y_left_expand:y_right_expand,x_left_expand:x_right_expand],
                                        (400,400))
    natural_light_face_img_shape = natural_light_img_gray.shape
    natural_light_rects = detector(natural_light_img_gray, 0)
    cv2.imshow('n',natural_light_img_gray)

    # 我们的数据都是一人的所以这里直接用rects[0]来表示
    natural_light_landmarks = np.matrix(
        [[p.x, p.y] for p in predictor(natural_light_img, natural_light_rects[0]).parts()])
    natural_light_landmarks = np.array(natural_light_landmarks, np.float32)
    # 光源照在人脸图片中人脸的位置-二维列表
    light_source_rects = detector(light_source_img_gray, 0)

    # 通过人脸检测获得的人脸位置将其扩展
    x_left = light_source_rects[0].left()
    y_left = light_source_rects[0].top()
    x_right = light_source_rects[0].right()
    y_right = light_source_rects[0].bottom()
    w = x_right - x_left
    h = y_right - y_left
    x_left_expand = x_left - w * 1 / 2
    y_left_expand = y_left - h * 1 / 2
    x_right_expand = x_right + w * 1 / 2
    y_right_expand = y_right + h * 1 / 2
    x_left_expand = int(np.maximum(x_left_expand, 0))
    y_left_expand = int(np.maximum(y_left_expand, 0))
    x_right_expand = int(np.minimum(x_right_expand, natural_light_img_shape[1] - 1))
    y_right_expand = int(np.minimum(y_right_expand, natural_light_img_shape[0] - 1))
    #cv2.imshow('l', light_source_img_gray[y_left_expand:y_right_expand, x_left_expand:x_right_expand])
    light_source_img_gray = cv2.resize(
        light_source_img_gray[y_left_expand:y_right_expand, x_left_expand:x_right_expand],
        (400, 400))
    cv2.imshow('l', light_source_img_gray)
    light_source_rects = detector(light_source_img_gray, 0)

    light_source_landmarks = np.matrix(
        [[p.x, p.y] for p in predictor(light_source_img, light_source_rects[0]).parts()])
    light_source_landmarks = np.array(light_source_landmarks, np.float32)
    #order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序


    #通过人脸68个关键点计算单应矩阵
    h,mask = cv2.findHomography(light_source_landmarks,natural_light_landmarks,cv2.RANSAC)
    #warped是对齐后的图像
    warped = cv2.warpPerspective(light_source_img_gray,h,(natural_light_face_img_shape[1],natural_light_face_img_shape[0]))
    cv2.imshow('w',warped)
    #warped = np.array(warped,dtype='int32')
    #natural_light_img_gray = np.array(natural_light_img_gray,dtype='int32')
    M = np.ones(natural_light_img_shape, dtype="uint8") * 255
    res = cv2.subtract(warped,natural_light_img_gray)

    # res = cv2.subtract(M,res)
    res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
    a = np.maximum(res,0)
    window_name = 'resBGR'
    cv2.imshow(window_name,res)
    #cv2.imwrite(result_img_path, res)
    cv2.waitKey(0)
    #cv2.destroyWindow(window_name)
#在获取差分图像后剪切数据

# file_name = ['left-nature-fake.jpg','top-nature-fake.jpg','right-nature-fake.jpg']
# result_img_path = 'result/flash_live_bright2_light1_scenes7/true/3/' + file_name[2]
# res, natural_cut_img, nose,face = get_difference_graph(
#     natural_light_img_path='flash_live/flash_live2/flash_live_bright2_light1_scenes7/true/3/69.jpg',
#     light_source_img_path='flash_live/flash_live2/flash_live_bright2_light1_scenes7/true/3/72.jpg')
# window_name = 'resBGR'
# cv2.imshow(window_name, res)
# cv2.imwrite(result_img_path, res)
# cv2.waitKey(0)
