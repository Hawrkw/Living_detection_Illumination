import numpy as np
import os
import cv2
import dlib
class FaceDataset:
    def __init__(self,root,transform = None,is_test = False):
        self.root = root
        #初步构想是以voc数据格式来进行读取，数据分为true与fake，fake下一个文件夹里包含四张图片
        # 每个人不同场景的文件夹中包括图像以及label，label中有光照方向以及真假类别
        #true--renming-changjing--
        self.transform = transform
        if is_test:
            image_sets_file = os.path.join(self.root,'test.txt')
        else:
            image_sets_file = os.path.join(self.root, 'train.txt')
        self.ids = FaceDataset.read_image_ids(image_sets_file)

        # 还有一个问题-label--写在txt里边？
    # 数据获取流程-通过item获取图片文件夹路径-读取图片文件夹中的图片以及光照方向-处理读取的数据得到差分图+法线图+反射率图——将其concat
    def __getitem__(self, item):
        # image_id = dataset/true/renming/changjing image_id是文件夹路径，通过该路径读取该文件夹下不同光照图像路径
        image_id = self.ids[item]
        left_light_source_path = os.path.join(image_id, 'left_light_source.jpg')
        # 获取差分图
        natural_light_path = os.path.join(image_id, 'natural_light.jpg')
        left_difference_graph = self.get_difference_graph(natural_light_path, left_light_source_path)
        top_light_source_path = os.path.join(image_id, 'top_light_source.jpg')
        top_difference_graph = self.get_difference_graph(natural_light_path, top_light_source_path)
        right_light_source_path = os.path.join(image_id, 'right_light_source.jpg')
        right_difference_graph = self.get_difference_graph(natural_light_path, right_light_source_path)
        #差分图的图像shape是（h*w）--需要转换为（h*w*channels）-该步骤是为了将3个差分图concate,所以要扩展一个维度
        # (1)在python中有歧义，如果一个元素要表示为元组则需要加,
        difference_graph_shape = left_difference_graph.shape + (1,)
        left_difference_graph = left_difference_graph.reshape(difference_graph_shape)
        top_difference_graph = top_difference_graph.reshape(difference_graph_shape)
        right_difference_graph = right_difference_graph.reshape(difference_graph_shape)
        #将差分图concate起来
        difference_graph = np.concatenate((left_difference_graph,top_difference_graph,right_difference_graph),axis=2)
        # 通过差分图与光照方向-平行光照-获取法线图与反射率图
        #训练阶段，光照方向是测量好的距离从txt文件读取，实际中光照方向需要通过测距来进行获得定位
        light_direction_path = os.path.join(image_id, 'light_direction.txt')
        #获取光照方向
        light_direction = []
        with open(light_direction_path, 'r') as f:
            for line in f.readlines():
                light_direction.append(list(map(float, line.split())))
        f.close()
        #对光照方向进行正则化
        light_direction_norm = np.linalg.norm(light_direction, axis=1, keepdims=True)
        light_direction = light_direction / light_direction_norm
        light_direction = np.matrix(light_direction)
        #normal_graph_shape-(h*w*3) reflectance_graph_shape-(h*w)
        normal_graph,reflectance_graph = self.get_normal_reflectance_graph(light_direction,difference_graph)
        #将差分图-法线图-反射率图concate起来
    #获取差分图以及法线图，反射率图的方法不应该写在这里-因为测试也需要用到这两个方法
    #人脸框后边不用dlib的人脸检测方法来获取
    def get_difference_graph(natural_light_img_path, light_source_img_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # cv2读取图像
        # 如果参数是str类型，则为图片路径，否则则是传入图片--这里应该做bug处理
        if type(natural_light_img_path) == str and type(light_source_img_path) == str:
            # 自然光图像
            natural_light_img = cv2.imread(natural_light_img_path)
            # 左光源图像是说光源的光打在左脸上
            light_source_img = cv2.imread(light_source_img_path)
        else:
            natural_light_img = natural_light_img_path
            # 左光源图像是说光源的光打在左脸上
            light_source_img = light_source_img_path

        # 图片转换为灰度图像
        natural_light_img_gray = cv2.cvtColor(natural_light_img, cv2.COLOR_RGB2GRAY)

        light_source_img_gray = cv2.cvtColor(light_source_img, cv2.COLOR_RGB2GRAY)

        # 我们的图像大小以自然光为准，带光源图像也要resize成自然光图像大小---暂时这么确定
        # opencv读出的图像（宽，高）
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
        # order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序

        # 通过人脸68个关键点计算单应矩阵
        h, mask = cv2.findHomography(light_source_landmarks, natural_light_landmarks, cv2.RANSAC)
        # warped是对齐后的图像
        warped = cv2.warpPerspective(light_source_img_gray, h, (natural_light_img_shape[1], natural_light_img_shape[0]))
        # warped = np.array(warped,dtype='int32')
        # natural_light_img_gray = np.array(natural_light_img_gray,dtype='int32')
        M = np.ones(natural_light_img_shape, dtype="uint8") * 255
        res = cv2.subtract(warped, natural_light_img_gray)

        # res = cv2.subtract(M,res)
        #res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        res = np.maximum(res, 0)

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

        res = res[y_left_expand:y_right_expand, x_left_expand:x_right_expand]
        return res

    def get_normal_reflectance_graph(A, q):
        q = q.reshape((q.shape[0], q.shape[1], 3, 1))
        # q = (q - np.min(q)) / (np.max(q) - np.min(q))
        # q = np.random.randint(0, 255, (480,640,3,1))
        a_inverse = A.I
        a_expand = np.expand_dims(a_inverse, 0).repeat(q.shape[0], axis=0)
        a_expand = np.expand_dims(a_expand, 1).repeat(q.shape[1], axis=1)
        # "2维以上"的尺寸必须完全对应相等；
        # "2维"具有实际意义的单位，只要满足矩阵相乘的尺寸规律即可。
        x = np.matmul(a_expand, q)

        # 求模长
        alpha = np.linalg.norm(x, axis=2)

        alpha = alpha.reshape((q.shape[0], q.shape[1], 1))
        alpha_reciprocal = (1 / alpha).repeat(3, axis=2)
        alpha_reciprocal_expand = np.expand_dims(alpha_reciprocal, 2)
        N = np.matmul(x, alpha_reciprocal_expand)
        # [1,2]表示1，2列
        N = np.delete(N, [1, 2], axis=3)
        N = N.reshape((q.shape[0], q.shape[1], 3))

        N[np.isnan(N)] = 1
        # N = N * 100
        # N[N < 0] = 0
        # N = N.astype(np.uint8)
        # N = (N - np.min(N)) / (np.max(N) - np.min(N))
        # N = np.ceil(N)

        # from PIL import Image
        # import matplotlib.pyplot as plt
        # after_img = tf.clip_by_value(N, 0.0, 1.0)

        # im = Image.fromarray(N)
        # im.save("faxian.jpg")

        alpha = alpha.reshape((q.shape[0], q.shape[1]))
        alpha[alpha < 0] = 0
        alpha[alpha > alpha.max()] = alpha.max()
        alpha /= alpha.max()

        return alpha, N

    @staticmethod
    def read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids
