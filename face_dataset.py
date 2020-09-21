import numpy as np
import os
import cv2
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
    def __getitem__(self, item):
        #image_id = dataset/true/renming/changjing image_id是个文件夹，通过文件夹里的图像获得差分图
        image_id = self.ids[item]
        #

    def get_difference_graph(natural_light_img_path, light_source_img_path):
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
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
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

        res = res[y_left_expand:y_right_expand, x_left_expand:x_right_expand, :]
        return res

    @staticmethod
    def read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids
