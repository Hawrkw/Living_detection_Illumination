#from sympy import *
import sympy.integrals.trigonometry
from sklearn import preprocessing
import cv2
import numpy as np
import random
import time
from FourierTransform import *
import os
#x是法线向量,类型为3*1的矩阵--我们需要解出来
#x = np.random
#alpha---反射率--一个实数--未知数
#alpha*A*x = B




#compute_with_metrix()
####
#先不用矩阵操作---没找到方法----直接解方程组
######
from sympy import symbols, nonlinsolve


def method1():

    start_time = time.time()
    for i in range(1 * 1):
        print(i)
        alpha = symbols('alpha')
        #不同差分图像相同像素点的反射率相同，法线相同，不同像素点则不同
        x,y,z = symbols('x,y,z')
        #不同图像相同像素点的光照方向不同
        #第一光照方向
        a1x = random.randint(0,10)
        a1y = random.randint(0,10)
        a1z = random.randint(0,10)
        #第二光照方向
        a2x = random.randint(0,10)
        a2y = random.randint(0,10)
        a2z = random.randint(0,10)
        #第三光照方向
        a3x = random.randint(0,10)
        a3y = random.randint(0,10)
        a3z = random.randint(0,10)
        #像素值
        q1 = random.randint(0,255)
        q2 = random.randint(0,255)
        q3 = random.randint(0,255)
        #方程组
        f1 = alpha * a1x * x + alpha * a1y * y + alpha * a1z * z - q1
        f2 = alpha * a2x * x + alpha * a2y * y + alpha * a2z * z - q2
        f3 = alpha * a3x * x + alpha * a3y * y + alpha * a3z * z - q3
        f4 = x * x + y * y + z * z - 1

        result = nonlinsolve([f1,f2,f3,f4],[alpha,x,y,z])

        print(result.args[0][1] * result.args[0][1] + result.args[0][2] * result.args[0][2] +
              result.args[0][3] * result.args[0][3])
    end_time = time.time()

    print(end_time - start_time)

###对像素点一个一个求，对矩阵求逆
def method2_1(A,y):
    #光照
    #A = np.matrix(A)
    y = y.reshape((3,1))
    a_inverse = A.I
    x = np.dot(a_inverse, y)
    # 求模长--反射率
    alpha = np.linalg.norm(x)
    #法线
    N = x / alpha
    print('1:' + str(alpha))
    print(alpha*np.dot(A,N))

#矩阵求逆法-将数据扩展到整个图像上
#该方法认为所有像素点的光照方向一致
#方法应该传入三个差分图像以及三个光照方向形成的3*3矩阵，返回一个w*h*3的法线图以及一个w*h的反射率图
def method2_2(A,q):
    start_time = time.time()
    q = q.reshape((q.shape[0],q.shape[1],3,1))
    #q = (q - np.min(q)) / (np.max(q) - np.min(q))
    #q = np.random.randint(0, 255, (480,640,3,1))
    a_inverse = A.I
    a_expand = np.expand_dims(a_inverse,0).repeat(q.shape[0],axis=0)
    a_expand = np.expand_dims(a_expand, 1).repeat(q.shape[1], axis=1)
    # "2维以上"的尺寸必须完全对应相等；
    # "2维"具有实际意义的单位，只要满足矩阵相乘的尺寸规律即可。
    x  = np.matmul(a_expand,q)

    #求模长
    alpha = np.linalg.norm(x,axis=2)

    alpha = alpha.reshape((q.shape[0],q.shape[1],1))
    alpha_reciprocal = (1 / alpha).repeat(3,axis=2)
    alpha_reciprocal_expand = np.expand_dims(alpha_reciprocal,2)
    N = np.matmul(x,alpha_reciprocal_expand)
    #[1,2]表示1，2列
    N = np.delete(N,[1,2],axis=3)
    N = N.reshape((q.shape[0],q.shape[1],3))

    N[np.isnan(N)] = 1
    #N = N * 100
    # N[N < 0] = 0
    #N = N.astype(np.uint8)
    #N = (N - np.min(N)) / (np.max(N) - np.min(N))
    #N = np.ceil(N)

    #from PIL import Image
    #import matplotlib.pyplot as plt
    # after_img = tf.clip_by_value(N, 0.0, 1.0)

    # im = Image.fromarray(N)
    # im.save("faxian.jpg")

    alpha = alpha.reshape((q.shape[0],q.shape[1]))
    alpha[alpha < 0] = 0
    alpha[alpha > alpha.max()] = alpha.max()
    alpha  /= alpha.max()
    # M = np.ones((q.shape[0],q.shape[1])) * alpha.max
    # alpha = cv2.subtract(M, alpha)
    # plt.subplot(231), plt.imshow(alpha, 'gray'), plt.title('Reflectivity')
    # plt.subplot(232), plt.imshow(N, 'gray'), plt.title('Normal')

    #plt.savefig('result/flash_live_bright_scenes1/true/3/c_bright_true_result.jpg')
    end_time = time.time()
    print("平行光光照计算时间：{}".format(end_time - start_time))
    cv2.imshow('Reflectivity',alpha)
    cv2.imshow('Normal',N)
    #cv2.imwrite('faxian.jpg',N)

    # im = im.astype('float64')
    #cv2.imshow('fa',im)
    cv2.waitKey(1000)

    return alpha,N

#该方法认为所有像素点的光照方向不一致
#方法应该传入三个差分图像以及所有光照方向形成的q.shape[0]*q.shape[1]*3*3矩阵，返回一个w*h*3的法线图以及一个w*h的反射率图
def method2_3(A,q):
    #start_time = time.time()
    q = q.reshape((q.shape[0],q.shape[1],3,1))
    #q = np.random.randint(0, 255, (480,640,3,1))

    # "2维以上"的尺寸必须完全对应相等；
    # "2维"具有实际意义的单位，只要满足矩阵相乘的尺寸规律即可。
    x  = np.matmul(A,q)

    #求模长
    alpha = np.linalg.norm(x,axis=2)

    alpha = alpha.reshape((q.shape[0],q.shape[1],1))
    alpha_reciprocal = (1 / alpha).repeat(3,axis=2)
    alpha_reciprocal_expand = np.expand_dims(alpha_reciprocal,2)
    N = np.matmul(x,alpha_reciprocal_expand)
    #[1,2]表示1，2列
    N = np.delete(N,[1,2],axis=3)
    N = N.reshape((q.shape[0],q.shape[1],3))

    N[np.isnan(N)] = 1
    #N = N * 100
    # N[N < 0] = 0
    #N = N.astype(np.uint8)
    #N = (N - np.min(N)) / (np.max(N) - np.min(N))
    #N = np.ceil(N)

    #from PIL import Image
    #import matplotlib.pyplot as plt
    # after_img = tf.clip_by_value(N, 0.0, 1.0)

    # im = Image.fromarray(N)
    # im.save("faxian.jpg")

    alpha = alpha.reshape((q.shape[0],q.shape[1]))
    alpha[alpha < 0] = 0
    alpha[alpha > alpha.max()] = alpha.max()
    alpha  /= alpha.max()
    # M = np.ones((q.shape[0],q.shape[1])) * alpha.max
    # alpha = cv2.subtract(M, alpha)
    # plt.subplot(231), plt.imshow(alpha, 'gray'), plt.title('Reflectivity')
    # plt.subplot(232), plt.imshow(N, 'gray'), plt.title('Normal')

    #plt.savefig('result/flash_live_bright_scenes1/true/3/c_bright_true_result.jpg')
    # end_time = time.time()
    # print(end_time - start_time)
    cv2.imshow('Reflectivity_all',alpha)
    cv2.imshow('Normal_all',N)
    #cv2.imwrite('faxian.jpg',N)

    # im = im.astype('float64')
    #cv2.imshow('fa',im)
    cv2.waitKey(1000)

    return alpha,N
#用qr分解方法
def method3(A,y):
    start_time = time.time()

    q,r = np.linalg.qr(A)

    r_T = r.T

    r_T_inverse = np.matrix(r_T).I
    r_inverse = r.I
    z = np.dot(r_T_inverse,y)
    alpha = np.linalg.norm(z)

    p = z / alpha
    # 法线
    N = np.dot(q,p)

    end_time = time.time()
    print('2:' + str(alpha))
    #print(N)
    print("验证：")
    res = alpha * np.dot(N.T,A)
    print(res)
    print(y)
    print(np.dot(q,r_T_inverse))
    print(A.I.T)
    #print(end_time - start_time)
def get_difference_image(paths:list):
    left_nature_img = cv2.imread(paths[0], 0)
    #cv2.imshow('left1',left_nature_img)
    image_shape = left_nature_img.shape
    # (1)在python中有歧义，如果一个元素要表示为元组则需要加,
    image_shape_expand = image_shape + (1,)
    image_shape_expand = image_shape + (1,)
    left_nature_img_expand = left_nature_img.reshape(image_shape_expand)
    top_nature_img = cv2.imread(paths[1], 0)
    top_nature_img_expand = top_nature_img.reshape(image_shape_expand)

    right_nature_img = cv2.imread(paths[2], 0)
    right_nature_img_expand = right_nature_img.reshape(image_shape_expand)
    q = np.concatenate((left_nature_img_expand, top_nature_img_expand, right_nature_img_expand), axis=2)
    return q

def verification(alpha,N,a1):
    a1_expand = np.expand_dims(a1, 0).repeat(alpha.shape[0], axis=0)
    a1_expand = np.expand_dims(a1_expand, 1).repeat(alpha.shape[1], axis=1)
    N_reshape = N.reshape((N.shape[0],N.shape[1],N.shape[2],1))
    alpha_reshape = alpha.reshape((alpha.shape[0],alpha.shape[1],1,1))

    res = np.matmul(a1_expand,N_reshape)

    res = np.matmul(alpha_reshape,res)
    res = res.reshape(res.shape[0],res.shape[1])
    #res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    cv2.imshow('left', res)
    cv2.waitKey(0)
if __name__ == '__main__':
    #计算差分图像，应传入两幅图像路径，返回可以是空，也可以是bool
    #计算法线图与反射率图，传入光照方向-3*3矩阵，传入三个差分图像--或者一个像素一个像素传入
    # 光照
    # A = np.random.randint(0, 255, (3, 3))
    # A = [[35, 20, 12],
    #      [35, 0, 0],
    #      [35,-12, 12]]
    #人脸鼻尖位置：wu_bright_true:[54,0,41],yang_bright_true:[54,0,35],chen_bright_true:[54,0,40]
    #wu_bright_fake:[10,0,30],yang_bright_fake:[10,0,31],chen_bright_fake:[10,0,32]
    #光源位置:left:[0,-22.5,13],top:[0,0,29.5],right:[0,20.5,13]



    # A_T = A.T
    # 像素
    # y = np.random.randint(0, 255, (3, 1))
    # method2_1(A, y)
    # method3(A_T,y)
    # "result/flash_live_fake/1/left-nature-fake.jpg",
    # "result/flash_live/亮/场景1/1/left-nature-true.jpg",
    bright_scenes1 = ["result/flash_live_bright_scenes1/true/1",
                            "result/flash_live_bright_scenes1/true/2",
                            "result/flash_live_bright_scenes1/true/3",
                            "result/flash_live_bright_scenes1/fake/1",
                            "result/flash_live_bright_scenes1/fake/2",
                            "result/flash_live_bright_scenes1/fake/3"]

    bright_scenes2 = ["result/flash_live_bright_scenes2/true/1",
                            "result/flash_live_bright_scenes2/true/2",
                            "result/flash_live_bright_scenes2/true/3",
                            "result/flash_live_bright_scenes2/fake/1",
                            "result/flash_live_bright_scenes2/fake/2",
                            "result/flash_live_bright_scenes2/fake/3",
                            "result/flash_live_bright_scenes2/fake/4",
                            "result/flash_live_bright_scenes2/fake/5"]
    bright_scenes3 = ["result/flash_live_bright_scenes3/true/3",
                      "result/flash_live_bright_scenes3/fake/3",
                      "result/flash_live_bright_scenes3/fake/4",
                      "result/flash_live_bright_scenes3/fake/5"]
    dark_scenes2 = ["result/flash_live_dark_scenes2/true/1",
                    "result/flash_live_dark_scenes2/true/2",
                    "result/flash_live_dark_scenes2/true/3",
                    "result/flash_live_dark_scenes2/fake/4",
                    "result/flash_live_dark_scenes2/fake/5"]

    # left_nature_img = cv2.imread(left_nature_img_path[4], 0)
    # image_shape = left_nature_img.shape
    # #(1)在python中有歧义，如果一个元素要表示为元组则需要加,
    # image_shape_expand = image_shape + (1,)
    # left_nature_img_expand = left_nature_img.reshape(image_shape_expand)
    # top_nature_img_path = ["result/flash_live_fake/1/top-nature-fake.jpg",
    #                         "result/flash_live/亮/场景1/1/top-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/true/1/top-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/true/2/top-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/true/3/top-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/fake/1/top-nature-fake.jpg",
    #                         "result/flash_live_bright_scenes1/fake/2/top-nature-fake.jpg",
    #                         "result/flash_live_bright_scenes1/fake/3/top-nature-fake.jpg"]

    # top_nature_img = cv2.imread(top_nature_img_path[4], 0)
    # top_nature_img_expand = top_nature_img.reshape(image_shape_expand)
    # right_nature_img_path = ["result/flash_live_fake/1/right-nature-fake.jpg",
    #                        "result/flash_live/亮/场景1/1/right-nature-true.jpg",
    #                        "result/flash_live_bright_scenes1/true/1/right-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/true/2/right-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/true/3/right-nature-true.jpg",
    #                         "result/flash_live_bright_scenes1/fake/1/right-nature-fake.jpg",
    #                         "result/flash_live_bright_scenes1/fake/2/right-nature-fake.jpg",
    #                         "result/flash_live_bright_scenes1/fake/3/right-nature-fake.jpg"]

    # right_nature_img = cv2.imread(right_nature_img_path[4], 0)
    # right_nature_img_expand = right_nature_img.reshape(image_shape_expand)
    # q = np.concatenate((left_nature_img_expand,top_nature_img_expand,right_nature_img_expand),axis=2)

    path = bright_scenes2[3]
    image_names = os.listdir(path)
    image_pathes = [""] * 3
    for j in range(len(image_names)):
        image_name = image_names[j]
        if image_name == 'left-nature-fake.jpg' or image_name == 'left-nature-true.jpg':
            image_pathes[0] = (os.path.join(path, image_name))
        elif image_name == 'top-nature-fake.jpg' or image_name == 'top-nature-true.jpg':
            image_pathes[1] = (os.path.join(path, image_name))
        elif image_name == 'right-nature-fake.jpg' or image_name == 'right-nature-true.jpg':
            image_pathes[2] = (os.path.join(path, image_name))
        elif image_name[-3:] == 'txt':
            A = []
            with open(os.path.join(path, image_name), 'r') as f:
                for line in f.readlines():
                    A.append(list(map(float, line.split())))
            f.close()
    A_norm = np.linalg.norm(A,axis=1,keepdims=True)
    A = A / A_norm
    A = np.matrix(A)


    #paths2.extend([left_nature_img_path, top_nature_img_path, right_nature_img_path])

    q1 = get_difference_image(image_pathes)
    #q2 = get_difference_image()
    #method2_1(A,q1[100][100])
    alpha,N = method2_2(A,q1)
    a1 = A[0]
    verification(alpha,N,a1)
    #alpha2 = method2_2(A,q2)
    #get_fourier_transform(alpha1,alpha2)
#compute()