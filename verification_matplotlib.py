import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_normal_direction_point(N):


    x = N[:,:,0]
    y = N[:,:,1]
    z = N[:,:,2]
    fig = plt.figure()
    ax = Axes3D(fig)
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         ax.plot(x[i][j],y[i][j],z[i][j])
    ax.scatter(x,y,z)
    ax.set_zlabel('Z',fontdict={'size':30,'color':'red'})
    ax.set_ylabel('Y', fontdict={'size': 30, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 30, 'color': 'red'})
    plt.show()

def draw_normal_direction_surface(N):
    x = N[:, :, 0]
    y = N[:, :, 1]
    z = N[:, :, 2]
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z, alpha=0.3, cmap='winter')  # 生成表面， alpha 用于控制透明度
    # ax.contour(x, y, z, zdir='z', offset=-3, cmap="rainbow")  # 生成z方向投影，投到x-y平面
    # ax.contour(x, y, z, zdir='x', offset=-6, cmap="rainbow")  # 生成x方向投影，投到y-z平面
    # ax.contour(x, y, z, zdir='y', offset=6, cmap="rainbow")  # 生成y方向投影，投到x-z平面
    plt.show()
def draw_hist(N,color,title,save_path):
    N = N * 100
    #N = np.fabs(N)
    #N = np.array(N,dtype=np.uint8)
    # b_hist = cv2.calcHist([N], [0], None, [10], [0, 100])
    # g_hist = cv2.calcHist([N], [1], None, [10], [0, 100])
    # r_hist = cv2.calcHist([N], [2], None, [10], [0, 100])
    #
    # # 显示3个通道的颜色直方图
    # plt.plot(b_hist, label='B', color='blue')
    # plt.plot(g_hist, label='G', color='green')
    # plt.plot(r_hist, label='R', color='red')
    # plt.legend(loc='best')
    # plt.xlim([0, 10])
    # plt.show()
    # 均值
    print("均值：" + str(np.mean(N.flatten())))
    # 方差
    print("方差：" + str(np.var(N.flatten())))
    # 标准差
    print("标准差：" + str(np.std(N.flatten())))
    bins = np.arange(-100, 101, 10)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # 直方图会进行统计各个区间的数值
    #flatten()二维数组转为一维数组
    plt.hist(N.flatten(), bins, color=color, alpha=0.9)  # alpha设置透明度，0为完全透明
    plt.xlabel('scores: M:' + str(round(np.mean(N.flatten()),2)) + ' V:' + str(round(np.var(N.flatten()),2))
               + ' S:' + str(round(np.std(N.flatten()),2)))
    plt.ylabel('count')
    plt.title(title)
    plt.xlim(-100, 100)  # 设置x轴分布范围
    plt.savefig(save_path)
    plt.show()

def draw_normal_direction_line(N,natrual_light_img_path,face_area,save_path):
    fig = plt.figure()
    ax = Axes3D(fig)
    #N = N * 50
    x = N[:, :, 0]
    y = N[:, :, 1]
    z = N[:, :, 2]
    from PIL import Image
    #Image.open的读取模式是rgb，cv2.imread是bgr
    natrual_light_img = cv2.imread(natrual_light_img_path)
    natrual_light_img = natrual_light_img[face_area[1]:face_area[3],face_area[0]:face_area[2],:]
    # difference_graph = Image.open(difference_graph_path)
    # difference_graph = difference_graph.crop(tuple(face_area))
    #Image._show(difference_graph)
    for i in range(natrual_light_img.shape[0]):
        for j in range(natrual_light_img.shape[1]):
            if i % 5 == 0 and j % 5 == 0:
                b = hex(natrual_light_img[i][j][0])[2:]
                g = hex(natrual_light_img[i][j][1])[2:]
                r = hex(natrual_light_img[i][j][2])[2:]
                if len(r) == 1:
                    r = '0' + r
                if len(b) == 1:
                    b = '0' + b
                if len(g) == 1:
                    g = '0' + g
                col = '#' + r + g + b
                ax.scatter(i,j,0,c=col,alpha=0.5)
                point = (y[i][j] + i,z[i][j] + j,-x[i][j])
                line = zip(point, (i, j, 0))
                ax.plot3D(*line,
                          ls=':',  #
                          c='g',  # line color
                          marker='o',  # 标记点符号
                          mfc='r',  # marker facecolor
                          # mec='',  # marker edgecolor
                          ms=5,  # mfc size
                          )
    plt.savefig(save_path)
    plt.show()

def face_size_distance(data):
    data = np.array(data)
    fig = plt.figure()
    #ax = Axes3D(fig)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    #ax.scatter(x, y, z, c='r', label='顺序点')
    # 添加坐标轴(顺序是Z, Y, X)
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.plot(z, x, marker='o', mec='r', mfc='w')
    plt.savefig('face_size_distance_c.png')
    plt.show()

data_path = 'Ranging/data/data_c.txt'
data = []
with open(data_path, 'r') as f:
    for line in f.readlines():
        item = list(map(float,line.split(" ")))
        data.append(item)


face_size_distance(data)
