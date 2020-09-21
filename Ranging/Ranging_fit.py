from sklearn import linear_model
import matplotlib.pyplot as plt  # 用于作图
import numpy as np  # 用于创建向量
from scipy.optimize import curve_fit
data_path = 'data/data_w.txt'
data = []
with open(data_path, 'r') as f:
    for line in f.readlines():
        item = list(map(float,line.split(" ")))
        data.append(item)

def linear_regression(data):
    reg = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    data= np.array(data)
    data = data[:,0:2]
    data = list(data)
    y = [34.41,41.34,49.61,59.59,71.59,86.02,86.02,102.56,124.03,122.01,148.15,148.15,148.15]
    reg.fit(data, y)
    k = reg.coef_  # 获取斜率w1,w2,w3,...,wn
    b = reg.intercept_  # 获取截距w0
    predict_data = [[310,310]]
    out = reg.predict(data)
    print(out)
    # x0 = np.arange(0, 10, 0.2)
    # y0 = k * x0 + b
    data = np.array(data)
    plt.scatter(data[:,0], y)
    plt.scatter(data[:,0], out)
    plt.show()
    #plt.plot(x0, y0)

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


#
def exp_fit(data):
    xdata = []
    for i in range(len(data)):
        item = data[i][0] * data[i][1]
        xdata.append(item)
    xdata = np.array(xdata)
    data = np.array(data)
    ydata = data[:,2]
    popt, pcov = curve_fit(func, xdata, ydata)
    print(popt)
    yp = func(xdata,popt[0],popt[1],popt[2])
    plt.plot(xdata, yp, label="Fitted Curve")
    plt.legend()
    plt.show()

def one_variable_multi_order_fitting(data):
    xdata = []
    for i in range(len(data)):
        item = data[i][0] * data[i][1]
        xdata.append(item)
    xdata = np.array(xdata)
    data = np.array(data)
    ydata = data[:, 2]
    # coef 为系数，poly_fit 拟合函数
    coef1 = np.polyfit(xdata, ydata, 1)
    poly_fit1 = np.poly1d(coef1)
    plt.plot(xdata, poly_fit1(xdata), 'g', label="one_order")
    print(poly_fit1)

    coef2 = np.polyfit(xdata, ydata, 2)
    poly_fit2 = np.poly1d(coef2)
    plt.plot(xdata, poly_fit2(xdata), 'b', label="two_order")
    print(poly_fit2)

    coef3 = np.polyfit(xdata, ydata, 3)
    poly_fit3 = np.poly1d(coef3)
    plt.plot(xdata, poly_fit3(xdata), 'y', label="three_order")
    print(poly_fit3)

    coef4 = np.polyfit(xdata, ydata, 4)
    poly_fit4 = np.poly1d(coef4)
    plt.plot(xdata, poly_fit4(xdata), 'k', label="four_order")
    print(poly_fit4)

    coef5 = np.polyfit(xdata, ydata, 5)
    poly_fit5 = np.poly1d(coef5)
    plt.plot(xdata, poly_fit5(xdata), 'r:', label="five_order")
    print(poly_fit5)

    plt.scatter(xdata, ydata, color='black')
    plt.legend(loc=1)
    plt.show()


one_variable_multi_order_fitting(data)