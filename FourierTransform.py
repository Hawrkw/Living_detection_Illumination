import cv2
import numpy as np
import matplotlib.pyplot as plt
def get_fourier_transform(img1,img2):

    img_true = img1
    img_fake = img2
    f_true = np.fft.fft2(img_true)
    f_fake = np.fft.fft2(img_fake)
    f_shift_true = np.fft.fftshift(f_true)
    f_shift_fake = np.fft.fftshift(f_fake)
    res_true = np.log(np.abs(f_shift_true))
    plt.subplot(231),plt.imshow(res_true,'gray'),plt.title('res_true')
    res_fake = np.log(np.abs(f_shift_fake))
    plt.subplot(232),plt.imshow(res_fake,'gray'),plt.title('res_fake')
    plt.show()
# cv2.imshow('res_true',res_true)
# cv2.imshow("res_fake",res_fake)
# cv2.waitKey(0)

