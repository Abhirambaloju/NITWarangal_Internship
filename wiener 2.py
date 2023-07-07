import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import cv2
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def blur(img, kernel_size = 3):
	dummy = np.copy(img)
	h = np.eye(kernel_size) / kernel_size
	dummy = convolve2d(dummy, h, mode = 'valid')
	return dummy

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy


if __name__ == '__main__':
    file_name = os.path.join('lena512.jpg') 
    img= rgb2gray(cv2.imread(file_name))
    blurred_img = blur(img, kernel_size = 15)
    kernel = blur(img,3)
    filtered_img = wiener_filter(img, kernel, K = 10)
    display = [img, blurred_img, filtered_img]
    label=['Original Image','Motion Blurred Image','Wiener Filter applied']
    fig = plt.figure(figsize=(12, 10))
# cv2.imshow('img', filtered_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    for i in range(len(display)):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])

    plt.show()
