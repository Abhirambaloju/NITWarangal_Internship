import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    
    print('MSE=',mse)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# def add_noise(img):
#     row,col = img.shape
#     number_of_pixels = random.randint(300,10000)
#     for i in range(number_of_pixels):
#         y_coord=random.randint(0, row - 1)
#         x_coord=random.randint(0, col - 1)
#         img[y_coord][x_coord] = 255
#     for i in range(number_of_pixels):
#         y_coord=random.randint(0, row - 1)
#         x_coord=random.randint(0, col - 1)
#         img[y_coord][x_coord] = 0
#     return img

img = cv2.imread('mamo1.jpg',cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread('mamo.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('mamo12.jpg',add_noise(img))


gauss = cv2.GaussianBlur(img, (3,3), 3)
median = cv2.medianBlur(img,3)
kernal=np.ones((5,5))/25
blur=cv2.filter2D(img,-1,kernal)


cv2.imwrite('mamo21.jpg',median)
cv2.imwrite('mamo22.jpg',gauss)
cv2.imwrite('mamo23.jpg',blur)

value = PSNR(img, median)
print("PSNR for original - median=",value)

value = PSNR(img, gauss)
print("PSNR for original - gauss=",value)

value = PSNR(img, blur)
print("PSNR for original - blur=",value)



# MSE =np.square(np.subtract(img,median)).mean()
# print('Mse original - median=',MSE)

# MSE1=np.square(np.subtract(img,gauss)).mean()
# print('Mse original - Gauss=',MSE1)

# MSE2=np.square(np.subtract(img,blur)).mean()
# print('Mse original - Mean=',MSE2)

cv2.imshow('original img',img)
# cv2.imshow('Noise img',img)
cv2.imshow('Median',median)
cv2.imshow('Gaussian',gauss)
cv2.imshow('Mean',blur)

cv2.waitKey(0)
cv2.destroyAllWindows()