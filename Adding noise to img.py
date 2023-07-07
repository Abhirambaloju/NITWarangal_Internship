import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(img):
    row,col = img.shape
    number_of_pixels = random.randint(300,10000)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img

img = cv2.imread('mamo.jpg',cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('mamo.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('mamo1.jpg',add_noise(img))


gauss = cv2.GaussianBlur(img, (3,3), 3)
median = cv2.medianBlur(img,5)
kernal=np.ones((5,5))/25
blur=cv2.filter2D(img,-1,kernal)

cv2.imshow('original img',img1)
cv2.imshow('Noise img',img)
cv2.imshow('Median',median)
cv2.imshow('Gaussian',gauss)
cv2.imshow('Mean',blur)

cv2.waitKey(0)
cv2.destroyAllWindows()