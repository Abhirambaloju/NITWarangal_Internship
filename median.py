import cv2 
import numpy as np
from skimage.util import random_noise

noise_img=cv2.imread("img.png")

# Load an image
#im_arr = cv2.imread("D:/downloads/opencv_logo.PNG")

# Add salt and pepper noise to the image
#noise_img = random_noise(im_arr, mode="s&p",amount=0.3)
#noise_img = np.array(255*noise_img, dtype = 'uint8')

# Apply median filter
#median=cv.medianBlur(noise_img,3)
median = cv2.medianBlur(noise_img,5)

# Display the image
cv2.imshow('blur',noise_img)
cv2.imshow('blur1',median)
# images = np.concatenate((noise_img,median), axis=1)
# cv2.imshow('img',images)
cv2.waitKey(0)
cv2.destroyAllWindows()


# a=median-noise_img
# b=a*a
# mse=b.sum()/828054


# from sklearn.metrics import mean_squared_error
