import cv2
import numpy as np
from matplotlib import pyplot as plt

image=cv2.imread('mamo.jpg',0)
cv2.imshow('Original img',image)

clahe = cv2.createCLAHE(clipLimit=5.0)
cl1 = clahe.apply(image)

histogram_img=cv2.equalizeHist(image)

cv2.imshow('Clahe img',cl1)
cv2.imshow('Histogram equalization img',histogram_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
