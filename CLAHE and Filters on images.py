import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('moon1.tif',0)
median = cv2.medianBlur(img,1)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
cl1 = clahe.apply(median)

#cv2.imwrite('GRAY SCALE.jpg',img)
cv2.imshow('original',img)
cv2.imshow('clahe img with median fil',cl1)

cl2 = clahe.apply(img)
cv2.imshow('clahe img with out median filter',cl2)
cv2.waitKey(0)
cv2.destroyAllWindows()