import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('mamo.jpg',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cl = clahe.apply(cl1)
#c=clahe.apply(cl)
#cv2.imshow('3 stage clahe img',c)
#cv2.imwrite('GRAY SCALE.jpg',img)
cv2.imshow('original',img)
cv2.imshow('clahe img',cl1)
cv2.imshow('2 stage clahe img',cl)
cv2.waitKey(0)
cv2.destroyAllWindows()