import cv2
from matplotlib import pyplot as plt
import numpy as np
image1 = cv2.imread('mamo.jpg')
rows = 1
columns = 2
image2 = 255 - image1
plt.subplot(rows,columns,1),plt.imshow(image1)
plt.subplot(rows,columns,2),plt.imshow(image2)
plt.show()

#cv2.imshow('Ngetivity',image1)
cv2.waitKey(0)
cv2.destoryAllWindows()