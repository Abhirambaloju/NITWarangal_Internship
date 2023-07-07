import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mdb003ll.pgm')
kernal_sharpening = np.array([[-1,-1,-1,-1,-1],
                              [-1,-1,-1,-1,-1],
                              [-1,-1,25,-1,-1],
                              [-1,-1,-1,-1,-1],
                              [-1,-1,-1,-1,-1]])

sharpened = cv2.filter2D(img,-1,kernal_sharpening)
plt.subplot(1,2,1),plt.imshow(img)

plt.subplot(1,2,2),plt.imshow(sharpened)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

