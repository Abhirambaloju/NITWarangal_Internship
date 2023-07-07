import cv2
from matplotlib import pyplot as plt
import numpy as np
image1 = cv2.imread('mdb003ll.pgm')
#image1 = cv2.imread('mdb003ll.pgm')

grayimage=cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
print(grayimage.shape)


[threshold,binaryimage]=cv2.threshold(grayimage,127,255,cv2.THRESH_BINARY)
print(binaryimage.shape)

#Image negitivity
image2 = 255 - image1

gauss = cv2.GaussianBlur(image1, (3,3), 3)
median = cv2.medianBlur(image1,5)
kernal=np.ones((5,5))/25
blur=cv2.filter2D(image1,-1,kernal)

plt.subplot(3,3,1),plt.imshow(image1),plt.title('Original image')
plt.subplot(3,3,2),plt.imshow(grayimage),plt.title('grayimage')
plt.subplot(3,3,3),plt.imshow(image2),plt.title('Negitivity')
plt.subplot(3,3,4),plt.imshow(binaryimage),plt.xlabel('Binary ')
plt.subplot(3,3,5),plt.imshow(gauss),plt.xlabel('gauss')
plt.subplot(3,3,6),plt.imshow(median),plt.xlabel('median')
plt.subplot(3,3,7),plt.imshow(blur),plt.xlabel('Mean')
plt.show()

