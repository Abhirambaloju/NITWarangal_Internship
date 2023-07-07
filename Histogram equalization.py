import cv2
#import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('mamo.jpg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Calculating / Showing of pixel values in image 
hist=cv2.calcHist([image],[0],None,[256],[0,256])
#plt.subplot(1,2,1)
plt.plot(hist)
plt.show()
cv2.imshow('IMage',image)


#Applying histogram equalization to image
img_hist=cv2.equalizeHist(image)


#Calculating / Showing of pixel values in image 
hist_img=cv2.calcHist([img_hist],[0],None,[256],[0,256])
#plt.subplot(1,2,2)
plt.plot(hist_img)
plt.show()
cv2.imshow('histgram image',img_hist)

cv2.waitKey(0)
cv2.destroyAllWindows()
