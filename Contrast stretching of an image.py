import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('area.jpg')



norm_img1 = cv2.normalize(img, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# scale to uint8
norm_img1 = (255*norm_img1).astype(np.uint8)
norm_img2 = np.clip(norm_img2,0,1)
norm_img2 = (255*norm_img2).astype(np.uint8)
rows = 1
columns = 3
plt.subplot(rows,columns,1),plt.imshow(img),plt.title('original image')
plt.subplot(rows,columns,2),plt.imshow(norm_img1),plt.title('min  fun')
plt.subplot(rows,columns,3),plt.imshow(norm_img2),plt.title(' max fun ')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()