import cv2
#from scipy.signal import gaussian, convolve2d
#from skimage import color, data, restoration
import numpy as np
from matplotlib import pyplot as plt

from scipy import signal


# face = misc.face(gray=True)
# face = face[:512, -512:]  # crop out square on right
# noisy_face = np.copy(face).astype(np.float)
# noisy_face += face.std() * 0.5 * np.random.standard_normal(face.shape)



noise_img=cv2.imread("saltpepper.png")
#noise_img=cv2.imread("img.png")
#noise_img=cv2.imread("brain.jpg")
gauss = cv2.GaussianBlur(noise_img, (3,3), 3)
median = cv2.medianBlur(noise_img,5)
kernal=np.ones((5,5))/25
blur=cv2.filter2D(noise_img,-1,kernal)
#wiener_face = signal.wiener(noise_img, (5, 5))
#images = np.concatenate((noise_img,median, gauss), axis=1)
cv2.imshow('Original',noise_img)
cv2.imshow('Median',median)
cv2.imshow('Gaussian',gauss)
cv2.imshow('Mean',blur)


# plt.subplot(2,2,1),plt.imshow(noise_img),plt.title('Original image')
# plt.subplot(2,2,2),plt.imshow(gauss),plt.title('gauss')
# plt.subplot(2,2,3),plt.imshow(median),plt.xlabel('median')
# plt.subplot(2,2,4),plt.imshow(blur),plt.xlabel('Mean')
# plt.show()



# psf = np.ones((5,5)) / 25
# img6 = convolve2d(noise_img,psf,'same')
# img6 += 0.1 * img6.std() * np.random.standard_normal(img6.shape)
# Wiener_filtered = restoration.wiener(img6,psf,1100) 


# cv2.imshow('img', Wiener_filtered)
#cv2.imshow('img', images)
cv2.waitKey(0)
cv2.destroyAllWindows()


