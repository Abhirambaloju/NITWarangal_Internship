from scipy.misc import face
from scipy import signal
from scipy import misc
# face = misc.face(gray=True)
# face = face[:512, -512:] 

from scipy.signal import wiener
import matplotlib.pyplot as plt
import cv2
import numpy as np
face=cv2.imread("blur.png")
#rng = np.random.default_rng()

noisy_face = np.copy(face).astype(np.float)
noisy_face += face.std() * 0.5 * np.random.standard_normal(face.shape)
#img = rng.random((40, 40))    #Create a random image
wiener_face = signal.wiener(face, (5, 5))
# filtered_img = wiener(noise_img, (5, 5))  #Filter the image
plt.subplot(1,2,1),plt.imshow(face),plt.title('Original image')
plt.subplot(2,1,2),plt.imshow(wiener_face),plt.title('wienerimage')
# f, (plot1, plot2) = plt.subplots(1, 2)
# plot1.imshow(noise_img)
# plot2.imshow(filtered_img)
# plt.show()
