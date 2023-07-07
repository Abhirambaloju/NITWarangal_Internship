import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
#from sklearn.metrics import pairwise_distances_argmin

n_colors = 128

# Load the Summer Palace photo
#img = load_sample_image("flower.jpg")

from matplotlib import image
#from matplotlib import pyplot

img = image.imread('C:/Users/Abhiram/Downloads/IMG_20220520_144745.jpg')

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = np.array(img, dtype=np.float64)/255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))

print("Fitting model on a small sub-sample of the data")
image_array_sample = shuffle(image_array, random_state=1)[:1000]

kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
#kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
#image_array_sample = shuffle(image_array, random_state=0)[:1000]
#kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)


# Get labels for all points
print("Predicting color indices on the full image (k-means)")

labels = kmeans.predict(image_array)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
  
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(img)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (128 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))