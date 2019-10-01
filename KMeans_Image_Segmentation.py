# -*- coding: utf-8 -*-


from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cv2
import time
import numpy as np

COLORS = [[255, 0, 0],
          [0, 255, 0],
          [0, 0, 255],
          [255, 255, 0],  # yello
          [0, 255, 255],  # Cyan
          [255, 0, 255],  # Magenta
          [192, 192, 192],  # Silver
          [128, 128, 128],  # Gray
          [128, 0, 0],  # Maroon
          [128, 128, 0],  # Olive
          [0, 128, 0],  # Green
          [128, 0, 128],  # Purple
          [0, 128, 128],  # Teal
          [0, 0, 128],
          [255, 64, 0],  #
          [0, 255, 64],  #
          [0, 64, 128],
          [255, 64, 128],  #
          [64, 128, 64],  #
          [128, 0, 64]]  #

# %% Read Image


if __name__ == '__main__':
img_path = '/Volumes/medvedev/_SP/2019_09_06_BMWi3_natika_camera_images/empty_pad/empty_pad3/image00081.png'
image = cv2.imread(img_path)
#image = cv2.bilateralFilter(image, 15, 75, 75)
h, w, c = image.shape
image = cv2.resize(image, (256, 256))

h, w, c = image.shape
copy = image.copy()
print(image.shape)
X = np.reshape(image, (h * w, c))
print(X.shape)

# %% K-Means
n_clusters = 20
print("Computing K-Means")
start = time.time()
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=100).fit(X)
print("Done Computing")
print("time Taken:", time.time() - start)

# %% Display Labels
print(kmeans.labels_)
original = np.reshape(kmeans.labels_, (h, w))
print(original.shape)

for i in range(n_clusters):
    copy[original == i] = COLORS[i]

# %% Display Picture
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)
cv2.imshow("Image", copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
