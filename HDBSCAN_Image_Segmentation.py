# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cv2
import time
import numpy as np

from sklearn.cluster import DBSCAN

COLORS = [[255,0,0],
          [0,255,0],
          [0,0,255],
          [255,255,0], #yello
          [0,255,255], #Cyan
          [255,0,255], #Magenta
          [192,192,192], #Silver
          [128,128,128], #Gray
          [128,0,0], #Maroon
          [128,128,0], #Olive
          [0,128,0], #Green
          [128,0,128], #Purple
          [0,128,128], #Teal
          [0,0,128]] #Navy

#%% Read Image
img = cv2.imread('/Volumes/medvedev/_SP/car_reference_position/car_1m5/image00241.png')
h,w,c = img.shape
img = cv2.resize(img, (int(h/2),int(w/2)))
image = img
cv2.medianBlur(img, 7, image)


#h,w,c = image.shape
print(image.shape)
h,w,c = image.shape
#%% Reshape Image for clustering
X = np.reshape(image, (h*w, c))
print(X.shape)



#%% DBSCAN
print("Computing Clusters")
start = time.time()
clustering = DBSCAN(eps=1, min_samples=3).fit(X)
print("Done Computing")
print("time Taken:",time.time() - start)



unique, counts = np.unique(clustering.labels_, return_counts=True)
unique = np.reshape(unique, (-1,1))
counts = np.reshape(counts, (-1,1))

unique = np.hstack((unique, counts))
print(unique.shape)
sorted_list = unique[np.argsort(unique[:, 1])[::-1]]
print(sorted_list[0:10])


#%% Display Labels DBSCAN

print(clustering.labels_)
original = np.reshape(clustering.labels_, (h,w))
print(original.shape)

copy_DBSCAN = image.copy()
for i in range(10):
    copy_DBSCAN[original == sorted_list[i][0]] = COLORS[i]
    
#%% Display Picture
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image_DBSCAN", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)
cv2.imshow("Image_DBSCAN", copy_DBSCAN)
cv2.waitKey(0)
cv2.destroyAllWindows()

