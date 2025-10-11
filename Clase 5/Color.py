import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversión a RGB
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #conversión a RGB
img = cv2.resize(img, (1200, 800))

    # 1. Segmentación por Color
lower_f = np.array([100, 70, 100])
upper_f = np.array([190, 110, 130])
mask_color = cv2.inRange( img, lower_f, upper_f)
seg_color = cv2.bitwise_and(img, img, mask=mask_color)
plt.imshow(img)

plt.imshow(img2)
plt.figure("Segmentación")
plt.imshow(mask_color)
plt.show()
