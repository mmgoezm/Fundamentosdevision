import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversión a RGB
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #conversión a RGB
img = cv2.resize(img, (1200, 800))

    # 1. Segmentación por Color
lower_f = np.array([50, 50, 200])
upper_f = np.array([180, 150, 255])
mask_colorB = cv2.inRange( img, lower_f, upper_f)
lower_f = np.array([200, 50, 50])
upper_f = np.array([255, 155, 150])
mask_colorR = cv2.inRange( img, lower_f, upper_f)
mask_color = cv2.bitwise_or(mask_colorB, mask_colorR )
seg_color = cv2.bitwise_and(img, img, mask=mask_color)


plt.figure("imagen")
plt.imshow(img)
plt.figure("imagen HSV")
plt.imshow(img2)
plt.figure("Segmentación")
plt.imshow(seg_color)
plt.figure("mask")
plt.imshow(mask_color)
plt.show()
