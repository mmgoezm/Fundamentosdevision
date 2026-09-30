import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversión a RGB
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #conversión a RGB
img = cv2.resize(img, (1200, 800))

    # 1. Segmentación por Color
lower_f = np.array([90, 200, 0])
upper_f = np.array([120, 255, 125])
mask_color1 = cv2.inRange( img2, lower_f, upper_f)

lower_f = np.array([99, 70, 59])
upper_f = np.array([114, 143, 188])
mask_color2 = cv2.inRange( img2, lower_f, upper_f)

lower_f = np.array([110, 100, 20])
upper_f = np.array([130, 255, 100])
mask_color6 = cv2.inRange( img2, lower_f, upper_f)


lower_f = np.array([150, 100, 100])
upper_f = np.array([190, 255, 255])
mask_color3 = cv2.inRange( img2, lower_f, upper_f)


lower_f = np.array([105, 153, 51])
upper_f = np.array([125, 255, 153])
mask_color4 = cv2.inRange( img2, lower_f, upper_f)


lower_f = np.array([90, 50, 50])
upper_f = np.array([135, 255, 255])
mask_color5 = cv2.inRange( img2, lower_f, upper_f)





plt.figure("imagen")
plt.imshow(img)

plt.figure("mask1")
plt.imshow(mask_color1)

plt.figure("mask2")
plt.imshow(mask_color2)

plt.figure("mask3")
plt.imshow(mask_color3)

plt.figure("mask4")
plt.imshow(mask_color4)

plt.figure("mask5")
plt.imshow(mask_color5)

plt.figure("mask 6")
plt.imshow(mask_color6)
plt.show()
