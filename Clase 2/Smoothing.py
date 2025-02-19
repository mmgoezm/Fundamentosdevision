import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.cvtColor(cv2.imread("lena.png"), cv2.COLOR_BGR2RGB)

Kmean= np.array([
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9]
])

Ksovel= np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
])


kemboss = np.array([
  [-2, -1, 0],
  [-1, 1, 1],
  [0, 1, 2]
])



img1 = cv2.filter2D(img,-1,Ksovel)
img2 = cv2.filter2D(img,-1,kemboss)
img3 = cv2.medianBlur (img,5)
img4 = cv2.bilateralFilter (img,9,75,75)

plt.subplot(1,5,1),  plt.imshow(img), plt.title("Original")
plt.subplot(1,5,2),  plt.imshow(img1), plt.title("mean (3x3)")
plt.subplot(1,5,3),  plt.imshow(img2), plt.title("Blurring (4x4)")
plt.subplot(1,5,4),  plt.imshow(img3), plt.title("Blurring (5x5)")
plt.subplot(1,5,5),  plt.imshow(img4), plt.title("Blurring (6x6)")

plt.show()