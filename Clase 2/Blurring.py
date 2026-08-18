import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.cvtColor(cv2.imread("lena.png"), cv2.COLOR_BGR2RGB)

img1 = cv2.blur(img,(3,3))
img2 = cv2.blur(img,(4,4))
img3 = cv2.blur(img,(5,5))
img4 = cv2.blur(img,(6,6))

plt.subplot(1,5,1),  plt.imshow(img), plt.title("Original")
plt.subplot(1,5,2),  plt.imshow(img1), plt.title("Blurring (3x3)")
plt.subplot(1,5,3),  plt.imshow(img2), plt.title("Blurring (4x4)")
plt.subplot(1,5,4),  plt.imshow(img3), plt.title("Blurring (5x5)")
plt.subplot(1,5,5),  plt.imshow(img4), plt.title("Blurring (6x6)")

plt.show()

img = cv2.cvtColor(cv2.imread("tablero.jpg"), cv2.COLOR_BGR2RGB)


img1 = cv2.blur(img,(3,3))
img2 = cv2.blur(img,(6,6))
img3 = cv2.blur(img,(12,12))
img4 = cv2.blur(img,(24,24))

plt.subplot(1,5,1),  plt.imshow(img), plt.title("Original")
plt.subplot(1,5,2),  plt.imshow(img1), plt.title("Blurring (3x3)")
plt.subplot(1,5,3),  plt.imshow(img2), plt.title("Blurring (6x6)")
plt.subplot(1,5,4),  plt.imshow(img3), plt.title("Blurring (12x12)")
plt.subplot(1,5,5),  plt.imshow(img4), plt.title("Blurring (24x24)")

plt.show()