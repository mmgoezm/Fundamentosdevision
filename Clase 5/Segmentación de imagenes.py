import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversión a RGB
img = cv2.resize(img, (1200, 800))

    # 1. Segmentación por Color
lower_f = np.array([20, 50, 50])
upper_f = np.array([200, 200, 200])
mask_color = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), lower_f, upper_f)
seg_color = cv2.bitwise_and(img, img, mask=mask_color)

    # 2. Segmentación por Textura
def Gabor():
    filters = []
    ksize = 9
    for theta in np.arange(0, np.pi, np.pi / 3):
        kern = cv2.getGaborKernel((ksize, ksize), 5.0, theta, 10.0, 0.8, 0, ktype=cv2.CV_32F)
        kern /= 0.5 * kern.sum()
        filters.append(kern)
    return filters
filters = Gabor()
accum = np.zeros_like(img)
for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
seg_texture = accum

    # 3. Segmentación por Forma (usando detección de contornos)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
seg_shape = np.zeros_like(img)
cv2.drawContours(seg_shape, contours, -1, (100, 200, 0), 2)

    # 4. Segmentación por Thresholding
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, mask_thresh = cv2.threshold(gray, 160, 205, cv2.THRESH_BINARY)
seg_thresh = cv2.bitwise_and(img, img, mask=mask_thresh[:, :, np.newaxis])

    # 5. Segmentación por Clustering (K-means)

pixels = img.reshape((-1, 3))
kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
seg_kmeans = kmeans.cluster_centers_[kmeans.labels_].reshape(img.shape).astype(np.uint8)

    # Mostrar resultados



plt.figure("Segmentación")
plt.subplot(2,3,1),  plt.imshow(img), plt.title("Imagen Inicial")
plt.subplot(2,3,2),  plt.imshow(seg_color), plt.title("Color")
plt.subplot(2,3,3),  plt.imshow(seg_texture), plt.title("Textura ")
plt.subplot(2,3,4),  plt.imshow(seg_shape), plt.title("Contornos")
plt.subplot(2,3,5),  plt.imshow(seg_thresh), plt.title("Thresholding")
plt.subplot(2,3,6),  plt.imshow(seg_kmeans), plt.title("Clustering")
plt.show()


