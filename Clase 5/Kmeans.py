import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans


num_clusters = 8
imagen = cv2.imread("2.jpg")
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) # Cambiar BGR a RGB
imagen = cv2.resize(imagen, (1200, 800))
original_shape = imagen.shape[:2] # Guardar la forma original
pixeles = imagen.reshape((-1, 3)) # Convertir la imagen a un array de píxeles RGB
pixeles = np.float32(pixeles)

centroides_por_iteracion = [] #

kmeans = KMeans(n_clusters=num_clusters, n_init="auto", max_iter=10)

        # Para cada iteracion, visualizar el resultado.
for i in range(kmeans.max_iter):
  kmeans.fit(pixeles)
  labels = kmeans.labels_
  centroides = kmeans.cluster_centers_
  centroides_por_iteracion.append(centroides)

  imagen_segmentada = centroides[labels].reshape(original_shape[0], original_shape[1], 3).astype(np.uint8)
  plt.imshow(imagen_segmentada)
  plt.title(f"Iteración {i + 1}")
  plt.show()

   # Mostrar el resultado final
plt.imshow(imagen_segmentada)
plt.title("Resultado Final")
for i in range(kmeans.max_iter):
    plt.figure("Centroides")
    rgb_data = centroides_por_iteracion[i]
    num_pixels = rgb_data.shape[0]
    side_length = int(np.ceil(np.sqrt(num_pixels)))
    image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    for j in range(num_pixels):
      row = j // side_length
      col = j % side_length
      image[row, col] = rgb_data[j]

    plt.subplot(2, 5, i+1), plt.imshow(image), plt.title(i)
plt.show()




