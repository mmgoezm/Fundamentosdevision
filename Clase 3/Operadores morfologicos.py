# LLamado de Librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Lectura de la imagen
inicial = cv2.imread('A.jpg',1)
#Tamaño del kernel
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(inicial,kernel,iterations = 1)
erosion2 = cv2.erode(inicial,kernel,iterations = 5)

plt.figure("Erosion")
plt.subplot(1,3,1),  plt.imshow(inicial), plt.title("Imagen Inicial")
plt.subplot(1,3,2),  plt.imshow(erosion), plt.title("Imagen Erosionada i=1")
plt.subplot(1,3,3),  plt.imshow(erosion2), plt.title("Imagen Erosionada i=5")
plt.show()

dilation = cv2.dilate(inicial,kernel,iterations = 1)
dilation2 = cv2.dilate(inicial,kernel,iterations = 5)

plt.figure("Dilatacion")
plt.subplot(1,3,1),  plt.imshow(inicial), plt.title("Imagen Inicial")
plt.subplot(1,3,2),  plt.imshow(dilation), plt.title("Imagen Dilatada i=1")
plt.subplot(1,3,3),  plt.imshow(dilation2), plt.title("Imagen Dilatada i=5")
plt.show()

# Agregar ruido Sal and Peppers
ICR= inicial.copy() # se crea una copia de la imagen original nombrada como ICR (Imagen Con Ruido)
h, w, c = inicial.shape

for _ in range(1000):
    row, col = np.random.randint(0, h), np.random.randint(0, w)
    if np.random.rand() < 0.5:
        ICR[row, col] = [0, 0, 0]
    else:
        ICR[row, col] = [255, 255, 255]

opening = cv2.morphologyEx(ICR, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(ICR, cv2.MORPH_CLOSE, kernel)

plt.figure("Apertura y Cierre")
plt.subplot(1,3,1),  plt.imshow(ICR), plt.title("Imagen con ruido")
plt.subplot(1,3,2),  plt.imshow(opening), plt.title("erosion - dilation (Apertura)")
plt.subplot(1,3,3),  plt.imshow(closing), plt.title("dilation - erosion (Cierre)")
plt.show()


gradient = cv2.morphologyEx(inicial, cv2.MORPH_GRADIENT, kernel)

plt.figure("Gradiente Morfologico")
plt.subplot(1,2,1),  plt.imshow(inicial), plt.title("Imagen Inicial")
plt.subplot(1,2,2),  plt.imshow(gradient), plt.title(" Diferencia = erosion - dilation")

plt.show()

# ejemplo con una imagen

inicial = cv2.imread('Dogs.jpg',1)
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(inicial,kernel,iterations = 1)
erosion2 = cv2.erode(inicial,kernel,iterations = 5)

plt.figure("Erosion")
plt.subplot(1,3,1),  plt.imshow(inicial), plt.title("Imagen Inicial")
plt.subplot(1,3,2),  plt.imshow(erosion), plt.title("Imagen Erosionada i=1")
plt.subplot(1,3,3),  plt.imshow(erosion2), plt.title("Imagen Erosionada i=5")
plt.show()

dilation = cv2.dilate(inicial,kernel,iterations = 1)
dilation2 = cv2.dilate(inicial,kernel,iterations = 5)

plt.figure("Dilatacion")
plt.subplot(1,3,1),  plt.imshow(inicial), plt.title("Imagen Inicial")
plt.subplot(1,3,2),  plt.imshow(dilation), plt.title("Imagen Dilatada i=1")
plt.subplot(1,3,3),  plt.imshow(dilation2), plt.title("Imagen Dilatada i=5")
plt.show()

# Agregar ruido Sal and Peppers
ICR= inicial.copy() # se crea una copia de la imagen original nombrada como ICR (Imagen Con Ruido)
h, w, c = inicial.shape

for _ in range(1000):
    row, col = np.random.randint(0, h), np.random.randint(0, w)
    if np.random.rand() < 0.5:
        ICR[row, col] = [0, 0, 0]
    else:
        ICR[row, col] = [255, 255, 255]

opening = cv2.morphologyEx(ICR, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(ICR, cv2.MORPH_CLOSE, kernel)

plt.figure("Apertura y Cierre")
plt.subplot(1,3,1),  plt.imshow(ICR), plt.title("Imagen con ruido")
plt.subplot(1,3,2),  plt.imshow(opening), plt.title("erosion - dilation (Apertura)")
plt.subplot(1,3,3),  plt.imshow(closing), plt.title("dilation - erosion (Cierre)")
plt.show()


gradient = cv2.morphologyEx(inicial, cv2.MORPH_GRADIENT, kernel)

plt.figure("Gradiente Morfologico")
plt.subplot(1,2,1),  plt.imshow(inicial), plt.title("Imagen Inicial")
plt.subplot(1,2,2),  plt.imshow(gradient), plt.title(" Diferencia = erosion - dilation")

plt.show()
