# LLamado de Librerias

import cv2
import matplotlib.pyplot as plt
import numpy as np


#Lectura de la imagen

Imagen = cv2.imread("Coins.jpg",0)

grad_x_S= cv2.Sobel(Imagen , cv2.CV_64F, 1, 0, ksize=3)
grad_y_S = cv2.Sobel(Imagen , cv2.CV_64F, 0, 1, ksize=3)

abs_grad_x_S = cv2.convertScaleAbs(grad_x_S)
abs_grad_y_S = cv2.convertScaleAbs(grad_y_S)
grad_S = cv2.addWeighted(abs_grad_x_S, 0.5, abs_grad_y_S, 0.5, 0)

plt.figure("Sobel")
plt.subplot(2,2,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(2,2,2),  plt.imshow(abs_grad_x_S,cmap="gray"), plt.title("Gradiente X")
plt.subplot(2,2,3),  plt.imshow(abs_grad_y_S,cmap="gray"), plt.title("Gradiente Y")
plt.subplot(2,2,4),  plt.imshow(grad_S,cmap="gray"), plt.title("Gradiente")
plt.show()

# Prewitt
Kernel_x_P= np.array([  [1, 1, 1], [0, 0, 0], [-1, -1, -1]])
Kernel_y_P= np.array([  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

abs_grad_x_P = cv2.filter2D(Imagen,-1, Kernel_x_P)
abs_grad_y_P = cv2.filter2D(Imagen,-1, Kernel_y_P)

grad_P = cv2.addWeighted(abs_grad_x_P, 0.5, abs_grad_y_P, 0.5, 0)

plt.figure("Prewitt")
plt.subplot(2,2,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(2,2,2),  plt.imshow(abs_grad_x_P,cmap="gray"), plt.title("Gradiente X")
plt.subplot(2,2,3),  plt.imshow(abs_grad_y_P,cmap="gray"), plt.title("Gradiente Y")
plt.subplot(2,2,4),  plt.imshow(grad_P,cmap="gray"), plt.title("Gradiente")
plt.show()

# Canny
grad_C = cv2.Canny(Imagen,100,200) # Se debe asignar valores del Threshold

plt.figure("Comparación")
plt.subplot(2,2,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(2,2,2),  plt.imshow(grad_S,cmap="gray"), plt.title("Gradiente Sobel")
plt.subplot(2,2,3),  plt.imshow(grad_P,cmap="gray"), plt.title("Gradiente Prewitt")
plt.subplot(2,2,4),  plt.imshow(grad_C,cmap="gray"), plt.title("Gradiente Canny")
plt.show()


Imagen = cv2.imread("Dogs.jpg",0)

grad_x_S= cv2.Sobel(Imagen , cv2.CV_64F, 1, 0, ksize=3)
grad_y_S = cv2.Sobel(Imagen , cv2.CV_64F, 0, 1, ksize=3)

abs_grad_x_S = cv2.convertScaleAbs(grad_x_S)
abs_grad_y_S = cv2.convertScaleAbs(grad_y_S)
grad_S = cv2.addWeighted(abs_grad_x_S, 0.5, abs_grad_y_S, 0.5, 0)

plt.figure("Sobel")
plt.subplot(2,2,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(2,2,2),  plt.imshow(abs_grad_x_S,cmap="gray"), plt.title("Gradiente X")
plt.subplot(2,2,3),  plt.imshow(abs_grad_y_S,cmap="gray"), plt.title("Gradiente Y")
plt.subplot(2,2,4),  plt.imshow(grad_S,cmap="gray"), plt.title("Gradiente")
plt.show()

# Prewitt
Kernel_x_P= np.array([  [1, 1, 1], [0, 0, 0], [-1, -1, -1]])
Kernel_y_P= np.array([  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

abs_grad_x_P = cv2.filter2D(Imagen,-1, Kernel_x_P)
abs_grad_y_P = cv2.filter2D(Imagen,-1, Kernel_y_P)

grad_P = cv2.addWeighted(abs_grad_x_P, 0.5, abs_grad_y_P, 0.5, 0)

plt.figure("Prewitt")
plt.subplot(2,2,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(2,2,2),  plt.imshow(abs_grad_x_P,cmap="gray"), plt.title("Gradiente X")
plt.subplot(2,2,3),  plt.imshow(abs_grad_y_P,cmap="gray"), plt.title("Gradiente Y")
plt.subplot(2,2,4),  plt.imshow(grad_P,cmap="gray"), plt.title("Gradiente")
plt.show()

# Canny
grad_C = cv2.Canny(Imagen,100,200) # Se debe asignar valores del Threshold

plt.figure("Comparación")
plt.subplot(2,2,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(2,2,2),  plt.imshow(grad_S,cmap="gray"), plt.title("Gradiente Sobel")
plt.subplot(2,2,3),  plt.imshow(grad_P,cmap="gray"), plt.title("Gradiente Prewitt")
plt.subplot(2,2,4),  plt.imshow(grad_C,cmap="gray"), plt.title("Gradiente Canny")
plt.show()


