# LLamado de Librerias

import cv2
import matplotlib.pyplot as plt
import numpy as np


#Lectura de la imagen

Imagen = cv2.imread("Dogs.jpg",0)

cv2.imshow("Imagen inicial",Imagen)
plt.show()
cv2.waitKey(0) # Se espera a ingreso de una tecla

Media_intensidades=np.mean(Imagen)
Histograma=cv2.calcHist(Imagen,[0],None,[256],[0,256])

plt.figure("Histograma")
plt.title("Histograma con media Inicial")
plt.plot(Histograma)
plt.axvline(x=Media_intensidades,color="r",label="Media")
plt.show()

#Proceso de equalizacion

ImagenE=cv2.equalizeHist(Imagen)
Media_intensidadesE=np.mean(ImagenE)
HistogramaE=cv2.calcHist(ImagenE,[0],None,[256],[0,256])

plt.figure("Histogramas")
plt.subplot(1,2,1),  plt.plot(Histograma),plt.axvline(x=Media_intensidades,color="r",label="Media"), plt.title("Histograma con media Inicial")
plt.subplot(1,2,2),  plt.plot(HistogramaE),plt.axvline(x=Media_intensidadesE,color="r",label="Media"), plt.title("Histograma Ecualizado con media Inicial")


plt.figure("Imagenes")
plt.subplot(1,3,1),  plt.imshow(Imagen,cmap="gray"), plt.title("Imagen Inicial")
plt.subplot(1,3,2),  plt.imshow(ImagenE,cmap="gray"), plt.title("Imagen Ecualizada")
plt.subplot(1,3,3),  plt.imshow(ImagenE,cmap="inferno"), plt.title("Imagen Ecualizada inferno cmap")
plt.show()

cv2.destroyAllWindows()# se cierran las ventanas
