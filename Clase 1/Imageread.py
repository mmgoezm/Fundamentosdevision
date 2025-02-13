# Importar librerias
import cv2
import matplotlib.pyplot as plt

#Lecturas
imagegray=cv2.imread("Dogs.jpg",0)
imageBGR=cv2.imread("Dogs.jpg",1) # Nota RGB a BGR Esto por el formato de lectura de OPENCV
imageRGB=cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
print (imageRGB.shape)

fig = plt.figure(1)
ax1=fig.add_subplot(1,3,1)
ax1.imshow(imagegray, cmap='gray')
ax1.set_title("Imagen Gris")
ax2=fig.add_subplot(1,3,2)
ax2.imshow(imageBGR)
ax2.set_title("Imagen BGR")
ax3=fig.add_subplot(1,3,3)
ax3.imshow(imageRGB)
ax3.set_title("Imagen RGB")
plt.show()

ImageR,ImageG,ImageB = cv2.split(imageBGR)

fig = plt.figure(2)
ax1=fig.add_subplot(2,2,1)
ax1.imshow(ImageR,cmap="gray")
ax1.set_title("Imagen Rojo")
ax2=fig.add_subplot(2,2,2)
ax2.imshow(ImageG,cmap="gray")
ax2.set_title("Imagen Verde")
ax3=fig.add_subplot(2,2,3)
ax3.imshow(ImageB,cmap="gray")
ax3.set_title("Imagen Azul")
ax4=fig.add_subplot(2,2,4)
ax4.imshow(imageRGB,cmap="gray")
ax4.set_title("Imagen RGB")
plt.show()
