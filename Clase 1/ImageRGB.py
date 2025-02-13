# Importar librerias
import numpy as np
import matplotlib.pyplot as plt


# Imagen con valores aleatorios

Image= np.random.randint(255, size=(2,4,3))

plt.figure(1)
plt.imshow(Image)


Red11= Image[1,1,0]
Green11= Image[1,1,1]
Blue11= Image[1,1,2]

print(Red11,Green11,Blue11)
plt.show()

Image[0,0,0]=0
Image[0,0,1]=0
Image[0,0,2]=0


plt.figure(2)
plt.imshow(Image)
plt.show()
Image[0,1,0]=255
Image[0,1,1]=0
Image[0,1,2]=0


plt.figure(3)
plt.imshow(Image)
plt.show()

Image[0,2,0]=0
Image[0,2,1]=255
Image[0,2,2]=0

plt.figure(4)
plt.imshow(Image)
plt.show()

Image[0,3,0]=0
Image[0,3,1]=0
Image[0,3,2]=255

plt.figure(5)
plt.imshow(Image)
plt.show()

Image[1,0,0]=255
Image[1,0,1]=0
Image[1,0,2]=255

plt.figure(6)
plt.imshow(Image)
plt.show()

Image[1,1,0]=0
Image[1,1,1]=255
Image[1,1,2]=255

plt.figure(7)
plt.imshow(Image)
plt.show()

Image[1,2,0]=255
Image[1,2,1]=255
Image[1,2,2]=0

plt.figure(8)
plt.imshow(Image)
plt.show()

Image[1,3,0]=255
Image[1,3,1]=255
Image[1,3,2]=255

plt.figure(9)
plt.imshow(Image)
plt.show()