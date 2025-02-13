# Importar librerias

import numpy as np
import matplotlib.pyplot as plt


# Imagen con valores Ceros = Negro

Image = np.zeros((15,15), np.uint8)
print (Image)

plt.figure(1)
plt.imshow(Image, interpolation ='none', aspect = 'auto', cmap='gray')
for (j,i),label in np.ndenumerate(Image):
    plt.text(i,j,label,ha='center',va='center',c='g')
Image[0,1]=55
Image[0,2]=155
Image[0,5]=120
Image[5,0]=220
Image[5,0]=90
Image[5,5]=255
Image[6,6]=100
Image[7,7]=200
Image[8,14]=180
Image[11,9]=50
Image[14,14]=60
Image[12,13]=80

plt.figure(2)
plt.imshow(Image, interpolation ='none', aspect = 'auto', cmap='gray')
for (j,i),label in np.ndenumerate(Image):
    plt.text(i,j,label,ha='center',va='center',c='g')
plt.show()