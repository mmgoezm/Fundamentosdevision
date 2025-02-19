import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Greydegree.png',0)
u,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
u,th2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
u,th3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
u,th4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
u,th5=cv2.threshold(img,127,255,cv2.THRESH_TRIANGLE)

x=np.linspace(0,255,num=256, dtype= np.uint8) # se genera un vector incremental de tamaño 256
y= np.zeros(256) # se crea un vector vacio

Width, Height =img.shape
for w in range (Width):
    for h in range (Height):
        v= img[w,h]
        y[v]=y[v] + 1
plt.bar(x,y)
plt.show()


Imagenes=[img,th1,th2,th3,th4,th5]
Titulos=['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TRIANGLE']

for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(Imagenes[i],'gray',vmin=0,vmax=255)
    plt.title(Titulos[i])
    plt.xticks([]),plt.yticks([])

plt.show()

img = cv2.imread('lena.png',0)
u,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
u,th2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
u,th3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
u,th4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
u,th5=cv2.threshold(img,127,255,cv2.THRESH_TRIANGLE)



Imagenes=[img,th1,th2,th3,th4,th5]
Titulos=['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TRIANGLE']

plt.hist(img)
plt.show()


img = cv2.imread('Greydegree.png',0)
u,th1=cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
u,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
u,th3=cv2.threshold(img,0,255,cv2.THRESH_TRUNC+ cv2.THRESH_OTSU)
u,th4=cv2.threshold(img,0,255,cv2.THRESH_TOZERO+ cv2.THRESH_OTSU)
u,th5=cv2.threshold(img,0,255,cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)


Imagenes=[img,th1,th2,th3,th4,th5]
Titulos=['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(Imagenes[i],'gray',vmin=0,vmax=255)
    plt.title(Titulos[i])
    plt.xticks([]),plt.yticks([])
plt.show()

img = cv2.imread('lena.png',0)
u,th1=cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
u,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
u,th3=cv2.threshold(img,0,255,cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
u,th4=cv2.threshold(img,0,255,cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
u,th5=cv2.threshold(img,0,255,cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)


Imagenes=[img,th1,th2,th3,th4,th5]
Titulos=['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(Imagenes[i],'gray',vmin=0,vmax=255)
    plt.title(Titulos[i])
    plt.xticks([]),plt.yticks([])

plt.show()
