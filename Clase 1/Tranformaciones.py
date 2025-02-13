# Importar librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Imagen con valores aleatorios

imageBGR=cv2.imread("PirateDog.png",1) # Nota RGB a BGR Esto por el formato de lectura de OPENCV
ResizeImage= cv2.resize(imageBGR,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST)
#cv2.imwrite('Grises.png',ResizeImage) #Guardar imagen
cv2.imshow('Resize Image',ResizeImage)
cv2.waitKey(0)
ResizeImage2= cv2.resize(ResizeImage,None,fx=2,fy=2, interpolation=cv2.INTER_CUBIC)
print (imageBGR.shape)
print (ResizeImage2.shape)
numpy_horizontal = np.hstack((imageBGR, ResizeImage2))
cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.imshow("image", numpy_horizontal)
cv2.waitKey(0)

im=cv2.imread(("lena.png"),1)
ResizeImage= cv2.resize(im,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST)
cv2.imshow('Resize Image',ResizeImage)
cv2.waitKey(0)
ResizeImage2= cv2.resize(ResizeImage,None,fx=2,fy=2, interpolation=cv2.INTER_CUBIC)
numpy_horizontal = np.hstack((im, ResizeImage2))
cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.imshow("image", numpy_horizontal)
cv2.waitKey(0)