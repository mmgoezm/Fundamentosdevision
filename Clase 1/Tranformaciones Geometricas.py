# Importar librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Imagen con valores aleatorios

imageBGR=cv2.imread("PirateDog.png",0) # Nota RGB a BGR Esto por el formato de lectura de OPENCV
rows,cols = imageBGR.shape
M=np.float32([[1,0,40],[0,1,80]]) #Translacion
dst = cv2.warpAffine(imageBGR,M,(cols,rows))

cv2.imshow('Resize Image',imageBGR)
cv2.waitKey(0)
cv2.imshow('Resize Image',dst)
cv2.waitKey(0)

M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1) #Rotación
dst = cv2.warpAffine(imageBGR,M,(cols,rows))
cv2.imshow('Resize Image',dst)
cv2.waitKey(0)

Visionangle1=np.float32([[20,80],[140,40],[5,270],[380,310]])
Visionangle2=np.float32([[0,0],[300,0],[0,300],[300,300]])

M=cv2.getPerspectiveTransform(Visionangle1,Visionangle2) #Rotación
dst = cv2.warpPerspective(imageBGR,M,(cols,rows))
plt.subplot(121),plt.imshow(imageBGR),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()