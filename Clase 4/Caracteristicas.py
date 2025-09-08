import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('1.jpg')
img = cv2.resize(img,(1200, 800))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)  # escala de grises en float32

# Detector de Esquinas Harris
img_harris = img.copy()
corners_harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
corners_harris = cv2.dilate(corners_harris, None)
img_harris[corners_harris > 0.01 * corners_harris.max()] = [0, 0, 255]
cv2.imshow('Esquinas Harris', img_harris)

# Detector de Esquinas Shi-Tomasi
img_shi = img.copy()
corners_shi = cv2.goodFeaturesToTrack(gray, maxCorners=5000, qualityLevel=0.01, minDistance=20)

if corners_shi is not None:
    corners_shi = np.float32(corners_shi) #Se cambia el tipo de datos a float para cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners_shi = cv2.cornerSubPix(gray, corners_shi, (5, 5), (-1, -1), criteria)
    corners_shi = np.int32(corners_shi) # Se covierte de nuevo a int para dibujar los circulos.
    for i in corners_shi:
        x, y = i.ravel()
        cv2.circle(img_shi, (x, y), 3, 255, -1)
    cv2.imshow('Esquinas Shi-Tomasi', img_shi)

# Detector SIFT
img_sift = img.copy()
sift = cv2.SIFT_create()
keypoints_sift, _ = sift.detectAndCompute(gray.astype(np.uint8), None)  # SIFT requiere uint8
cv2.drawKeypoints(img_sift, keypoints_sift, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Puntos Clave SIFT', img_sift)

# Detector ORB
img_orb = img.copy()
orb = cv2.ORB_create()
keypoints_orb, _ = orb.detectAndCompute(gray.astype(np.uint8), None)  # ORB requiere uint8
cv2.drawKeypoints(img_orb, keypoints_orb, img_orb, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Puntos Clave ORB', img_orb)

# Detector FAST
img_fast = img.copy()
fast = cv2.FastFeatureDetector_create()
keypoints_fast = fast.detect(gray.astype(np.uint8), None)  # FAST requiere uint8
cv2.drawKeypoints(img_fast, keypoints_fast, img_fast, color=(255, 0, 0))
cv2.imshow('Puntos Clave FAST', img_fast)


# Detector AKAZE
img_akaze = img.copy()
akaze = cv2.AKAZE_create()
keypoints_akaze, _ = akaze.detectAndCompute(gray.astype(np.uint8), None)  # AKAZE requiere uint8
cv2.drawKeypoints(img_akaze, keypoints_akaze, img_akaze, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Puntos Clave AKAZE', img_akaze)

# Detector BRISK
img_brisk = img.copy()
brisk = cv2.BRISK_create()
keypoints_brisk, _ = brisk.detectAndCompute(gray.astype(np.uint8), None)  # BRISK requiere uint8
cv2.drawKeypoints(img_brisk, keypoints_brisk, img_brisk, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Puntos Clave BRISK', img_brisk)

cv2.waitKey(0)
cv2.destroyAllWindows()


