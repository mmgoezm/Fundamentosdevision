# IMPORTACIÓN DE LIBRERÍAS FUNDAMENTALES
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==============================================================================
# PARTE 1: LA IMAGEN COMO UNA MATRIZ DE NÚMEROS
# Creamos una imagen negra de 15x15 píxeles usando una matriz de ceros.
# 'np.uint8' indica que los valores van de 0 a 255 (formato estándar de 8 bits).
imagen_matriz = np.zeros((15, 15), dtype=np.uint8)

# Modificamos el valor (brillo) de píxeles específicos (0 = Negro, 255 = Blanco)
imagen_matriz[0, 1] = 55
imagen_matriz[7, 7] = 200
imagen_matriz[14, 14] = 60
imagen_matriz[12, 13] = 80

# Visualizamos la matriz con matplotlib
plt.figure(figsize=(6, 6))
plt.title("Imagen como Matriz (Escala de Grises)")
plt.imshow(imagen_matriz, interpolation='none', aspect='auto', cmap='gray')

# Ciclo didáctico: Imprimimos el número sobre cada píxel para entender la matriz
for (j, i), label in np.ndenumerate(imagen_matriz):
    plt.text(i, j, label, ha='center', va='center', color='green', fontsize=8)
plt.show()

# Ahora creamos una imagen a color (3 canales: Rojo, Verde, Azul - RGB)
# Llenamos una matriz de 2x4 con valores aleatorios entre 0 y 255
imagen_color = np.random.randint(255, size=(2, 4, 3), dtype=np.uint8)

plt.figure(figsize=(6, 3))
plt.title("Imagen a Color (Matriz 3D: RGB)")
plt.imshow(imagen_color)
plt.show()

# Modificamos píxeles manualmente para ver colores puros
imagen_color[0, 0] = [0, 0, 0]  # Negro (R=0, G=0, B=0)
imagen_color[0, 1] = [255, 0, 0]  # Rojo puro
imagen_color[0, 2] = [0, 255, 0]  # Verde puro
imagen_color[0, 3] = [0, 0, 255]  # Azul puro

plt.figure(figsize=(6, 3))
plt.title("Imagen a Color (Matriz 3D: RGB)")
plt.imshow(imagen_color)
plt.show()

# Modificamos píxeles manualmente para ver colores diversos
imagen_color[1,0,0]=255
imagen_color[1,0,1]=0
imagen_color[1,0,2]=255

imagen_color[1,1,0]=255
imagen_color[1,1,1]=255
imagen_color[1,1,2]=0

imagen_color[1,2,0]=0
imagen_color[1,2,1]=255
imagen_color[1,2,2]=255

imagen_color[1,3,0]=255
imagen_color[1,3,1]=255
imagen_color[1,3,2]=255

plt.figure(figsize=(6, 3))
plt.title("Imagen a Color (Matriz 3D: RGB)")
plt.imshow(imagen_color)
plt.show()

# ==============================================================================
# PARTE 2: LECTURA DE IMÁGENES Y ESPACIOS DE COLOR
# NOTA CLAVE: OpenCV lee las imágenes a color en formato BGR (Azul, Verde, Rojo).
# Matplotlib, por el contrario, espera formato RGB (Rojo, Verde, Azul).
image_gray = cv2.imread("Dogs.jpg", 0)  # El '0' fuerza la lectura en escala de grises
image_bgr = cv2.imread("Dogs.jpg", 1)  # El '1' indica lectura a color (BGR)

# Validamos que la imagen exista antes de continuar
if image_bgr is not None:
    # Convertimos de BGR a RGB para visualizarla correctamente en Matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Podemos separar la imagen en sus 3 canales individuales
    canal_b, canal_g, canal_r = cv2.split(image_bgr)

    # Visualizamos las diferencias
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].imshow(canal_b, cmap='gray');
    axs[0].set_title("1.canal_b")
    axs[1].imshow(canal_g, cmap='gray');
    axs[1].set_title("2. canal_g")
    axs[2].imshow(canal_r, cmap='gray');
    axs[2].set_title("3. canal_r")
    plt.show()

    # Visualizamos La imagen unificada
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].imshow(image_gray, cmap='gray');
    axs[0].set_title("1. Escala de Grises")
    axs[1].imshow(image_bgr);
    axs[1].set_title("2. Formato BGR (OpenCV original)")
    axs[2].imshow(image_rgb);
    axs[2].set_title("3. Formato RGB (Corregido)")
    plt.show()

else:

    print("Error: No se encontró 'Dogs.jpg'")

# ==============================================================================
# PARTE 3: REDIMENSIONAMIENTO (RESIZE) E INTERPOLACIÓN
# Usamos OpenCV para visualizar (cv2.imshow) en lugar de Matplotlib

image = cv2.imread("PirateDog.png", 1)
# 1. Reducimos la imagen a la mitad (0.5) usando interpolación 'Nearest'
img_reducida = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

# 2. Volvemos a ampliar la imagen reducida al doble (2.0) usando interpolación 'Cubic'
img_ampliada = cv2.resize(img_reducida, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Unimos ambas imágenes horizontalmente para compararlas directamente
comparacion_resize = np.hstack((image, img_ampliada))

cv2.imshow("Izquierda: Original | Derecha: Reducida y vuelta a ampliar", comparacion_resize)
cv2.waitKey(0)  # Espera a que el usuario presione una tecla
cv2.destroyAllWindows()  # Cierra las ventanas de OpenCV

# 1. Reducimos la imagen  (0.4) usando interpolación 'Nearest'
img_reducida = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

# 2. Volvemos a ampliar la imagen reducida (5.0) usando interpolación 'Cubic'
img_ampliada = cv2.resize(img_reducida, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

# Unimos ambas imágenes horizontalmente para compararlas directamente
comparacion_resize = np.hstack((image, img_ampliada))

cv2.imshow("Izquierda: Original | Derecha: Reducida y vuelta a ampliar", comparacion_resize)
cv2.waitKey(0)  # Espera a que el usuario presione una tecla
cv2.destroyAllWindows()  # Cierra las ventanas de OpenCV

# ==============================================================================
# PARTE 4: TRANSFORMACIONES GEOMÉTRICAS
# ==============================================================================
print("--- Ejecutando Parte 4: Transformaciones ---")
image_geo = cv2.imread("PirateDog.png", 0)  # Leemos en escala de grises

if image_geo is not None:
    filas, columnas = image_geo.shape

    # --- A. TRASLACIÓN (Mover la imagen en X y Y) ---
    # Matriz M: [[1, 0, Desplazamiento_X], [0, 1, Desplazamiento_Y]]
    M_traslacion = np.float32([[1, 0, 40], [0, 1, 80]])
    img_trasladada = cv2.warpAffine(image_geo, M_traslacion, (columnas, filas))

    # --- B. ROTACIÓN ---
    # Centro de rotación (mitad de la imagen), Ángulo (45 grados), Escala (1.0)
    M_rotacion = cv2.getRotationMatrix2D((columnas / 2, filas / 2), 45, 1)
    img_rotada = cv2.warpAffine(image_geo, M_rotacion, (columnas, filas))

    # --- C. PERSPECTIVA ---
    # Mapeamos 4 puntos de la imagen original hacia 4 puntos de destino
    puntos_origen = np.float32([[20, 80], [140, 40], [5, 270], [380, 310]])
    puntos_destino = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M_perspectiva = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)
    img_perspectiva = cv2.warpPerspective(image_geo, M_perspectiva, (300, 300))  # Salida de 300x300

    # Mostramos todos los resultados juntos
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(image_geo, cmap='gray');
    axs[0].set_title("Original")
    axs[1].imshow(img_trasladada, cmap='gray');
    axs[1].set_title("Traslación")
    axs[2].imshow(img_rotada, cmap='gray');
    axs[2].set_title("Rotación")
    axs[3].imshow(img_perspectiva, cmap='gray');
    axs[3].set_title("Perspectiva")
    plt.show()

print("--- FIN DEL SCRIPT ---")