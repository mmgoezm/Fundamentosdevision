import cv2
import numpy as np

def detectar_ojos(cuadro, clasificador_rostros, clasificador_ojos):
    gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    rostros = clasificador_rostros.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #Detecta rostros

    ojos = []
    for (x, y, ancho, alto) in rostros:
        roi_gris = gris[y:y + int(alto / 2), x:x + ancho]  # Solo la mitad superior del rostro en escala de grises
        ojos_en_rostro = clasificador_ojos.detectMultiScale(roi_gris, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)) #Detecta ojos en la región de interés
        for (ex, ey, eancho, ealto) in ojos_en_rostro:
            ojos.append((x + ex, y + ey, eancho, ealto))

    return ojos, rostros


def calcular_relacion_aspecto_ojo(ojo):

        # Calcula las distancias euclidianas entre los puntos de referencia verticales del ojo
    A = np.linalg.norm(ojo[1] - ojo[5])
    B = np.linalg.norm(ojo[2] - ojo[4])

    # Calcula la distancia euclidiana entre los puntos de referencia horizontales del ojo
    C = np.linalg.norm(ojo[0] - ojo[3])

    # Calcula la relación de aspecto del ojo
    ear = (A + B) / (2.0 * C)
    return ear


def Imagen():

    clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #Carga el clasificador de rostros
    clasificador_ojos = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') #Carga el clasificador de ojos

    # Carga la imagen
    ruta_imagen = 'lena.png'  # Reemplaza con la ruta de tu imagen
    cuadro = cv2.imread(ruta_imagen)

    if cuadro is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Redimensiona la imagen si es demasiado grande
    dimension_maxima = 800
    alto, ancho = cuadro.shape[:2]
    if max(alto, ancho) > dimension_maxima:
        escala = dimension_maxima / max(alto, ancho)
        cuadro = cv2.resize(cuadro, None, fx=escala, fy=escala)

    ojos, rostros = detectar_ojos(cuadro, clasificador_rostros, clasificador_ojos)

    # Dibuja rectángulos alrededor de los rostros
    for (x, y, ancho, alto) in rostros:
        cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (255, 0, 0), 2)

    # Dibuja rectángulos alrededor de los ojos
    for (x, y, ancho, alto) in ojos:
        cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

    # Muestra el conteo de ojos
    cv2.putText(cuadro, f"Ojos: {len(ojos)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Muestra el conteo de rostros
    cv2.putText(cuadro, f"Rostros: {len(rostros)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Detección de Rostros y Ojos', cuadro)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Camara():
    """
    Función principal para detectar rostros y ojos en un video desde la cámara.
    """
    clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #Carga el clasificador de rostros
    clasificador_ojos = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') #Carga el clasificador de ojos

    # Inicializa la cámara
    captura = cv2.VideoCapture(0)

    while True:
        ret, cuadro = captura.read()
        if not ret:
            print("Falló la captura del cuadro")
            break

        ojos, rostros = detectar_ojos(cuadro, clasificador_rostros, clasificador_ojos)

        # Dibuja rectángulos alrededor de los rostros
        for (x, y, ancho, alto) in rostros:
            cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (255, 0, 0), 2)

        # Dibuja rectángulos alrededor de los ojos
        for (x, y, ancho, alto) in ojos:
            cv2.rectangle(cuadro, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

        # Muestra el conteo de ojos
        cv2.putText(cuadro, f"Ojos: {len(ojos)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Muestra el conteo de rostros
        cv2.putText(cuadro, f"Rostros: {len(rostros)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Deteccion de Rostros y Ojos', cuadro)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

Camara()