#
"""
======================================================================
==  Ejemplo de Juego Interactivo con Visión por Computador          ==
==               Universidad ITM                                    ==
==               Curso: Instrodución a vision por compitador        ==
==               Profesor: Manuel Mauricio Goez                     ==
======================================================================
"""

import cv2
import numpy as np
import random

"""Parametros de la Ventana y del Juego"""
WINDOW_WIDTH = 800; WINDOW_HEIGHT = 600; BALL_RADIUS = 20; BALL_SPEED = 5; MAX_BALLS = 10; SPAWN_INTERVAL = 30 ; INTERCEPTION_ZONE_SIZE = 100
BLACK = (0, 0, 0)       # Negro puro
WHITE = (255, 255, 255) # Blanco puro
RED = (0, 0, 255)       # Rojo puro

"""Variables del Juego"""

class Pelota:
    def __init__(self):
        self.x = random.randint(BALL_RADIUS, WINDOW_WIDTH - BALL_RADIUS)
        self.y = -BALL_RADIUS # Coordenada Y negativa

    def update(self):
        """
        Actualiza la posición de la pelota en cada fotograma.
        Simula el movimiento de caída vertical.
        """
        self.y += BALL_SPEED # Incrementa la coordenada Y según la velocidad definida.

    def draw(self, frame):
        """
        Dibuja la pelota sobre un fotograma.
       """
        # Usamos cv2.circle para dibujar.
        # Parámetros: imagen_destino, centro (x, y), radio, color, grosor (-1 para relleno)
        cv2.circle(frame, (self.x, int(self.y)), BALL_RADIUS, RED, -1)

"""captura de la cámara y dibujado del juego"""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Variables del estado del juego
balls = []             # Una lista para almacenar todas las pelotas activas en pantalla.
score = 0              # La puntuación del jugador, inicia en 0.
spawn_counter = 0      # Un contador para controlar cuándo generar la próxima pelota.

""""Bucle principal del juego"""
while True:
    # Creamos un 'lienzo' (frame) negro para cada iteración del juego.
    game_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)


    ret, camera_frame = cap.read() # Leemos la imagen de la cámara.
    if not ret:
        print("Error: No se pudo capturar el fotograma de la cámara.")
        break # Salimos del bucle si falla la captura.

    camera_frame = cv2.flip(camera_frame, 1) # Volteamos la imagen para que se vea bien.
    fram1hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV) # en formato HSV (Hue, Saturation, Value)
    lowerblack = np.array([0, 0, 0])     # Límite inferior [H, S, V]
    upperblack = np.array([180, 255, 50]) # Límite superior [H, S, V]

    mask = cv2.inRange(fram1hsv, lowerblack, upperblack)

    # Aplicamos operaciones morfológicas para limpiar la máscara.
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    interception_x, interception_y = None, None # Coordenadas del cuadro de intercepción.
    if contours:
        # Encontramos el contorno con el área más grande.
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculamos los 'momentos' del contorno. Son útiles para calcular propiedades como el área y el centroide.
        M = cv2.moments(largest_contour)
        # Calculamos el centroide (cx, cy) si el área (M['m00']) no es cero (para evitar división por cero).
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) # Coordenada X del centroide en la imagen de la cámara
            cy = int(M["m01"] / M["m00"]) # Coordenada Y del centroide en la imagen de la cámara

            # ----- ¡Mapeo de Coordenadas! -----
            # Convertimos las coordenadas del centroide (cx, cy) del espacio de la cámara
            # al espacio de la ventana del juego (interception_x, interception_y).
            # Usamos una regla de tres simple basada en las dimensiones de ambas 'pantallas'.
            interception_x = int(cx * WINDOW_WIDTH / camera_frame.shape[1]) # shape[1] es el ancho de la cámara
            interception_y = int(cy * WINDOW_HEIGHT / camera_frame.shape[0]) # shape[0] es la altura de la cámara

            # Dibujamos el contorno detectado en la imagen de la cámara (feedback visual).
            cv2.drawContours(camera_frame, [largest_contour], 0, (0, 255, 0), 2) # Contorno verde, grosor 2


    spawn_counter += 1 # Incrementamos el contador en cada fotograma.
    # Si el contador alcanza el intervalo Y no hemos superado el máximo de pelotas...
    if spawn_counter >= SPAWN_INTERVAL and len(balls) < MAX_BALLS:
        balls.append(Pelota()) # Creamos una nueva pelota y la añadimos a la lista.
        spawn_counter = 0      # Reiniciamos el contador.

    # Actualizamos y dibujamos cada pelota en la lista.
    for ball in balls[:]:
        ball.update() # Actualizamos la posición de la pelota (la hacemos caer).
        ball.draw(game_frame) # Dibujamos la pelota en el lienzo del juego.

        if (interception_x is not None and interception_y is not None and
            # Comprobamos si el centro de la pelota (ball.x, ball.y) está dentro del cuadrado definido por la zona de intercepción.
            # Comparamos la distancia absoluta en X e Y con la mitad del tamaño de la zona.
            abs(ball.x - interception_x) < INTERCEPTION_ZONE_SIZE // 2 and
            abs(ball.y - interception_y) < INTERCEPTION_ZONE_SIZE // 2):
            balls.remove(ball) # Eliminamos la pelota de la lista.
            score += 1         # Incrementamos la puntuación.

        # Eliminamos las pelotas que ya cayeron y salieron de la pantalla por abajo.
        if ball.y > WINDOW_HEIGHT + BALL_RADIUS:
            balls.remove(ball) # La eliminamos para optimizar.

    """se muestra el area de intercepción en la pantalla."""
    if interception_x is not None and interception_y is not None:
        pt1 = (interception_x - INTERCEPTION_ZONE_SIZE // 2, interception_y - INTERCEPTION_ZONE_SIZE // 2)
        pt2 = (interception_x + INTERCEPTION_ZONE_SIZE // 2, interception_y + INTERCEPTION_ZONE_SIZE // 2)
        cv2.rectangle(game_frame, pt1, pt2, WHITE, 2)

    """"Contrador de puntos en pantalla."""
    cv2.putText(game_frame,
                f"Score: {score}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,1, WHITE,2)                              # Grosor del texto


    cv2.imshow("Falling Balls Game", game_frame) # La ventana principal del juego.
    cv2.imshow("Camera Feed", camera_frame)     # La vista de la cámara con el contorno detectado.
    cv2.imshow("Mask", mask)                    # La máscara binaria (útil para depuración).

    key = cv2.waitKey(1) & 0xFF
    # Si la tecla presionada es 'q'...
    if key == ord('q'):
        break # ... rompemos el bucle principal y salimos.

print(f"Juego terminado. Puntuación final: {score}")
cap.release()
cv2.destroyAllWindows()
