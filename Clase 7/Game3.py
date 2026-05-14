#
"""
======================================================================
==  Ejemplo de Juego Interactivo con Visión por Computador          ==
==               "Defensor Espacial"                                ==
==               Basado en el ejemplo de Pelotas                    ==
======================================================================
"""

import cv2
import numpy as np
import random
import time

"""Parametros de la Ventana y del Juego"""
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ASTEROID_RADIUS = 15  # Radio del asteroide
ASTEROID_SPEED = 4  # Velocidad de caída
MAX_ASTEROIDS = 10
SPAWN_INTERVAL = 25
INTERCEPTION_ZONE_SIZE = 120  # Hacemos el escudo un poco más grande
BASE_Y_POSITION = WINDOW_HEIGHT - 50  # Posición Y de nuestra base

# --- Colores ---
BLACK = (0, 0, 0)  # Negro
WHITE = (255, 255, 255)  # Blanco (Escudo)
GRAY = (128, 128, 128)  # Gris (Asteroides)
BLUE = (255, 0, 0)  # Azul (Base)
RED = (0, 0, 255)  # Rojo (para texto de alerta)

"""Variables del Juego"""


class Asteroide:
    def __init__(self):
        self.x = random.randint(ASTEROID_RADIUS, WINDOW_WIDTH - ASTEROID_RADIUS)
        self.y = -ASTEROID_RADIUS  # Empieza arriba, fuera de la pantalla
        self.color = GRAY
        self.radius = ASTEROID_RADIUS

    def update(self):
        """
        Actualiza la posición del asteroide. Solo cae.
        """
        self.y += ASTEROID_SPEED

    def draw(self, frame):
        """
        Dibuja el asteroide (círculo gris).
        """
        cv2.circle(frame, (self.x, int(self.y)), self.radius, self.color, -1)


"""captura de la cámara y dibujado del juego"""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Variables del estado del juego
asteroids = []  # Lista para los asteroides
score = 0  # Puntuación
lives = 10  # Vidas del jugador
spawn_counter = 0  # Contador para generar asteroides

""""Bucle principal del juego"""
while True:
    # Si te quedas sin vidas, termina el juego
    if lives <= 0:
        print(f"Juego terminado. Puntuación final: {score}")
        break

    # Creamos un 'lienzo' (frame) negro para cada iteración.
    game_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # --- Lógica de la Cámara (igual que antes) ---
    ret, camera_frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    camera_frame = cv2.flip(camera_frame, 1)
    fram1hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

    # Mismos rangos para detectar el objeto negro/oscuro
    lowerblack = np.array([0, 0, 0])
    upperblack = np.array([180, 255, 50])  # Ajusta el '50' si no detecta bien

    mask = cv2.inRange(fram1hsv, lowerblack, upperblack)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    interception_x, interception_y = None, None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Mapeo de coordenadas (igual que antes)
            interception_x = int(cx * WINDOW_WIDTH / camera_frame.shape[1])
            interception_y = int(cy * WINDOW_HEIGHT / camera_frame.shape[0])

            cv2.drawContours(camera_frame, [largest_contour], 0, (0, 255, 0), 2)

    # --- Lógica del Juego ---
    spawn_counter += 1
    # Generamos un nuevo asteroide si toca
    if spawn_counter >= SPAWN_INTERVAL and len(asteroids) < MAX_ASTEROIDS:
        asteroids.append(Asteroide())
        spawn_counter = 0

    # Actualizamos y dibujamos cada asteroide
    for asteroid in asteroids[:]:  # Usamos [:] para poder borrar de la lista
        asteroid.update()
        asteroid.draw(game_frame)

        # --- Lógica de Colisión ---

        # 1. Colisión con el ESCUDO (el jugador)
        if (interception_x is not None and interception_y is not None and
                # ¿El centro del asteroide está dentro del cuadrado del escudo?
                abs(asteroid.x - interception_x) < INTERCEPTION_ZONE_SIZE // 2 and
                abs(asteroid.y - interception_y) < INTERCEPTION_ZONE_SIZE // 2):
            asteroids.remove(asteroid)  # Destruimos el asteroide
            score += 1  # Sumamos puntos
            continue  # Saltamos al siguiente asteroide (importante)

        # 2. Colisión con la BASE
        # Si el asteroide cruza la línea Y de la base
        if asteroid.y > BASE_Y_POSITION - asteroid.radius:
            asteroids.remove(asteroid)  # Destruimos el asteroide
            lives -= 1  # Restamos una vida

    """Dibujamos la BASE (una línea azul gruesa)"""
    # Usamos rectangle para hacer una línea gruesa
    cv2.rectangle(game_frame,
                  (0, BASE_Y_POSITION),
                  (WINDOW_WIDTH, BASE_Y_POSITION + 5),
                  BLUE, -1)  # -1 para relleno

    """Dibujamos el ESCUDO (área de intercepción) en blanco"""
    if interception_x is not None and interception_y is not None:
        pt1 = (interception_x - INTERCEPTION_ZONE_SIZE // 2, interception_y - INTERCEPTION_ZONE_SIZE // 2)
        pt2 = (interception_x + INTERCEPTION_ZONE_SIZE // 2, interception_y + INTERCEPTION_ZONE_SIZE // 2)
        cv2.rectangle(game_frame, pt1, pt2, WHITE, 2)  # Borde blanco

    """Mostramos Puntuación y Vidas"""
    cv2.putText(game_frame,
                f"Score: {score}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    # Mostramos las vidas en rojo para que destaquen
    cv2.putText(game_frame,
                f"Lives: {lives}",
                (WINDOW_WIDTH - 150, 30),  # Arriba a la derecha
                cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    """Mostramos las ventanas"""
    cv2.imshow("Juego 'Defensor Espacial'", game_frame)
    cv2.imshow("Camera Feed", camera_frame)
    # cv2.imshow("Mask", mask) # Puedes descomentar esto para depurar

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Fin del Juego ---
# Mostramos la pantalla final
game_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
cv2.putText(game_frame, "GAME OVER", (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            RED, 3)
cv2.putText(game_frame, f"Puntuacion final: {score}", (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
cv2.imshow("Juego 'Defensor Espacial'", game_frame)
cv2.waitKey(3000)  # Espera 3 segundos

cap.release()
cv2.destroyAllWindows()