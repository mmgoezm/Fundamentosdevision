#
"""
======================================================================
==  Ejemplo de Juego Interactivo con Visión por Computador          ==
==               "Atrapa Mariposas"                                 ==
==               Basado en el ejemplo de Pelotas                    ==
======================================================================
"""

import cv2
import numpy as np
import random
import time  # Necesitamos 'time' para el temporizador

"""Parametros de la Ventana y del Juego"""
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BUTTERFLY_SIZE = 15  # Tamaño de la mariposa (radio para dibujar)
MAX_BUTTERFLIES = 12  # Máximo de mariposas en pantalla
SPAWN_INTERVAL = 20  # Qué tan rápido aparecen (menos es más rápido)
GAME_DURATION = 60  # Duración del juego en segundos
INTERCEPTION_ZONE_SIZE = 100  # Tamaño de la "red"

# --- Colores ---
BLACK = (0, 0, 0)  # Negro
WHITE = (255, 255, 255)  # Blanco
YELLOW = (0, 255, 255)  # Amarillo (para las mariposas)

"""Variables del Juego"""


class Mariposa:
    def __init__(self):
        # Aparece en un lugar aleatorio dentro de la ventana
        self.x = random.randint(BUTTERFLY_SIZE, WINDOW_WIDTH - BUTTERFLY_SIZE)
        self.y = random.randint(BUTTERFLY_SIZE, WINDOW_HEIGHT - BUTTERFLY_SIZE)
        # Velocidad inicial aleatoria
        self.vx = random.randint(-3, 3)
        self.vy = random.randint(-3, 3)
        # Aseguramos que no empiece quieta
        if self.vx == 0: self.vx = 1
        if self.vy == 0: self.vy = -1

        self.size = BUTTERFLY_SIZE
        self.color = YELLOW

    def update(self):
        """
        Actualiza la posición de la mariposa.
        Simula el movimiento de "revoloteo" y rebote.
        """
        self.x += self.vx
        self.y += self.vy

        # --- Lógica de "Revoloteo" (Flutter) ---
        # Con un 10% de probabilidad, cambia un poco su dirección
        if random.random() < 0.1:
            self.vx += random.randint(-1, 1)
            self.vy += random.randint(-1, 1)

            # Mantenemos la velocidad dentro de un límite (para que no se descontrole)
            self.vx = max(-4, min(4, self.vx))
            self.vy = max(-4, min(4, self.vy))

        # --- Lógica de "Rebote" en las paredes ---
        if self.x <= self.size or self.x >= WINDOW_WIDTH - self.size:
            self.vx *= -1  # Invierte la velocidad en X

        if self.y <= self.size or self.y >= WINDOW_HEIGHT - self.size:
            self.vy *= -1  # Invierte la velocidad en Y

    def draw(self, frame):
        """
        Dibuja la mariposa (como un corbatín) usando dos triángulos.
        """
        # Triángulo 1 (ala izquierda)
        p1_l = (self.x - self.size, self.y - self.size)
        p2_l = (self.x - self.size, self.y + self.size)
        p3_l = (self.x, self.y)
        triangle_left = np.array([p1_l, p2_l, p3_l], dtype=np.int32)

        # Triángulo 2 (ala derecha)
        p1_r = (self.x + self.size, self.y - self.size)
        p2_r = (self.x + self.size, self.y + self.size)
        p3_r = (self.x, self.y)
        triangle_right = np.array([p1_r, p2_r, p3_r], dtype=np.int32)

        # Rellenamos los triángulos
        cv2.fillPoly(frame, [triangle_left], self.color)
        cv2.fillPoly(frame, [triangle_right], self.color)


"""captura de la cámara y dibujado del juego"""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Variables del estado del juego
butterflies = []  # Lista para almacenar las mariposas
score = 0  # Puntuación
spawn_counter = 0  # Contador para generar mariposas
start_time = time.time()  # Momento exacto en que empieza el juego

""""Bucle principal del juego"""
while True:
    # Creamos un 'lienzo' (frame) negro para cada iteración.
    game_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # --- Lógica del Temporizador ---
    elapsed_time = time.time() - start_time
    time_remaining = max(0, int(GAME_DURATION - elapsed_time))

    # Si el tiempo se acaba, salimos del bucle
    if time_remaining <= 0:
        break

    # --- Lógica de la Cámara (igual que antes) ---
    ret, camera_frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    camera_frame = cv2.flip(camera_frame, 1)
    fram1hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

    # Mismos rangos para detectar el objeto negro/oscuro
    lowerblack = np.array([0, 0, 0])
    upperblack = np.array([180, 255, 50])

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
    # Generamos una nueva mariposa si toca y no hemos llegado al máximo
    if spawn_counter >= SPAWN_INTERVAL and len(butterflies) < MAX_BUTTERFLIES:
        butterflies.append(Mariposa())
        spawn_counter = 0

    # Actualizamos y dibujamos cada mariposa
    for bfly in butterflies[:]:  # Usamos [:] para poder borrar de la lista mientras iteramos
        bfly.update()
        bfly.draw(game_frame)

        # Comprobamos si la "red" (zona de intercepción) atrapa la mariposa
        if (interception_x is not None and interception_y is not None and
                abs(bfly.x - interception_x) < INTERCEPTION_ZONE_SIZE // 2 and
                abs(bfly.y - interception_y) < INTERCEPTION_ZONE_SIZE // 2):
            butterflies.remove(bfly)  # La eliminamos
            score += 1  # Sumamos puntos

        # YA NO necesitamos la parte de "si la pelota cae",
        # porque las mariposas rebotan y no se salen.

    """Mostramos la 'red' (área de intercepción) en blanco"""
    if interception_x is not None and interception_y is not None:
        pt1 = (interception_x - INTERCEPTION_ZONE_SIZE // 2, interception_y - INTERCEPTION_ZONE_SIZE // 2)
        pt2 = (interception_x + INTERCEPTION_ZONE_SIZE // 2, interception_y + INTERCEPTION_ZONE_SIZE // 2)
        cv2.rectangle(game_frame, pt1, pt2, WHITE, 2)

    """Mostramos Puntuación y Tiempo"""
    cv2.putText(game_frame,
                f"Score: {score}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    cv2.putText(game_frame,
                f"Time: {time_remaining}",
                (WINDOW_WIDTH - 150, 30),  # Arriba a la derecha
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    """Mostramos las ventanas"""
    cv2.imshow("Juego 'Atrapa Mariposas'", game_frame)
    cv2.imshow("Camera Feed", camera_frame)
    # cv2.imshow("Mask", mask) # Puedes descomentar esto para depurar la máscara

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Fin del Juego ---
print(f"¡Juego terminado! Puntuación final: {score}")

# Añadimos un mensaje final en la pantalla del juego
game_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
cv2.putText(game_frame, "FIN DEL JUEGO", (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, WHITE, 3)
cv2.putText(game_frame, f"Puntuacion final: {score}", (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
cv2.imshow("Juego 'Atrapa Mariposas'", game_frame)
cv2.waitKey(3000)  # Espera 3 segundos antes de cerrar

cap.release()
cv2.destroyAllWindows()