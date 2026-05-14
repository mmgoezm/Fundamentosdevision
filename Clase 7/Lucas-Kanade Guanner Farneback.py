import cv2
import numpy as np
import time


def flujo_optico_farneback(frame1, frame2, flujo=None):
    """ Calcula el flujo óptico Gunnar Farneback."""
    return cv2.calcOpticalFlowFarneback(frame1, frame2, flujo,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)


def flujo_optico_lucas_kanade(frame1, frame2, p0):
    """Calcula el flujo óptico de Lucas-Kanade."""
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)
    News = p1[st == 1]
    Prev = p0[st == 1]
    return News, Prev

def detectarmovimiento(flujo, umbral):
    magnitud, _ = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
    return np.mean(magnitud) > umbral


def principal():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    Grisprevio = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


    parametros_caracteristicas = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7) # Parámetros para la detección de esquinas de ShiTomasi

    color = np.random.randint(0, 255, (100, 3))  # Colores para las esquinas detectadas
    mascara = np.zeros_like(frame1)

    # Parámetros de detección de movimiento
    umbral_movimiento = 2.0  # Ajusta este valor para cambiar la sensibilidad
    duracion_alarma = 3  # Duración de la alarma en segundos
    alarma_activa = False
    tiempo_inicio_alarma = 0

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        gris = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flujo = flujo_optico_farneback(Grisprevio, gris)

        # Verifica si hay movimiento significativo
        if detectarmovimiento(flujo, umbral_movimiento):
            if not alarma_activa:
                alarma_activa = True
                tiempo_inicio_alarma = time.time()
                print("¡Movimiento detectado! Alarma activada.")

        if alarma_activa and time.time() - tiempo_inicio_alarma > duracion_alarma:
            alarma_activa = False
            print("Alarma desactivada.")

        magnitud, angulo = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        hsv[..., 0] = angulo * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)
        visualizacion_farneback = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Flujo Óptico Disperso de Lucas-Kanade
        p0 = cv2.goodFeaturesToTrack(Grisprevio, mask=None, **parametros_caracteristicas)
        buenos_nuevos, buenos_viejos = flujo_optico_lucas_kanade(Grisprevio, gris, p0)

        # Visualización para Lucas-Kanade
        for i, (nuevo, viejo) in enumerate(zip(buenos_nuevos, buenos_viejos)):
            a, b = nuevo.ravel()
            c, d = viejo.ravel()
            mascara = cv2.line(mascara, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame2 = cv2.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)
        visualizacion_lucas_kanade = cv2.add(frame2, mascara)

        # Agrega el indicador de alarma
        if alarma_activa:
            cv2.putText(visualizacion_farneback, "ALARMA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(visualizacion_lucas_kanade, "ALARMA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Muestra los resultados
        cv2.imshow('Flujo Farneback', visualizacion_farneback)
        cv2.imshow('Flujo Lucas-Kanade', visualizacion_lucas_kanade)

        Grisprevio = gris.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


principal()