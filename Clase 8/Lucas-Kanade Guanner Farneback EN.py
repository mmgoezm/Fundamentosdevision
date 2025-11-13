# Import libraries (tools) we need.
import cv2  # This is OpenCV, for computer vision.
import numpy as np  # This is NumPy, for math and arrays.
import time  # This helps us use time (like a clock).


def flujo_optico_farneback(frame1, frame2, flujo=None):
    """
    This function finds movement. It uses the Farneback method.
    It compares two pictures (frame1 and frame2).
    """
    return cv2.calcOpticalFlowFarneback(frame1, frame2, flujo,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)


def flujo_optico_lucas_kanade(frame1, frame2, p0):
    """
    This function also finds movement. It uses the Lucas-Kanade method.
    It follows specific points (p0) from frame1 to frame2.
    """
    # These are settings for the function.
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # This function finds the new points (p1).
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

    # We only want the "good" points. (st == 1) means the point is good.
    News = p1[st == 1]  # The new positions of good points.
    Prev = p0[st == 1]  # The old positions of good points.
    return News, Prev


def detectarmovimiento(flujo, umbral):
    """
    This function checks if the movement is "big".
    'flujo' is the movement data.
    'umbral' is the limit (a number).
    """
    # We measure the *size* (magnitude) of the movement.
    magnitud, _ = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])

    # We check if the average size is bigger than the limit ('umbral').
    # If yes, it returns True (big movement).
    # If no, it returns False (small movement).
    return np.mean(magnitud) > umbral


def principal():
    """
    This is the main function. The program starts here.
    """
    cap = cv2.VideoCapture(0)  # Start the camera (webcam 0).
    ret, frame1 = cap.read()  # Get the first picture (frame) from the camera.

    # Make the picture black and white (grayscale).
    # 'Grisprevio' means "previous gray".
    Grisprevio = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # These are settings to find good points (corners) for Lucas-Kanade.
    parametros_caracteristicas = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Make 100 random colors. We use this to draw lines.
    color = np.random.randint(0, 255, (100, 3))

    # Make a new black image. It is the same size as the camera picture.
    # We will draw lines on this 'mascara' (mask).
    mascara = np.zeros_like(frame1)

    # --- Alarm Settings ---
    umbral_movimiento = 2.0  # This is the limit for "big" movement.
    duracion_alarma = 3  # Alarm stays on for 3 seconds.
    alarma_activa = False  # The alarm is 'Off' (False) at the start.
    tiempo_inicio_alarma = 0  # The alarm start time is 0.
    # --- End Alarm Settings ---

    # This is the main loop. It runs forever (until you press 'q').
    while True:
        # Get a new picture (frame) from the camera.
        ret, frame2 = cap.read()
        if not ret:
            # If there is no picture, stop the loop.
            break

        # Make the new picture black and white (grayscale).
        gris = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # --- Farneback Calculation ---
        # Compare the old gray picture ('Grisprevio') and the new gray picture ('gris').
        # 'flujo' has the movement data for all pixels.
        flujo = flujo_optico_farneback(Grisprevio, gris)

        # --- Alarm Check ---
        # Check if the movement is "big".
        if detectarmovimiento(flujo, umbral_movimiento):
            # If movement is big AND the alarm is 'Off' (not alarma_activa):
            if not alarma_activa:
                alarma_activa = True  # Turn the alarm 'On'.
                tiempo_inicio_alarma = time.time()  # Save the start time.
                print("Movement detected! Alarm ON.")

        # If the alarm is 'On' AND 3 seconds have passed:
        if alarma_activa and (time.time() - tiempo_inicio_alarma > duracion_alarma):
            alarma_activa = False  # Turn the alarm 'Off'.
            print("Alarm OFF.")

        # --- Farneback Visualization (How to *see* the movement) ---
        # 'magnitud' is the *speed* of movement.
        # 'angulo' is the *direction* of movement.
        magnitud, angulo = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
        hsv = np.zeros_like(frame1)  # Make a new empty image (HSV color).
        hsv[..., 1] = 255  # Set saturation to maximum.
        hsv[..., 0] = angulo * 180 / np.pi / 2  # Color (Hue) shows direction.
        hsv[..., 2] = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)  # Brightness (Value) shows speed.

        # Change the HSV image back to a normal BGR image (color).
        visualizacion_farneback = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # --- Lucas-Kanade Calculation & Visualization ---
        # 1. Find good points (corners) in the *old* gray image.
        p0 = cv2.goodFeaturesToTrack(Grisprevio, mask=None, **parametros_caracteristicas)

        # 2. Find where those points moved in the *new* gray image.
        buenos_nuevos, buenos_viejos = flujo_optico_lucas_kanade(Grisprevio, gris, p0)

        # 3. Draw the movement.
        # Loop for all "good" points.
        for i, (nuevo, viejo) in enumerate(zip(buenos_nuevos, buenos_viejos)):
            a, b = nuevo.ravel()  # Get the (x, y) of the new position.
            c, d = viejo.ravel()  # Get the (x, y) of the old position.

            # Draw a line from the old position (c, d) to the new position (a, b).
            # We draw on the black 'mascara'.
            mascara = cv2.line(mascara, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)

            # Draw a circle at the new position (a, b).
            # We draw on the normal camera picture ('frame2').
            frame2 = cv2.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Add the lines (from 'mascara') to the normal picture ('frame2').
        visualizacion_lucas_kanade = cv2.add(frame2, mascara)

        # --- Show Alarm Text ---
        # If the alarm is 'On':
        if alarma_activa:
            # Write "ALARMA!" in red text on the Farneback window.
            cv2.putText(visualizacion_farneback, "ALARMA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Write "ALARMA!" in red text on the Lucas-Kanade window.
            cv2.putText(visualizacion_lucas_kanade, "ALARMA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Show the Windows ---
        # Show the Farneback image in a window.
        cv2.imshow('Flujo Farneback', visualizacion_farneback)
        # Show the Lucas-Kanade image in a window.
        cv2.imshow('Flujo Lucas-Kanade', visualizacion_lucas_kanade)

        # --- Prepare for Next Loop ---
        # IMPORTANT: The new gray frame ('gris') becomes the old gray frame ('Grisprevio')
        # for the next time the loop runs.
        Grisprevio = gris.copy()

        # Wait for 1 millisecond. If the user presses the 'q' key...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # ...stop the 'while True' loop.

    # --- Clean Up ---
    # When the loop stops, turn off the camera.
    cap.release()
    # Close all the windows.
    cv2.destroyAllWindows()


# This line runs the 'principal' function to start the program.
principal()