import cv2
import numpy as np

def Detector_de_objetos(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Mismo ancho que el video de referencia
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# Misma altura que el video de referencia
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para generación del video
    out = cv2.VideoWriter('output_detected.mp4', fourcc, 30, (frame_width, frame_height)) #parametrización del video

    while cap.isOpened():
        if not ret or frame2 is None: # Finalizan los fotogramas
            break
        frameaux= frame1.copy() #copia auxiliar para el video de salida

        # Detección de movimiento
        diff = cv2.absdiff(frame1, frame2) #Diferrencia
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=9)
        erode = cv2.erode(thresh, None, iterations=2)
        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 2100:
                continue
            cv2.rectangle(frameaux, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frameaux, "Movimiento", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


        # Detección de objetos negros
        lowerblack = np.array([0, 0, 0])
        upperblack = np.array([100, 100, 150])
        maskblack = cv2.inRange(frame1, lowerblack, upperblack)
        contoursblack, _ = cv2.findContours(maskblack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour_black in contoursblack:
            (x, y, w, h) = cv2.boundingRect(contour_black)
            if cv2.contourArea(contour_black) < 2100:
                continue
            cv2.rectangle(frameaux, (x, y), (x + w, y + h), (0, 0, 0), 2)  # rectangulo rojo para los objetos negros.
            cv2.putText(frameaux, "Negros", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)



        # Detección de objetos rojos
        lowerred1 = np.array([0, 100, 100])
        upperred1 = np.array([10, 255, 255])
        lowerred2 = np.array([160, 100, 100])
        upperred2 = np.array([200, 255, 255])
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        maskred1 = cv2.inRange(hsv, lowerred1, upperred1)
        maskred2 = cv2.inRange(hsv, lowerred2, upperred2)
        maskred = cv2.bitwise_or(maskred1, maskred2)

        contoursred, _ = cv2.findContours(maskred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contourred in contoursred:
            (x, y, w, h) = cv2.boundingRect(contourred)
            if cv2.contourArea(contourred) < 700:
                continue
            cv2.rectangle(frameaux, (x, y), (x + w, y + h), (0, 0, 255), 2)  # rectangulo rojo para los objetos negros.
            cv2.putText(frameaux, "Rojos", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Video", frameaux)
        out.write(frameaux)  # Grabar el frame modificado
        cv2.imshow("Movimiento2", erode)
        cv2.imshow("Movimiento1", dilated)
        cv2.imshow("Mascara negra", maskblack)
        cv2.imshow("Mascara roja", maskred)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()

Detector_de_objetos('Video.mov') #