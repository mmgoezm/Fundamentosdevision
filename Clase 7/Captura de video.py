import cv2
cap = cv2.VideoCapture(0) # Captura de video de la cámara por defecto
if not cap.isOpened():
    print("Puerto no disponible.")
else:
    while True:
        ret, frame = cap.read() # tomar un frame del video

        if not ret:
            print("Error: No se pudo leer el frame.")
            break

        cv2.imshow("Video en vivo", frame)  # Muestra el frame en una ventana
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Sale del bucle si se presiona la tecla 'q'
            break
    cap.release()
    cv2.destroyAllWindows()