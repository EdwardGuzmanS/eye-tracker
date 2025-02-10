import cv2
import numpy as np

# Iniciar la captura de video (puede ser también la carga de una imagen)
cap = cv2.VideoCapture(1)  # Ajusta el índice o la fuente según corresponda

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Asumimos que 'frame' es un primer plano del ojo
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Umbralización inversa para resaltar la pupila (ajusta el valor de 'thresh_value' según tus condiciones)
    thresh_value = 30  # Valor de ejemplo; puede variar según iluminación y cámara
    _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    # Operaciones morfológicas para mejorar la imagen binaria (opcional)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=4)
    thresh = cv2.dilate(thresh, kernel, iterations=6)
    
    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Suponemos que el contorno de mayor área corresponde a la pupila
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Dibujar el centro de la pupila en la imagen original
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    
    # Mostrar las imágenes de depuración
    cv2.imshow("Imagen Original", frame)
    cv2.imshow("Umbralizado", thresh)
    
    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
