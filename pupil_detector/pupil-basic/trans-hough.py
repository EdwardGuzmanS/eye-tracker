import cv2
import numpy as np

# Iniciar la captura de video. 
# Asegúrate de que la cámara esté enfocada al ojo y que la imagen se centre en él.
cap = cv2.VideoCapture(1)  # Cambia el índice o la fuente según necesites

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un desenfoque para reducir el ruido y mejorar la detección de bordes
    # El tamaño del kernel y el sigma se pueden ajustar según tus necesidades
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Opcional: aplicar ecualización de histograma para mejorar el contraste
    # blurred = cv2.equalizeHist(blurred)

    # Aplicar la Transformada de Hough para detectar círculos
    # Los parámetros son:
    #   dp: la relación inversa de la resolución del acumulador respecto a la imagen original.
    #   minDist: distancia mínima entre los centros de los círculos detectados.
    #   param1: umbral alto para el detector de bordes de Canny.
    #   param2: umbral para el acumulador de círculos (cuanto menor, más círculos se detectan, pero aumenta el riesgo de falsos positivos).
    #   minRadius y maxRadius: rango de radios a buscar (ajusta estos valores en función del tamaño esperado de la pupila).
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    
    # Si se detecta al menos un círculo, se procesa la información
    if circles is not None:
        # Convertir los valores flotantes a enteros
        circles = np.uint16(np.around(circles))
        # Usualmente, la pupila es el círculo más prominente; aquí se recorren todos los detectados
        for i in circles[0, :]:
            x, y, r = i
            # Dibujar el círculo detectado (contorno) en color verde
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # Dibujar el centro del círculo (punto) en color rojo
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            # Si sólo esperas un círculo (la pupila), puedes hacer un break aquí

    # Mostrar la imagen procesada
    cv2.imshow("Seguimiento Ocular con Transformada de Hough", frame)
    
    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
