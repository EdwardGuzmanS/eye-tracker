import cv2
import pypupilext as pp
import numpy as np
from collections import deque

class Camera:
    def __init__(self, camera_index, window_name):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Error: No se pudo abrir la cámara con índice {camera_index}.")
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: No se pudo leer el fotograma.")
        return frame
    
    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)
    
    def release(self):
        self.cap.release()


class PupilDetector(Camera):
    def __init__(self, camera_index=1):
        super().__init__(camera_index, "Detección de Pupilas")
        self.algorithm = pp.PuReST()
    
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil = self.algorithm.runWithConfidence(gray)
        pupil_data = None
        
        if pupil is not None and pupil.center is not None:
            x, y = int(pupil.center[0]), int(pupil.center[1])
            major_axis = abs(int(pupil.majorAxis() / 2))
            minor_axis = abs(int(pupil.minorAxis() / 2))
            angle = pupil.angle
            confidence = pupil.outline_confidence

            if minor_axis > 0 and major_axis > 0 and not np.isnan(minor_axis) and not np.isnan(major_axis):
                if minor_axis > major_axis:
                    minor_axis, major_axis = major_axis, minor_axis
                
                pupil_data = (x, y, minor_axis, major_axis, angle, confidence)
                cv2.ellipse(frame, (x, y), (minor_axis, major_axis), angle, 0, 360, (0, 0, 255), 2)
            else:
                print("Advertencia: Se detectaron valores inválidos en los ejes de la elipse.")
        
        return frame, pupil_data
    
    def getpupildata(self):
        # Capturamos un fotograma de la cámara
        frame = self.read_frame()
        # Procesamos el fotograma para obtener los datos de la pupila
        _, pupil_data = self.process_frame(frame)
        # Si se detectó la pupila, extraemos las coordenadas (x, y)
        if pupil_data is not None:
            x, y = pupil_data[0], pupil_data[1]
            return (x, y)
        else:
            return None
        

class WorldCamera(Camera):
    def __init__(self, camera_index=0):
        super().__init__(camera_index, "Camara del entorno")
        self.click_detection_active = False  # Bandera para controlar la activación del click
        self.click_coords = deque()

    def activar_click_detection(self):
        """Activa la opción de detectar clicks en la imagen."""
        self.click_detection_active = True
        cv2.setMouseCallback(self.window_name, self.handle_click)

    def desactivar_click_detection(self):
        """Desactiva la opción de detectar clicks en la imagen."""
        self.click_detection_active = False
        # Asignamos un callback vacío para que no haga nada
        cv2.setMouseCallback(self.window_name, lambda *args: None)

    def handle_click(self, event, x, y, flags, param):
        """
        Callback para manejar eventos de mouse.
        Solo actúa si la detección de click está activada.
        """
        if not self.click_detection_active:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_coords.append((x, y))

