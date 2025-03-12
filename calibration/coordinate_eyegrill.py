import threading
import sys
import cv2
import pygame
from collections import deque  # Importamos deque
from cameras import PupilDetector
from paterns import CalibrationApp

# Variable global para almacenar las coordenadas actuales de la pupila
pupil_coords = None

def camera_loop():
    global pupil_coords
    detector = PupilDetector()  # Esto crea la ventana "Detección de Pupilas"
    while True:
        # Capturamos y procesamos el fotograma
        frame = detector.read_frame()
        processed_frame, pupil_data = detector.process_frame(frame)
        if pupil_data is not None:
            # Se actualizan las coordenadas (x, y)
            pupil_coords = (pupil_data[0], pupil_data[1])
        # Mostramos el fotograma con OpenCV
        cv2.imshow(detector.window_name, processed_frame)
        # Salimos si se presiona 'q' en la ventana de la cámara
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    detector.release()
    cv2.destroyAllWindows()

# Extendemos la clase de la interfaz de Pygame para integrar el dato de la cámara
class IntegratedCalibrationApp(CalibrationApp):
    def __init__(self):
        super().__init__()
        self.calibration_data = deque() 

    def run(self):
        global pupil_coords
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    # Al presionar el botón de calibración se guarda el dato actual de la pupila
                    if self.calibrate_button.collidepoint(mouse_pos):
                        if pupil_coords is not None:
                            print("Guardando datos de la pupila:", pupil_coords)
                            # Se guarda en el deque únicamente la coordenada de la pupila
                            self.calibration_data.append(pupil_coords)
                        else:
                            print("No se detectaron coordenadas de la pupila.")
                        self.start_calibration()
                    # Botón de recalibrar: reinicia el proceso y limpia los datos almacenados
                    elif self.recalibrate_button.collidepoint(mouse_pos):
                        self.reset_calibration()
                        self.calibration_data.clear()

            self.draw_screen()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        # Al cerrar el programa, imprimimos el contenido del deque
        print("Deque de coordenadas de la pupila capturadas:")
        print(self.calibration_data)

if __name__ == "__main__":
    # Inicia el procesamiento de la cámara en un hilo separado
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Inicia la aplicación de calibración con Pygame en el hilo principal
    app = IntegratedCalibrationApp()
    app.run()
