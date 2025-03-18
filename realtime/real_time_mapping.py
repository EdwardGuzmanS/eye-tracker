# realtime/real_time_mapping.py
import pickle
import cv2
from training.model import apply_polynomial_2d
from common.cameras import WorldCamera, PupilDetector

def real_time_mapping(shared_pupil_data):
    # Cargar el modelo entrenado
    with open("model.pkl", "rb") as f:
        params = pickle.load(f)
    
    # Inicializar las cámaras
    world_camera = WorldCamera()
    pupil_detector = PupilDetector()  # Instancia única para detectar la pupila

    while True:
        ret, frame = world_camera.cap.read()
        if not ret:
            break

        # Capturamos y procesamos el frame de la cámara del ojo
        frame_eye = pupil_detector.read_frame()
        processed_frame, pupil_data = pupil_detector.process_frame(frame_eye)
        
        if pupil_data is not None:
            # Actualizamos el diccionario compartido con los datos actuales
            shared_pupil_data['x'] = pupil_data[0]
            shared_pupil_data['y'] = pupil_data[1]
        else:
            # Si no se detectó pupila, se mantiene el último valor
            pass

        # Leer la posición actual de la pupila desde el diccionario compartido
        current_pupil = (shared_pupil_data.get('x', 250), shared_pupil_data.get('y', 180))
        
        # Aplicar el modelo para mapear la posición de la pupila a la escena
        x_mapped, y_mapped = apply_polynomial_2d(params, [current_pupil[0]], [current_pupil[1]])
        x_mapped = int(round(x_mapped[0]))
        y_mapped = int(round(y_mapped[0]))
        
        # Dibujar un punto rojo en la posición mapeada en la imagen del entorno
        cv2.circle(frame, (x_mapped, y_mapped), 10, (0, 0, 255), -1)
        cv2.imshow(pupil_detector.window_name, frame_eye)
        cv2.imshow("Mapping Real-Time", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    world_camera.release()
    pupil_detector.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from multiprocessing import Manager
    manager = Manager()
    shared_pupil_data = manager.dict()
    # Inicializamos con un valor por defecto
    shared_pupil_data['x'] = 250
    shared_pupil_data['y'] = 180
    real_time_mapping(shared_pupil_data)
