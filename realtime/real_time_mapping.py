import pickle
import cv2
import time
from training.model import apply_polynomial_2d
from common.cameras import WorldCamera, PupilDetector
from ultralytics import YOLO
import requests

# Dirección IP de la ESP32 
esp_ip = "192.168.50.106"  
model = YOLO("modelo_final.pt")


def send_signal(signal: str):
    try:
        resp = requests.get(f"http://{esp_ip}/?signal={signal}", timeout=1)
        print(f"[ESP32] Señal {signal} enviada, respuesta: {resp.text}")
    except Exception as e:
        print(f"[ESP32] Error enviando señal {signal}: {e}")

def real_time_mapping(shared_pupil_data):
    # Cargar el modelo entrenado
    with open("model.pkl", "rb") as f:
        params = pickle.load(f)
    
    # Inicializar las cámaras
    world_camera = WorldCamera()
    pupil_detector = PupilDetector()  # Instancia única para detectar la pupila

    # Variables para la lógica de dwell time
    TIEMPO_FIJACION = 1  # segundos
    objeto_actual = None
    tiempo_inicio = None

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
            # Si no se detectó la pupila, se mantiene el último valor
            pass

        # Leer la posición actual de la pupila desde el diccionario compartido
        current_pupil = (shared_pupil_data.get('x', 250), shared_pupil_data.get('y', 180))
        
        # Aplicar el modelo para mapear la posición de la pupila a la escena
        x_mapped, y_mapped = apply_polynomial_2d(params, [current_pupil[0]], [current_pupil[1]])
        x_mapped = int(round(x_mapped[0]))
        y_mapped = int(round(y_mapped[0]))
        
        # Modelo de detección de objetos
        results = model.predict(source=frame, conf=0.4, iou=0.45, show=False, verbose=False, device="cpu")
        last_results = results[0]  

        # Inicializar variable para almacenar el objeto bajo la mirada (si existe)
        objeto_enfocado = None

        # Extraer bounding boxes del resultado (asumiendo que se tiene la propiedad boxes.xyxy)
        if hasattr(last_results, "boxes") and last_results.boxes is not None:
            # Convertir los bounding boxes a un array NumPy, si es necesario
            boxes = last_results.boxes.xyxy.cpu().numpy() if hasattr(last_results.boxes.xyxy, 'cpu') else last_results.boxes.xyxy
            # Obtener los índices de clase para cada detección
            cls_ids = last_results.boxes.cls.cpu().numpy() if hasattr(last_results.boxes.cls, 'cpu') else last_results.boxes.cls
            # Iterar sobre cada bounding box
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                # Verificar si el punto mapeado se encuentra dentro del bounding box
                if x_mapped >= x_min and x_mapped <= x_max and y_mapped >= y_min and y_mapped <= y_max:
                    # Obtener el nombre del objeto a partir del índice de clase
                    class_id = int(cls_ids[i])
                    object_name = model.names[class_id]
                    objeto_enfocado = {'name': object_name, 'bbox': (x_min, y_min, x_max, y_max)}
                    break  # Se asume que solo interesa el primer objeto que cumpla la condición

        # Obtener el tiempo actual
        tiempo_actual = time.time()

        # Lógica para determinar el dwell time
        if objeto_enfocado:
            # Si se cambia de objeto o no había objeto antes, se reinicia el contador
            if objeto_actual is None or objeto_actual['name'] != objeto_enfocado['name']:
                objeto_actual = objeto_enfocado
                tiempo_inicio = tiempo_actual
            else:
                # Si supera el tiempo de fijación, dibuja y envía señal
                if tiempo_actual - tiempo_inicio >= TIEMPO_FIJACION:
                    texto = f"Fijación en {objeto_actual['name']}"
                    cv2.putText(frame, texto, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    mapping = {
                        'heater': '0',
                        'lamp': '1',
                        'television': '2',
                        'mechanical fan' : '3',
                        'speaker' : '4'
                    }
                    signal = mapping.get(objeto_actual['name'], '0')
                    send_signal(signal)
        else:
            # Si la mirada no se encuentra sobre ningún objeto, reiniciamos las variables
            objeto_actual = None
            tiempo_inicio = None

        # Dibujar un punto rojo en la posición mapeada sobre la imagen del entorno
        cv2.circle(frame, (x_mapped, y_mapped), 5, (0, 0, 255), -1)
        
        # Obtener el frame anotado a partir del modelo YOLO (si está disponible)
        if last_results:
            annotated_frame = last_results.plot()
        else:
            annotated_frame = frame

        # Opcional: Dibujar el bounding box del objeto en foco y mostrar el tiempo acumulado
        if objeto_actual is not None and tiempo_inicio is not None:
            x_min, y_min, x_max, y_max = objeto_actual['bbox']
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            dwell_time_elapsed = tiempo_actual - tiempo_inicio
            cv2.putText(annotated_frame, f"Tiempo: {dwell_time_elapsed:.1f}s", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow(pupil_detector.window_name, frame_eye)
        cv2.imshow("Mapping Real-Time", annotated_frame)
        
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
