import pickle
import cv2
import time
from training.model import apply_polynomial_2d
from common.cameras import WorldCamera, PupilDetector
from ultralytics import YOLO
import requests

# Dirección IP de la ESP32 
esp_ip = "192.168.50.106"  #IP Edu = "192.168.1.153"
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

    # Parámetros de fijacion y baja confianza
    TIEMPO_FIJACION            = 1.0   # segundos para fijación en objeto
    CONFIANZA_MINIMA           = 0.9
    TIEMPO_CONFIANZA_BAJA      = 5   # segundos de confianza baja
    objeto_actual              = None
    tiempo_inicio              = None
    señal_enviada              = False
    low_conf_start             = None
    low_conf_signal_sent       = False
    last_turned_on = None
    fixation_achieved = False

    # Mapeo de nombres a señales
    mapping = {
        'Heater':         '0',
        'Lamp':           '1',
        'Television':     '2',
        'Mechanical fan': '3',
        'Speaker':        '4'
    }
    try:
        while True:
            ret, frame = world_camera.cap.read()
            if not ret:
                break

            # Capturamos y procesamos el frame de la cámara del ojo
            frame_eye, pupil_data = pupil_detector.process_frame(pupil_detector.read_frame())
            if pupil_data is None:
                # no detectó pupila: saltamos ciclo
                continue
            
            # Extraer x, y, confidence
            x_pup, y_pup, *_ , confidence = pupil_data
            shared_pupil_data['x'] = x_pup
            shared_pupil_data['y'] = y_pup
            shared_pupil_data['confidence'] = confidence

            now = time.time()
            if confidence < CONFIANZA_MINIMA:
                # primer frame de baja confianza
                if low_conf_start is None:
                    low_conf_start = now
                    low_conf_signal_sent = False

                elapsed_low = now - low_conf_start

                # sólo una vez cuando pasa TIEMPO_CONFIANZA_BAJA
                if not low_conf_signal_sent and elapsed_low >= TIEMPO_CONFIANZA_BAJA:
                    print("Apagar Todo")
                    send_signal("5")
                    low_conf_signal_sent = True
                    last_turned_on = None

                # mostrar contador
                
                
                #'''
                if low_conf_signal_sent:
                    cv2.putText(frame,
                                "Apagar todo",
                                (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255),
                                2)


                cv2.putText(frame, f"Ojo cerrado: {elapsed_low:.1f}s",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                #'''
                
                
                # reiniciar lógica de dwell-time
                objeto_actual = None
                tiempo_inicio = None
                señal_enviada = False

                # dibujar y continuar
                cv2.imshow(pupil_detector.window_name, frame_eye)
                cv2.imshow("Mapping Real-Time", frame)
                cv2.waitKey(1)
                continue

            else:
                # ojo abierto: resetea para el próximo cierre
                low_conf_start       = None
                low_conf_signal_sent = False


            # --- Mapeo de coordenadas ---
            x_map, y_map = apply_polynomial_2d(params, [x_pup], [y_pup])
            x_map, y_map = int(round(x_map[0])), int(round(y_map[0]))

            # --- Detección de objetos ---
            results      = model.predict(source=frame, conf=0.4, iou=0.45,
                                         show=False, verbose=False, device="cpu")
            last_results = results[0]
            objeto_enfocado = None

            if hasattr(last_results, "boxes") and last_results.boxes is not None:
                boxes   = last_results.boxes.xyxy.cpu().numpy()
                cls_ids = last_results.boxes.cls.cpu().numpy()
                for i, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = map(int, box)
                    if x_min <= x_map <= x_max and y_min <= y_map <= y_max:
                        name = model.names[int(cls_ids[i])]
                        objeto_enfocado = {'name': name, 'bbox': (x_min, y_min, x_max, y_max)}
                        break


            # --- Fiajcion time y señal única ---
            if objeto_enfocado:
                if objeto_actual is None or objeto_actual['name'] != objeto_enfocado['name']:
                    objeto_actual = objeto_enfocado
                    tiempo_inicio = now
                    señal_enviada = False
                    fixation_achieved = False

                elif (now - tiempo_inicio >= TIEMPO_FIJACION and not señal_enviada):
                    nombre = objeto_actual['name']
                    # sólo enviamos si aún no lo encendimos
                    if last_turned_on != nombre:
                        print(f"Prender {nombre}")
                        send_signal(mapping.get(nombre, '0'))
                        señal_enviada   = True
                        last_turned_on  = nombre
                        fixation_achieved = True
                    # en otro caso, ya estaba encendido: no hacemos nada
            else:
                objeto_actual = None
                tiempo_inicio = None
                señal_enviada = False
                fixation_achieved = False
            
            # --- Visualización ---
            # punto de mirada
            cv2.circle(frame, (x_map, y_map), 5, (0, 0, 255), -1)
            # overlay YOLO
            annotated = last_results.plot() if last_results else frame
            # cuadro y tiempo dwell
            if objeto_actual and tiempo_inicio:
                x_min, y_min, x_max, y_max = objeto_actual['bbox']
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                elapsed = now - tiempo_inicio
                cv2.putText(annotated,
                            f"{elapsed:.1f}s",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            if fixation_achieved and objeto_actual:
                # texto fijo en la esquina superior
                cv2.putText(annotated,
                            f"Fijacion en {objeto_actual['name']}: {TIEMPO_FIJACION:.1f}s",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow(pupil_detector.window_name, frame_eye)      # muestro frame procesado
            cv2.imshow("Mapping Real-Time", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        world_camera.release()
        pupil_detector.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    from multiprocessing import Manager
    manager = Manager()
    shared_pupil_data = manager.dict(x=250, y=180, confidence=1.0)
    real_time_mapping(shared_pupil_data)
