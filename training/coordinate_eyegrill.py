# training/coordinate_eyegrill.py
import cv2
from common.cameras import PupilDetector

def camera_process(pupil_queue):
    print("Iniciando cámara de pupila en proceso separado...")
    detector = PupilDetector()  # Ventana "Detección de Pupilas"
    while True:
        frame = detector.read_frame()
        processed_frame, pupil_data = detector.process_frame(frame)
        if pupil_data is not None:
            # Vaciar la cola para mantener solo el último dato
            while not pupil_queue.empty():
                try:
                    pupil_queue.get_nowait()
                except Exception:
                    break
            pupil_queue.put((pupil_data[0], pupil_data[1]))
        cv2.imshow(detector.window_name, processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    detector.release()
    cv2.destroyAllWindows()
    print("Proceso de cámara de pupila finalizado.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    q = multiprocessing.Queue(maxsize=1)
    camera_process(q)
