# pupil_logger_sessions.py

import csv
import cv2
from cameras import PupilDetector

def record_pupil_data(output_csv="pupil_data.csv"):
    detector = PupilDetector()
    recording = False
    data = []
    session_idx = 0

    print("Controles:")
    print("  s = Iniciar grabación (nueva sesión)")
    print("  e = Detener grabación (fin de sesión)")
    print("  q = Salir y guardar CSV")

    while True:
        frame = detector.read_frame()
        processed_frame, pupil = detector.process_frame(frame)

        cv2.imshow(detector.window_name, processed_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not recording:
            session_idx += 1
            recording = True
            print(f"[*] Sesión {session_idx} iniciada")
        elif key == ord('e') and recording:
            recording = False
            print(f"[*] Sesión {session_idx} finalizada")
        elif key == ord('q'):
            print("[*] Saliendo...")
            break

        # Si estamos grabando y hay detección, guardamos (sesión, x, y)
        if recording and pupil is not None:
            x, y = pupil[0], pupil[1]
            data.append((session_idx, x, y))

    detector.release()
    cv2.destroyAllWindows()

    # Guardar CSV con sesión como índice
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "pupil_x", "pupil_y"])
        writer.writerows(data)

    print(f"[+] Datos guardados en '{output_csv}' ({len(data)} muestras, {session_idx} sesiones)")

if __name__ == "__main__":
    record_pupil_data()
