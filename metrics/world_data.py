# world_logger_sessions.py

import csv
import cv2
from cameras import WorldCamera

def record_world_data(output_csv="world_data.csv"):
    cam = WorldCamera()
    cam.activar_click_detection()
    recording = False
    data = []
    session_idx = 0

    print("Controles:")
    print("  s = Iniciar grabación (nueva sesión)")
    print("  e = Detener grabación (fin de sesión)")
    print("  q = Salir y guardar CSV")

    while True:
        ret, frame = cam.cap.read()
        if not ret:
            break

        cv2.imshow(cam.window_name, frame)
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

        # Mientras grabamos, volcamos todos los clicks acumulados
        if recording and cam.click_coords:
            # vaciamos la cola de clicks para no repetirlos
            while cam.click_coords:
                x, y = cam.click_coords.popleft()
                data.append((session_idx, x, y))

    cam.release()
    cv2.destroyAllWindows()

    # Guardar CSV con columnas: session, world_x, world_y
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "world_x", "world_y"])
        writer.writerows(data)

    print(f"[+] Datos guardados en '{output_csv}' ({len(data)} clicks, {session_idx} sesiones)")

if __name__ == "__main__":
    record_world_data()
