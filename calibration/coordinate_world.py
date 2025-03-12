import cv2
from cameras import WorldCamera
from collections import deque

def run_coordinate_world():
    world_camera = WorldCamera()
    world_camera.activar_click_detection()
    
    while True:
        ret, frame = world_camera.cap.read()
        world_camera.show_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    world_camera.release()
    # Devuelve el deque de coordenadas
    return world_camera.click_coords