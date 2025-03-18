# training/coordinate_world.py
import cv2
from common.cameras import WorldCamera

def run_coordinate_world(shared_coords):
    world_camera = WorldCamera(0)
    world_camera.activar_click_detection()
    while True:
        ret, frame = world_camera.cap.read()
        #world_camera.show_frame(frame)
        cv2.imshow(world_camera.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    world_camera.release()
    cv2.destroyAllWindows()
    # Guarda los puntos de click en la lista compartida
    shared_coords.extend(world_camera.click_coords)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    from multiprocessing import Manager
    manager = Manager()
    shared_coords = manager.list()
    run_coordinate_world(shared_coords)
    print("Coordenadas de clic registradas:", list(shared_coords))
