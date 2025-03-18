# training/train_model.py
import multiprocessing
import time
import numpy as np
from collections import deque

from .coordinate_world import run_coordinate_world
from .coordinate_eyegrill import camera_process
from .paterns import run_calibration
from .model import calibrate_polynomial_2d

def train_model():
    manager = multiprocessing.Manager()
    shared_coords = manager.list()             # Puntos del mundo: (X_si, Y_si)
    shared_calibration_data = manager.list()     # Puntos del ojo: (X_ei, Y_ei)
    pupil_queue = multiprocessing.Queue(maxsize=1)
    
    # 1. Capturar la escena
    print("Ejecutando coordinate_world...")
    world_proc = multiprocessing.Process(target=run_coordinate_world, args=(shared_coords,))
    world_proc.start()
    world_proc.join()
    print("Captura del entorno finalizada.")
    print("Coordenadas del mundo:", list(shared_coords))
    
    # 2. Capturar datos de la pupila mediante la cámara y la interfaz de calibración
    print("Iniciando procesos de detección de pupila y calibración...")
    cam_proc = multiprocessing.Process(target=camera_process, args=(pupil_queue,))
    calib_proc = multiprocessing.Process(target=run_calibration, args=(pupil_queue, shared_calibration_data))
    cam_proc.start()
    calib_proc.start()
    cam_proc.join()
    calib_proc.join()
    print("Proceso de calibración finalizado.")
    print("Coordenadas del ojo (pupil):", list(shared_calibration_data))
    
    # 3. Emparejar puntos
    if len(shared_coords) != len(shared_calibration_data):
        print("Error: La cantidad de puntos no coincide.")
        return None
    x_si, y_si = [], []
    x_ei, y_ei = [], []
    for (ws_x, ws_y), (ei_x, ei_y) in zip(shared_coords, shared_calibration_data):
        x_si.append(ws_x)
        y_si.append(ws_y)
        x_ei.append(ei_x)
        y_ei.append(ei_y)
    
    # 4. Entrenar el modelo
    params = calibrate_polynomial_2d(x_ei, y_ei, x_si, y_si)
    print("Modelo entrenado. Coeficientes:")
    print("Mx =", params['Mx'])
    print("My =", params['My'])
    return params

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    params = train_model()
    if params is not None:
        # Guardar el modelo usando pickle
        import pickle
        with open("model.pkl", "wb") as f:
            pickle.dump(params, f)
        print("Modelo guardado en 'model.pkl'")
