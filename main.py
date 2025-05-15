# 01_main.py
import sys
from multiprocessing import Manager

def main():
    print("Seleccione el modo:")
    print("1. Entrenamiento (Calibración)")
    print("2. Estimación en Tiempo Real")
    mode = input("Ingrese 1 o 2: ")
    
    if mode == "1":
        # Ejecutar el entrenamiento
        import training.train_model as tm
        params = tm.train_model()
        if params is not None:
            import pickle
            with open("model.pkl", "wb") as f:
                pickle.dump(params, f)
            print("Modelo guardado en 'model.pkl'")
    elif mode == "2":
        # Ejecutar el mapeo en tiempo real
        import realtime.real_time_mapping as rtm
        manager = Manager()
        shared_pupil_data = manager.dict()
        # Inicializar con valores por defecto
        shared_pupil_data['x'] = 250
        shared_pupil_data['y'] = 180
        rtm.real_time_mapping(shared_pupil_data)
    else:
        print("Modo no reconocido.")

if __name__ == '__main__':
    main()
