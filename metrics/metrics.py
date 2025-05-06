# metrics.py

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import apply_polynomial_2d

def compute_gaze_metrics(x_true, y_true, x_pred, y_pred):
    """
    Devuelve un dict con MAE, RMSE, STD, MaxE y array de errores euclídeos.
    """
    errors = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
    return {
        'MAE':   np.mean(errors),
        'RMSE':  np.sqrt(np.mean(errors**2)),
        'STD':   np.std(errors),
        'MaxE':  np.max(errors),
        'errors': errors
    }

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_and_merge_data(world_csv="world_data.csv", pupil_csv="pupil_data.csv"):
    """
    Carga los CSV de ground-truth (world) y pupila, asigna un índice
    por sesión y muestra, y los mergea para devolver arrays alineados.
    """
    # Carga
    wd = pd.read_csv(world_csv)
    pd_ = pd.read_csv(pupil_csv)

    # Índice por orden dentro de cada sesión
    wd['idx'] = wd.groupby('session').cumcount()
    pd_['idx'] = pd_.groupby('session').cumcount()

    # Merge por sesión e índice
    merged = pd.merge(wd, pd_, on=['session','idx'], how='inner',
                      suffixes=('_world','_pupil'))

    # Extraer arrays
    x_true = merged['world_x'].to_numpy()
    y_true = merged['world_y'].to_numpy()
    x_pup  = merged['pupil_x'].to_numpy()
    y_pup  = merged['pupil_y'].to_numpy()

    return x_true, y_true, x_pup, y_pup

def main():
    # 1) Carga modelo y datos
    params = load_model("model.pkl")
    x_true, y_true, x_e, y_e = load_and_merge_data("world_data.csv", "pupil_data.csv")

    # 2) Predicción del modelo
    x_pred, y_pred = apply_polynomial_2d(params, x_e, y_e)

    # 3) Cálculo de métricas
    metrics = compute_gaze_metrics(x_true, y_true, x_pred, y_pred)
    print("=== Métricas de Precisión ===")
    print(f"MAE   = {metrics['MAE']:.2f} px")
    print(f"RMSE  = {metrics['RMSE']:.2f} px")
    print(f"STD   = {metrics['STD']:.2f} px")
    print(f"MaxE  = {metrics['MaxE']:.2f} px")

    errs = metrics['errors']

    # 4) Visualizaciones

    # 4.1 Error vs. índice de muestra
    plt.figure()
    plt.plot(errs, marker='o', linestyle='-', label='Error por muestra')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Error (px)')
    plt.title('Error Euclídeo vs. muestra')
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_vs_muestra.png")

    # 4.2 Histograma de errores
    plt.figure()
    plt.hist(errs, bins=20)
    plt.xlabel('Error (px)')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de errores')
    plt.tight_layout()
    plt.savefig("histograma_errores.png")

    # 4.3 Scatter Predicho vs Real
    plt.figure()
    plt.scatter(x_true, x_pred, label='Eje X', alpha=0.6)
    plt.scatter(y_true, y_pred, label='Eje Y', alpha=0.6)
    m = min(x_true.min(), y_true.min(), x_pred.min(), y_pred.min())
    M = max(x_true.max(), y_true.max(), x_pred.max(), y_pred.max())
    plt.plot([m, M], [m, M], 'k--')
    plt.xlabel('Real (px)')
    plt.ylabel('Predicción (px)')
    plt.title('Predicho vs. Real')
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatter_predicho_vs_real.png")

    plt.show()

if __name__ == "__main__":
    main()
