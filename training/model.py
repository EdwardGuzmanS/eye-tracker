# training/model.py
import numpy as np

def calibrate_polynomial_2d(x_ei, y_ei, x_si, y_si):
    """
    Ajusta un modelo polinómico de segundo orden que mapea (x_ei, y_ei) -> (x_si, y_si),
    aplicando normalización.
    Retorna un diccionario con coeficientes y parámetros de normalización.
    """
    x_ei = np.array(x_ei, dtype=float)
    y_ei = np.array(y_ei, dtype=float)
    x_si = np.array(x_si, dtype=float)
    y_si = np.array(y_si, dtype=float)
    n = len(x_ei)
    assert n == len(y_ei) == len(x_si) == len(y_si), "Las listas deben tener la misma longitud."

    mu_xe, std_xe = x_ei.mean(), x_ei.std()
    mu_ye, std_ye = y_ei.mean(), y_ei.std()
    mu_xs, std_xs = x_si.mean(), x_si.std()
    mu_ys, std_ys = y_si.mean(), y_si.std()

    epsilon = 1e-12
    std_xe = max(std_xe, epsilon)
    std_ye = max(std_ye, epsilon)
    std_xs = max(std_xs, epsilon)
    std_ys = max(std_ys, epsilon)

    x_ei_norm = (x_ei - mu_xe) / std_xe
    y_ei_norm = (y_ei - mu_ye) / std_ye
    x_si_norm = (x_si - mu_xs) / std_xs
    y_si_norm = (y_si - mu_ys) / std_ys

    A = np.column_stack([
        np.ones(n),
        x_ei_norm,
        y_ei_norm,
        x_ei_norm * y_ei_norm,
        x_ei_norm**2,
        y_ei_norm**2
    ])

    Mx, _, _, _ = np.linalg.lstsq(A, x_si_norm, rcond=None)
    My, _, _, _ = np.linalg.lstsq(A, y_si_norm, rcond=None)

    params = {
        'Mx': Mx,
        'My': My,
        'mu_xe': mu_xe, 'std_xe': std_xe,
        'mu_ye': mu_ye, 'std_ye': std_ye,
        'mu_xs': mu_xs, 'std_xs': std_xs,
        'mu_ys': mu_ys, 'std_ys': std_ys
    }
    return params

def apply_polynomial_2d(params, x_ei, y_ei):
    """
    Aplica el modelo polinómico para mapear (x_ei, y_ei) a (x_si, y_si),
    usando la misma normalización que en calibrate_polynomial_2d.
    """
    x_ei = np.array(x_ei, dtype=float)
    y_ei = np.array(y_ei, dtype=float)

    Mx = params['Mx']
    My = params['My']
    mu_xe, std_xe = params['mu_xe'], params['std_xe']
    mu_ye, std_ye = params['mu_ye'], params['std_ye']
    mu_xs, std_xs = params['mu_xs'], params['std_xs']
    mu_ys, std_ys = params['mu_ys'], params['std_ys']

    x_ei_norm = (x_ei - mu_xe) / std_xe
    y_ei_norm = (y_ei - mu_ye) / std_ye

    A = np.column_stack([
        np.ones_like(x_ei_norm),
        x_ei_norm,
        y_ei_norm,
        x_ei_norm * y_ei_norm,
        x_ei_norm**2,
        y_ei_norm**2
    ])

    x_si_norm = A @ Mx
    y_si_norm = A @ My

    x_si = x_si_norm * std_xs + mu_xs
    y_si = y_si_norm * std_ys + mu_ys

    return x_si, y_si
