#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:46:03 2025

@author: edu
"""
    
import cv2

def thresholding_pupil_detection(frame, threshold_value=16):
    """
    Método basado en umbral y centro de masa para detectar la pupila.

    Parámetros:
    - frame: Imagen en escala de grises.
    - threshold_value: Valor del umbral (opcional, por defecto 16).
    
    Retorna:
    - (cx, cy): Coordenadas del centro de masa de la pupila.
    - thresh: Imagen binarizada con el umbral aplicado.
    """
    
    # Aplicar umbral binario con el valor especificado (por defecto 16)
    _, thresh = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Calcular momentos de la imagen binaria
    moments = cv2.moments(thresh)
    cx, cy = None, None

    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    
    return (cx, cy), thresh


def starburst():
    """Metodo Starbust basado en deteccion de bordes y luego usando un algoritmo de RASNAC para buscar la mejor eclipse"""
    pass

def paralelogramos():
    """Se busca una ubicacion aproximada de la pupila y  despues se usan paralelogramas para encontrar el centro de masa"""
    pass

def umbral_paralelogramo():
    """Es similar a def paralelogramo pero antes se umbraliza y se hacen operaciones morfologicas"""
    pass

def reflejos_corneales():
    """Se buscan los reflejos y luego se expande una mancha en busca del centroide"""
    pass

def umbral_ajuste():
    """Se umbraliza, se extraen los puntos de contorno entre pupila e iris  luego se ajusta con el metodo RASNAC"""
    pass

def features_hass():
    """Se ubica la pupila con caracteristicas de haar, se escogio la posicion con algoritmo k means, y luego ajuste RASNAC"""
    pass



""" Metodos mas recientes para ambientes naturales """

def SET():
    """Extrae los pixeles de pupilas basados en luminancia, se extraen los bordes con Convex Hull luego se ajustan las elipses"""
    pass
def ExCuSe():
    """Se detectan los bordes, Operaciones morfologicas y luego una funcion de proyeccion angular integral, y ajustan la elipse con minimos cuadrados"""
    pass

def ElSe():
    """Similar al ExCuSe pero con mejores operaciones morfologicas se cambia la funcion angular por reflejos ponderados"""
    pass


"""Metodos con redes neuronales"""