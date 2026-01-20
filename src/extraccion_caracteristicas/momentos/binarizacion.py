import cv2
import numpy as np


def binarizar_otsu(img_gris):
    """
    Binariza una imagen en escala de grises usando el metodo de Otsu.
    
    El metodo de Otsu calcula automaticamente el umbral optimo para
    separar el objeto del fondo maximizando la varianza entre clases.
    
    Parametros:
        img_gris: Imagen en escala de grises (numpy array)
        
    Retorna:
        numpy array: Imagen binarizada (valores 0 y 255)
    """
    _, img_bin = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_bin


def binarizar_adaptativa(img_gris, block_size=11, C=2):
    """
    Binariza una imagen usando umbralización adaptativa.
    
    Util cuando hay variaciones de iluminacion en la imagen.
    Calcula el umbral para cada region pequeña de la imagen.
    
    Parametros:
        img_gris: Imagen en escala de grises (numpy array)
        block_size: Tamaño del vecindario para calcular el umbral (debe ser impar)
        C: Constante que se resta del umbral calculado
        
    Retorna:
        numpy array: Imagen binarizada (valores 0 y 255)
    """
    img_bin = cv2.adaptiveThreshold(
        img_gris, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        C
    )
    return img_bin


def binarizar_imagen(img, metodo='otsu'):
    """
    Binariza una imagen usando el metodo especificado.
    
    Convierte la imagen a escala de grises si es necesario y aplica
    el algoritmo de binarizacion seleccionado.
    
    Parametros:
        img: Imagen en formato BGR o escala de grises (numpy array)
        metodo: Metodo de binarizacion ('otsu' o 'adaptativa')
        
    Retorna:
        numpy array: Imagen binarizada (valores 0 y 255) o None si falla
    """
    if img is None:
        return None
    
    # Convertir a escala de grises si es necesario
    if len(img.shape) == 3:
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gris = img.copy()
    
    # Aplicar metodo de binarizacion
    if metodo == 'otsu':
        return binarizar_otsu(img_gris)
    elif metodo == 'adaptativa':
        return binarizar_adaptativa(img_gris)
    else:
        # Por defecto usar umbral fijo en 127
        _, img_bin = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
        return img_bin
