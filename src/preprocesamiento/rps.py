import cv2
import numpy as np

def procesar_rps_gris(img, size=(256, 256)):
    """
    Convierte la imagen completa a escala de grises y aplica una limpieza de ruido.
    NO elimina el fondo. Mantiene toda la información visual.
    
    Ideal para algoritmos de extracción de características (SIFT, SURF, ORB)
    que necesitan el contexto completo.
    """
    if img is None: 
        return None

    # 1. Redimensionar (Estandarizar tamaño)
    img_resized = cv2.resize(img, size)

    # 2. Convertir a Escala de Grises (Mantiene el fondo)
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. Limpieza (Suavizado Gaussiano)
    gris_limpia = cv2.GaussianBlur(gris, (5, 5), 0)

    return gris_limpia