import cv2
import numpy as np

def procesar_imagen_sperm(img, size=(256, 256)):
    """
    1. Limpieza de ruido con Filtro de Mediana.
    2. Realce de bordes (Sharpening) con kernel de convolución.
    
    Retorna:
        img_resized: Imagen a color original (para dibujar los resultados).
        img_enfocada: Imagen gris procesada (para entregar al algoritmo SURF).
    """
    if img is None:
        return None, None

    # 1. Redimensionar (Para visualización y estandarización)
    img_resized = cv2.resize(img, size)
    
    # 2. Convertir a Gris
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # PASO A: LIMPIEZA (Filtro de Mediana)
    # ---------------------------------------------------------
    gris_limpia = cv2.medianBlur(gris, 3)

    # ---------------------------------------------------------
    # PASO B: REALCE DE BORDES 
    # ---------------------------------------------------------
    # Este kernel hace que los bordes del espermatozoide resalten más,
    # compensando lo borroso de la imagen original.
    kernel_enfoque = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

    # Aplicamos el filtro de realce sobre la imagen limpia
    img_enfocada = cv2.filter2D(gris_limpia, -1, kernel_enfoque)

    return img_resized, img_enfocada