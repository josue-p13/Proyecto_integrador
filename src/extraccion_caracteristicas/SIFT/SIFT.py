import cv2
import os
import numpy as np


def crear_sift(nfeatures=0):
    """
    nfeatures = 0 → sin límite de puntos clave
    """
    return cv2.SIFT_create(nfeatures=nfeatures)


def extraer_descriptores_imagen(ruta_imagen, sift):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

    _, descriptores = sift.detectAndCompute(imagen, None)
    return descriptores


def resumir_descriptores(descriptores, dimension=128):
    """
    SIFT genera descriptores de 128 dimensiones
    """
    if descriptores is None:
        return np.zeros(dimension)
    return np.mean(descriptores, axis=0)
