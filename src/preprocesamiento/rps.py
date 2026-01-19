import cv2
import numpy as np

def procesar_rps(img, size=(256, 256)):
    """
    Preprocesamiento para RPS optimizado para SURF:
    1. Convierte a Grises (Mantiene fondo).
    2. Limpia ruido 'sal y pimienta' con Mediana.
    3. Aplica REALCE DE BORDES (Sharpening) para destacar características.
    """
    if img is None: 
        return None

    # 1. Redimensionar
    img_resized = cv2.resize(img, size)

    # 2. Escala de Grises (Sin eliminar el fondo)
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. Limpieza suave (Median Blur)
    # Usamos esto antes del realce para no "enfocar" el ruido.
    gris_limpia = cv2.medianBlur(gris, 3)

    # 4. REALCE DE BORDES (Sharpening)
    # Este kernel resalta las transiciones bruscas (dedos vs fondo).
    kernel_enfoque = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

    # Aplicamos el kernel
    img_enfocada = cv2.filter2D(gris_limpia, -1, kernel_enfoque)

    return img_enfocada