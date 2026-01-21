import cv2
import numpy as np

def procesar_imagen_sperm(img, size=(256, 256)):
    """
    Preprocesa una imagen de espermatozoide realzando bordes y suavizando ruido,
    para facilitar inspección manual o entrenamiento de modelos.
    Parámetros
    ----------
        img : np.ndarray
            Imagen original en formato BGR.
        size : tuple
            Tamaño de salida (width, height).
    Retorna
    -------
    img_resized : np.ndarray
        Imagen redimensionada en escala de grises (mantiene la información visual).
    img_enfocada : np.ndarray
        Imagen con bordes realzados y ruido reducido.
    """
    if img is None:
        return None, None

    # 1. Redimensionar MEJORADO
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convertir a Gris
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # PASO A: LIMPIEZA CONSERVADORA (Mediana)
    # ---------------------------------------------------------
    # Solo 3 para no borrar los pocos detalles que quedan.
    # "Mediana: preserva mejor los bordes".
    gris_limpia = cv2.medianBlur(gris, 3)

    # ---------------------------------------------------------
    # PASO B: REALCE DE BORDES (Filtro "Ganancia 1")
    # ---------------------------------------------------------
    kernel_fuerte = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    
    img_enfocada = cv2.filter2D(gris_limpia, -1, kernel_fuerte)

    return img_resized, img_enfocada


def procesar_imagen_sperm_bin(img, size=(256, 256)):
    """
    Preprocesa una imagen de espermatozoide para segmentar
    cabeza y cola, eliminando ruido y preservando estructuras finas.
    Parámetros
    ----------
        Imagen original en formato BGR.
    size : tuple
        Tamaño de salida (width, height).
    Retorna
    -------
    img_resized : np.ndarray
        Imagen original redimensionada.
    mascara_final : np.ndarray
        Máscara binaria del espermatozoide segmentado.
    """

    if img is None:
        return None, None
    # --- A. Redimensionar ---
    img_resized = cv2.resize(img, size)
    h, w = size
    centro_img = (w // 2, h // 2)
    # --- B. Escala de grises ---
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # --- C. Suavizado ligero (preserva cola) ---
    gris_suave = cv2.medianBlur(gris, 3)
    # --- D. Estimación de fondo ---
    fondo = np.median(gris_suave)
    # --- E. Umbral sensible (cola) ---
    _, binaria = cv2.threshold(
        gris_suave,
        fondo - 12,
        255,
        cv2.THRESH_BINARY_INV
    )
    # --- F. Bordes finos ---
    bordes = cv2.Canny(gris_suave, 20, 60)
    bordes = cv2.dilate(bordes, np.ones((2, 2), np.uint8), iterations=1)
    # --- G. Combinación ---
    combinado = cv2.bitwise_or(binaria, bordes)
    # --- H. Operaciones morfológicas ---
    combinado = cv2.morphologyEx(
        combinado,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )
    combinado = cv2.morphologyEx(
        combinado,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8)
    )
    # --- I. Selección del componente principal ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combinado,
        connectivity=8
    )
    mascara_final = np.zeros_like(combinado)
    if num_labels > 1:
        mejor_idx = -1
        mejor_distancia = float("inf")
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 200:
                continue
            cx, cy = centroids[i]
            distancia = np.sqrt(
                (cx - centro_img[0]) ** 2 +
                (cy - centro_img[1]) ** 2
            )
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_idx = i
        if mejor_idx > 0:
            mascara_final[labels == mejor_idx] = 255
    else:
        mascara_final = combinado

    return img_resized, mascara_final