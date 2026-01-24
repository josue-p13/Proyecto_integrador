import cv2
import numpy as np


def binarizar_espermatozoides(img, size=(256, 256)):
    """
    Binariza imagenes de espermatozoides con segmentacion avanzada.
    
    Segmenta cabeza y cola del espermatozoide, eliminando ruido y
    preservando estructuras finas. Usa umbralizacion adaptativa,
    deteccion de bordes y seleccion del componente mas cercano al centro.
    
    Parametros:
        img: Imagen en formato BGR o escala de grises (numpy array)
        size: Tamaño de salida (ancho, alto) para redimensionar
        
    Retorna:
        numpy array: Imagen binarizada (valores 0 y 255) o None si falla
    """
    if img is None:
        return None

    # A. Redimensionar
    img_resized = cv2.resize(img, size)
    h, w = size
    centro_img = (w // 2, h // 2)

    # B. Escala de grises
    if len(img_resized.shape) == 3:
        gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        gris = img_resized.copy()

    # C. Suavizado ligero (preserva cola)
    gris_suave = cv2.medianBlur(gris, 3)

    # D. Estimacion de fondo
    fondo = np.median(gris_suave)

    # E. Umbral sensible (cola)
    _, binaria = cv2.threshold(
        gris_suave,
        fondo - 12,
        255,
        cv2.THRESH_BINARY_INV
    )

    # F. Bordes finos
    bordes = cv2.Canny(gris_suave, 20, 60)
    bordes = cv2.dilate(bordes, np.ones((2, 2), np.uint8), iterations=1)

    # G. Combinacion
    combinado = cv2.bitwise_or(binaria, bordes)

    # H. Operaciones morfologicas
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

    # I. Seleccion del componente principal
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

    return mascara_final


def binarizar_rps(img, size=(256, 256)):
    """
    Binariza imagenes de piedra-papel-tijera con segmentacion de mano.
    
    Segmenta la mano en fondo verde usando resta de canales (G - R).
    Utiliza Otsu para umbralizar automaticamente y selecciona el
    componente mas grande (la mano).
    
    Parametros:
        img: Imagen en formato BGR (numpy array)
        size: Tamaño de salida (ancho, alto) para redimensionar
        
    Retorna:
        numpy array: Imagen binarizada (valores 0 y 255) o None si falla
    """
    if img is None:
        return None

    # A. Redimensionar
    img_resized = cv2.resize(img, size)

    # B. Separacion de canales
    b, g, r = cv2.split(img_resized)

    # C. Aritmetica de canales (fondo verde tiene G alto, piel tiene R alto)
    diferencia = cv2.subtract(g, r)

    # D. Filtrado espacial
    suave = cv2.GaussianBlur(diferencia, (5, 5), 0)

    # E. Umbralizacion de Otsu
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # F. Inversion (queremos la mano blanca)
    binaria = cv2.bitwise_not(binaria)

    # G. Limpieza morfologica
    kernel = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # H. Seleccion del componente principal
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binaria, 
        connectivity=8
    )

    mascara_final = np.zeros_like(binaria)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_mayor = np.argmax(areas) + 1
        mascara_final[labels == idx_mayor] = 255
    else:
        mascara_final = binaria

    return mascara_final


def binarizar_imagen(img, metodo='espermatozoides'):
    """
    Binariza una imagen usando el metodo especificado para cada tipo de dataset.
    
    Aplica el algoritmo de binarizacion apropiado segun el tipo de imagen.
    
    Parametros:
        img: Imagen en formato BGR o escala de grises (numpy array)
        metodo: Tipo de binarizacion ('espermatozoides' o 'rps')
        
    Retorna:
        numpy array: Imagen binarizada (valores 0 y 255) o None si falla
    """
    if img is None:
        return None
    
    if metodo == 'espermatozoides':
        return binarizar_espermatozoides(img)
    elif metodo == 'rps':
        return binarizar_rps(img)
    else:
        # Fallback: binarizacion simple con Otsu
        if len(img.shape) == 3:
            img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gris = img.copy()
        _, img_bin = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_bin
