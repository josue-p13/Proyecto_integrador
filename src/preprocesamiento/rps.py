import cv2
import numpy as np

def procesar_resta_canales(img, size=(256, 256)):
    """
    Segmenta la mano en imágenes de Piedra-Papel-Tijera (fondo verde)
    usando resta aritmética de canales (Verde - Rojo).

    Parámetros
    ----------
    img : np.ndarray
        Imagen original en formato BGR.
    size : tuple
        Tamaño de salida (width, height).

    Retorna
    -------
    mascara_final : np.ndarray
        Máscara binaria con la mano segmentada (255) y fondo negro (0).
        Retorna None si la imagen está vacía.
    """

    if img is None:
        return None

    # --- A. Redimensionar ---
    # Estandarizamos el tamaño para que los kernels funcionen igual siempre
    img_resized = cv2.resize(img, size)

    # --- B. Separación de Canales ---
    b, g, r = cv2.split(img_resized)

    # --- C. Aritmética de Canales (La clave del algoritmo) ---
    # El fondo verde tiene G alto y R bajo. La piel tiene R alto.
    # Restar (G - R) hace que el fondo brille y la mano se oscurezca.
    diferencia = cv2.subtract(g, r)

    # --- D. Filtrado Espacial ---
    # Suavizado para eliminar el ruido granulado de la cámara
    suave = cv2.GaussianBlur(diferencia, (5, 5), 0)

    # --- E. Umbralización de Otsu ---
    # Separa automáticamente el fondo (claro) de la mano (oscura)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- F. Inversión ---
    # Queremos la mano blanca (255) y el fondo negro (0)
    binaria = cv2.bitwise_not(binaria)

    # --- G. Limpieza Morfológica ---
    # Apertura para eliminar puntitos blancos del fondo
    kernel = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # --- H. Selección del Componente Principal ---
    # Nos quedamos solo con la mancha blanca más grande (la mano)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binaria, 
        connectivity=8
    )

    mascara_final = np.zeros_like(binaria)
    
    if num_labels > 1:
        # stats[1:] ignora el fondo (índice 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        # Buscamos el índice del área máxima
        idx_mayor = np.argmax(areas) + 1 
        
        # Pintamos solo ese componente
        mascara_final[labels == idx_mayor] = 255
    else:
        # Si no encontró nada, devolvemos lo que haya
        mascara_final = binaria

    return mascara_final