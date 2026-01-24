import cv2
import numpy as np

# --- FUNCION 1: BINARIZACION (Blanco y Negro puro) ---
def procesar_resta_canales(img, size=(256, 256)):
    """
    Segmenta la mano usando resta de canales y devuelve una MASCARA BINARIA.
    """
    if img is None: return None

    # A. Redimensionar
    img_resized = cv2.resize(img, size)

    # B. Separacion de Canales
    b, g, r = cv2.split(img_resized)

    # C. Aritmetica (G - R)
    diferencia = cv2.subtract(g, r)

    # D. Filtrado
    suave = cv2.GaussianBlur(diferencia, (5, 5), 0)

    # E. Otsu (BINARIZACION)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # F. Inversion
    binaria = cv2.bitwise_not(binaria)

    # G. Morfologia
    kernel = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # H. Componente Principal
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    mascara_final = np.zeros_like(binaria)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_mayor = np.argmax(areas) + 1 
        mascara_final[labels == idx_mayor] = 255
    else:
        mascara_final = binaria

    return mascara_final

# --- FUNCION 2: GRISES + REALCE (Sin Binarizar) ---
def procesar_rps_grises(img, size=(256, 256)):
    """
    Preprocesamiento: Grises + Limpieza + Realce de Bordes.
    Ideal para algoritmos de caracteristicas (SURF/SIFT).
    """
    if img is None: return None

    # 1. Redimensionar
    img_resized = cv2.resize(img, size)

    # 2. Escala de Grises (Mantiene fondo)
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. Limpieza suave
    gris_limpia = cv2.medianBlur(gris, 3)

    # 4. Realce de Bordes (Sharpening)
    kernel_enfoque = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

    img_enfocada = cv2.filter2D(gris_limpia, -1, kernel_enfoque)

    return img_enfocada