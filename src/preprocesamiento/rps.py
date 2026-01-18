import kagglehub
import os
import cv2
import numpy as np
import random
import shutil

# Configuracion inicial
SEED = 42
random.seed(SEED)
NUM_IMAGENES = 100  

BASE_DIR = os.getcwd()

# 1. Descarga del Dataset
print("Verificando/Descargando dataset...")
try:
    path_origen = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    print(f"Dataset ubicado en: {path_origen}")
except Exception as e:
    print(f"Error descargando dataset: {e}")
    exit()

# 2. Localizar carpetas originales
carpetas_encontradas = []
for root, dirs, _ in os.walk(path_origen):
    candidatos = [d for d in dirs if d.lower() in ["rock", "paper", "scissors"]]
    if len(candidatos) >= 2:
        ruta_base_dataset = root
        carpetas_encontradas = candidatos
        break

if not carpetas_encontradas:
    print("Error: No se encontraron las carpetas del dataset.")
    exit()

# Funcion de procesamiento
def procesar_resta_canales(img):
    if img is None: return None

    img = cv2.resize(img, (256, 256))

    # Separacion de Canales
    b, g, r = cv2.split(img)

    # Resta de Canales (Verde - Rojo)
    # El fondo verde tiene G alto, la piel tiene R alto.
    # Al restar G - R, el fondo se mantiene brillante y la mano se oscurece.
    diferencia = cv2.subtract(g, r)

    # Filtrado para reducir ruido de camara
    suave = cv2.GaussianBlur(diferencia, (5, 5), 0)

    # Otsu para segmentar automaticamente fondo vs mano
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertir para tener mano blanca sobre fondo negro
    binaria = cv2.bitwise_not(binaria)

    # Apertura morfologica para eliminar ruido pequeño
    kernel = np.ones((5,5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # Conservar solo el componente conectado mas grande (la mano)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, connectivity=8)

    mascara_final = np.zeros_like(binaria)
    if num_labels > 1:
        # stats[1:] ignora el fondo (indice 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_mayor = np.argmax(areas) + 1
        mascara_final[labels == idx_mayor] = 255
    else:
        mascara_final = binaria

    return mascara_final
