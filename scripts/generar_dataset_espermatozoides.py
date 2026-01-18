import os
import random
import cv2
import kagglehub
from tqdm import tqdm

from src.preprocesamiento.espermatozoides import procesar_imagen_sperm as procesar_imagen_espermatozoide

# ---------------- CONFIGURACIÓN ----------------
SEED = 56
random.seed(SEED)

NUM_MUESTRAS = 100  # <-- máximo de imágenes a procesar por clase
RUTA_SALIDA = "datos_procesados/espermatozoides"
EXT_VALIDAS = (".bmp", ".jpg", ".jpeg", ".png")

# ---------------- DESCARGA DATASET ----------------
print("⬇️ Descargando dataset de espermatozoides...")
path_origen = kagglehub.dataset_download(
    "orvile/sperm-morphology-image-data-set-smids"
)

# ---------------- BUSCAR CARPETAS ----------------
ruta_base = ""
clases = []

for root, dirs, _ in os.walk(path_origen):
    candidatos = [d for d in dirs if "sperm" in d.lower() or "normal" in d.lower()]
    if len(candidatos) >= 2:
        ruta_base = root
        clases = candidatos
        break

if not clases:
    raise RuntimeError("❌ No se encontraron las carpetas del dataset.")

print(f"✅ Clases encontradas: {clases}")

# ---------------- CREAR ESTRUCTURA DE SALIDA ----------------
for clase in clases:
    os.makedirs(os.path.join(RUTA_SALIDA, clase), exist_ok=True)

# ---------------- PROCESAMIENTO Y GUARDADO ----------------
print("\n🧪 Generando dataset procesado...")

for clase in clases:
    path_clase = os.path.join(ruta_base, clase)
    salida_clase = os.path.join(RUTA_SALIDA, clase)

    archivos = [
        f for f in os.listdir(path_clase)
        if os.path.splitext(f)[1].lower() in EXT_VALIDAS
    ]

    if not archivos:
        print(f"⚠️ No hay imágenes válidas en: {path_clase}")
        continue

    # Muestreo reproducible (siempre mismas 100 con la misma semilla)
    archivos = sorted(archivos)  # estabiliza el muestreo entre PCs
    muestras = random.sample(archivos, k=min(NUM_MUESTRAS, len(archivos)))

    print(f"Procesando clase: {clase} ({len(muestras)} de {len(archivos)} imágenes)")

    for nombre in tqdm(muestras):
        ruta_img = os.path.join(path_clase, nombre)
        img = cv2.imread(ruta_img)

        if img is None:
            continue

        _, mascara = procesar_imagen_espermatozoide(img)

        if mascara is None:
            continue

        ruta_salida = os.path.join(salida_clase, nombre)
        cv2.imwrite(ruta_salida, mascara)

print("\n✅ Dataset de espermatozoides generado correctamente.")
print(f"📁 Ubicación: {RUTA_SALIDA}")