import os
import random
import cv2
import kagglehub
from tqdm import tqdm

from src.preprocesamiento.espermatozoides import (
    procesar_imagen_sperm,
    procesar_imagen_sperm_bin,
)

def generar_datos():
    # ---------------- CONFIGURACIÓN ----------------
    SEED = 56
    random.seed(SEED)

    NUM_MUESTRAS = 100  # <-- máximo de imágenes a procesar por clase
    RUTA_SALIDA_BASE = "datos_procesados"
    EXT_VALIDAS = (".bmp", ".jpg", ".jpeg", ".png")
    PROCESADORES = (
        ("espermatozoides", procesar_imagen_sperm),
        ("espermatozoides_binarizados", procesar_imagen_sperm_bin),
    )

    # ---------------- DESCARGA DATASET ----------------
    print("⬇Descargando dataset de espermatozoides...")
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
        raise RuntimeError("No se encontraron las carpetas del dataset.")

    print(f"Clases encontradas: {clases}")

    # ---------------- CREAR ESTRUCTURA DE SALIDA ----------------
    for nombre_tipo, _ in PROCESADORES:
        for clase in clases:
            os.makedirs(os.path.join(RUTA_SALIDA_BASE, nombre_tipo, clase), exist_ok=True)

    muestras_por_clase = {}
    for clase in clases:
        path_clase = os.path.join(ruta_base, clase)
        archivos = [
            f for f in os.listdir(path_clase)
            if os.path.splitext(f)[1].lower() in EXT_VALIDAS
        ]

        if not archivos:
            print(f"No hay imágenes válidas en: {path_clase}")
            continue

        archivos = sorted(archivos)
        muestras = random.sample(archivos, k=min(NUM_MUESTRAS, len(archivos)))
        muestras_por_clase[clase] = {
            "muestras": muestras,
            "total": len(archivos)
        }

    if not muestras_por_clase:
        raise RuntimeError("No se pudo construir ninguna muestra procesable para las clases encontradas.")

    # ---------------- PROCESAMIENTO Y GUARDADO ----------------
    print("\nGenerando dataset procesado...")

    for nombre_tipo, procesar in PROCESADORES:
        print(f"\nProcesando conjunto: {nombre_tipo}")
        for clase, datos in muestras_por_clase.items():
            path_clase = os.path.join(ruta_base, clase)
            salida_clase = os.path.join(RUTA_SALIDA_BASE, nombre_tipo, clase)
            muestras = datos["muestras"]
            total_archivos = datos["total"]

            print(f"Procesando clase: {clase} ({len(muestras)} de {total_archivos} imágenes)")

            for nombre in tqdm(muestras):
                ruta_img = os.path.join(path_clase, nombre)
                img = cv2.imread(ruta_img)

                if img is None:
                    continue

                _, mascara = procesar(img)

                if mascara is None:
                    continue

                ruta_salida = os.path.join(salida_clase, nombre)
                cv2.imwrite(ruta_salida, mascara)

    print("\nDataset de espermatozoides generado correctamente.")
    for nombre_tipo, _ in PROCESADORES:
        print(f"Ubicación ({nombre_tipo}): {os.path.join(RUTA_SALIDA_BASE, nombre_tipo)}")