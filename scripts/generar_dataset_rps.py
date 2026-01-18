import os
import random
import cv2
import kagglehub
import shutil
from tqdm import tqdm
from src.preprocesamiento.rps import procesar_resta_canales

def generar_datos():
    SEED = 42
    random.seed(SEED)

    NUM_MUESTRAS = 100  
    RUTA_SALIDA = "datos_procesados/piedra_papel_tijera"
    EXT_VALIDAS = (".png", ".jpg", ".jpeg")
    
    TRADUCCION = {
        'rock': 'piedra', 
        'paper': 'papel', 
        'scissors': 'tijeras'
    }

    print("Descargando dataset Rock-Paper-Scissors...")
    try:
        path_origen = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    except Exception as e:
        print(f"Error descargando: {e}")
        return

    ruta_base = ""
    clases_encontradas = []

    for root, dirs, _ in os.walk(path_origen):
        candidatos = [d for d in dirs if d.lower() in ["rock", "paper", "scissors"]]
        if len(candidatos) >= 2:
            ruta_base = root
            clases_encontradas = candidatos
            break

    if not clases_encontradas:
        print("No se encontraron las carpetas del dataset.")
        return 

    print(f"Clases encontradas: {clases_encontradas}")

    if os.path.exists(RUTA_SALIDA):
        shutil.rmtree(RUTA_SALIDA)

    print("\nGenerando dataset procesado...")

    for clase_ingles in clases_encontradas:
        nombre_espanol = TRADUCCION.get(clase_ingles.lower(), clase_ingles)
        
        path_clase_in = os.path.join(ruta_base, clase_ingles)
        salida_clase = os.path.join(RUTA_SALIDA, nombre_espanol)
        
        os.makedirs(salida_clase, exist_ok=True)

        archivos = [
            f for f in os.listdir(path_clase_in)
            if os.path.splitext(f)[1].lower() in EXT_VALIDAS
        ]

        if not archivos:
            print(f"No hay imagenes validas en: {path_clase_in}")
            continue

        archivos = sorted(archivos)
        muestras = random.sample(archivos, k=min(NUM_MUESTRAS, len(archivos)))

        print(f"Procesando: {nombre_espanol} ({len(muestras)} imagenes)")

        for nombre in tqdm(muestras):
            ruta_img = os.path.join(path_clase_in, nombre)
            img = cv2.imread(ruta_img)

            if img is None: continue

            mascara = procesar_resta_canales(img)

            if mascara is not None:
                ruta_final = os.path.join(salida_clase, nombre)
                cv2.imwrite(ruta_final, mascara)

    print("\nDataset RPS generado correctamente.")
    print(f"Ubicacion: {os.path.abspath(RUTA_SALIDA)}")

