import os
import random
import cv2
import kagglehub
import shutil
from tqdm import tqdm
from src.preprocesamiento.rps import procesar_rps_gris

def generar_datos():
    SEED = 42 # Semilla
    random.seed(SEED)
    NUM_MUESTRAS = 100  
    
    BASE_DIR = os.getcwd()
    RUTA_SALIDA = os.path.join(BASE_DIR, "datos_procesados", "piedra_papel_tijera")
    EXT_VALIDAS = (".png", ".jpg", ".jpeg")
    
    TRADUCCION = {
        'rock': 'piedra', 
        'paper': 'papel', 
        'scissors': 'tijeras'
    }

    # ================= EJECUCIÓN =================
    # 1. Limpiar carpeta anterior
    if os.path.exists(RUTA_SALIDA):
        print(f"🧹 Limpiando carpeta de salida: {RUTA_SALIDA}")
        shutil.rmtree(RUTA_SALIDA)

    # 2. Descargar Dataset
    print("Descargando dataset Rock-Paper-Scissors...")
    try:
        path_origen = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    except Exception as e:
        print(f"Error descargando: {e}")
        return

    # 3. Buscar las carpetas correctas
    ruta_base_img = ""
    carpetas_encontradas = []

    for root, dirs, _ in os.walk(path_origen):
        candidatos = [d for d in dirs if d.lower() in ["rock", "paper", "scissors"]]
        if len(candidatos) >= 2:
            ruta_base_img = root
            carpetas_encontradas = candidatos
            break
    
    if not carpetas_encontradas:
        print("No se encontraron las carpetas de imágenes.")
        return

    print(f"Carpetas encontradas: {carpetas_encontradas}")
    print("\nIniciando procesamiento (Solo Grises + Limpieza)...")
    # 4. Procesar cada clase
    for clase_ingles in carpetas_encontradas:
        nombre_espanol = TRADUCCION.get(clase_ingles.lower(), clase_ingles)
        
        dir_entrada = os.path.join(ruta_base_img, clase_ingles)
        dir_salida = os.path.join(RUTA_SALIDA, nombre_espanol)
        
        os.makedirs(dir_salida, exist_ok=True)

        # Listar imágenes
        archivos = [f for f in os.listdir(dir_entrada) if f.lower().endswith(EXT_VALIDAS)]
        
        # Seleccionar muestras
        archivos = sorted(archivos)
        cantidad = len(archivos) if NUM_MUESTRAS == -1 else min(NUM_MUESTRAS, len(archivos))
        muestras = random.sample(archivos, k=cantidad)
        
        print(f"   -> Procesando '{nombre_espanol}': {cantidad} imágenes...")

        for nombre_archivo in tqdm(muestras):
            ruta_completa = os.path.join(dir_entrada, nombre_archivo)
            
            # Leer
            img_original = cv2.imread(ruta_completa)
            
            # Procesar
            img_procesada = procesar_rps_gris(img_original)
            
            # Guardar
            if img_procesada is not None:
                cv2.imwrite(os.path.join(dir_salida, nombre_archivo), img_procesada)

    print("\n" + "="*50)
    print(f"Imágenes guardadas en:\n -> {RUTA_SALIDA}")
    print("="*50)

