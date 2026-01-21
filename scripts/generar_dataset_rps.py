import os
import random
import cv2
import kagglehub
import shutil
from tqdm import tqdm
from src.preprocesamiento.rps import procesar_resta_canales, procesar_rps_grises

def generar_datos():
    SEED = 42
    random.seed(SEED)
    NUM_MUESTRAS = 100  # semilla
    BASE_DIR = os.getcwd()
    
    # Rutas de Salida
    RUTA_RAIZ = os.path.join(BASE_DIR, "datos_procesados")
    RUTA_BINARIAS = os.path.join(RUTA_RAIZ, "piedra_papel_tijera_binarizados")
    RUTA_GRISES = os.path.join(RUTA_RAIZ, "piedra_papel_tijera")
        
    EXT_VALIDAS = (".png", ".jpg", ".jpeg")
    TRADUCCION = {'rock': 'piedra', 'paper': 'papel', 'scissors': 'tijeras'}
    os.makedirs(RUTA_BINARIAS, exist_ok=True)
    os.makedirs(RUTA_GRISES, exist_ok=True)

    # --- Descarga ---
    print("Descargando dataset Rock-Paper-Scissors...")
    try:
        path_origen = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    except Exception as e:
        print(f"Error descargando: {e}")
        return

    # --- Buscar Carpetas ---
    ruta_base_img = ""
    carpetas_encontradas = []
    for root, dirs, _ in os.walk(path_origen):
        cands = [d for d in dirs if d.lower() in ["rock", "paper", "scissors"]]
        if len(cands) >= 2:
            ruta_base_img = root
            carpetas_encontradas = cands
            break
    
    if not carpetas_encontradas:
        print("No se encontraron carpetas.")
        return

    print(f"Carpetas encontradas: {carpetas_encontradas}")
    print("\nIniciando procesamiento DOBLE (Binarizadas y Grises)...")

    # --- Loop Principal ---
    for clase_ingles in carpetas_encontradas:
        nombre_espanol = TRADUCCION.get(clase_ingles.lower(), clase_ingles)
        
        path_in = os.path.join(ruta_base_img, clase_ingles)
        
        # Rutas especificas para esta clase
        path_out_bin = os.path.join(RUTA_BINARIAS, nombre_espanol)
        path_out_gris = os.path.join(RUTA_GRISES, nombre_espanol)
        
        # Creamos las carpetas si no existen
        os.makedirs(path_out_bin, exist_ok=True)
        os.makedirs(path_out_gris, exist_ok=True)

        # Seleccionar imagenes
        archivos = [f for f in os.listdir(path_in) if f.lower().endswith(EXT_VALIDAS)]
        archivos = sorted(archivos)
        cantidad = len(archivos) if NUM_MUESTRAS == -1 else min(NUM_MUESTRAS, len(archivos))
        muestras = random.sample(archivos, k=cantidad)
        
        print(f"   -> Procesando '{nombre_espanol}': {cantidad} imagenes...")

        for nombre in tqdm(muestras):
            ruta_img = os.path.join(path_in, nombre)
            img_original = cv2.imread(ruta_img)
            
            if img_original is None: continue

            # 1. Generar y Guardar BINARIA
            res_binaria = procesar_resta_canales(img_original)
            if res_binaria is not None:
                cv2.imwrite(os.path.join(path_out_bin, nombre), res_binaria)

            # 2. Generar y Guardar GRISES (Realce de bordes)
            res_gris = procesar_rps_grises(img_original)
            if res_gris is not None:
                cv2.imwrite(os.path.join(path_out_gris, nombre), res_gris)

    print("\n" + "="*50)
    print("PROCESO FINALIZADO.")
    print(f"Ubicacion Binarizadas: {RUTA_BINARIAS}")
    print(f"Ubicacion Grises:      {RUTA_GRISES}")
    print("="*50)

