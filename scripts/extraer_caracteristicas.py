import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.extraccion_caracteristicas.momentos.momentos import calcular_momentos
from src.extraccion_caracteristicas.momentos.hu import calcular_hu_momentos
from src.extraccion_caracteristicas.momentos.zernike import calcular_zernike_momentos
from src.extraccion_caracteristicas.SIFT.SIFT import crear_sift, extraer_descriptores_imagen, resumir_descriptores
from src.extraccion_caracteristicas.HOG.HOG import extraer_hog_imagen


def escalar_logaritmicamente(datos):
    """
    Aplica escala logaritmica a todos los valores numericos de un diccionario.
    
    Usa log10(abs(x) + 1) para manejar valores negativos y ceros.
    Preserva el signo del valor original.
    
    Parametros:
        datos: Diccionario con valores numericos
        
    Retorna:
        dict: Diccionario con valores escalados logaritmicamente
    """
    datos_escalados = {}
    for key, value in datos.items():
        if isinstance(value, (int, float)):
            if value == 0:
                datos_escalados[key] = 0.0
            else:
                signo = np.sign(value)
                datos_escalados[key] = signo * np.log10(abs(value) + 1)
        else:
            datos_escalados[key] = value
    return datos_escalados


def extraer_caracteristicas_dataset(ruta_imagenes_bin, ruta_salida_csv, nombre_dataset):
    """
    Extrae momentos, Hu y Zernike de imagenes binarizadas.
    
    Lee imagenes binarizadas generadas por generar_dataset_espermatozoides
    o generar_dataset_rps, calcula los tres tipos de momentos aplicando
    escala logaritmica y guarda los resultados en CSV.
    
    Parametros:
        ruta_imagenes_bin: Ruta donde estan las imagenes binarizadas
        ruta_salida_csv: Ruta donde se guardaran los archivos CSV
        nombre_dataset: Nombre del dataset para mensajes
    """
    os.makedirs(ruta_salida_csv, exist_ok=True)
    
    clases = [d for d in os.listdir(ruta_imagenes_bin) 
              if os.path.isdir(os.path.join(ruta_imagenes_bin, d))]
    
    if not clases:
        print(f"No se encontraron clases en {ruta_imagenes_bin}")
        return
    
    print(f"\nExtrayendo caracteristicas de {nombre_dataset}...")
    print(f"Clases encontradas: {clases}")
    
    datos_momentos = []
    datos_hu = []
    datos_zernike = []
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_imagenes_bin, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nProcesando clase: {clase} ({len(archivos)} imagenes)")
        
        for archivo in tqdm(archivos):
            ruta_img = os.path.join(ruta_clase, archivo)
            img_bin = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
            
            if img_bin is None:
                continue
            
            momentos_reg = calcular_momentos(img_bin)
            momentos_reg = escalar_logaritmicamente(momentos_reg)
            momentos_reg['clase'] = clase
            datos_momentos.append(momentos_reg)
            
            hu = calcular_hu_momentos(img_bin)
            hu = escalar_logaritmicamente(hu)
            hu['clase'] = clase
            datos_hu.append(hu)
            
            zernike = calcular_zernike_momentos(img_bin)
            if zernike:
                zernike = escalar_logaritmicamente(zernike)
                zernike['clase'] = clase
                datos_zernike.append(zernike)
    
    print("\nGuardando archivos CSV...")
    
    if datos_momentos:
        with open(os.path.join(ruta_salida_csv, 'momentos.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=datos_momentos[0].keys())
            writer.writeheader()
            writer.writerows(datos_momentos)
        print(f"{len(datos_momentos)} filas guardadas en momentos.csv")
    
    if datos_hu:
        with open(os.path.join(ruta_salida_csv, 'hu_momentos.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=datos_hu[0].keys())
            writer.writeheader()
            writer.writerows(datos_hu)
        print(f"{len(datos_hu)} filas guardadas en hu_momentos.csv")
    
    if datos_zernike:
        with open(os.path.join(ruta_salida_csv, 'zernike.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=datos_zernike[0].keys())
            writer.writeheader()
            writer.writerows(datos_zernike)
        print(f"{len(datos_zernike)} filas guardadas en zernike.csv")
    
    print(f"\nExtraccion completada para {nombre_dataset}")
    print(f"Archivos guardados en: {os.path.abspath(ruta_salida_csv)}")


def guardar_dataset_sift_csv(ruta_imagenes, ruta_csv, nombre_dataset):
    """Extrae descriptores SIFT de las imagenes y los guarda en CSV"""
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    
    sift = crear_sift()
    datos = []
    clases = [d for d in os.listdir(ruta_imagenes) if os.path.isdir(os.path.join(ruta_imagenes, d))]
    
    print(f"\nExtrayendo descriptores SIFT de {nombre_dataset}...")
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_imagenes, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Procesando clase: {clase} ({len(archivos)} imagenes)")
        
        for archivo in tqdm(archivos):
            ruta_img = os.path.join(ruta_clase, archivo)
            
            descriptores = extraer_descriptores_imagen(ruta_img, sift)
            if descriptores is not None:
                resumen = resumir_descriptores(descriptores)
                fila = {f'sift_{i}': val for i, val in enumerate(resumen)}
                fila['clase'] = clase
                fila['archivo'] = archivo
                datos.append(fila)
    
    if datos:
        df = pd.DataFrame(datos)
        df.to_csv(ruta_csv, index=False, encoding='utf-8')
        print(f"\n{len(datos)} filas guardadas en {ruta_csv}")


def guardar_dataset_hog_csv(ruta_imagenes, ruta_csv, nombre_dataset):
    """Extrae descriptores HOG de las imagenes y los guarda en CSV"""
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    
    datos = []
    clases = [d for d in os.listdir(ruta_imagenes) if os.path.isdir(os.path.join(ruta_imagenes, d))]
    
    print(f"\nExtrayendo descriptores HOG de {nombre_dataset}...")
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_imagenes, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Procesando clase: {clase} ({len(archivos)} imagenes)")
        
        for archivo in tqdm(archivos):
            ruta_img = os.path.join(ruta_clase, archivo)
            
            hog_desc = extraer_hog_imagen(ruta_img)
            if hog_desc is not None:
                fila = {f'hog_{i}': val for i, val in enumerate(hog_desc)}
                fila['clase'] = clase
                fila['archivo'] = archivo
                datos.append(fila)
    
    if datos:
        df = pd.DataFrame(datos)
        df.to_csv(ruta_csv, index=False, encoding='utf-8')
        print(f"\n{len(datos)} filas guardadas en {ruta_csv}")


def extraer_todas_caracteristicas():
    """
    Funcion principal que extrae caracteristicas de ambos datasets.
    
    Usa las imagenes binarizadas generadas por generar_dataset_espermatozoides
    y generar_dataset_rps, extrae momentos, Hu y Zernike aplicando
    escala logaritmica y guarda los resultados en CSV.
    """
    print("\n--- EXTRAYENDO CARACTERISTICAS DE IMAGENES BINARIZADAS ---")
    
    # Momentos de imagenes binarizadas
    extraer_caracteristicas_dataset(
        ruta_imagenes_bin="datos_procesados/espermatozoides_binarizados",
        ruta_salida_csv="caracteristicas_extraidas/momentos/espermatozoides",
        nombre_dataset="espermatozoides"
    )
    
    extraer_caracteristicas_dataset(
        ruta_imagenes_bin="datos_procesados/piedra_papel_tijera_binarizados",
        ruta_salida_csv="caracteristicas_extraidas/momentos/piedra_papel_tijera",
        nombre_dataset="piedra-papel-tijera"
    )
    
    # SIFT y HOG de imagenes en escala de grises
    print("\n--- EXTRAYENDO SIFT Y HOG ---")
    
    guardar_dataset_sift_csv(
        ruta_imagenes="datos_procesados/espermatozoides",
        ruta_csv="caracteristicas_extraidas/sift/espermatozoides/sift.csv",
        nombre_dataset="espermatozoides"
    )
    
    guardar_dataset_sift_csv(
        ruta_imagenes="datos_procesados/piedra_papel_tijera",
        ruta_csv="caracteristicas_extraidas/sift/piedra_papel_tijera/sift.csv",
        nombre_dataset="piedra-papel-tijera"
    )
    
    guardar_dataset_hog_csv(
        ruta_imagenes="datos_procesados/espermatozoides",
        ruta_csv="caracteristicas_extraidas/hog/espermatozoides/hog.csv",
        nombre_dataset="espermatozoides"
    )
    
    guardar_dataset_hog_csv(
        ruta_imagenes="datos_procesados/piedra_papel_tijera",
        ruta_csv="caracteristicas_extraidas/hog/piedra_papel_tijera/hog.csv",
        nombre_dataset="piedra-papel-tijera"
    )


if __name__ == "__main__":
    extraer_todas_caracteristicas()
