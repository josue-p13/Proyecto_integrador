import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

from src.extraccion_caracteristicas.momentos.momentos import calcular_momentos
from src.extraccion_caracteristicas.momentos.hu import calcular_hu_momentos
from src.extraccion_caracteristicas.momentos.zernike import calcular_zernike_momentos
from src.extraccion_caracteristicas.momentos.binarizacion import binarizar_imagen


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


def binarizar_dataset(ruta_imagenes, ruta_salida_bin, nombre_dataset):
    """
    Binariza todas las imagenes de un dataset y las guarda.
    
    Lee imagenes procesadas, aplica binarizacion de Otsu y guarda
    las imagenes binarizadas manteniendo la estructura de carpetas.
    
    Parametros:
        ruta_imagenes: Ruta donde estan las carpetas con imagenes por clase
        ruta_salida_bin: Ruta donde se guardaran las imagenes binarizadas
        nombre_dataset: Nombre del dataset para mensajes
    """
    clases = [d for d in os.listdir(ruta_imagenes) 
              if os.path.isdir(os.path.join(ruta_imagenes, d))]
    
    if not clases:
        print(f"No se encontraron clases en {ruta_imagenes}")
        return
    
    print(f"Binarizando {nombre_dataset}...")
    print(f"Clases: {clases}")
    
    for clase in clases:
        ruta_clase_in = os.path.join(ruta_imagenes, clase)
        ruta_clase_out = os.path.join(ruta_salida_bin, clase)
        os.makedirs(ruta_clase_out, exist_ok=True)
        
        archivos = [f for f in os.listdir(ruta_clase_in) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nBinarizando clase: {clase} ({len(archivos)} imagenes)")
        
        for archivo in tqdm(archivos):
            ruta_img = os.path.join(ruta_clase_in, archivo)
            img = cv2.imread(ruta_img)
            
            if img is None:
                continue
            
            img_bin = binarizar_imagen(img, metodo='otsu')
            
            if img_bin is not None:
                ruta_salida = os.path.join(ruta_clase_out, archivo)
                cv2.imwrite(ruta_salida, img_bin)
    
    print(f"Binarizacion completada. Guardado en: {os.path.abspath(ruta_salida_bin)}")


def extraer_caracteristicas_dataset(ruta_imagenes_bin, ruta_salida_csv, nombre_dataset):
    """
    Extrae momentos, Hu y Zernike de imagenes binarizadas.
    
    Lee imagenes binarizadas, calcula los tres tipos de momentos
    aplicando escala logaritmica y guarda los resultados en CSV.
    
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


def extraer_todas_caracteristicas():
    """
    Funcion principal que extrae caracteristicas de ambos datasets.
    
    Primero binariza las imagenes procesadas y las guarda.
    Luego extrae momentos, Hu y Zernike de las imagenes binarizadas
    aplicando escala logaritmica antes de guardar en CSV.
    """
    print("\n--- PASO 1: BINARIZANDO IMAGENES ---")
    
    binarizar_dataset(
        ruta_imagenes="datos_procesados/espermatozoides",
        ruta_salida_bin="caracteristicas_extraidas/momentos/binarizacion/espermatozoides",
        nombre_dataset="espermatozoides"
    )
    
    binarizar_dataset(
        ruta_imagenes="datos_procesados/piedra_papel_tijera",
        ruta_salida_bin="caracteristicas_extraidas/momentos/binarizacion/piedra_papel_tijera",
        nombre_dataset="piedra-papel-tijera"
    )
    
    print("\n--- PASO 2: EXTRAYENDO CARACTERISTICAS ---")
    
    extraer_caracteristicas_dataset(
        ruta_imagenes_bin="caracteristicas_extraidas/momentos/binarizacion/espermatozoides",
        ruta_salida_csv="caracteristicas_extraidas/momentos/espermatozoides",
        nombre_dataset="espermatozoides"
    )
    
    extraer_caracteristicas_dataset(
        ruta_imagenes_bin="caracteristicas_extraidas/momentos/binarizacion/piedra_papel_tijera",
        ruta_salida_csv="caracteristicas_extraidas/momentos/piedra_papel_tijera",
        nombre_dataset="piedra-papel-tijera"
    )


if __name__ == "__main__":
    extraer_todas_caracteristicas()
