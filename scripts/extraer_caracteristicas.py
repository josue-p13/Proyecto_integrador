import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

from src.extraccion_caracteristicas.momentos.momentos import calcular_momentos
from src.extraccion_caracteristicas.momentos.hu import calcular_hu_momentos
from src.extraccion_caracteristicas.momentos.zernike import calcular_zernike_momentos


def aplicar_escala_logaritmica(datos):
    """
    Aplica escala logaritmica a todos los valores numericos.
    
    Maneja valores positivos, negativos y ceros usando la formula:
    sign(x) * log(abs(x) + 1)
    
    Parametros:
        datos: Lista de diccionarios con los datos
        
    Retorna:
        list: Lista de diccionarios con valores escalados logaritmicamente
    """
    datos_escalados = []
    
    for fila in datos:
        fila_escalada = {}
        for key, value in fila.items():
            if key == 'clase':
                fila_escalada[key] = value
            else:
                fila_escalada[key] = np.sign(value) * np.log(np.abs(value) + 1)
        datos_escalados.append(fila_escalada)
    
    return datos_escalados


def extraer_caracteristicas_dataset(ruta_imagenes, ruta_salida, nombre_dataset):
    """
    Extrae momentos, Hu y Zernike de todas las imagenes de un dataset.
    
    Procesa cada clase del dataset, calcula los tres tipos de momentos
    y guarda los resultados en archivos CSV separados.
    
    Parametros:
        ruta_imagenes: Ruta donde estan las carpetas con imagenes por clase
        ruta_salida: Ruta donde se guardaran los archivos CSV
        nombre_dataset: Nombre del dataset para mensajes (ej: 'espermatozoides')
    """
    os.makedirs(ruta_salida, exist_ok=True)
    
    clases = [d for d in os.listdir(ruta_imagenes) 
              if os.path.isdir(os.path.join(ruta_imagenes, d))]
    
    if not clases:
        print(f"No se encontraron clases en {ruta_imagenes}")
        return
    
    print(f"Clases encontradas: {clases}")
    
    datos_momentos = []
    datos_hu = []
    datos_zernike = []
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_imagenes, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nProcesando clase: {clase} ({len(archivos)} imagenes)")
        
        for archivo in tqdm(archivos):
            ruta_img = os.path.join(ruta_clase, archivo)
            img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            momentos_reg = calcular_momentos(img_bin)
            momentos_reg['clase'] = clase
            datos_momentos.append(momentos_reg)
            
            hu = calcular_hu_momentos(img_bin)
            hu['clase'] = clase
            datos_hu.append(hu)
            
            zernike = calcular_zernike_momentos(img_bin)
            if zernike:
                zernike['clase'] = clase
                datos_zernike.append(zernike)
    
    print("\nAplicando escala logaritmica...")
    datos_momentos = aplicar_escala_logaritmica(datos_momentos)
    datos_hu = aplicar_escala_logaritmica(datos_hu)
    datos_zernike = aplicar_escala_logaritmica(datos_zernike)
    
    print("Guardando archivos CSV...")
    
    if datos_momentos:
        with open(os.path.join(ruta_salida, 'momentos.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=datos_momentos[0].keys())
            writer.writeheader()
            writer.writerows(datos_momentos)
        print(f"{len(datos_momentos)} filas guardadas en momentos.csv")
    
    if datos_hu:
        with open(os.path.join(ruta_salida, 'hu_momentos.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=datos_hu[0].keys())
            writer.writeheader()
            writer.writerows(datos_hu)
        print(f"{len(datos_hu)} filas guardadas en hu_momentos.csv")
    
    if datos_zernike:
        with open(os.path.join(ruta_salida, 'zernike.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=datos_zernike[0].keys())
            writer.writeheader()
            writer.writerows(datos_zernike)
        print(f"{len(datos_zernike)} filas guardadas en zernike.csv")
    
    print(f"\nExtraccion completada para {nombre_dataset}")
    print(f"Archivos guardados en: {os.path.abspath(ruta_salida)}")


def extraer_todas_caracteristicas():
    """
    Funcion principal que extrae caracteristicas de ambos datasets.
    
    Procesa espermatozoides y piedra-papel-tijera, extrayendo momentos,
    Hu y Zernike de cada uno y guardando en sus respectivas carpetas.
    """
    print("\n--- EXTRAYENDO CARACTERISTICAS: ESPERMATOZOIDES ---")
    extraer_caracteristicas_dataset(
        ruta_imagenes="datos_procesados/espermatozoides",
        ruta_salida="caracteristicas_extraidas/momentos/espermatozoides",
        nombre_dataset="espermatozoides"
    )
    
    print("\n--- EXTRAYENDO CARACTERISTICAS: PIEDRA-PAPEL-TIJERA ---")
    extraer_caracteristicas_dataset(
        ruta_imagenes="datos_procesados/piedra_papel_tijera",
        ruta_salida="caracteristicas_extraidas/momentos/piedra_papel_tijera",
        nombre_dataset="piedra-papel-tijera"
    )


if __name__ == "__main__":
    extraer_todas_caracteristicas()
