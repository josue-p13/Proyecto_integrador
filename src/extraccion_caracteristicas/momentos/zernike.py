import numpy as np
import mahotas


def calcular_zernike_momentos(img_bin):
    """
    Calcula los momentos de Zernike de una imagen binaria.
    
    Los momentos de Zernike son ortogonales y robustos al ruido,
    utilizados en analisis de forma y reconocimiento de patrones.
    
    Parametros:
        img_bin: Imagen binaria en formato numpy array
        
    Retorna:
        dict: Diccionario con momentos de Zernike (z00-zNN) o None si hay error
    """
    try:
        zernike_moments = mahotas.features.zernike_moments(
            img_bin.astype(np.uint8), 
            radius=min(img_bin.shape) // 2
        )
        
        resultado = {}
        for i, val in enumerate(zernike_moments):
            resultado[f'z{i:02d}'] = val
        
        return resultado
    except Exception as e:
        print(f"Error calculando Zernike: {e}")
        return None
