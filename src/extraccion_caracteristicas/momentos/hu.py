import cv2


def calcular_hu_momentos(img_bin):
    """
    Calcula los 7 momentos de Hu de una imagen binaria.
    
    Los momentos de Hu son invariantes a traslacion, escala y rotacion,
    lo que los hace utiles para reconocimiento de patrones.
    
    Parametros:
        img_bin: Imagen binaria en formato numpy array
        
    Retorna:
        dict: Diccionario con los 7 momentos de Hu (hu1-hu7)
    """
    momentos = cv2.moments(img_bin)
    hu_moments = cv2.HuMoments(momentos).flatten()
    
    return {
        'hu1': hu_moments[0],
        'hu2': hu_moments[1],
        'hu3': hu_moments[2],
        'hu4': hu_moments[3],
        'hu5': hu_moments[4],
        'hu6': hu_moments[5],
        'hu7': hu_moments[6]
    }
