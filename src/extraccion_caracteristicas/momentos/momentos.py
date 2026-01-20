import cv2


def calcular_momentos(img_bin):
    """
    Calcula los 24 momentos regulares de una imagen binaria.
    
    Utiliza cv2.moments para extraer momentos espaciales (m), 
    centrales (mu) y normalizados (nu).
    
    Parametros:
        img_bin: Imagen binaria en formato numpy array
        
    Retorna:
        dict: Diccionario con los 24 momentos (m00-m03, mu20-mu03, nu20-nu03)
    """
    momentos = cv2.moments(img_bin)
    
    return {
        'm00': momentos['m00'],
        'm10': momentos['m10'],
        'm01': momentos['m01'],
        'm20': momentos['m20'],
        'm11': momentos['m11'],
        'm02': momentos['m02'],
        'm30': momentos['m30'],
        'm21': momentos['m21'],
        'm12': momentos['m12'],
        'm03': momentos['m03'],
        'mu20': momentos['mu20'],
        'mu11': momentos['mu11'],
        'mu02': momentos['mu02'],
        'mu30': momentos['mu30'],
        'mu21': momentos['mu21'],
        'mu12': momentos['mu12'],
        'mu03': momentos['mu03'],
        'nu20': momentos['nu20'],
        'nu11': momentos['nu11'],
        'nu02': momentos['nu02'],
        'nu30': momentos['nu30'],
        'nu21': momentos['nu21'],
        'nu12': momentos['nu12'],
        'nu03': momentos['nu03']
    }
