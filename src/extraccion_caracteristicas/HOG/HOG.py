import cv2
from skimage.feature import hog

def extraer_hog_imagen(
    ruta_imagen,
    resize=(128, 64)
):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

    imagen = cv2.resize(imagen, resize)

    caracteristicas = hog(
        imagen,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

    return caracteristicas