# Proyecto Integrador - 7mo Semestre
## Ingeniería en Ciencias de la Computación
### Universidad Politécnica Salesiana - Sede Quito

**Integrantes:**
* Josue Pinza
* Cristian Ati
* Bryan Gonzales
* Jhon Cordova
* Pablo Paucar

---

## 📝 Descripción del Proyecto

Este proyecto se enfoca en el desarrollo de un sistema de procesamiento y clasificación de imágenes con un enfoque en **aprendizaje no-supervisado**. El objetivo es construir un flujo completo que transforme imágenes crudas en datos estructurados para su posterior análisis y clasificación en una plataforma web.

### Metodología del Proyecto:
1.  **Limpieza de Ruido:** Aplicación de técnicas de procesamiento digital para eliminar impurezas y mejorar la calidad de las imágenes originales.
2.  **Extracción de Características:** Identificación y obtención de descriptores morfológicos y geométricos de los objetos segmentados.
3.  **Estructuración de Datos:** Almacenamiento de las características obtenidas en un formato estructurado (archivo .CSV).
4.  **Algoritmo de Aprendizaje Semisupervisado:** Entrenamiento de un modelo que utilice tanto datos etiquetados como no etiquetados para realizar la clasificación.
5.  **Despliegue Web:** Implementación de una interfaz web que reciba imágenes y realice la clasificación "en caliente" (procesamiento en tiempo real).

---

## 📊 Selección de Datasets

Para este proyecto se han seleccionado dos datasets específicos debido a su pertinencia técnica en el análisis de formas.

### 1. Rock Paper Scissors
* **Fuente:** [Kaggle - Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* **Descripción:**
    * **Instancias:** 2,188 imágenes.
    * **Clases:** Rock, Paper, Scissors.
    * **Formato:** .PNG (RGB).
    * **Resolución:** 300x200 píxeles.
* 
### 2. Sperm Morphology Image Data Set (SMIDS)
* **Fuente:** [Kaggle - SMIDS](https://www.kaggle.com/datasets/orvile/sperm-morphology-image-data-set-smids)
* **Descripción:**
    * **Instancias:** 3,000 instancias.
    * **Clases:** Normal, Anormal, Nada.
    * **Formato:** .BMP.
* 
---

## 🚀 Objetivo Final
Desarrollar una aplicación web funcional donde el usuario cargue una imagen y el sistema decida instantáneamente a qué grupo pertenece, aplicando todo el proceso de limpieza y clasificación desarrollado.