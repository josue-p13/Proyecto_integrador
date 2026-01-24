from scripts.generar_dataset_espermatozoides import generar_datos as generar_dataset_espermatozoides
from scripts.generar_dataset_rps import generar_datos as generar_dataset_rps
from scripts.extraer_caracteristicas import extraer_todas_caracteristicas
from scripts.generar_embeddings_espermatozoides import generar_embeddings_espermatozoides
from scripts.generar_embeddings_rps import generar_embeddings_rps
def main():
    print("\n--- INICIANDO PIPELINE: ESPERMATOZOIDES ---")
    generar_dataset_espermatozoides()

    print("\n--- INICIANDO PIPELINE: PIEDRA, PAPEL O TIJERA ---")
    generar_dataset_rps()
    
    print("\n--- EXTRAYENDO TODAS LAS CARACTERISTICAS ---")
    extraer_todas_caracteristicas()

    print("\n--- EXTRAYENDO LOS EMBEDDINGS ---")
    generar_embeddings_espermatozoides()
    generar_embeddings_rps()
    print("\n--- TODOS LOS PROCESOS FINALIZADOS ---")

if __name__ == "__main__":
    main()