from scripts.generar_dataset_espermatozoides import generar_datos as generar_dataset_espermatozoides
from scripts.generar_dataset_rps import generar_datos as generar_dataset_rps

def main():
    print("\n--- INICIANDO PIPELINE: ESPERMATOZOIDES ---")
    generar_dataset_espermatozoides()

    print("--- INICIANDO PIPELINE: PIEDRA, PAPEL O TIJERA ---")
    generar_dataset_rps()

    print("\n--- TODOS LOS PROCESOS FINALIZADOS ---")

if __name__ == "__main__":
    main()