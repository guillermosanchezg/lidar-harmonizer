import pandas as pd
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()
    
    print("="*60)
    print("🔎 DIAGNÓSTICO DE ETIQUETAS V2X")
    print("="*60)

    # 1. Leer el JSON de Stats
    stats_path = f"./outputs/stats/{args.model_name}.pth_stats.json"
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            class_dist = stats.get("class_distribution", {})
            print("\n[1] ETIQUETAS EN EL JSON (stats):")
            print(f"Tipo de dato de las llaves: {type(list(class_dist.keys())[0])}")
            print(f"Contenido: {class_dist}")
    except Exception as e:
        print(f"Error leyendo JSON: {e}")

    # 2. Leer el CSV de K-NN
    csv_path = f"./outputs/knn_results/k_neighbors_all_{args.model_name}.csv"
    try:
        df = pd.read_csv(csv_path)
        
        # Original Labels
        print("\n[2] COLUMNA 'Original Label' (Query):")
        primer_valor = df["Original Label"].iloc[0]
        print(f"Tipo de dato: {type(primer_valor)}")
        print(f"Valores únicos en el CSV: {df['Original Label'].unique()[:10]}")
        
        if "Mapped Label" in df.columns:
            print("\n[3] COLUMNA 'Mapped Label':")
            print(f"Tipo de dato: {type(df['Mapped Label'].iloc[0])}")
            print(f"Valores únicos: {df['Mapped Label'].unique()[:10]}")

        # Neighbor Labels (Lo que escupe K-NN)
        print("\n[4] COLUMNA 'Neighbor Labels' (Vecinos):")
        vecinos_str = df["Neighbor Labels"].iloc[0]
        print(f"Tipo de dato crudo (CSV): {type(vecinos_str)}")
        print(f"Contenido crudo: {vecinos_str}")
        vecinos_eval = eval(vecinos_str)
        print(f"Tipo de dato tras eval(): {type(vecinos_eval[0])} (Elemento 0)")
        print(f"Contenido tras eval(): {vecinos_eval}")
        
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
