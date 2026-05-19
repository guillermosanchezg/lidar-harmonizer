import pandas as pd
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Generar lista de candidatos a reetiquetar")
    parser.add_argument("model_name", type=str)
    parser.add_argument("--k", type=int, default=10, help="Número de vecinos a evaluar")
    args = parser.parse_args()

    results_csv = f"./outputs/knn_results/k_neighbors_all_{args.model_name}.csv"
    stats_path = f"./outputs/stats/{args.model_name}_stats.json"
    out_path = f"./outputs/knn_results/relabel_candidates_{args.model_name}.csv"

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"No se encuentra el CSV de KNN: {results_csv}")

    # Cargar mapeo de nombres para que sea legible por humanos
    idx_to_name = {}
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        idx_to_name = {int(v): str(k) for k, v in stats.get("class_mapping", {}).items()}

    print(f"Analizando predicciones con K={args.k}...")
    df = pd.read_csv(results_csv)
    
    relabel_list = []

    for _, row in df.iterrows():
        true_label = int(row["Original Label"])
        # Limitar a K vecinos
        neighbors = eval(row["Neighbor Labels"])[:args.k] if isinstance(row["Neighbor Labels"], str) else row["Neighbor Labels"][:args.k]
        distances = eval(row["Distances"])[:args.k] if isinstance(row["Distances"], str) else row["Distances"][:args.k]

        # Calcular predicción por distancia inversa
        w_inv = [1 / (d + 1e-6) if d > 0 else 1.0 for d in distances]
        scores = {}
        for n, w in zip(neighbors, w_inv):
            scores[n] = scores.get(n, 0) + w
            
        pred_label = max(scores, key=scores.get)

        # Si hay fallo, lo guardamos para revisar/reetiquetar
        if true_label != pred_label:
            # Calcular una "confianza" básica (0 a 1) para saber cuán seguro está el modelo
            confidence = scores[pred_label] / sum(scores.values())
            
            relabel_list.append({
                "Path": row["Query"],
                "Old_Label_ID": true_label,
                "Old_Label_Name": idx_to_name.get(true_label, str(true_label)),
                "Predicted_Label_ID": pred_label,
                "Predicted_Label_Name": idx_to_name.get(pred_label, str(pred_label)),
                "Model_Confidence": round(confidence, 3)
            })

    out_df = pd.DataFrame(relabel_list)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    print("\n=== RESUMEN DE REETIQUETADO ===")
    print(f"Total de nubes evaluadas: {len(df)}")
    print(f"Total de discrepancias (candidatos a reetiquetar): {len(out_df)}")
    print(f"Lista guardada en: {out_path}")
    
    # Mostrar un top de los cambios más sugeridos
    if len(out_df) > 0:
        print("\nTop 5 cambios más sugeridos por el modelo:")
        cambios = out_df.groupby(["Old_Label_Name", "Predicted_Label_Name"]).size().reset_index(name="Count")
        cambios = cambios.sort_values("Count", ascending=False).head(5)
        for _, r in cambios.iterrows():
            print(f"  De '{r['Old_Label_Name']}' a '{r['Predicted_Label_Name']}': {r['Count']} archivos")

if __name__ == "__main__":
    main()

