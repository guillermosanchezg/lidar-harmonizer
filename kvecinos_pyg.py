import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import yaml

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

DEVICE_CONFIG = params["DEVICE_CONFIG"]
device = torch.device(DEVICE_CONFIG["device"] if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Cálculo de k vecinos usando Embeddings precalculados")
    parser.add_argument("model_name", type=str, help="Nombre del modelo, ej. pointnet2_1024d_XXX.pth")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    model_name = args.model_name
    results_path = f"./outputs/knn_results/k_neighbors_all_{model_name}.csv"

    # 1. Definir rutas a los embeddings ya calculados
    knowledge_emb_csv = f"./outputs/embeddings/embeddings_knowledge_{model_name.replace('.pth', '.csv')}"
    unseen_emb_csv = f"./outputs/embeddings/embeddings_unseen_{model_name.replace('.pth', '.csv')}"

    if not os.path.exists(knowledge_emb_csv) or not os.path.exists(unseen_emb_csv):
        raise FileNotFoundError("Faltan los CSV de embeddings. Ejecuta generate_embeddings_pyg.py primero.")

    # 2. Cargar Base de Conocimiento (La galería de búsqueda)
    print(f"Cargando Base de Conocimiento desde {knowledge_emb_csv}...")
    df_know = pd.read_csv(knowledge_emb_csv)
    know_paths = df_know["Path"].tolist()
    know_labels = df_know["Label"].tolist()
    
    # 3. Cargar Consultas (Las nubes 'unseen' a etiquetar)
    print(f"Cargando Consultas (Unseen) desde {unseen_emb_csv}...")
    df_unseen = pd.read_csv(unseen_emb_csv)
    unseen_paths = df_unseen["Path"].tolist()
    unseen_labels = df_unseen["Label"].tolist()

    # Extraer dinámicamente las columnas que contienen los vectores latentes (emb_0, emb_1...)
    emb_cols = [c for c in df_know.columns if c.startswith('emb_')]
    
    print(f"Moviendo vectores de {len(emb_cols)} dimensiones al dispositivo: {device}...")
    know_tensor = torch.tensor(df_know[emb_cols].values, dtype=torch.float32).to(device)
    unseen_tensor = torch.tensor(df_unseen[emb_cols].values, dtype=torch.float32).to(device)

    num_queries = unseen_tensor.shape[0]
    actual_k = min(args.k, know_tensor.shape[0])
    
    # Matriz para guardar los índices de los vecinos
    neighbors = np.zeros((num_queries, actual_k), dtype=np.int32)
    # Lista para recolectar las distancias y no saturar la memoria RAM
    distances_list = []

    print("Calculando distancias K-NN masivas en la GPU...")
    batch_size = 512
    for i in tqdm(range(0, num_queries, batch_size), desc="Distancias GPU"):
        batch_end = min(i + batch_size, num_queries)
        batch_emb = unseen_tensor[i:batch_end]
        
        # cdist compara el lote de unseen contra toda la base de conocimiento
        batch_distances = torch.cdist(batch_emb, know_tensor, p=2)

        # Ordenar para obtener los K más cercanos
        sorted_indices = torch.argsort(batch_distances, dim=1)[:, :actual_k]
        neighbors[i:batch_end] = sorted_indices.cpu().numpy()
        
        # Recolectar distancias correspondientes
        for j in range(batch_end - i):
            idx_row = sorted_indices[j]
            dists = batch_distances[j, idx_row].cpu().numpy()
            distances_list.append(dists)

    print("Formateando y guardando resultados CSV...")
    rows = []
    for i in tqdm(range(num_queries), desc="Guardando"):
        query_path = unseen_paths[i]
        query_label = int(unseen_labels[i])
        
        neighbor_indices = neighbors[i]
        n_paths = [know_paths[n] for n in neighbor_indices]
        n_labels = [int(know_labels[n]) for n in neighbor_indices]
        n_dists = distances_list[i].tolist()

        rows.append({
            "Query": query_path,
            "Original Label": query_label,
            "Neighbor Labels": n_labels,
            "Distances": n_dists,
            "Neighbors": n_paths
        })

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"✅ Resultados guardados en: {results_path}")

if __name__ == "__main__":
    main()  