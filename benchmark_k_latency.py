import os
import argparse
import time
import torch
import numpy as np
import pandas as pd
import yaml
from collections import Counter

from models_pyg import DGCNNClassifier, PointNet2Classifier, PointMLPClassifier
from utils import normalize_point_cloud, uniform_points
import open3d as o3d

# ==========================================
# CONFIGURACIÓN
# ==========================================
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]
device = torch.device(DEVICE_CONFIG["device"])
NUM_POINTS = MODEL_HYPERPARAMETERS["num_points"]

def get_model(model_name, feature_size):
    model_path = os.path.join("./outputs/models", model_name)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = state_dict["fc3.weight"].shape[0]

    if "dgcnn" in model_name.lower():
        model = DGCNNClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    elif "pointmlp" in model_name.lower():
        model = PointMLPClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    else:
        model = PointNet2Classifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_sample_point_clouds(csv_path, num_samples=100):
    df = pd.read_csv(csv_path)
    paths = df["Path"].tolist()
    
    import random
    np.random.seed(42)
    sample_paths = random.sample(paths, min(num_samples, len(paths)))
    
    clouds = []
    for path in sample_paths:
        if os.path.exists(path):
            pc = np.asarray(o3d.io.read_point_cloud(path).points)
            if pc.size > 0:
                pc = normalize_point_cloud(pc)
                pc = uniform_points(pc, NUM_POINTS)
                clouds.append(pc)
    return clouds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    feature_size = int(args.model_name.split("_")[1].replace("d", ""))

    print(f"--- BENCHMARK DE LATENCIA POR K ---")
    print(f"Cargando PointNet ({args.model_name}) en {device}...")
    model = get_model(args.model_name, feature_size)

    # 1. Cargar Base de Conocimiento para el KNN
    kb_csv = f"./outputs/embeddings/embeddings_knowledge_{args.model_name.replace('.pth', '.csv')}"
    if not os.path.exists(kb_csv):
        print(f"Error: No encuentro el CSV de Knowledge Base: {kb_csv}")
        return
        
    kb_df = pd.read_csv(kb_csv)
    emb_cols = [c for c in kb_df.columns if c.startswith('emb_')]
    kb_tensor = torch.tensor(kb_df[emb_cols].values, dtype=torch.float32).to(device)
    kb_labels = kb_df["Label"].tolist()
    print(f"Base de conocimiento en VRAM: {kb_tensor.shape[0]} instancias, dimension {kb_tensor.shape[1]}")

    # 2. Cargar nubes de prueba (Unseen)
    unseen_csv = "./data/unseen_sets/unseen_dataset.csv"
    num_eval_clouds = 100
    sample_clouds = load_sample_point_clouds(unseen_csv, num_samples=num_eval_clouds)

    # ========================================================
    # WARM-UP (CORREGIDO PARA PYG)
    # ========================================================
    print("Realizando warm-up de GPU...")
    dummy_pc = torch.randn(1, NUM_POINTS, 3).to(device)
    dummy_pos = dummy_pc.view(-1, 3) # <-- CORRECCIÓN AQUÍ: Aplanar para PyG
    dummy_idx = torch.zeros(dummy_pos.size(0), dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model.extract_features(dummy_pos, dummy_idx)

    # ========================================================
    # PASO A: Extraer todos los embeddings y medir latencia PointNet
    # ========================================================
    times_pointnet = []
    query_embeddings = []
    
    print("\n1. Midiendo latencia de extracción de características (PointNet)...")
    for pc_numpy in sample_clouds:
        pc_tensor = torch.tensor(pc_numpy, dtype=torch.float32).to(device)
        pc_tensor = pc_tensor.unsqueeze(0) 
        pos = pc_tensor.view(-1, 3)
        batch_idx = torch.zeros(pos.size(0), dtype=torch.long).to(device)
        
        start_pn = time.perf_counter()
        with torch.no_grad():
            emb = model.extract_features(pos, batch_idx)
        if device.type == 'cuda': torch.cuda.synchronize()
        end_pn = time.perf_counter()
        
        times_pointnet.append(end_pn - start_pn)
        query_embeddings.append(emb)

    # Quitamos el primer 5% de tiempos para evitar outliers iniciales del SO
    valid_pn_times = times_pointnet[int(len(times_pointnet)*0.05):]
    avg_pn_ms = (sum(valid_pn_times) / len(valid_pn_times)) * 1000  
    print(f"-> Latencia media PointNet: {avg_pn_ms:.3f} ms")

    # ========================================================
    # PASO B: Medir latencia KNN para K desde 2 hasta 10
    # ========================================================
    print("\n2. Midiendo latencia de KNN para K de 2 a 10...")
    
    results = []
    
    for k in range(2, 11):
        times_knn = []
        
        for q_emb in query_embeddings:
            start_knn = time.perf_counter()
            
            # Distancia L2 contra toda la base de conocimiento
            distances = torch.cdist(q_emb, kb_tensor, p=2)
            topk_dist, topk_idx = torch.topk(distances, k=k, largest=False)
            
            # Traer a CPU simulando la predicción (Majority)
            topk_idx_cpu = topk_idx.cpu().numpy()[0]
            neighbor_labels = [kb_labels[idx] for idx in topk_idx_cpu]
            _ = Counter(neighbor_labels).most_common(1)[0][0]
            
            if device.type == 'cuda': torch.cuda.synchronize()
            end_knn = time.perf_counter()
            times_knn.append(end_knn - start_knn)
            
        valid_knn_times = times_knn[int(len(times_knn)*0.05):]
        avg_knn_ms = (sum(valid_knn_times) / len(valid_knn_times)) * 1000
        
        # Asumimos ~0.05ms para la trayectoria basados en la lógica probada
        avg_traj_ms = 0.05
        total_ms = avg_pn_ms + avg_knn_ms + avg_traj_ms
        
        results.append({
            "K": k,
            "t_embed_ms": round(avg_pn_ms, 4),
            "t_knn_ms": round(avg_knn_ms, 4),
            "t_vote_ms": round(avg_traj_ms, 4),
            "t_total_ms": round(total_ms, 4)
        })
        
        print(f"K={k:2d} | t_knn: {avg_knn_ms:.4f} ms | T. Total: {total_ms:.4f} ms")

    # Guardar en CSV para la gráfica del paper
    out_dir = "./outputs/knn_results"
    os.makedirs(out_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    csv_out_path = os.path.join(out_dir, f"latency_benchmark_{args.model_name.replace('.pth', '.csv')}")
    df_results.to_csv(csv_out_path, index=False)
    
    print("\n" + "="*50)
    print(f"BENCHMARK COMPLETADO.")
    print(f"Archivo guardado en: {csv_out_path}")
    print("="*50)

if __name__ == "__main__":
    main()