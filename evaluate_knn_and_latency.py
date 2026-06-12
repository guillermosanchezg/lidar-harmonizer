import os
import json
import pandas as pd
import argparse
from collections import Counter
import time

def extract_gantry_distance(path):
    try:
        dist_str = str(path).replace('\\\\', '/').split('/')[-5] 
        clean_dist = ''.join([c for c in dist_str if c.isdigit() or c == '.' or c == '-'])
        return float(clean_dist) if clean_dist else 0.0
    except Exception:
        return 0.0

def extract_event_id(path):
    try:
        path_parts = str(path).replace('\\\\', '/').split('/')
        if len(path_parts) >= 5:
            cls = path_parts[-4]
            date = path_parts[-3]
            hour = path_parts[-2]
            return f"{cls}_{date}_{hour}"
        return os.path.basename(os.path.dirname(str(path)))
    except Exception:
        return "Unknown"

def calculate_combined_weight(latent_distance, gantry_difference, alpha=0.7, beta=0.3):
    w_latent = 1 / (1 + latent_distance)
    w_gantry = 1 / (1 + gantry_difference)
    return alpha * w_latent + beta * w_gantry

def safe_eval(val, k):
    if isinstance(val, str):
        return eval(val)[:k]
    elif isinstance(val, list):
        return val[:k]
    return val

def calculate_predictions_and_latency(data, k):
    data["Neighbor Labels"] = data["Neighbor Labels"].apply(lambda x: safe_eval(x, k))
    data["Distances"] = data["Distances"].apply(lambda x: safe_eval(x, k))
    data["Neighbors"] = data["Neighbors"].apply(lambda x: safe_eval(x, k))

    pred_maj, pred_inv, pred_sq, pred_gant = [], [], [], []
    query_gantries = []

    # BLOQUE 3: MEDIR LATENCIA KNN
    start_knn = time.perf_counter()

    for _, row in data.iterrows():
        neighbors = row["Neighbor Labels"] 
        distances = row["Distances"]
        query_gantry = extract_gantry_distance(row["Query"])
        query_gantries.append(query_gantry)

        # Majority
        p_maj = Counter(neighbors).most_common(1)[0][0]
        pred_maj.append(p_maj)

        # Inverse
        w_inv = [1 / (d + 1e-6) if d > 0 else 1.0 for d in distances]
        scores_inv = {}
        for n, w in zip(neighbors, w_inv):
            scores_inv[n] = scores_inv.get(n, 0) + w
        p_inv = max(scores_inv, key=scores_inv.get)
        pred_inv.append(p_inv)

        # Squared
        w_sq = [1 / ((d ** 2) + 1e-6) if d > 0 else 1.0 for d in distances]
        scores_sq = {}
        for n, w in zip(neighbors, w_sq):
            scores_sq[n] = scores_sq.get(n, 0) + w
        p_sq = max(scores_sq, key=scores_sq.get)
        pred_sq.append(p_sq)

        # Gantry
        neighbor_gantries = [extract_gantry_distance(p) for p in row["Neighbors"]]
        w_gant = [
            calculate_combined_weight(
                d if d > 0 else 1e-6,
                abs(query_gantry - ng) if abs(query_gantry - ng) > 0 else 1e-6,
                alpha=0.7, beta=0.3
            )
            for d, ng in zip(distances, neighbor_gantries)
        ]
        scores_gant = {}
        for n, w in zip(neighbors, w_gant):
            scores_gant[n] = scores_gant.get(n, 0) + w
        p_gant = max(scores_gant, key=scores_gant.get)
        pred_gant.append(p_gant)

    end_knn = time.perf_counter()
    knn_latency_total = end_knn - start_knn
    knn_latency_per_query_ms = (knn_latency_total / len(data)) * 1000

    data["Query_Gantry"] = query_gantries
    
    # SINGLE FRAME PREDICTIONS (Evaluación de tu reivindicación)
    data["Predicted_Majority_Base"] = pred_maj
    data["TP_Majority_Base"] = data["Original Label"] == data["Predicted_Majority_Base"]
    data["Predicted_Inverse_Base"] = pred_inv
    data["TP_Inverse_Base"] = data["Original Label"] == data["Predicted_Inverse_Base"]
    data["Predicted_Squared_Base"] = pred_sq
    data["TP_Squared_Base"] = data["Original Label"] == data["Predicted_Squared_Base"]
    data["Predicted_Gantry_Base"] = pred_gant
    data["TP_Gantry_Base"] = data["Original Label"] == data["Predicted_Gantry_Base"]

    # BLOQUE 3 Y 4: VOTACIÓN TRAYECTORIA Y LATENCIA
    data['event_id'] = data['Query'].apply(extract_event_id)
    
    start_vote = time.perf_counter()
    for method in ['Majority', 'Inverse', 'Squared', 'Gantry']:
        base_col = f"Predicted_{method}_Base"
        final_col = f"Predicted_{method}_Trajectory"
        tp_col = f"TP_{method}_Trajectory"
        
        voted_preds = data.groupby('event_id')[base_col].agg(lambda x: x.mode()[0])
        data[final_col] = data['event_id'].map(voted_preds)
        data[tp_col] = data["Original Label"] == data[final_col]
        
    end_vote = time.perf_counter()
    vote_latency_total = end_vote - start_vote
    num_events = data['event_id'].nunique()
    vote_latency_per_event_ms = (vote_latency_total / num_events) * 1000

    latencies = {
        "knn_latency_ms_per_query": knn_latency_per_query_ms,
        "vote_latency_ms_per_event": vote_latency_per_event_ms,
        "total_queries": len(data),
        "total_events": num_events
    }

    return data, latencies

def get_metrics_for_method(data, pred_col, tp_col, idx_to_name_map):
    metrics = {}
    class_metrics = {}
    
    ID_PASSENGER_CAR = 0  
    ID_VAN = 1           
    
    mask_to_ignore = (data["Original Label"] == ID_PASSENGER_CAR) & (data[pred_col] == ID_VAN)
    casos_ignorados = mask_to_ignore.sum()
    data_filtered = data[~mask_to_ignore].copy()
    
    total_samples = len(data_filtered)
    unique_classes = sorted(data_filtered["Original Label"].unique())
    total_classes = len(unique_classes)

    weighted_f1 = []
    total_precision = 0
    total_recall = 0

    for label in unique_classes:
        label_data = data_filtered[data_filtered["Original Label"] == label]
        total_queries = len(label_data)
        if total_queries == 0: continue

        tp = label_data[tp_col].sum()
        pred_count = (data_filtered[pred_col] == label).sum()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / total_queries if total_queries > 0 else 0.0
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

        human_name = idx_to_name_map.get(label, str(label))
        class_metrics[human_name] = {"precision": precision, "recall": recall, "f1": f1, "total_queries": total_queries}

        w = total_queries / total_samples
        weighted_f1.append(f1 * w)
        total_precision += precision
        total_recall += recall

    metrics["accuracy"] = data_filtered[tp_col].mean() if total_samples > 0 else 0.0
    metrics["weighted_f1"] = sum(weighted_f1)
    metrics["average_f1"] = sum([m["f1"] for m in class_metrics.values()]) / max(total_classes, 1)
    metrics["class_metrics"] = class_metrics
    metrics["ignoring_car_to_van_errors"] = int(casos_ignorados)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()
    model_name = args.model_name

    csv_path = f"./outputs/knn_results/k_neighbors_all_{model_name}.csv"
    output_path = f"./outputs/knn_results/metrics_complete_{model_name}.json"
    stats_path = f"./outputs/stats/{model_name}_stats.json"

    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV no encontrado en {csv_path}")
        return

    idx_to_name_map = {}
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        for k, v in stats.get("class_mapping", {}).items():
            if v not in idx_to_name_map:
                idx_to_name_map[v] = k

    data_full = pd.read_csv(csv_path)
    data_full["Original Label"] = data_full["Original Label"].astype(int)

    results = {}

    for k in range(2, 11):
        data_k = data_full.copy()
        data_k, latencies = calculate_predictions_and_latency(data_k, k)

        k_metrics = {"latencies": latencies, "single_frame": {}, "trajectory": {}}

        for method in ['Majority', 'Inverse', 'Squared', 'Gantry']:
            # Single Frame
            m_base = get_metrics_for_method(data_k, f"Predicted_{method}_Base", f"TP_{method}_Base", idx_to_name_map)
            k_metrics["single_frame"][method.lower()] = m_base
            
            # Trajectory
            m_traj = get_metrics_for_method(data_k, f"Predicted_{method}_Trajectory", f"TP_{method}_Trajectory", idx_to_name_map)
            k_metrics["trajectory"][method.lower()] = m_traj

        results[k] = k_metrics
        print(f"K={k:2d} | Inv F1 (SF): {k_metrics['single_frame']['inverse']['weighted_f1']:.4f} | Inv F1 (Traj): {k_metrics['trajectory']['inverse']['weighted_f1']:.4f} | Latencia KNN: {latencies['knn_latency_ms_per_query']:.3f} ms")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Métricas unificadas y de latencia guardadas en {output_path}")

if __name__ == "__main__":
    main()

