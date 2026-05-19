import os
import json
import pandas as pd
import argparse
from collections import Counter


def extract_gantry_distance(path):
    try:
        dist_str = str(path).replace('\\', '/').split('/')[-5] 
        # Añadimos el signo '-' para no perder las distancias negativas
        clean_dist = ''.join([c for c in dist_str if c.isdigit() or c == '.' or c == '-'])
        return float(clean_dist) if clean_dist else 0.0
    except Exception:
        return 0.0

def extract_event_id(path):
    """
    Extrae el ID único del evento para la Votación de Trayectoria.
    Asume la estructura: .../<gantry>/<class>/<date>/<hour>/<file.pcd>
    """
    try:
        path_parts = str(path).replace('\\', '/').split('/')
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
    """Evalúa de forma segura por si pandas ya lo convirtió a lista"""
    if isinstance(val, str):
        return eval(val)[:k]
    elif isinstance(val, list):
        return val[:k]
    return val


def calculate_predictions(data, k):
    data["Neighbor Labels"] = data["Neighbor Labels"].apply(lambda x: safe_eval(x, k))
    data["Distances"] = data["Distances"].apply(lambda x: safe_eval(x, k))
    data["Neighbors"] = data["Neighbors"].apply(lambda x: safe_eval(x, k))

    pred_maj, pred_inv, pred_sq, pred_gant = [], [], [], []
    query_gantries = []

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

    data["Query_Gantry"] = query_gantries
    data["Predicted_Majority_Base"] = pred_maj
    data["Predicted_Inverse_Base"] = pred_inv
    data["Predicted_Squared_Base"] = pred_sq
    data["Predicted_Gantry_Base"] = pred_gant

    # ==========================================
    # VOTACIÓN DE TRAYECTORIA APLICADA AQUÍ
    # ==========================================
    data['event_id'] = data['Query'].apply(extract_event_id)
    
    for method in ['Majority', 'Inverse', 'Squared', 'Gantry']:
        base_col = f"Predicted_{method}_Base"
        final_col = f"Predicted_{method}"
        tp_col = f"TP_{method}"
        
        # Agrupar por evento de vehículo y elegir la más votada en esa trayectoria
        voted_preds = data.groupby('event_id')[base_col].agg(lambda x: x.mode()[0])
        
        # Asignar la final y calcular TPs
        data[final_col] = data['event_id'].map(voted_preds)
        data[tp_col] = data["Original Label"] == data[final_col]

    return data


def get_metrics_for_method(data, pred_col, tp_col, idx_to_name_map):
    metrics = {}
    class_metrics = {}
    
    # Ajusta estos IDs si en el futuro cambian
    ID_PASSENGER_CAR = 0  
    ID_VAN = 1           
    
    # Ignorar errores donde era Coche y predijo Furgoneta
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

        if total_queries == 0:
            continue

        tp = label_data[tp_col].sum()
        pred_count = (data_filtered[pred_col] == label).sum()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / total_queries if total_queries > 0 else 0.0
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

        human_name = idx_to_name_map.get(label, str(label))
        
        class_metrics[human_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_queries": total_queries
        }

        w = total_queries / total_samples
        weighted_f1.append(f1 * w)
        total_precision += precision
        total_recall += recall

    metrics["accuracy"] = data_filtered[tp_col].mean() if total_samples > 0 else 0.0
    metrics["weighted_f1"] = sum(weighted_f1)
    metrics["average_f1"] = sum([m["f1"] for m in class_metrics.values()]) / max(total_classes, 1)
    metrics["average_precision"] = total_precision / max(total_classes, 1)
    metrics["average_recall"] = total_recall / max(total_classes, 1)
    metrics["class_metrics"] = class_metrics
    metrics["ignoring_car_to_van_errors"] = int(casos_ignorados)

    return metrics


def save_csv(data, path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data[columns].to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()
    model_name = args.model_name

    csv_path = f"./outputs/knn_results/k_neighbors_all_{model_name}.csv"
    output_path = f"./outputs/knn_results/metrics_trajectory_{model_name}.json"
    stats_path = f"./outputs/stats/{model_name}_stats.json"
    out_dir = "./outputs/knn_results"

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

    results = {"inverse": {}, "squared": {}, "gantry": {}}
    best = {"inverse": (None, 0), "squared": (None, 0), "gantry": (None, 0)}
    gantry_metrics_best_k = {"inverse": {}, "squared": {}, "gantry": {}}

    for k in range(2, 11):
        data_k = data_full.copy()
        data_k = calculate_predictions(data_k, k)

        m_maj = get_metrics_for_method(data_k, "Predicted_Majority", "TP_Majority", idx_to_name_map)
        m_inv = get_metrics_for_method(data_k, "Predicted_Inverse", "TP_Inverse", idx_to_name_map)
        m_sq = get_metrics_for_method(data_k, "Predicted_Squared", "TP_Squared", idx_to_name_map)
        m_gantry = get_metrics_for_method(data_k, "Predicted_Gantry", "TP_Gantry", idx_to_name_map)

        results["inverse"][k] = {"majority": m_maj, "inverse_distance": m_inv}
        results["squared"][k] = {"squared_distance": m_sq}
        results["gantry"][k] = {"gantry": m_gantry}

        for variant, metric_val, tp_col, pred_col in [
            ("inverse", m_inv["weighted_f1"], "TP_Inverse", "Predicted_Inverse"),
            ("squared", m_sq["weighted_f1"], "TP_Squared", "Predicted_Squared"),
            ("gantry", m_gantry["weighted_f1"], "TP_Gantry", "Predicted_Gantry"),
        ]:
            if metric_val >= best[variant][1]:
                best[variant] = (k, metric_val)
                cols = ["Query", "Original Label", pred_col, "Neighbor Labels", "Distances", "Neighbors", "Query_Gantry"]
                save_csv(data_k[~data_k[tp_col]], f"{out_dir}/misses_trajectory_{model_name}_{variant}.csv", cols)
                save_csv(data_k[data_k[tp_col]], f"{out_dir}/matches_trajectory_{model_name}_{variant}.csv", cols)
                
                gantry_breakdown = {}
                for gantry_val in sorted(data_k["Query_Gantry"].unique()):
                    gantry_subset = data_k[data_k["Query_Gantry"] == gantry_val]
                    if len(gantry_subset) > 0:
                        metrics_for_gantry = get_metrics_for_method(gantry_subset, pred_col, tp_col, idx_to_name_map)
                        gantry_breakdown[f"{gantry_val}m"] = {
                            "weighted_f1": metrics_for_gantry["weighted_f1"],
                            "accuracy": metrics_for_gantry["accuracy"],
                            "total_queries": len(gantry_subset),
                            "class_breakdown": metrics_for_gantry["class_metrics"]
                        }
                gantry_metrics_best_k[variant] = gantry_breakdown

        print(f"k={k:2d} | Inv F1: {m_inv['weighted_f1']:.4f} | Sq F1: {m_sq['weighted_f1']:.4f} | Gantry F1: {m_gantry['weighted_f1']:.4f}")

    final = {
        "best_k_inverse": best["inverse"][0],
        "best_k_squared": best["squared"][0],
        "best_k_gantry": best["gantry"][0],
        "metrics_inverse": results["inverse"],
        "metrics_squared": results["squared"],
        "metrics_gantry": results["gantry"],
        "gantry_breakdown_for_best_k": gantry_metrics_best_k 
    }

    with open(output_path, "w") as f:
        json.dump(final, f, indent=4)
    print(f"\n✅ Métricas de trayectoria guardadas en {output_path}")

if __name__ == "__main__":
    main()