import os
import json
import pandas as pd
import argparse
from collections import Counter


def calculate_predictions(data, k):
    data["Neighbor Labels"] = data["Neighbor Labels"].apply(lambda x: eval(x)[:k] if isinstance(x, str) else x[:k])
    data["Distances"]       = data["Distances"].apply(lambda x: eval(x)[:k] if isinstance(x, str) else x[:k])
    data["Neighbors"]       = data["Neighbors"].apply(lambda x: eval(x)[:k] if isinstance(x, str) else x[:k])

    pred_maj, pred_inv, pred_sq = [], [], []
    tp_maj,   tp_inv,   tp_sq   = [], [], []

    for _, row in data.iterrows():
        query_label = row["Original Label"]
        neighbors   = row["Neighbor Labels"]
        distances   = row["Distances"]

        # Majority
        p_maj = Counter(neighbors).most_common(1)[0][0]
        pred_maj.append(p_maj)
        tp_maj.append(query_label == p_maj)

        # Inverse distance
        w_inv = [1 / (d + 1e-6) if d > 0 else 1.0 for d in distances]
        scores_inv = {}
        for n, w in zip(neighbors, w_inv):
            scores_inv[n] = scores_inv.get(n, 0) + w
        p_inv = max(scores_inv, key=scores_inv.get)
        pred_inv.append(p_inv)
        tp_inv.append(query_label == p_inv)

        # Squared distance
        w_sq = [1 / ((d ** 2) + 1e-6) if d > 0 else 1.0 for d in distances]
        scores_sq = {}
        for n, w in zip(neighbors, w_sq):
            scores_sq[n] = scores_sq.get(n, 0) + w
        p_sq = max(scores_sq, key=scores_sq.get)
        pred_sq.append(p_sq)
        tp_sq.append(query_label == p_sq)

    data["Predicted_Majority"] = pred_maj
    data["TP_Majority"]        = tp_maj
    data["Predicted_Inverse"]  = pred_inv
    data["TP_Inverse"]         = tp_inv
    data["Predicted_Squared"]  = pred_sq
    data["TP_Squared"]         = tp_sq

    return data


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
        if total_queries == 0:
            continue

        tp = label_data[tp_col].sum()
        pred_count = (data_filtered[pred_col] == label).sum()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall    = tp / total_queries if total_queries > 0 else 0.0
        f1        = (2 * precision * recall) / max(precision + recall, 1e-6)

        human_name = idx_to_name_map.get(label, str(label))
        class_metrics[human_name] = {
            "precision":     precision,
            "recall":        recall,
            "f1":            f1,
            "support":       total_queries,
            "total_queries": total_queries,
        }

        w = total_queries / total_samples
        weighted_f1.append(f1 * w)
        total_precision += precision
        total_recall    += recall

    metrics["accuracy"]          = data_filtered[tp_col].mean() if total_samples > 0 else 0.0
    metrics["weighted_f1"]       = sum(weighted_f1)
    metrics["average_f1"]        = sum(m["f1"] for m in class_metrics.values()) / max(total_classes, 1)
    metrics["average_precision"] = total_precision / max(total_classes, 1)
    metrics["average_recall"]    = total_recall    / max(total_classes, 1)
    metrics["class_metrics"]     = class_metrics
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

    csv_path    = f"./outputs/knn_results/k_neighbors_all_{model_name}.csv"
    output_path = f"./outputs/knn_results/metrics_{model_name}.json"
    stats_path  = f"./outputs/stats/{model_name}_stats.json"
    out_dir     = "./outputs/knn_results"

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

    results = {"majority": {}, "inverse": {}, "squared": {}}
    best    = {"majority": (None, 0), "inverse": (None, 0), "squared": (None, 0)}

    for k in range(2, 11):
        data_k = data_full.copy()
        data_k = calculate_predictions(data_k, k)

        m_maj = get_metrics_for_method(data_k, "Predicted_Majority", "TP_Majority", idx_to_name_map)
        m_inv = get_metrics_for_method(data_k, "Predicted_Inverse",  "TP_Inverse",  idx_to_name_map)
        m_sq  = get_metrics_for_method(data_k, "Predicted_Squared",  "TP_Squared",  idx_to_name_map)

        results["majority"][k] = m_maj
        results["inverse"][k]  = m_inv
        results["squared"][k]  = m_sq

        for variant, metric_val, tp_col, pred_col in [
            ("majority", m_maj["weighted_f1"], "TP_Majority", "Predicted_Majority"),
            ("inverse",  m_inv["weighted_f1"], "TP_Inverse",  "Predicted_Inverse"),
            ("squared",  m_sq["weighted_f1"],  "TP_Squared",  "Predicted_Squared"),
        ]:
            if metric_val >= best[variant][1]:
                best[variant] = (k, metric_val)
                cols = ["Query", "Original Label", pred_col, "Neighbor Labels", "Distances", "Neighbors"]
                save_csv(data_k[~data_k[tp_col]], f"{out_dir}/misses_{model_name}_{variant}.csv", cols)
                save_csv(data_k[ data_k[tp_col]], f"{out_dir}/matches_{model_name}_{variant}.csv", cols)

        print(f"k={k:2d} | Maj F1: {m_maj['weighted_f1']:.4f} | Inv F1: {m_inv['weighted_f1']:.4f} | Sq F1: {m_sq['weighted_f1']:.4f}")

    final = {
        "best_k_majority": best["majority"][0],
        "best_k_inverse":  best["inverse"][0],
        "best_k_squared":  best["squared"][0],
        "metrics_majority": results["majority"],
        "metrics_inverse":  results["inverse"],
        "metrics_squared":  results["squared"],
    }

    with open(output_path, "w") as f:
        json.dump(final, f, indent=4)
    print(f"\n✅ Métricas guardadas en {output_path}")

    # ---- Tabla de soporte por clase (útil en fine-grained con clases minoritarias) ----
    # Usamos la mejor regla por average_f1 al mejor K
    best_rule, best_avg_f1, best_k_found = "inverse", 0.0, best["inverse"][0]
    for rule in ["majority", "inverse", "squared"]:
        k_candidate = best[rule][0]
        if k_candidate is None:
            continue
        avg_f1 = results[rule][k_candidate].get("average_f1", 0.0)
        if avg_f1 > best_avg_f1:
            best_avg_f1   = avg_f1
            best_rule     = rule
            best_k_found  = k_candidate

    if best_k_found is not None:
        class_m = results[best_rule][best_k_found].get("class_metrics", {})
        print(f"\n{'='*62}")
        print(f"SOPORTE POR CLASE EN UNSEEN  (regla={best_rule}, K={best_k_found})")
        print(f"{'Clase':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'n':>6}")
        print(f"{'-'*62}")
        for cname, cm in sorted(class_m.items(), key=lambda x: -x[1].get("total_queries", 0)):
            print(f"{cname:<25} {cm['f1']:>6.4f} {cm['precision']:>6.4f} "
                  f"{cm['recall']:>6.4f} {cm['total_queries']:>6d}")
        print(f"{'='*62}")


if __name__ == "__main__":
    main()
