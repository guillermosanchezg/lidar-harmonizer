import os
import json
import pandas as pd
import argparse
from collections import Counter


def extract_event_id(path):
    """
    ID único del paso de un vehículo para la Votación de Trayectoria.
    Estructura: .../<gantry>/<class>/<date>/<hour>/<prefix>_<id>.pcd
    Usa fecha + hora + prefijo del fichero (SIN clase: usarla sería leakage).
    El prefijo se resetea al reiniciar el sistema de captura, por eso se
    combina con date+hour para garantizar unicidad entre sesiones.
    """
    try:
        path_parts = str(path).replace('\\', '/').split('/')
        filename = path_parts[-1]           # e.g. "1234_20240315123456.pcd"
        prefix = filename.split('_')[0]     # e.g. "1234"
        date = path_parts[-3]              # e.g. "20240315"
        hour = path_parts[-2]              # e.g. "14"
        return f"{date}_{hour}_{prefix}"
    except Exception:
        return os.path.basename(str(path))


def safe_eval(val, k):
    """Evalúa de forma segura por si pandas ya lo convirtió a lista."""
    if isinstance(val, str):
        return eval(val)[:k]
    elif isinstance(val, list):
        return val[:k]
    return val


def calculate_predictions(data, k):
    data["Neighbor Labels"] = data["Neighbor Labels"].apply(lambda x: safe_eval(x, k))
    data["Distances"] = data["Distances"].apply(lambda x: safe_eval(x, k))
    data["Neighbors"] = data["Neighbors"].apply(lambda x: safe_eval(x, k))

    pred_maj, pred_inv, pred_sq = [], [], []

    for _, row in data.iterrows():
        neighbors = row["Neighbor Labels"]
        distances = row["Distances"]

        # Majority
        p_maj = Counter(neighbors).most_common(1)[0][0]
        pred_maj.append(p_maj)

        # Inverse distance
        w_inv = [1 / (d + 1e-6) if d > 0 else 1.0 for d in distances]
        scores_inv = {}
        for n, w in zip(neighbors, w_inv):
            scores_inv[n] = scores_inv.get(n, 0) + w
        pred_inv.append(max(scores_inv, key=scores_inv.get))

        # Squared distance
        w_sq = [1 / ((d ** 2) + 1e-6) if d > 0 else 1.0 for d in distances]
        scores_sq = {}
        for n, w in zip(neighbors, w_sq):
            scores_sq[n] = scores_sq.get(n, 0) + w
        pred_sq.append(max(scores_sq, key=scores_sq.get))

    data["Predicted_Majority_Base"] = pred_maj
    data["Predicted_Inverse_Base"] = pred_inv
    data["Predicted_Squared_Base"] = pred_sq

    # ==========================================
    # VOTACIÓN DE TRAYECTORIA
    # ==========================================
    data['event_id'] = data['Query'].apply(extract_event_id)

    for method in ['Majority', 'Inverse', 'Squared']:
        base_col = f"Predicted_{method}_Base"
        final_col = f"Predicted_{method}"
        tp_col = f"TP_{method}"

        voted_preds = data.groupby('event_id')[base_col].agg(lambda x: x.mode()[0])
        data[final_col] = data['event_id'].map(voted_preds)
        data[tp_col] = data["Original Label"] == data[final_col]

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
        recall = tp / total_queries if total_queries > 0 else 0.0
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

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
        total_recall += recall

    metrics["accuracy"] = data_filtered[tp_col].mean() if total_samples > 0 else 0.0
    metrics["weighted_f1"] = sum(weighted_f1)
    metrics["average_f1"] = sum(m["f1"] for m in class_metrics.values()) / max(total_classes, 1)
    metrics["average_precision"] = total_precision / max(total_classes, 1)
    metrics["average_recall"] = total_recall / max(total_classes, 1)
    metrics["class_metrics"] = class_metrics
    metrics["ignoring_car_to_van_errors"] = int(casos_ignorados)

    return metrics


def save_csv(data, path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data[columns].to_csv(path, index=False)


def _track_length_bin(n):
    """Bins de longitud de tracking: 1, 2, 3, 4+."""
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def compute_f1_by_track_length(data, pred_col, tp_col, idx_to_name_map):
    """
    Desglose de F1 de TRAYECTORIA por longitud de tracking (nº de frames del
    event_id). Se evalúa a nivel VEHÍCULO: una muestra por event_id, con su
    predicción ya votada y su Original Label (constante dentro del vehículo).

    El bin de longitud 1 es el CONTROL: un vehículo con un solo frame no tiene
    nada que votar, así que su F1 coincide con el frame-level de esos frames.

    Devuelve {bin: {"n_vehicles": int, "average_f1": float, "weighted_f1": float}}.
    """
    # Una fila por vehículo (todas las filas de un event_id comparten predicción
    # votada y Original Label, así que .first() es representativo).
    sizes = data.groupby("event_id").size()
    veh   = data.groupby("event_id").first().copy()
    veh["track_length"] = sizes
    veh["__bin"]        = veh["track_length"].apply(_track_length_bin)
    veh[tp_col]         = veh["Original Label"] == veh[pred_col]

    out = {}
    for b in ["1", "2", "3", "4+"]:
        sub = veh[veh["__bin"] == b]
        if len(sub) == 0:
            out[b] = {"n_vehicles": 0, "average_f1": None, "weighted_f1": None}
            continue
        m = get_metrics_for_method(sub, pred_col, tp_col, idx_to_name_map)
        out[b] = {
            "n_vehicles":  int(len(sub)),
            "average_f1":  m["average_f1"],
            "weighted_f1": m["weighted_f1"],
        }
    return out


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

    results = {"majority": {}, "inverse": {}, "squared": {}}
    best     = {"majority": (None, 0), "inverse": (None, 0), "squared": (None, 0)}
    best_f1tl = {"majority": None,    "inverse": None,       "squared": None}

    for k in range(2, 11):
        data_k = data_full.copy()
        data_k = calculate_predictions(data_k, k)

        m_maj = get_metrics_for_method(data_k, "Predicted_Majority", "TP_Majority", idx_to_name_map)
        m_inv = get_metrics_for_method(data_k, "Predicted_Inverse", "TP_Inverse", idx_to_name_map)
        m_sq  = get_metrics_for_method(data_k, "Predicted_Squared",  "TP_Squared",  idx_to_name_map)

        results["majority"][k] = m_maj
        results["inverse"][k]  = m_inv
        results["squared"][k]  = m_sq

        save_cols = ["Query", "Original Label", None, "Neighbor Labels", "Distances", "Neighbors"]

        for variant, metric_val, tp_col, pred_col in [
            ("majority", m_maj["weighted_f1"], "TP_Majority", "Predicted_Majority"),
            ("inverse",  m_inv["weighted_f1"], "TP_Inverse",  "Predicted_Inverse"),
            ("squared",  m_sq["weighted_f1"],  "TP_Squared",  "Predicted_Squared"),
        ]:
            if metric_val >= best[variant][1]:
                best[variant] = (k, metric_val)
                best_f1tl[variant] = compute_f1_by_track_length(
                    data_k, pred_col, tp_col, idx_to_name_map)
                cols = ["Query", "Original Label", pred_col, "Neighbor Labels", "Distances", "Neighbors"]
                save_csv(data_k[~data_k[tp_col]], f"{out_dir}/misses_trajectory_{model_name}_{variant}.csv", cols)
                save_csv(data_k[ data_k[tp_col]], f"{out_dir}/matches_trajectory_{model_name}_{variant}.csv", cols)

        print(f"k={k:2d} | Maj F1: {m_maj['weighted_f1']:.4f} | Inv F1: {m_inv['weighted_f1']:.4f} | Sq F1: {m_sq['weighted_f1']:.4f}")

    final = {
        "best_k_majority": best["majority"][0],
        "best_k_inverse":  best["inverse"][0],
        "best_k_squared":  best["squared"][0],
        "metrics_majority": results["majority"],
        "metrics_inverse":  results["inverse"],
        "metrics_squared":  results["squared"],
        "f1_by_track_length": {v: best_f1tl[v] for v in ("majority", "inverse", "squared")
                                if best_f1tl[v] is not None},
    }

    with open(output_path, "w") as f:
        json.dump(final, f, indent=4)
    print(f"\n✅ Métricas de trayectoria guardadas en {output_path}")

    # --- Tabla terminal: F1 por longitud de tracking ---
    print("\nF1 por longitud de tracking (mejor k por regla):")
    print(f"{'Regla':<12} {'Bin':<5} {'N_Veh':>7} {'avg_F1':>8} {'wF1':>8}")
    print("-" * 45)
    for rule in ("majority", "inverse", "squared"):
        tl = best_f1tl.get(rule)
        if tl is None:
            continue
        for b in ("1", "2", "3", "4+"):
            entry   = tl.get(b, {})
            n       = entry.get("n_vehicles", 0)
            avg_f1  = entry.get("average_f1")
            wf1     = entry.get("weighted_f1")
            avg_str = f"{avg_f1:.4f}" if avg_f1 is not None else "—"
            wf1_str = f"{wf1:.4f}"   if wf1    is not None else "—"
            print(f"{rule:<12} {b:<5} {n:>7} {avg_str:>8} {wf1_str:>8}")

    # --- CSV: f1_by_track_length_{model_name}.csv ---
    tl_rows = []
    for rule in ("majority", "inverse", "squared"):
        tl = best_f1tl.get(rule)
        if tl is None:
            continue
        for b in ("1", "2", "3", "4+"):
            entry = tl.get(b, {})
            tl_rows.append({
                "rule":        rule,
                "bin":         b,
                "n_vehicles":  entry.get("n_vehicles", 0),
                "average_f1":  entry.get("average_f1"),
                "weighted_f1": entry.get("weighted_f1"),
            })
    if tl_rows:
        tl_csv_path = f"{out_dir}/f1_by_track_length_{model_name}.csv"
        pd.DataFrame(tl_rows).to_csv(tl_csv_path, index=False)
        print(f"✅ F1 por longitud de tracking guardado en {tl_csv_path}")


if __name__ == "__main__":
    main()
