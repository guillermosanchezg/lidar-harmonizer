import os
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def extract_gantry_distance(path):
    try:
        dist_str = path.replace('\\', '/').split('/')[-5]
        clean_dist = ''.join([c for c in dist_str if c.isdigit() or c == '.' or c == '-'])
        return float(clean_dist) if clean_dist else 0.0
    except Exception:
        return 0.0


def calculate_combined_weight(latent_distance, gantry_difference, alpha=0.7, beta=0.3):
    w_latent = 1 / (1 + latent_distance)
    w_gantry = 1 / (1 + gantry_difference)
    return alpha * w_latent + beta * w_gantry


def get_idx_to_name_map(model_name):
    stats_path = f"./outputs/stats/{model_name}_stats.json"
    idx_to_name_map = {}
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        for k, v in stats.get("class_mapping", {}).items():
            idx_to_name_map[int(v)] = k
    return idx_to_name_map


def get_embeddings_paths(model_name):
    knowledge_emb_csv = f"./outputs/embeddings/embeddings_knowledge_{model_name.replace('.pth', '.csv')}"
    unseen_emb_csv = f"./outputs/embeddings/embeddings_unseen_{model_name.replace('.pth', '.csv')}"
    if not os.path.exists(knowledge_emb_csv):
        raise FileNotFoundError(f"No existe: {knowledge_emb_csv}")
    if not os.path.exists(unseen_emb_csv):
        raise FileNotFoundError(f"No existe: {unseen_emb_csv}")
    return knowledge_emb_csv, unseen_emb_csv


def parse_embedding_columns(df):
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    if not emb_cols:
        raise ValueError("No se han encontrado columnas emb_ en el CSV")
    return emb_cols


def majority_vote(neighbors):
    return Counter(neighbors).most_common(1)[0][0]


def inverse_vote(neighbors, distances):
    w_inv = [1 / (d + 1e-6) if d > 0 else 1.0 for d in distances]
    scores = {}
    for n, w in zip(neighbors, w_inv):
        scores[n] = scores.get(n, 0) + w
    return max(scores, key=scores.get)


def squared_vote(neighbors, distances):
    w_sq = [1 / ((d ** 2) + 1e-6) if d > 0 else 1.0 for d in distances]
    scores = {}
    for n, w in zip(neighbors, w_sq):
        scores[n] = scores.get(n, 0) + w
    return max(scores, key=scores.get)


def gantry_vote(query_path, neighbor_paths, neighbors, distances):
    query_gantry = extract_gantry_distance(query_path)
    neighbor_gantries = [extract_gantry_distance(p) for p in neighbor_paths]
    w_gant = [
        calculate_combined_weight(
            d if d > 0 else 1e-6,
            abs(query_gantry - ng) if abs(query_gantry - ng) > 0 else 1e-6,
            alpha=0.7,
            beta=0.3,
        )
        for d, ng in zip(distances, neighbor_gantries)
    ]
    scores = {}
    for n, w in zip(neighbors, w_gant):
        scores[n] = scores.get(n, 0) + w
    return max(scores, key=scores.get)


def get_metrics_for_method(data, pred_col, tp_col, idx_to_name_map):
    metrics = {}
    class_metrics = {}

    ID_PASSENGER_CAR = 0
    ID_VAN = 1

    mask_to_ignore = (data["Original Label"] == ID_PASSENGER_CAR) & (data[pred_col] == ID_VAN)
    casos_ignorados = int(mask_to_ignore.sum())
    data_filtered = data[~mask_to_ignore].copy()

    total_samples = len(data_filtered)
    if total_samples == 0:
        return {
            "accuracy": 0.0,
            "weighted_f1": 0.0,
            "average_f1": 0.0,
            "average_precision": 0.0,
            "average_recall": 0.0,
            "class_metrics": {},
            "ignoring_car_to_van_errors": casos_ignorados,
        }

    unique_classes = sorted(data_filtered["Original Label"].unique())
    total_classes = len(unique_classes)

    weighted_f1 = []
    total_precision = 0.0
    total_recall = 0.0

    for label in unique_classes:
        label_data = data_filtered[data_filtered["Original Label"] == label]
        total_queries = len(label_data)
        if total_queries == 0:
            continue

        tp = int(label_data[tp_col].sum())
        pred_count = int((data_filtered[pred_col] == label).sum())

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / total_queries if total_queries > 0 else 0.0
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

        human_name = idx_to_name_map.get(int(label), str(label))
        class_metrics[human_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_queries": total_queries,
        }

        w = total_queries / total_samples
        weighted_f1.append(f1 * w)
        total_precision += precision
        total_recall += recall

    metrics["accuracy"] = float(data_filtered[tp_col].mean()) if total_samples > 0 else 0.0
    metrics["weighted_f1"] = float(sum(weighted_f1))
    metrics["average_f1"] = float(sum([m["f1"] for m in class_metrics.values()]) / max(total_classes, 1))
    metrics["average_precision"] = float(total_precision / max(total_classes, 1))
    metrics["average_recall"] = float(total_recall / max(total_classes, 1))
    metrics["class_metrics"] = class_metrics
    metrics["ignoring_car_to_van_errors"] = casos_ignorados
    return metrics


def build_eval_dataframe(unseen_df, pred_majority, pred_inverse, pred_squared, pred_gantry):
    data = pd.DataFrame({
        "Query": unseen_df["Path"].tolist(),
        "Original Label": unseen_df["Label"].astype(int).tolist(),
        "Predicted_Majority": pred_majority,
        "Predicted_Inverse": pred_inverse,
        "Predicted_Squared": pred_squared,
        "Predicted_Gantry": pred_gantry,
    })
    data["TP_Majority"] = data["Original Label"] == data["Predicted_Majority"]
    data["TP_Inverse"] = data["Original Label"] == data["Predicted_Inverse"]
    data["TP_Squared"] = data["Original Label"] == data["Predicted_Squared"]
    data["TP_Gantry"] = data["Original Label"] == data["Predicted_Gantry"]
    return data


def evaluate_against_knowledge(knowledge_df, unseen_df, emb_cols, k, idx_to_name_map):
    know_emb = knowledge_df[emb_cols].values
    unseen_emb = unseen_df[emb_cols].values

    distances = pairwise_distances(unseen_emb, know_emb, metric='euclidean')
    actual_k = min(k, know_emb.shape[0])
    knn_idx = np.argsort(distances, axis=1)[:, :actual_k]

    pred_majority, pred_inverse, pred_squared, pred_gantry = [], [], [], []

    know_labels = knowledge_df["Label"].astype(int).values
    know_paths = knowledge_df["Path"].values
    unseen_paths = unseen_df["Path"].values

    for i in range(len(unseen_df)):
        idxs = knn_idx[i]
        neighbor_labels = know_labels[idxs].tolist()
        neighbor_dists = distances[i, idxs].tolist()
        neighbor_paths = know_paths[idxs].tolist()
        query_path = unseen_paths[i]

        pred_majority.append(majority_vote(neighbor_labels))
        pred_inverse.append(inverse_vote(neighbor_labels, neighbor_dists))
        pred_squared.append(squared_vote(neighbor_labels, neighbor_dists))
        pred_gantry.append(gantry_vote(query_path, neighbor_paths, neighbor_labels, neighbor_dists))

    data_eval = build_eval_dataframe(unseen_df, pred_majority, pred_inverse, pred_squared, pred_gantry)

    return {
        "majority": get_metrics_for_method(data_eval, "Predicted_Majority", "TP_Majority", idx_to_name_map),
        "inverse_distance": get_metrics_for_method(data_eval, "Predicted_Inverse", "TP_Inverse", idx_to_name_map),
        "squared_distance": get_metrics_for_method(data_eval, "Predicted_Squared", "TP_Squared", idx_to_name_map),
        "gantry": get_metrics_for_method(data_eval, "Predicted_Gantry", "TP_Gantry", idx_to_name_map),
    }


def compute_acceptance_mask(labels_neighbors, dist_neighbors, unanimity_k, margin_ratio=None):
    unanimous = np.all(labels_neighbors == labels_neighbors[:, [0]], axis=1)
    if margin_ratio is None:
        return unanimous, labels_neighbors[:, 0]

    if dist_neighbors.shape[1] < 2:
        margin_ok = np.ones(dist_neighbors.shape[0], dtype=bool)
    else:
        d1 = dist_neighbors[:, 0]
        d2 = dist_neighbors[:, 1]
        margin_ok = (d2 - d1) / np.maximum(d2, 1e-6) >= margin_ratio

    return unanimous & margin_ok, labels_neighbors[:, 0]


def run_incremental_policy(policy_name, knowledge_df, unseen_df, emb_cols, idx_to_name_map,
                           k_eval=5, unanimity_k=5, margin_ratio=None, batch_size=None, max_iterations=50):
    current_knowledge = knowledge_df.copy().reset_index(drop=True)
    remaining_pool = unseen_df.copy().reset_index(drop=True)
    iteration_rows = []

    for iteration in range(max_iterations + 1):
        eval_metrics = evaluate_against_knowledge(current_knowledge, unseen_df, emb_cols, k_eval, idx_to_name_map)

        iteration_info = {
            "iteration": iteration,
            "policy": policy_name,
            "knowledge_size": int(len(current_knowledge)),
            "remaining_pool": int(len(remaining_pool)),
            "accepted": 0,
            "accepted_correct": 0,
            "acceptance_precision": None,
            "eval_weighted_f1_inverse": eval_metrics["inverse_distance"]["weighted_f1"],
            "eval_average_f1_inverse": eval_metrics["inverse_distance"]["average_f1"],
            "eval_accuracy_inverse": eval_metrics["inverse_distance"]["accuracy"],
            "eval_weighted_f1_squared": eval_metrics["squared_distance"]["weighted_f1"],
            "eval_average_f1_squared": eval_metrics["squared_distance"]["average_f1"],
            "eval_accuracy_squared": eval_metrics["squared_distance"]["accuracy"],
            "eval_weighted_f1_gantry": eval_metrics["gantry"]["weighted_f1"],
            "eval_average_f1_gantry": eval_metrics["gantry"]["average_f1"],
            "eval_accuracy_gantry": eval_metrics["gantry"]["accuracy"],
        }

        if iteration == max_iterations or len(remaining_pool) == 0:
            iteration_rows.append(iteration_info)
            break

        candidate_df = remaining_pool.iloc[:batch_size].copy().reset_index(drop=True) if batch_size else remaining_pool.copy().reset_index(drop=True)

        know_emb = current_knowledge[emb_cols].values
        cand_emb = candidate_df[emb_cols].values
        distances = pairwise_distances(cand_emb, know_emb, metric='euclidean')
        actual_k = min(unanimity_k, know_emb.shape[0])
        knn_idx = np.argsort(distances, axis=1)[:, :actual_k]

        know_labels = current_knowledge["Label"].astype(int).values
        labels_neighbors = know_labels[knn_idx]
        dist_neighbors = np.take_along_axis(distances, knn_idx, axis=1)

        accept_mask, pseudo_labels = compute_acceptance_mask(labels_neighbors, dist_neighbors, actual_k, margin_ratio)
        accepted_df = candidate_df[accept_mask].copy()

        if len(accepted_df) > 0:
            accepted_df.loc[:, "Pseudo_Label"] = pseudo_labels[accept_mask]
            accepted_df.loc[:, "Pseudo_Correct"] = accepted_df["Label"].astype(int).values == accepted_df["Pseudo_Label"].astype(int).values
            accepted_df.loc[:, "Label"] = accepted_df["Pseudo_Label"].astype(int)

            current_knowledge = pd.concat([current_knowledge, accepted_df.drop(columns=["Pseudo_Correct"])], ignore_index=True)
            iteration_info["accepted"] = int(len(accepted_df))
            iteration_info["accepted_correct"] = int(accepted_df["Pseudo_Correct"].sum())
            iteration_info["acceptance_precision"] = float(accepted_df["Pseudo_Correct"].mean())

        accepted_paths = set(accepted_df["Path"].tolist()) if len(accepted_df) > 0 else set()
        if accepted_paths:
            remaining_pool = remaining_pool[~remaining_pool["Path"].isin(accepted_paths)].reset_index(drop=True)
        else:
            iteration_rows.append(iteration_info)
            break

        iteration_rows.append(iteration_info)

    final_metrics = evaluate_against_knowledge(current_knowledge, unseen_df, emb_cols, k_eval, idx_to_name_map)

    return iteration_rows, final_metrics, current_knowledge


def main():
    parser = argparse.ArgumentParser(description="Bloque 5: Autoetiquetado incremental sobre embeddings precalculados")
    parser.add_argument("model_name", type=str, help="Nombre del modelo de referencia, ej. pointnet2_1024d_20260528-113836.pth")
    parser.add_argument("--k_eval", type=int, default=5, help="K para evaluación KNN")
    parser.add_argument("--unanimity_k", type=int, default=5, help="K para criterio de unanimidad")
    parser.add_argument("--margin_ratio", type=float, default=0.10, help="Margen relativo mínimo entre d1 y d2 para la política unanimity_plus_margin")
    parser.add_argument("--batch_size", type=int, default=1000, help="Número de muestras del pool procesadas por iteración; 0 = todo el pool")
    parser.add_argument("--max_iterations", type=int, default=50, help="Máximo número de iteraciones")
    args = parser.parse_args()

    model_name = args.model_name
    batch_size = None if args.batch_size == 0 else args.batch_size

    knowledge_emb_csv, unseen_emb_csv = get_embeddings_paths(model_name)
    idx_to_name_map = get_idx_to_name_map(model_name)

    knowledge_df = pd.read_csv(knowledge_emb_csv)
    unseen_df = pd.read_csv(unseen_emb_csv)

    emb_cols = parse_embedding_columns(knowledge_df)
    knowledge_df["Label"] = knowledge_df["Label"].astype(int)
    unseen_df["Label"] = unseen_df["Label"].astype(int)

    policy_results = {}
    summary_rows = []

    policies = [
        ("unanimity", None),
        ("unanimity_plus_margin", args.margin_ratio),
    ]

    for policy_name, margin_ratio in policies:
        iteration_rows, final_metrics, final_knowledge = run_incremental_policy(
            policy_name=policy_name,
            knowledge_df=knowledge_df,
            unseen_df=unseen_df,
            emb_cols=emb_cols,
            idx_to_name_map=idx_to_name_map,
            k_eval=args.k_eval,
            unanimity_k=args.unanimity_k,
            margin_ratio=margin_ratio,
            batch_size=batch_size,
            max_iterations=args.max_iterations,
        )

        out_dir = "./outputs/active_learning"
        os.makedirs(out_dir, exist_ok=True)

        iter_csv = os.path.join(out_dir, f"incremental_iterations_{policy_name}_{model_name.replace('.pth', '.csv')}")
        pd.DataFrame(iteration_rows).to_csv(iter_csv, index=False)

        final_knowledge_csv = os.path.join(out_dir, f"incremental_knowledge_{policy_name}_{model_name.replace('.pth', '.csv')}")
        final_knowledge.to_csv(final_knowledge_csv, index=False)

        last_row = iteration_rows[-1]
        summary_rows.append({
            "policy": policy_name,
            "final_knowledge_size": int(last_row["knowledge_size"]),
            "remaining_pool": int(last_row["remaining_pool"]),
            "final_weighted_f1_inverse": final_metrics["inverse_distance"]["weighted_f1"],
            "final_average_f1_inverse": final_metrics["inverse_distance"]["average_f1"],
            "final_accuracy_inverse": final_metrics["inverse_distance"]["accuracy"],
            "final_weighted_f1_squared": final_metrics["squared_distance"]["weighted_f1"],
            "final_average_f1_squared": final_metrics["squared_distance"]["average_f1"],
            "final_accuracy_squared": final_metrics["squared_distance"]["accuracy"],
            "final_weighted_f1_gantry": final_metrics["gantry"]["weighted_f1"],
            "final_average_f1_gantry": final_metrics["gantry"]["average_f1"],
            "final_accuracy_gantry": final_metrics["gantry"]["accuracy"],
        })

        policy_results[policy_name] = {
            "iteration_csv": iter_csv,
            "final_knowledge_csv": final_knowledge_csv,
            "final_metrics": final_metrics,
            "iterations": iteration_rows,
        }

    summary_csv = f"./outputs/active_learning/incremental_summary_{model_name.replace('.pth', '.csv')}"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    output_json = f"./outputs/active_learning/incremental_results_{model_name}.json"
    with open(output_json, 'w') as f:
        json.dump({
            "model_name": model_name,
            "k_eval": args.k_eval,
            "unanimity_k": args.unanimity_k,
            "margin_ratio": args.margin_ratio,
            "batch_size": args.batch_size,
            "max_iterations": args.max_iterations,
            "policies": policy_results,
            "summary_csv": summary_csv,
        }, f, indent=4)

    print(f"✅ Resultados guardados en {output_json}")


if __name__ == "__main__":
    main()
