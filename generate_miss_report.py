"""
Genera un reporte por vehículo de todos los fallos de clasificación
de trayectoria, con la distribución de votos de vecinos para ayudar
a identificar errores de etiquetado.
"""
import os
import json
import ast
import argparse
from collections import Counter

import pandas as pd


def parse_path(p):
    parts = str(p).replace("\\", "/").split("/")
    fname    = parts[-1]
    prefix   = fname.split("_")[0]
    date     = parts[-3]
    hour     = parts[-2]
    gantry   = parts[-5] if len(parts) >= 5 else "?"
    event_id = f"{date}_{hour}_{prefix}"
    return event_id, gantry, date, hour, str(p)


def neighbor_vote_dist(labels_str, k, idx2name):
    try:
        labs = ast.literal_eval(labels_str)[:k]
        c = Counter(labs)
        total = sum(c.values())
        return {idx2name.get(l, str(l)): round(v / total, 3) for l, v in c.most_common()}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--rule", default="majority",
                        choices=["majority", "inverse", "squared"])
    parser.add_argument("--k", type=int, default=9,
                        help="k usado para calcular la distribución de vecinos")
    args = parser.parse_args()

    model = args.model_name
    rule  = args.rule
    k     = args.k

    miss_path  = f"outputs/knn_results/misses_trajectory_{model}_{rule}.csv"
    knn_path   = f"outputs/knn_results/k_neighbors_all_{model}.csv"
    stats_path = f"outputs/stats/{model}_stats.json"
    out_path   = f"outputs/knn_results/miss_report_vehicles_{model}_{rule}.csv"

    if not os.path.exists(miss_path):
        print(f"No existe: {miss_path}")
        return

    # --- Mapas de etiquetas ---
    idx2name = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        for cls_name, idx in stats.get("class_mapping", {}).items():
            if idx not in idx2name:
                idx2name[idx] = cls_name

    pred_col = f"Predicted_{rule.capitalize()}"

    miss = pd.read_csv(miss_path)
    knn  = pd.read_csv(knn_path)

    # --- Parsear paths ---
    for df in [miss, knn]:
        parsed = df["Query"].apply(lambda p: pd.Series(parse_path(p),
                                   index=["event_id","gantry","date","hour","path"]))
        df[["event_id","gantry","date","hour","path"]] = parsed

    # --- Track length desde el CSV completo ---
    track_len = knn.groupby("event_id").size().rename("track_length")

    # --- Distribución de vecinos ---
    miss["_nbr_dist"] = miss["Neighbor Labels"].apply(
        lambda s: neighbor_vote_dist(s, k, idx2name))

    # --- Construir una fila por vehículo ---
    rows = []
    for eid, grp in miss.groupby("event_id"):
        true_lbl = int(grp["Original Label"].iloc[0])
        pred_lbl = int(grp[pred_col].iloc[0])
        true_name = idx2name.get(true_lbl, str(true_lbl))
        pred_name = idx2name.get(pred_lbl, str(pred_lbl))

        gantries = sorted(grp["gantry"].unique().tolist())
        dates    = sorted(grp["date"].unique().tolist())
        tlen     = int(track_len.get(eid, len(grp)))

        # Agregar distribuciones de todos los frames del vehículo
        merged = Counter()
        for d in grp["_nbr_dist"]:
            for cls, frac in d.items():
                merged[cls] += frac
        total_w = sum(merged.values())
        agg_dist = ({c: round(v / total_w, 3) for c, v in merged.most_common()}
                    if total_w > 0 else {})

        confidence = agg_dist.get(pred_name, 0.0)

        # ¿Qué clase domina en los vecinos?
        top_neighbor = list(agg_dist.keys())[0] if agg_dist else ""
        top_neighbor_pct = list(agg_dist.values())[0] if agg_dist else 0.0

        # Rutas de muestra (hasta 3)
        paths = grp["path"].head(3).tolist()

        rows.append({
            "event_id":           eid,
            "true_class":         true_name,
            "predicted_class":    pred_name,
            "confusion":          f"{true_name} → {pred_name}",
            "track_length":       tlen,
            "n_frames_missed":    len(grp),
            "gantries":           "|".join(gantries),
            "dates":              "|".join(dates),
            "neighbor_vote_dist": json.dumps(agg_dist),
            "pred_confidence":    round(confidence, 3),
            "top_neighbor_class": top_neighbor,
            "top_neighbor_pct":   round(top_neighbor_pct, 3),
            # Vecinos apuntan mayoritariamente a clase distinta de la etiqueta = posible error
            # top_neighbor_pct > 0.5 significa que >50% de los k vecinos son de otra clase
            "label_error_suspect": (top_neighbor != true_name and top_neighbor_pct > 0.5),
            "sample_path_1":      paths[0] if len(paths) > 0 else "",
            "sample_path_2":      paths[1] if len(paths) > 1 else "",
            "sample_path_3":      paths[2] if len(paths) > 2 else "",
        })

    df_out = pd.DataFrame(rows).sort_values(["confusion", "pred_confidence"])
    df_out.to_csv(out_path, index=False)

    print(f"Reporte generado: {out_path}")
    print(f"Total vehículos fallados: {len(df_out)}")
    print()

    print("=== Resumen por confusión ===")
    summary = (df_out.groupby("confusion")
               .agg(n_veh=("event_id", "count"),
                    avg_confidence=("pred_confidence", "mean"),
                    avg_track_len=("track_length", "mean"),
                    suspects=("label_error_suspect", "sum"))
               .sort_values("n_veh", ascending=False))
    print(summary.to_string())

    print()
    suspects = df_out[df_out["label_error_suspect"]].sort_values("pred_confidence")
    print(f"=== Posibles errores de etiquetado ({len(suspects)} vehículos) ===")
    print("(Vecinos apuntan mayoritariamente a otra clase y confianza < 35%)")
    cols = ["event_id", "confusion", "track_length",
            "pred_confidence", "neighbor_vote_dist", "gantries", "sample_path_1"]
    print(suspects[cols].to_string())


if __name__ == "__main__":
    main()
