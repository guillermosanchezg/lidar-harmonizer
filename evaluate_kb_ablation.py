"""
Ablation de la Knowledge Base (label efficiency).

Mantiene el modelo y el conjunto Unseen FIJOS y submuestrea la Knowledge Base
a distintos niveles para medir la degradación del rendimiento.

Niveles: 100%, 50%, 20%, cap=200, cap=50, cap=10 instancias por clase (min 5).
Cada nivel se repite con 3 semillas de submuestreo.
Salida: JSON + CSV con average_f1 y weighted_f1 por nivel/regla/K (K=2..10).
"""

import os
import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
with open("params.yaml", "r") as _f:
    _params = yaml.safe_load(_f)

DEVICE_CONFIG = _params["DEVICE_CONFIG"]
device = torch.device(DEVICE_CONFIG["device"] if torch.cuda.is_available() else "cpu")

ABLATION_LEVELS = [
    {"label": "100pct",  "mode": "pct",  "value": 1.00},
    {"label": "50pct",   "mode": "pct",  "value": 0.50},
    {"label": "20pct",   "mode": "pct",  "value": 0.20},
    {"label": "cap200",  "mode": "cap",  "value": 200},
    {"label": "cap50",   "mode": "cap",  "value": 50},
    {"label": "cap10",   "mode": "cap",  "value": 10},
]
ABLATION_SEEDS     = [0, 1, 2]
MIN_PER_CLASS      = 5    # ninguna clase puede extinguirse


# ---------------------------------------------------------------------------
# 1. Submuestreo de la Knowledge Base
# ---------------------------------------------------------------------------
def subsample_kb(df_know: pd.DataFrame, mode: str, value, seed: int) -> pd.DataFrame:
    """
    Submuestrea df_know por clase según el nivel especificado.

    mode='pct': conserva `value` fracción de cada clase (ej. 0.5 = 50%).
    mode='cap': conserva como máximo `value` instancias por clase.
    Mínimo MIN_PER_CLASS por clase siempre.
    """
    rng = random.Random(seed)
    chunks = []
    for label, grp in df_know.groupby("Label"):
        idxs = grp.index.tolist()
        if mode == "pct":
            n = max(MIN_PER_CLASS, int(len(idxs) * value))
        else:
            n = max(MIN_PER_CLASS, min(int(value), len(idxs)))
        chosen = rng.sample(idxs, min(n, len(idxs)))
        chunks.append(grp.loc[chosen])
    return pd.concat(chunks).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Cómputo de kNN (lógica de kvecinos_pyg.py, reutilizada inline)
# ---------------------------------------------------------------------------
def compute_knn(df_know_sub: pd.DataFrame, df_unseen: pd.DataFrame,
                k: int) -> pd.DataFrame:
    """
    Calcula los k vecinos más próximos de cada punto de df_unseen
    contra df_know_sub usando distancia L2 en GPU/CPU.
    """
    emb_cols       = [c for c in df_know_sub.columns if c.startswith("emb_")]
    know_tensor    = torch.tensor(df_know_sub[emb_cols].values, dtype=torch.float32).to(device)
    unseen_tensor  = torch.tensor(df_unseen[emb_cols].values,   dtype=torch.float32).to(device)

    know_paths     = df_know_sub["Path"].tolist()
    know_labels    = df_know_sub["Label"].tolist()
    unseen_paths   = df_unseen["Path"].tolist()
    unseen_labels  = df_unseen["Label"].tolist()

    num_queries = unseen_tensor.shape[0]
    actual_k    = min(k, know_tensor.shape[0])

    rows = []
    batch_size = 512
    for i in range(0, num_queries, batch_size):
        end      = min(i + batch_size, num_queries)
        batch_q  = unseen_tensor[i:end]
        dists    = torch.cdist(batch_q, know_tensor, p=2)
        sorted_i = torch.argsort(dists, dim=1)[:, :actual_k]

        for j in range(end - i):
            q_idx   = i + j
            ni      = sorted_i[j].cpu().numpy()
            n_paths = [know_paths[n] for n in ni]
            n_labs  = [int(know_labels[n]) for n in ni]
            n_dists = dists[j, sorted_i[j]].cpu().numpy().tolist()
            rows.append({
                "Query":          unseen_paths[q_idx],
                "Original Label": int(unseen_labels[q_idx]),
                "Neighbor Labels": n_labs,
                "Distances":       n_dists,
                "Neighbors":       n_paths,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Evaluación frame-level (3 reglas)
# ---------------------------------------------------------------------------
def _frame_preds(data: pd.DataFrame) -> pd.DataFrame:
    pred_maj, pred_inv, pred_sq = [], [], []
    tp_maj,   tp_inv,   tp_sq   = [], [], []

    for _, row in data.iterrows():
        ql  = row["Original Label"]
        nbs = row["Neighbor Labels"]
        ds  = row["Distances"]

        p_maj = Counter(nbs).most_common(1)[0][0]
        pred_maj.append(p_maj); tp_maj.append(ql == p_maj)

        w_inv = [1 / (d + 1e-6) if d > 0 else 1.0 for d in ds]
        sc = {}
        for n, w in zip(nbs, w_inv):
            sc[n] = sc.get(n, 0) + w
        p_inv = max(sc, key=sc.get)
        pred_inv.append(p_inv); tp_inv.append(ql == p_inv)

        w_sq = [1 / (d**2 + 1e-6) if d > 0 else 1.0 for d in ds]
        sc = {}
        for n, w in zip(nbs, w_sq):
            sc[n] = sc.get(n, 0) + w
        p_sq = max(sc, key=sc.get)
        pred_sq.append(p_sq); tp_sq.append(ql == p_sq)

    data = data.copy()
    data["Pred_Majority"] = pred_maj;  data["TP_Majority"] = tp_maj
    data["Pred_Inverse"]  = pred_inv;  data["TP_Inverse"]  = tp_inv
    data["Pred_Squared"]  = pred_sq;   data["TP_Squared"]  = tp_sq
    return data


def _weighted_f1(data: pd.DataFrame, pred_col: str, tp_col: str) -> float:
    total = len(data)
    if total == 0:
        return 0.0
    wf1 = 0.0
    for lbl in data["Original Label"].unique():
        sub     = data[data["Original Label"] == lbl]
        n       = len(sub)
        tp      = sub[tp_col].sum()
        pred_n  = (data[pred_col] == lbl).sum()
        prec    = tp / pred_n if pred_n > 0 else 0.0
        rec     = tp / n      if n      > 0 else 0.0
        f1      = (2 * prec * rec) / max(prec + rec, 1e-6)
        wf1    += f1 * (n / total)
    return wf1


def _avg_f1(data: pd.DataFrame, pred_col: str, tp_col: str) -> float:
    f1s = []
    for lbl in data["Original Label"].unique():
        sub    = data[data["Original Label"] == lbl]
        tp     = sub[tp_col].sum()
        pred_n = (data[pred_col] == lbl).sum()
        prec   = tp / pred_n if pred_n > 0 else 0.0
        rec    = tp / len(sub) if len(sub) > 0 else 0.0
        f1s.append((2 * prec * rec) / max(prec + rec, 1e-6))
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate_frame(data_k: pd.DataFrame) -> dict:
    df = _frame_preds(data_k)
    return {
        rule: {
            "weighted_f1": _weighted_f1(df, f"Pred_{rule}", f"TP_{rule}"),
            "average_f1":  _avg_f1(df,      f"Pred_{rule}", f"TP_{rule}"),
        }
        for rule in ["Majority", "Inverse", "Squared"]
    }


# ---------------------------------------------------------------------------
# 4. Evaluación trayectoria (usa extract_event_id corregido del CAMBIO 1)
# ---------------------------------------------------------------------------
def extract_event_id(path: str) -> str:
    """
    ID único del paso de un vehículo.
    Estructura: .../<gantry>/<class>/<date>/<hour>/<prefix>_<id>.pcd
    Sin clase (evita leakage).
    """
    try:
        parts    = str(path).replace("\\", "/").split("/")
        filename = parts[-1]
        prefix   = filename.split("_")[0]
        date     = parts[-3]
        hour     = parts[-2]
        return f"{date}_{hour}_{prefix}"
    except Exception:
        return os.path.basename(str(path))


def evaluate_trajectory(data_k: pd.DataFrame) -> dict:
    df = _frame_preds(data_k)
    df["event_id"] = df["Query"].apply(extract_event_id)

    results = {}
    for rule in ["Majority", "Inverse", "Squared"]:
        base_col  = f"Pred_{rule}"
        voted     = df.groupby("event_id")[base_col].agg(lambda x: x.mode()[0])
        final_col = f"Voted_{rule}"
        tp_col    = f"VTP_{rule}"
        df[final_col] = df["event_id"].map(voted)
        df[tp_col]    = df["Original Label"] == df[final_col]
        results[rule] = {
            "weighted_f1": _weighted_f1(df, final_col, tp_col),
            "average_f1":  _avg_f1(df,      final_col, tp_col),
        }
    return results


# ---------------------------------------------------------------------------
# 5. Bucle principal
# ---------------------------------------------------------------------------
def run_ablation(df_know: pd.DataFrame, df_unseen: pd.DataFrame) -> list:
    """
    Para cada nivel de ablación, semilla y K: calcula kNN, evalúa frame+traj.
    Devuelve lista de dicts lista para convertir en DataFrame.
    """
    rows = []
    total = len(ABLATION_LEVELS) * len(ABLATION_SEEDS) * 9  # K=2..10
    done  = 0

    for level_cfg in ABLATION_LEVELS:
        level_label = level_cfg["label"]
        mode        = level_cfg["mode"]
        value       = level_cfg["value"]

        print(f"\n{'='*60}")
        print(f"Nivel: {level_label}  (mode={mode}, value={value})")

        # Resultados por semilla y K
        seed_results = defaultdict(lambda: defaultdict(dict))  # seed -> K -> rule -> metric

        for seed in ABLATION_SEEDS:
            df_sub = subsample_kb(df_know, mode, value, seed)
            print(f"  seed={seed}: {len(df_sub)} instancias KB "
                  f"({df_sub.groupby('Label').size().to_dict()})")

            for k in range(2, 11):
                df_knn        = compute_knn(df_sub, df_unseen, k)
                frame_metrics = evaluate_frame(df_knn)
                traj_metrics  = evaluate_trajectory(df_knn)

                for rule in ["Majority", "Inverse", "Squared"]:
                    seed_results[seed][k][rule] = {
                        "frame_weighted_f1": frame_metrics[rule]["weighted_f1"],
                        "frame_average_f1":  frame_metrics[rule]["average_f1"],
                        "traj_weighted_f1":  traj_metrics[rule]["weighted_f1"],
                        "traj_average_f1":   traj_metrics[rule]["average_f1"],
                    }

                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] K={k}, seed={seed}")

        # Agregar media ± std sobre semillas
        for k in range(2, 11):
            for rule in ["Majority", "Inverse", "Squared"]:
                for metric in ["frame_weighted_f1", "frame_average_f1",
                                "traj_weighted_f1",  "traj_average_f1"]:
                    vals = [seed_results[s][k][rule][metric] for s in ABLATION_SEEDS]
                    rows.append({
                        "level":         level_label,
                        "mode":          mode,
                        "value":         str(value),
                        "K":             k,
                        "rule":          rule.lower(),
                        "metric":        metric,
                        "mean":          float(np.mean(vals)),
                        "std":           float(np.std(vals)),
                        "values":        vals,
                    })

    return rows


# ---------------------------------------------------------------------------
# 6. Punto de entrada
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ablación de la Knowledge Base (label efficiency)"
    )
    parser.add_argument("model_name", type=str,
                        help="Nombre del modelo, ej. pointnet2_1024d_a0.5_XXX.pth")
    parser.add_argument("--out_dir", type=str, default="./outputs/ablation",
                        help="Directorio de salida")
    args = parser.parse_args()

    model_name = args.model_name
    out_dir    = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Rutas a embeddings precalculados
    emb_dir    = "./outputs/embeddings"
    know_csv   = os.path.join(emb_dir, f"embeddings_knowledge_{model_name.replace('.pth', '.csv')}")
    unseen_csv = os.path.join(emb_dir, f"embeddings_unseen_{model_name.replace('.pth', '.csv')}")

    for p in [know_csv, unseen_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No encontrado: {p}\n"
                "Ejecuta generate_embeddings_pyg.py primero."
            )

    print(f"Cargando Knowledge: {know_csv}")
    df_know   = pd.read_csv(know_csv)
    print(f"  -> {len(df_know)} instancias, {df_know['Label'].nunique()} clases")

    print(f"Cargando Unseen:    {unseen_csv}")
    df_unseen = pd.read_csv(unseen_csv)
    print(f"  -> {len(df_unseen)} instancias")

    # Ejecutar ablación
    rows = run_ablation(df_know, df_unseen)

    # Guardar CSV
    df_out    = pd.DataFrame(rows)
    csv_path  = os.path.join(out_dir, f"kb_ablation_{model_name.replace('.pth', '.csv')}")
    df_out.to_csv(csv_path, index=False)
    print(f"\n✅ CSV guardado en {csv_path}")

    # Guardar JSON (resumen pivotado: level/rule/K → mean ± std)
    summary = {}
    for _, r in df_out.iterrows():
        key = f"{r['level']}/{r['rule']}/K{r['K']}/{r['metric']}"
        summary[key] = {"mean": r["mean"], "std": r["std"]}

    json_path = os.path.join(out_dir, f"kb_ablation_{model_name.replace('.pth', '.json')}")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ JSON guardado en {json_path}")

    # Tabla resumen en consola (traj_weighted_f1, K=5, seed-mean)
    print("\n--- Resumen: traj_weighted_f1 @ K=5 ---")
    pivot = df_out[(df_out["K"] == 5) & (df_out["metric"] == "traj_weighted_f1")]
    for level in [l["label"] for l in ABLATION_LEVELS]:
        for rule in ["majority", "inverse", "squared"]:
            row = pivot[(pivot["level"] == level) & (pivot["rule"] == rule)]
            if not row.empty:
                m, s = row["mean"].values[0], row["std"].values[0]
                print(f"  {level:<10} {rule:<15} {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
