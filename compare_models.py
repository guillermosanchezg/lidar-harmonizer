"""
Leaderboard de modelos entrenados.

Lee los JSON de métricas (frame y trayectoria) y los CSV de latencia que existan
en ./outputs/ y genera:
  - Una tabla resumen CSV  →  ./outputs/leaderboard.csv
  - Una tabla markdown     →  stdout
  - El campeón seleccionado según el criterio predefinido (máx average_f1 trayectoria;
    desempate por latencia si la diferencia es < 0.01).

Funciona de forma incremental: con los resultados que haya en cada momento.
"""

import os
import json
import re
import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Rutas de outputs
# ---------------------------------------------------------------------------
RESULTS_DIR = "./outputs/knn_results"
STATS_DIR   = "./outputs/stats"
OUT_CSV     = "./outputs/leaderboard.csv"

# Clases minoritarias de interés para el paper
MINORITY_CLASSES = ["articulated truck", "other vehicles", "other"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_model_meta(model_id: str) -> dict:
    """
    Extrae metadatos del nombre del modelo.
    Formato esperado: {type}_{dim}d_a{alpha}_{mode}_{timestamp}
    Ejemplo: pointnet2_1024d_a0.5_sc_20240315-143022
    """
    meta = {
        "model_id":   model_id,
        "model_type": "unknown",
        "dim":        None,
        "alpha":      None,
        "mode":       "sc",    # default retrocompatible
    }
    parts = model_id.split("_")

    # Tipo (puede ser multi-token: pointnet2)
    if parts[0] in ("dgcnn", "pointmlp"):
        meta["model_type"] = parts[0]
        offset = 1
    elif len(parts) > 1 and parts[0] + "_" + parts[1] == "pointnet2":
        meta["model_type"] = "pointnet2"
        offset = 2
    else:
        meta["model_type"] = parts[0]
        offset = 1

    # dim: Xd
    for i, p in enumerate(parts[offset:], start=offset):
        if re.match(r"^\d+d$", p):
            meta["dim"] = int(p[:-1])
            offset = i + 1
            break

    # alpha: aX.X
    for i, p in enumerate(parts[offset:], start=offset):
        m = re.match(r"^a([\d.]+)$", p)
        if m:
            meta["alpha"] = float(m.group(1))
            offset = i + 1
            break

    # modo: sc / fg
    for i, p in enumerate(parts[offset:], start=offset):
        if p in ("sc", "fg"):
            meta["mode"] = p
            break

    return meta


def _best_over_all(metrics_dict: dict) -> tuple:
    """
    Dado el dict de métricas {rule: {k: {average_f1, weighted_f1, ...}}},
    devuelve (average_f1, weighted_f1, best_rule, best_k).
    """
    best_avg_f1, best_wf1, best_rule, best_k = 0.0, 0.0, None, None
    for rule, ks in metrics_dict.items():
        for k_str, m in ks.items():
            try:
                k   = int(k_str)
                avg = float(m.get("average_f1",  0))
                wf1 = float(m.get("weighted_f1", 0))
            except (ValueError, TypeError):
                continue
            if avg > best_avg_f1:
                best_avg_f1, best_wf1 = avg, wf1
                best_rule, best_k     = rule, k
    return best_avg_f1, best_wf1, best_rule, best_k


def _minority_f1(metrics_dict: dict, best_rule: str, best_k: int) -> dict:
    """Extrae F1 de las clases minoritarias en el mejor (rule, K)."""
    result = {}
    if best_rule is None or best_k is None:
        return result
    class_m = (metrics_dict
               .get(best_rule, {})
               .get(str(best_k), {})
               .get("class_metrics", {}))
    for cname, cm in class_m.items():
        for target in MINORITY_CLASSES:
            if target in cname.lower():
                result[cname] = round(cm.get("f1", 0.0), 4)
    return result


def _avg_latency(latency_csv: str) -> float | None:
    if not os.path.exists(latency_csv):
        return None
    try:
        df = pd.read_csv(latency_csv)
        if "t_total_ms" in df.columns:
            return float(df["t_total_ms"].mean())
    except Exception:
        pass
    return None


def _load_stats(stats_path: str) -> dict:
    if not os.path.exists(stats_path):
        return {}
    try:
        with open(stats_path) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Recopilación de modelos
# ---------------------------------------------------------------------------
def collect_models() -> list[dict]:
    """Detecta todos los model_id a partir de ficheros en outputs/."""
    model_ids = set()

    for fname in Path(RESULTS_DIR).glob("metrics_*.json"):
        stem = fname.stem
        if stem.startswith("metrics_trajectory_"):
            mid = stem[len("metrics_trajectory_"):]
        else:
            mid = stem[len("metrics_"):]
        model_ids.add(mid)

    rows = []

    for model_id in sorted(model_ids):
        meta = _parse_model_meta(model_id)

        # Cargar stats JSON
        stats_path = os.path.join(STATS_DIR, f"{model_id}.pth_stats.json")
        if not os.path.exists(stats_path):
            # intentar sin .pth
            stats_path = os.path.join(STATS_DIR, f"{model_id}_stats.json")
        stats = _load_stats(stats_path)

        max_instances = (stats.get("training_hyperparameters", {})
                         .get("max_instances"))
        triplet_alpha_from_stats = stats.get("triplet_alpha")

        # Métricas frame-level
        frame_json = os.path.join(RESULTS_DIR, f"metrics_{model_id}.json")
        frame_metrics = {}
        if os.path.exists(frame_json):
            with open(frame_json) as f:
                raw = json.load(f)
            for key in ("metrics_majority", "metrics_inverse", "metrics_squared"):
                rule = key.replace("metrics_", "")
                if key in raw:
                    frame_metrics[rule] = raw[key]

        best_frame_avg, best_frame_wf1, best_frame_rule, best_frame_k = _best_over_all(frame_metrics)

        # Métricas trayectoria
        traj_json = os.path.join(RESULTS_DIR, f"metrics_trajectory_{model_id}.json")
        traj_metrics = {}
        if os.path.exists(traj_json):
            with open(traj_json) as f:
                raw = json.load(f)
            for key in ("metrics_majority", "metrics_inverse", "metrics_squared"):
                rule = key.replace("metrics_", "")
                if key in raw:
                    traj_metrics[rule] = raw[key]

        best_traj_avg, best_traj_wf1, best_traj_rule, best_traj_k = _best_over_all(traj_metrics)
        minority = _minority_f1(traj_metrics, best_traj_rule, best_traj_k)

        # Latencia
        # Los metrics se nombran metrics_{mid}.pth.json (de ahí que model_id lleve
        # el sufijo .pth), pero los CSV de latencia se guardan SIN .pth.
        mid_no_pth = model_id[:-4] if model_id.endswith(".pth") else model_id
        lat_csv = os.path.join(RESULTS_DIR, f"latency_benchmark_{mid_no_pth}.csv")
        avg_latency = _avg_latency(lat_csv)

        row = {
            "model_id":             model_id,
            "model_type":           meta["model_type"],
            "mode":                 meta["mode"],
            "dim":                  meta["dim"],
            "triplet_alpha":        meta["alpha"] if meta["alpha"] is not None
                                    else triplet_alpha_from_stats,
            "max_instances":        max_instances,
            # Frame
            "frame_avg_f1":         round(best_frame_avg, 4) if best_frame_avg else None,
            "frame_wf1":            round(best_frame_wf1, 4) if best_frame_wf1 else None,
            "frame_best_rule":      best_frame_rule,
            "frame_best_k":         best_frame_k,
            # Trayectoria
            "traj_avg_f1":          round(best_traj_avg, 4) if best_traj_avg else None,
            "traj_wf1":             round(best_traj_wf1, 4) if best_traj_wf1 else None,
            "traj_best_rule":       best_traj_rule,
            "traj_best_k":          best_traj_k,
            # Clases minoritarias (F1)
            "minority_class_f1":    str(minority) if minority else "",
            # Latencia
            "avg_latency_ms":       round(avg_latency, 4) if avg_latency is not None else None,
            "is_champion":          False,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Selección del campeón
# ---------------------------------------------------------------------------
CHAMPION_TIE_THRESHOLD = 0.01   # si la diferencia de avg_f1 es < esto → desempata latencia


def select_champion(rows: list[dict]) -> tuple[int, str]:
    """
    Devuelve (índice del campeón, razón textual).
    Criterio: máx traj_avg_f1; si diff < 0.01 → menor latencia.
    """
    candidates = [r for r in rows if r["traj_avg_f1"] is not None]
    if not candidates:
        return -1, "No hay modelos con métricas de trayectoria."

    best_f1 = max(r["traj_avg_f1"] for r in candidates)

    # Candidatos dentro del umbral
    within = [r for r in candidates if best_f1 - r["traj_avg_f1"] < CHAMPION_TIE_THRESHOLD]

    if len(within) == 1:
        champ = within[0]
        reason = (f"Máximo traj_avg_f1 = {champ['traj_avg_f1']:.4f} "
                  f"(regla={champ['traj_best_rule']}, K={champ['traj_best_k']})")
    else:
        # Desempate por latencia
        with_lat = [r for r in within if r["avg_latency_ms"] is not None]
        if with_lat:
            champ  = min(with_lat, key=lambda r: r["avg_latency_ms"])
            reason = (f"Empate técnico (diff < {CHAMPION_TIE_THRESHOLD}) en traj_avg_f1; "
                      f"menor latencia = {champ['avg_latency_ms']:.4f} ms → "
                      f"{champ['model_id']}")
        else:
            champ  = within[0]
            reason = (f"Empate técnico sin datos de latencia; primer candidato: "
                      f"{champ['model_id']}")

    # Marcar en rows
    for i, r in enumerate(rows):
        if r["model_id"] == champ["model_id"]:
            rows[i]["is_champion"] = True
            return i, reason

    return -1, "Error inesperado en selección."


# ---------------------------------------------------------------------------
# Formateo markdown
# ---------------------------------------------------------------------------
def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_Sin modelos._\n"

    cols = [
        ("model_id",       "Modelo"),
        ("mode",           "Modo"),
        ("model_type",     "Arq."),
        ("triplet_alpha",  "α"),
        ("max_instances",  "maxInst"),
        ("traj_avg_f1",    "Traj avgF1"),
        ("traj_wf1",       "Traj wF1"),
        ("traj_best_rule", "Regla"),
        ("traj_best_k",    "K"),
        ("frame_avg_f1",   "Frame avgF1"),
        ("frame_wf1",      "Frame wF1"),
        ("avg_latency_ms", "Lat(ms)"),
        ("is_champion",    "★"),
    ]

    header  = "| " + " | ".join(c[1] for c in cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    lines   = [header, divider]

    for r in sorted(rows, key=lambda x: -(x["traj_avg_f1"] or 0)):
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if key == "is_champion":
                cells.append("✓" if v else "")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Leaderboard de modelos V2X")
    parser.add_argument("--out", type=str, default=OUT_CSV,
                        help="Ruta de salida del CSV resumen")
    args = parser.parse_args()

    print("Escaneando resultados en ./outputs/ ...")
    rows = collect_models()

    if not rows:
        print("⚠️  No se encontró ningún modelo con métricas. "
              "Ejecuta el pipeline completo primero.")
        return

    champ_idx, champ_reason = select_champion(rows)

    # Guardar CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"✅ Leaderboard guardado en {args.out}")

    # Imprimir tabla markdown
    print("\n" + "=" * 70)
    print("LEADERBOARD  (ordenado por traj_avg_f1 desc)")
    print("=" * 70)
    print(_md_table(rows))

    # Imprimir campeón
    print("=" * 70)
    print("CAMPEÓN:")
    if champ_idx >= 0:
        c = rows[champ_idx]
        print(f"  → {c['model_id']}")
        print(f"     Razón : {champ_reason}")
        print(f"     traj_avg_f1 = {c['traj_avg_f1']}  |  "
              f"traj_wf1 = {c['traj_wf1']}  |  "
              f"Regla={c['traj_best_rule']}  K={c['traj_best_k']}")
        if c["minority_class_f1"]:
            print(f"     Clases minoritarias: {c['minority_class_f1']}")
        if c["avg_latency_ms"] is not None:
            print(f"     Latencia media: {c['avg_latency_ms']:.4f} ms")
    else:
        print(f"  No determinado: {champ_reason}")
    print("=" * 70)


if __name__ == "__main__":
    main()
