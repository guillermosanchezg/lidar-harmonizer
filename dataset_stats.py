"""
Estadísticas descriptivas del dataset V2X.

Itera sobre ../dataset_labeled/<gantry>/<class>/<date>/<hour>/<prefix>_<id>.pcd
y genera:
  - dataset_stats.json              → stats completas
  - tabla_por_superclase.csv        → una fila por superclase
  - tabla_por_clase_fina.csv        → una fila por clase original
  - visualizations/dataset/*.png    → histogramas (--plots)

Uso:
  python3 dataset_stats.py [--dataset_root ../dataset_labeled] [--plots]
"""

import os
import sys
import json
import argparse
import random
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import open3d as o3d
    OPEN3D_OK = True
except ImportError:
    OPEN3D_OK = False
    print("[WARN] open3d no disponible — se omiten stats de puntos por nube.")

# ---------------------------------------------------------------------------
# Mapeo de clases
# ---------------------------------------------------------------------------
SUPERCLASS_MAPPING = {
    2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,   # Passenger Car
    8: 1, 9: 1, 10: 1,                       # Van
    11: 2,                                   # Bus
    13: 3, 14: 3, 15: 3,                     # Rigid Truck
    17: 4, 18: 4, 19: 4,                     # Articulated Truck
}

SUPERCLASS_NAMES = {
    0: "Passenger Car",
    1: "Van",
    2: "Bus",
    3: "Rigid Truck",
    4: "Articulated Truck",
    5: "Other Vehicles",
}

CLASS_NAMES = {
    2: "Passenger Car Type 1",
    3: "Passenger Car Type 2",
    4: "Passenger Car Type 3",
    5: "Passenger Car Type 4",
    6: "Passenger Car Type 5",
    7: "Passenger Car Type 6",
    8: "Van Type 1",
    9: "Van Type 2",
    10: "Van Type 3",
    11: "Bus",
    13: "Rigid Truck Type 1",
    14: "Rigid Truck Type 2",
    15: "Rigid Truck Type 3",
    17: "Articulated Truck Type 1",
    18: "Articulated Truck Type 2",
    19: "Articulated Truck Type 3",
}

VALID_CLASSES = set(SUPERCLASS_MAPPING.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_event_id(path_parts: list[str]) -> str:
    """date_hour_prefix — mismo criterio que evaluate_KNN_trajectory.py."""
    filename = path_parts[-1]
    prefix   = filename.split("_")[0]
    date     = path_parts[-3]
    hour     = path_parts[-2]
    return f"{date}_{hour}_{prefix}"


def stats_1d(values: list) -> dict:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "median": float(np.median(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


def count_points(pcd_path: str) -> int | None:
    if not OPEN3D_OK:
        return None
    try:
        pc = o3d.io.read_point_cloud(pcd_path)
        return len(np.asarray(pc.points))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Escaneo principal
# ---------------------------------------------------------------------------
def scan_dataset(dataset_root: str, max_sample_per_class: int = 200) -> dict:
    """
    Recorre el dataset y recopila toda la información estadística.
    Estructura de retorno:
      {
        "classes": { str(cls): { "frames": n, "vehicles": set, "gantries": {g: n},
                                  "days": set, "hours": [h, ...],
                                  "point_counts": [n_pts, ...] } },
        "global": { "gantries": {g: n}, "dates": set, "hours": [h, ...],
                    "total_frames": int, "total_corrupted": int }
      }
    """
    root = Path(dataset_root)
    if not root.exists():
        print(f"❌ No existe el directorio: {dataset_root}")
        sys.exit(1)

    # Datos acumulados por clase original
    cls_data: dict[str, dict] = {}
    for cls in VALID_CLASSES:
        cls_data[str(cls)] = {
            "frames":       0,
            "vehicles":     set(),
            "gantries":     defaultdict(int),
            "days":         set(),
            "hours":        [],
            "point_counts": [],   # muestra de hasta max_sample_per_class
            "corrupted":    0,
        }

    global_gantries: dict[str, int] = defaultdict(int)
    global_dates:    set             = set()
    global_hours:    list            = []
    total_frames:    int             = 0
    total_corrupted: int             = 0

    # Iteración: gantry / class / date / hour / file
    gantry_dirs = [d for d in root.iterdir() if d.is_dir()]
    print(f"Escaneando {len(gantry_dirs)} gantries en {dataset_root} ...")

    for gantry_dir in sorted(gantry_dirs):
        gantry = gantry_dir.name

        for class_dir in sorted(gantry_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                cls_int = int(class_dir.name)
            except ValueError:
                continue

            cls_str = str(cls_int)
            if cls_str not in cls_data:
                continue

            for date_dir in sorted(class_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                date = date_dir.name
                global_dates.add(date)
                cls_data[cls_str]["days"].add(date)

                for hour_dir in sorted(date_dir.iterdir()):
                    if not hour_dir.is_dir():
                        continue
                    hour = hour_dir.name
                    try:
                        hour_int = int(hour)
                    except ValueError:
                        hour_int = -1

                    for pcd_file in sorted(hour_dir.glob("*.pcd")):
                        path_parts = str(pcd_file).replace("\\", "/").split("/")
                        try:
                            event_id = extract_event_id(path_parts)
                        except Exception:
                            event_id = pcd_file.name

                        cls_data[cls_str]["frames"] += 1
                        cls_data[cls_str]["vehicles"].add(event_id)
                        cls_data[cls_str]["gantries"][gantry] += 1
                        if hour_int >= 0:
                            cls_data[cls_str]["hours"].append(hour_int)

                        global_gantries[gantry] += 1
                        global_hours.append(hour_int) if hour_int >= 0 else None

                        # Muestro de puntos
                        if (OPEN3D_OK and
                                len(cls_data[cls_str]["point_counts"]) < max_sample_per_class):
                            n_pts = count_points(str(pcd_file))
                            if n_pts is not None and n_pts > 0:
                                cls_data[cls_str]["point_counts"].append(n_pts)
                            else:
                                cls_data[cls_str]["corrupted"] += 1
                                total_corrupted += 1

                        total_frames += 1

    global_info = {
        "gantries":       dict(global_gantries),
        "dates":          sorted(global_dates),
        "hours":          global_hours,
        "total_frames":   total_frames,
        "total_corrupted": total_corrupted,
    }

    return {"classes": cls_data, "global": global_info}


# ---------------------------------------------------------------------------
# Construcción de tablas
# ---------------------------------------------------------------------------
def build_class_rows(cls_data: dict) -> list[dict]:
    rows = []
    for cls_str, d in sorted(cls_data.items(), key=lambda x: int(x[0])):
        cls_int = int(cls_str)
        sc      = SUPERCLASS_MAPPING.get(cls_int, 5)
        name    = CLASS_NAMES.get(cls_int, cls_str)
        sc_name = SUPERCLASS_NAMES.get(sc, "Other Vehicles")

        vehicles = d["vehicles"]
        n_frames = d["frames"]
        n_veh    = len(vehicles)

        # frames per vehicle
        frames_per_veh = (n_frames / n_veh) if n_veh > 0 else 0.0

        # point stats
        pt_stats = stats_1d(d["point_counts"])

        row = {
            "class_id":          cls_int,
            "class_name":        name,
            "superclass_id":     sc,
            "superclass_name":   sc_name,
            "frames":            n_frames,
            "unique_vehicles":   n_veh,
            "frames_per_vehicle": round(frames_per_veh, 2),
            "capture_days":      len(d["days"]),
            "gantries":          json.dumps(dict(sorted(d["gantries"].items()))),
            "pts_mean":          round(pt_stats["mean"], 1) if pt_stats["mean"] is not None else None,
            "pts_std":           round(pt_stats["std"],  1) if pt_stats["std"]  is not None else None,
            "pts_median":        round(pt_stats["median"], 1) if pt_stats["median"] is not None else None,
            "pts_min":           pt_stats["min"],
            "pts_max":           pt_stats["max"],
        }
        rows.append(row)
    return rows


def build_superclass_rows(class_rows: list[dict]) -> list[dict]:
    sc_accum: dict[int, dict] = defaultdict(lambda: {
        "frames": 0, "vehicles": set(), "days": set(),
        "gantries": defaultdict(int), "pt_counts": []
    })

    # Re-escanear cls_data para acumular vehicles y days exactos por SC
    # En este punto solo tenemos class_rows → no podemos agregar sets fácilmente.
    # Por tanto la función de build espera recibir también cls_data.
    raise RuntimeError("Use build_superclass_rows_from_cls instead")


def build_superclass_rows_from_cls(cls_data: dict) -> list[dict]:
    sc_accum: dict[int, dict] = {}
    for sc_id in SUPERCLASS_NAMES:
        sc_accum[sc_id] = {
            "frames":     0,
            "vehicles":   set(),
            "days":       set(),
            "gantries":   defaultdict(int),
            "pt_counts":  [],
        }

    for cls_str, d in cls_data.items():
        cls_int = int(cls_str)
        sc      = SUPERCLASS_MAPPING.get(cls_int, 5)
        a       = sc_accum[sc]
        a["frames"]   += d["frames"]
        a["vehicles"] |= d["vehicles"]
        a["days"]     |= d["days"]
        a["pt_counts"] += d["point_counts"]
        for g, n in d["gantries"].items():
            a["gantries"][g] += n

    rows = []
    for sc_id in sorted(sc_accum.keys()):
        a       = sc_accum[sc_id]
        n_veh   = len(a["vehicles"])
        n_frames = a["frames"]
        fpv     = (n_frames / n_veh) if n_veh > 0 else 0.0
        pt_stats = stats_1d(a["pt_counts"])

        rows.append({
            "superclass_id":      sc_id,
            "superclass_name":    SUPERCLASS_NAMES[sc_id],
            "frames":             n_frames,
            "unique_vehicles":    n_veh,
            "frames_per_vehicle": round(fpv, 2),
            "capture_days":       len(a["days"]),
            "gantries":           json.dumps(dict(sorted(a["gantries"].items()))),
            "pts_mean":           round(pt_stats["mean"], 1) if pt_stats["mean"] is not None else None,
            "pts_std":            round(pt_stats["std"],  1) if pt_stats["std"]  is not None else None,
            "pts_median":         round(pt_stats["median"], 1) if pt_stats["median"] is not None else None,
            "pts_min":            pt_stats["min"],
            "pts_max":            pt_stats["max"],
        })
    return rows


# ---------------------------------------------------------------------------
# Histogramas opcionales
# ---------------------------------------------------------------------------
def make_plots(cls_data: dict, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib no disponible — se omiten histogramas.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1. Frames por clase
    cls_ids   = sorted(cls_data.keys(), key=int)
    frames    = [cls_data[c]["frames"] for c in cls_ids]
    labels    = [CLASS_NAMES.get(int(c), c) for c in cls_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(cls_ids)), frames)
    ax.set_xticks(range(len(cls_ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frames")
    ax.set_title("Frames por clase original")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "frames_per_class.png"), dpi=150)
    plt.close()

    # 2. Vehículos únicos por clase
    vehicles = [len(cls_data[c]["vehicles"]) for c in cls_ids]
    fig, ax  = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(cls_ids)), vehicles, color="orange")
    ax.set_xticks(range(len(cls_ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Vehículos únicos (event_id)")
    ax.set_title("Vehículos únicos por clase original")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vehicles_per_class.png"), dpi=150)
    plt.close()

    # 3. Distribución hora del día (global)
    all_hours = []
    for d in cls_data.values():
        all_hours.extend(d["hours"])

    if all_hours:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(all_hours, bins=range(0, 25), edgecolor="black")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Frames")
        ax.set_title("Distribución por hora del día (global)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hour_distribution.png"), dpi=150)
        plt.close()

    # 4. Distribución de puntos por nube (por clase)
    for cls_str in cls_ids:
        pts = cls_data[cls_str]["point_counts"]
        if len(pts) < 5:
            continue
        cls_int = int(cls_str)
        name    = CLASS_NAMES.get(cls_int, cls_str)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pts, bins=30, edgecolor="black")
        ax.set_xlabel("Puntos por nube")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de puntos — {name}")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"pts_{safe_name}.png"), dpi=120)
        plt.close()

    print(f"✅ Histogramas guardados en {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Estadísticas del dataset V2X")
    parser.add_argument("--dataset_root", type=str,
                        default="../dataset_labeled",
                        help="Ruta raíz del dataset (default: ../dataset_labeled)")
    parser.add_argument("--out_dir", type=str,
                        default="./outputs/dataset_stats",
                        help="Directorio de salida para JSON y CSVs")
    parser.add_argument("--plots", action="store_true",
                        help="Generar histogramas con matplotlib")
    parser.add_argument("--max_sample", type=int, default=200,
                        help="Máximo de nubes a leer por clase para stats de puntos")
    args = parser.parse_args()

    # Escaneo
    scan = scan_dataset(args.dataset_root, max_sample_per_class=args.max_sample)
    cls_data = scan["classes"]
    global_info = scan["global"]

    # Tabla por clase fina
    class_rows = build_class_rows(cls_data)
    df_class   = pd.DataFrame(class_rows)

    # Tabla por superclase
    sc_rows  = build_superclass_rows_from_cls(cls_data)
    df_sc    = pd.DataFrame(sc_rows)

    # Estadísticas globales compactas para el JSON
    all_pt_counts = []
    for d in cls_data.values():
        all_pt_counts.extend(d["point_counts"])

    hour_dist = defaultdict(int)
    for h in global_info["hours"]:
        hour_dist[h] += 1

    stats_json = {
        "global": {
            "total_frames":        global_info["total_frames"],
            "total_corrupted":     global_info["total_corrupted"],
            "n_gantries":          len(global_info["gantries"]),
            "gantry_frame_counts": global_info["gantries"],
            "date_range":          {
                "first": global_info["dates"][0]  if global_info["dates"] else None,
                "last":  global_info["dates"][-1] if global_info["dates"] else None,
                "n_days": len(global_info["dates"]),
            },
            "hour_distribution": dict(sorted(hour_dist.items())),
            "point_counts_global": stats_1d(all_pt_counts),
        },
        "per_superclass": {
            str(row["superclass_id"]): {
                "name":             row["superclass_name"],
                "frames":           row["frames"],
                "unique_vehicles":  row["unique_vehicles"],
                "frames_per_vehicle": row["frames_per_vehicle"],
                "capture_days":     row["capture_days"],
                "point_count_stats": {
                    "mean":   row["pts_mean"],
                    "std":    row["pts_std"],
                    "median": row["pts_median"],
                    "min":    row["pts_min"],
                    "max":    row["pts_max"],
                },
            }
            for row in sc_rows
        },
        "per_class": {
            str(row["class_id"]): {
                "name":             row["class_name"],
                "superclass_id":    row["superclass_id"],
                "frames":           row["frames"],
                "unique_vehicles":  row["unique_vehicles"],
                "frames_per_vehicle": row["frames_per_vehicle"],
                "capture_days":     row["capture_days"],
                "gantries":         json.loads(row["gantries"]),
                "point_count_stats": {
                    "mean":   row["pts_mean"],
                    "std":    row["pts_std"],
                    "median": row["pts_median"],
                    "min":    row["pts_min"],
                    "max":    row["pts_max"],
                },
            }
            for row in class_rows
        },
    }

    # Guardar salidas
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "dataset_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"✅ Stats JSON guardado en {json_path}")

    sc_csv_path = os.path.join(args.out_dir, "tabla_por_superclase.csv")
    df_sc.to_csv(sc_csv_path, index=False)
    print(f"✅ Tabla por superclase guardada en {sc_csv_path}")

    cls_csv_path = os.path.join(args.out_dir, "tabla_por_clase_fina.csv")
    df_class.to_csv(cls_csv_path, index=False)
    print(f"✅ Tabla por clase fina guardada en {cls_csv_path}")

    # Resumen rápido en terminal
    print("\n" + "=" * 62)
    print(f"DATASET STATS — {global_info['total_frames']:,} frames totales")
    print(f"Gantries: {len(global_info['gantries'])} | "
          f"Días: {len(global_info['dates'])} | "
          f"Clases: {sum(1 for r in class_rows if r['frames'] > 0)}")
    print("=" * 62)
    print(f"{'Superclase':<22} {'Frames':>8} {'Vehículos':>10} {'Días':>5}")
    print("-" * 62)
    for row in sc_rows:
        print(f"{row['superclass_name']:<22} {row['frames']:>8,} "
              f"{row['unique_vehicles']:>10,} {row['capture_days']:>5}")
    print("=" * 62)

    if args.plots:
        plot_dir = "./visualizations/dataset"
        make_plots(cls_data, plot_dir)


if __name__ == "__main__":
    main()
