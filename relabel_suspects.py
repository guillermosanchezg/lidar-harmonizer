"""
Mueve todos los PCD de los vehículos sospechosos de error de etiquetado
a la carpeta de clase correcta, según la predicción del modelo kNN.

Genera un log CSV completo para hacer rollback si fuera necesario.
Borra split.csv para forzar regeneración en el próximo entrenamiento.

Uso:
    python relabel_suspects.py [--model MODEL_NAME] [--rule majority] [--dry-run]
"""
import os
import shutil
import csv
import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Mapeo: "TrueSuper → PredSuper" → clase fina de destino
# ---------------------------------------------------------------------------
RELABEL_MAP = {
    "Bus → Articulated Truck":      9,   # bus (4) → camión articulado
    "Bus → Rigid Truck":            3,   # mini bus (5) → camión rígido
    "Bus → Van":                    13,  # mini bus (5) → L3H3
    "Other Vehicles → Rigid Truck": 3,   # caravana/grúa/cabeza tractora → camión rígido
    "Passenger Car → Rigid Truck":  3,   # turismo → camión rígido
    "Passenger Car → Van":          13,  # turismo → L3H3
    "Rigid Truck → Bus":            4,   # camión rígido → bus
    "Rigid Truck → Other Vehicles": 15,  # camión de obra → grúa/otros
    "Rigid Truck → Passenger Car":  2,   # camión rígido/hormigonera → turismo
    "Van → Bus":                    5,   # L3H3 → mini bus
    "Van → Passenger Car":          2,   # L3H3 → turismo
    "Van → Rigid Truck":            3,   # L3H3 → camión rígido
}

DATASET_ROOT = Path("../dataset_labeled")
SPLIT_CSV    = Path("data/splits/split.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet2_1024d_a0.5_sc_20260613-120231.pth")
    parser.add_argument("--rule",  default="majority")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra lo que haría sin mover nada")
    args = parser.parse_args()

    report_path = (f"outputs/knn_results/"
                   f"miss_report_vehicles_{args.model}_{args.rule}.csv")

    df = pd.read_csv(report_path)
    suspects = df[df["label_error_suspect"]].copy()
    print(f"Sospechosos a re-etiquetar: {len(suspects)}")
    if args.dry_run:
        print("[DRY-RUN] No se mueve nada.\n")

    log_rows   = []
    moved_files = 0
    moved_vehs  = 0
    skipped     = 0

    for _, row in suspects.iterrows():
        confusion  = row["confusion"]
        target_cls = RELABEL_MAP.get(confusion)
        if target_cls is None:
            print(f"[SKIP] Sin mapeo para: '{confusion}'")
            skipped += 1
            continue

        # Parsear sample_path_1 para extraer clase actual, date, hour, prefix
        sample = Path(str(row["sample_path_1"]))
        parts  = sample.parts
        try:
            dl_idx = next(i for i, p in enumerate(parts) if p == "dataset_labeled")
        except StopIteration:
            print(f"[SKIP] No se encuentra 'dataset_labeled' en: {sample}")
            skipped += 1
            continue

        src_cls  = parts[dl_idx + 2]
        date     = parts[dl_idx + 3]
        hour     = parts[dl_idx + 4]
        filename = parts[-1]
        prefix   = filename.split("_")[0]

        if str(src_cls) == str(target_cls):
            skipped += 1
            continue

        # Todos los gantries en los que se vio este vehículo
        gantries = [g.strip() for g in str(row["gantries"]).split("|")]

        veh_moved = 0
        for gantry in gantries:
            src_dir = DATASET_ROOT / gantry / src_cls / date / hour
            dst_dir = DATASET_ROOT / gantry / str(target_cls) / date / hour

            if not src_dir.exists():
                continue

            pcd_files = sorted(src_dir.glob(f"{prefix}_*.pcd"))
            if not pcd_files:
                continue

            if not args.dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)

            for pcd in pcd_files:
                dst_path = dst_dir / pcd.name
                if args.dry_run:
                    print(f"  MOVE {pcd} → {dst_path}")
                else:
                    shutil.move(str(pcd), str(dst_path))
                log_rows.append({
                    "event_id":  row["event_id"],
                    "confusion": confusion,
                    "src_cls":   src_cls,
                    "dst_cls":   str(target_cls),
                    "gantry":    gantry,
                    "date":      date,
                    "hour":      hour,
                    "file":      pcd.name,
                    "src_path":  str(pcd),
                    "dst_path":  str(dst_path),
                })
                moved_files += 1
                veh_moved   += 1

        if veh_moved > 0:
            moved_vehs += 1

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Resultado:")
    print(f"  Vehículos re-etiquetados : {moved_vehs}")
    print(f"  Archivos PCD movidos     : {moved_files}")
    print(f"  Saltados                 : {skipped}")

    if log_rows:
        log_path = f"outputs/relabel_log_{args.model}_{args.rule}.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"  Log de rollback         : {log_path}")

    # Borrar split.csv para forzar regeneración con clases corregidas
    if not args.dry_run and SPLIT_CSV.exists():
        SPLIT_CSV.unlink()
        print(f"  Split eliminado         : {SPLIT_CSV}  (se regenerará en el próximo entrenamiento)")


if __name__ == "__main__":
    main()
