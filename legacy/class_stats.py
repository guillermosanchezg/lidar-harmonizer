import os
import csv
import numpy as np
import open3d as o3d
from utils import iterate_dataset

def count_points_and_save_by_gantry_and_class(dataset_path, output_dir):
    """
    Cuenta el número de puntos por cada nube de puntos y guarda resultados en CSVs separados por gantry_dir y clase.

    Args:
        dataset_path (str): Ruta al directorio del dataset.
        output_dir (str): Directorio donde se guardarán los archivos CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    overall_results = []

    # Iterar sobre el dataset
    for gantry_dir in os.listdir(dataset_path):
        gantry_path = os.path.join(dataset_path, gantry_dir)
        if os.path.isdir(gantry_path):
            for class_dir in os.listdir(gantry_path):
                class_path = os.path.join(gantry_path, class_dir)
                if os.path.isdir(class_path):
                    results = []
                    for date_dir in os.listdir(class_path):
                        date_path = os.path.join(class_path, date_dir)
                        if os.path.isdir(date_path):
                            for hour_dir in os.listdir(date_path):
                                hour_path = os.path.join(date_path, hour_dir)
                                if os.path.isdir(hour_path):
                                    for file in os.listdir(hour_path):
                                        if file.endswith(".pcd"):
                                            file_path = os.path.join(hour_path, file)

                                            # Cargar la nube de puntos usando Open3D
                                            try:
                                                pcd = o3d.io.read_point_cloud(file_path)
                                                points = np.asarray(pcd.points)
                                                num_points = points.shape[0]

                                                # Guardar resultados locales y generales
                                                results.append({
                                                    "file": file,
                                                    "num_points": num_points
                                                })
                                                overall_results.append(num_points)
                                            except Exception as e:
                                                print(f"Error al procesar {file_path}: {e}")

                    # Guardar CSV por gantry_dir y clase
                    if results:
                        output_file = os.path.join(output_dir, f"{gantry_dir}_{class_dir}_points.csv")
                        with open(output_file, mode="w", newline="") as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=["file", "num_points"])
                            writer.writeheader()
                            writer.writerows(results)
                        print(f"Resultados guardados en {output_file}")

    # Calcular y guardar la media total
    if overall_results:
        mean_points = np.mean(overall_results)
        with open(os.path.join(output_dir, "overall_mean_points.csv"), mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Mean_Total_Points"])
            writer.writerow([mean_points])
        print(f"Media total de puntos guardada en overall_mean_points.csv: {mean_points:.2f}")

if __name__ == "__main__":
    DATASET_PATH = "../dataset_labeled"
    OUTPUT_DIR = "point_counts_by_gantry_and_class"

    print("Contando puntos por nube de puntos...")
    count_points_and_save_by_gantry_and_class(DATASET_PATH, OUTPUT_DIR)
    print("Proceso completado.")

