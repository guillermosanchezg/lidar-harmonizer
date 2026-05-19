import argparse
import torch
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import yaml
import os
from tqdm import tqdm
from utils import generate_embeddings, load_point_cloud, normalize_point_cloud, uniform_points, iterate_dataset
from pointnet import PointNetClassifier
from torch.utils.data import Dataset

# Cargar hiperparámetros desde params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]

# Configuración del dispositivo (GPU si está disponible)
device = torch.device(DEVICE_CONFIG["device"] if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Dataset personalizado con normalización y uniformización restauradas
class PointCloudDataset(Dataset):
    def __init__(self, data, labels, num_points=MODEL_HYPERPARAMETERS["num_points"]):
        self.data = data
        self.labels = labels
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        label = self.labels[idx]

        point_cloud = normalize_point_cloud(point_cloud)
        point_cloud = uniform_points(point_cloud, self.num_points)

        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(int(label), dtype=torch.long)

# Definir función principal
def main():
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Cálculo de k vecinos cercanos")
    parser.add_argument("model_name", type=str, help="Nombre del modelo preentrenado (archivo en ./models)")
    parser.add_argument("test_set", type=str, help="Archivo CSV con el conjunto de prueba")
    parser.add_argument("--k", type=int, default=5, help="Número de vecinos más cercanos a considerar")
    args = parser.parse_args()

    # Construir rutas
    model_path = os.path.join('./models', args.model_name)
    dataset_path = "../dataset_labeled"
    test_csv_path = args.test_set
    results_path = f"./results/k_neighbors_{args.model_name}_test.csv"

    # Cargar modelo preentrenado
    print(f"Cargando modelo desde {model_path}...")
    model = PointNetClassifier(
        num_classes=MODEL_HYPERPARAMETERS["num_classes"],
        feature_vector_size=MODEL_HYPERPARAMETERS["feature_vector_size"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Cargar dataset completo
    print("Cargando datos del dataset completo...")
    data, labels, paths, _ = iterate_dataset(dataset_path)

    # Cargar conjunto de prueba desde CSV
    print(f"Cargando conjunto de prueba desde {test_csv_path}...")
    test_data_df = pd.read_csv(test_csv_path)
    test_data_df.columns = test_data_df.columns.str.strip()  # ✅ Elimina espacios extra
    test_paths = test_data_df["Path"].tolist()

    test_labels = test_data_df["Original Label"].tolist()

    # Crear datasets y generar embeddings con el preprocesamiento correcto
    print("Generando embeddings para el conjunto de prueba...")
    test_data = [load_point_cloud(path) for path in test_paths]
    test_dataset = PointCloudDataset(test_data, test_labels)
    test_embeddings_tensor = torch.zeros((len(test_dataset), MODEL_HYPERPARAMETERS["feature_vector_size"]), device=device)

    with torch.no_grad():
        for i, (cloud, _) in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Procesando embeddings de prueba"):
            cloud = cloud.unsqueeze(0).to(device)  # [1, num_puntos, 3]
            test_embeddings_tensor[i] = model.feature_extractor(cloud.transpose(2, 1))  # Extraer embedding

    print("Generando embeddings para el dataset completo...")
    full_dataset = PointCloudDataset(data, labels)
    embeddings_tensor = torch.zeros((len(full_dataset), MODEL_HYPERPARAMETERS["feature_vector_size"]), device=device)

    with torch.no_grad():
        for i, (cloud, _) in tqdm(enumerate(full_dataset), total=len(full_dataset), desc="Procesando embeddings del dataset"):
            cloud = cloud.unsqueeze(0).to(device)  # [1, num_puntos, 3]
            embeddings_tensor[i] = model.feature_extractor(cloud.transpose(2, 1))  # Extraer embedding

    print("Cálculo de embeddings finalizado.")

    # Calcular vecinos más cercanos en GPU con torch.cdist
    print("Calculando vecinos más cercanos con torch.cdist en GPU...")
    
    num_test_embeddings = test_embeddings_tensor.shape[0]
    neighbors = np.zeros((num_test_embeddings, args.k), dtype=np.int32)
    distances_list = []

    for i in tqdm(range(0, num_test_embeddings, 512), desc="Procesando en GPU"):
        batch_indices = range(i, min(i + 512, num_test_embeddings))
        batch_embeddings = test_embeddings_tensor[batch_indices]  # [batch_size, feature_dim]

        batch_distances = torch.cdist(batch_embeddings, embeddings_tensor, p=2)  # Euclidean distance

        # Ordenar y obtener los K vecinos más cercanos
        sorted_indices = torch.argsort(batch_distances, dim=1)[:, :args.k]  # NO excluir la instancia misma

        # Mover a CPU inmediatamente después del cálculo
        neighbors[batch_indices] = sorted_indices.cpu().numpy()
        distances_list.extend(batch_distances.cpu().numpy())  # Guardamos distancias en una lista

    print("Cálculo de vecinos finalizado.")

    # Guardar vecinos más cercanos en CSV
    print("Guardando vecinos más cercanos...")
    rows = []

    for i in tqdm(range(num_test_embeddings), desc="Procesando para CSV"):
        query_path = test_paths[i]
        query_label = test_labels[i]
        neighbor_indices = neighbors[i]
        neighbor_paths = [paths[n] for n in neighbor_indices]
        neighbor_labels = [int(labels[n]) for n in neighbor_indices]
        neighbor_distances = distances_list[i][neighbor_indices].tolist()  # Obtener distancias correspondientes

        rows.append({
            "Query": query_path,
            "Original Label": query_label,
            "Neighbor Labels": neighbor_labels,
            "Distances": neighbor_distances,
            "Neighbors": neighbor_paths
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False, columns=["Query", "Original Label", "Neighbor Labels", "Distances", "Neighbors"])

    print(f"Resultados guardados en {results_path}")

# Ejecutar si el script es llamado directamente
if __name__ == "__main__":
    main()
