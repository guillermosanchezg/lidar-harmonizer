import argparse
import os
import torch
from utils import iterate_dataset, generate_embeddings, apply_tsne, plot_tsne, normalize_point_cloud, uniform_points, limit_instances_per_class
from pointnet import PointNetClassifier
import yaml
from torch.utils.data import Dataset

# Cargar hiperparámetros desde params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
TRAINING_HYPERPARAMETERS = params["TRAINING_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]

# Configuración del dispositivo
device = torch.device(DEVICE_CONFIG["device"])

# Dataset personalizado
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

        # Normalizar y uniformizar el número de puntos
        point_cloud = normalize_point_cloud(point_cloud)
        point_cloud = uniform_points(point_cloud, self.num_points)

        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_model_params(model_name):
    """
    Carga los parámetros del modelo desde un archivo .pth.
    """
    checkpoint = torch.load(model_name, map_location=device)
    feature_vector_size = checkpoint['feature_extractor.conv4.weight'].shape[0]
    num_classes = checkpoint['fc3.weight'].shape[0]
    return feature_vector_size, num_classes

if __name__ == "__main__":
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Generar T-SNE con embeddings de PointNet preentrenado")
    parser.add_argument("--model_name", type=str, help="Nombre del modelo preentrenado (archivo en ./models)")
    args = parser.parse_args()

    model_name = os.path.join('./models', args.model_name)

    # Cargar parámetros del modelo
    print(f"Cargando modelo desde {os.path.join('./models', args.model_name)}...")
    feature_vector_size, num_classes = load_model_params(model_name)
    print(f"Parámetros del modelo: feature_vector_size={feature_vector_size}, num_classes={num_classes}")

    # Cargar datos desde el dataset
    dataset_path = "../dataset_labeled"  # Ruta al dataset
    print("Cargando datos...")
    data, labels, paths, gantry_distances = iterate_dataset(dataset_path)

    # Filtrar las clases 0 y 1
    filtered_data = []
    filtered_labels = []
    for point_cloud, label in zip(data, labels):
        if label not in ["0", "1", "12", "16"]:  # Excluir clases 0 y 1
            filtered_data.append(point_cloud)
            filtered_labels.append(int(label))  # Convertir etiqueta a entero

    # Limitar las instancias a 4000 por clase
    max_instances = 1000
    filtered_data, filtered_labels = limit_instances_per_class(filtered_data, filtered_labels, max_instances)
    print(f"Datos limitados a {max_instances} instancias por clase.")

    # Obtener batch_size desde params.yaml
    batch_size = TRAINING_HYPERPARAMETERS["batch_size"]

    # Cargar modelo preentrenado
    model = PointNetClassifier(
        num_classes=num_classes,  # Número de clases del modelo
        feature_vector_size=feature_vector_size
    ).to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    print(f"Modelo preentrenado cargado desde {model_name}")

    # Generar embeddings
    full_dataset = PointCloudDataset(filtered_data, filtered_labels)  # Dataset filtrado
    embeddings, labels = generate_embeddings(model, full_dataset, device, batch_size=batch_size)
    print(f"Embeddings generados: {embeddings.shape}")

    # Aplicar T-SNE
    tsne_result = apply_tsne(embeddings, labels)
    print("T-SNE aplicado.")

    # Guardar la visualización
    unique_labels = sorted(set(labels))
    os.makedirs('./tsne', exist_ok=True)
    tsne_output_path = os.path.join('./tsne', os.path.basename(args.model_name).replace('.pth', '.png'))
    plot_tsne(
        tsne_result,
        labels,
        unique_labels,
        title=f"T-SNE Visualization\nModel: {args.model_name}\nFeature Vector: {feature_vector_size}\nClasses: {num_classes}",
        save_path=tsne_output_path
    )
    print(f"Visualización T-SNE guardada en {tsne_output_path}")
