import argparse
import os
import torch
import open3d as o3d
import numpy as np
from wpce import WPCE
from utils import iterate_dataset, preprocess_point_cloud

def load_model(model_path, input_dim, embedding_dim, num_points):
    """
    Carga un modelo WPCE desde un archivo.
    Args:
        model_path (str): Ruta al archivo del modelo.
        input_dim (int): Dimensiones de entrada.
        embedding_dim (int): Dimensión del embedding.
        num_points (int): Número de puntos en la reconstrucción.
    Returns:
        WPCE: El modelo cargado.
    """
    model = WPCE(input_dim=input_dim, embedding_dim=embedding_dim, num_points=num_points).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict, strict=False)
        print("Modelo cargado con éxito.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise
    model.eval()
    return model

def save_reconstructions(original_cloud, reconstructed_cloud, index, output_dir="reconstructions"):
    """
    Guarda las nubes de puntos originales y reconstruidas como archivos PLY.

    Args:
        original_cloud (np.ndarray): Nube de puntos original.
        reconstructed_cloud (np.ndarray): Nube de puntos reconstruida.
        index (int): Índice de la nube para nombrar los archivos.
        output_dir (str): Directorio donde se guardarán los archivos.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_cloud)
    o3d.io.write_point_cloud(os.path.join(output_dir, f"original_{index}.ply"), original_pcd)

    reconstructed_pcd = o3d.geometry.PointCloud()
    reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed_cloud)
    o3d.io.write_point_cloud(os.path.join(output_dir, f"reconstructed_{index}.ply"), reconstructed_pcd)

def visualize_reconstruction(model, dataset_path, num_points=1500, exclude_classes=None, max_visualizations=10):
    """
    Visualiza las reconstrucciones de un modelo WPCE y guarda las nubes reconstruidas.

    Args:
        model (torch.nn.Module): Modelo WPCE entrenado.
        dataset_path (str): Ruta al dataset etiquetado.
        num_points (int): Número de puntos por nube después de preprocesar.
        exclude_classes (list): Clases a excluir del dataset.
        max_visualizations (int): Número máximo de visualizaciones a guardar.
    """
    exclude_classes = exclude_classes or []

    # Cargar dataset
    print("Cargando datos del dataset...")
    data, labels = iterate_dataset(dataset_path)

    # Filtrar las clases excluidas
    filtered_data = [(cloud, label) for cloud, label in zip(data, labels) if int(label) not in exclude_classes]

    # Seleccionar aleatoriamente hasta un máximo de visualizaciones por clase
    class_groups = {}
    for cloud, label in filtered_data:
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(cloud)

    selected_data = []
    for label, clouds in class_groups.items():
        selected_data.extend(clouds[:max_visualizations])

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, cloud in enumerate(selected_data):
            # Preprocesar la nube de puntos
            cloud = preprocess_point_cloud(cloud, num_points)
            cloud_tensor = torch.tensor(cloud, dtype=torch.float32).unsqueeze(0).to(device)

            # Reconstrucción
            _, reconstructed = model(cloud_tensor)
            reconstructed = reconstructed.cpu().squeeze(0).numpy()

            # Guardar reconstrucciones
            save_reconstructions(cloud, reconstructed, i)

    print(f"Reconstrucciones guardadas en el directorio 'reconstructions'.")

def main():
    parser = argparse.ArgumentParser(description="Visualización aleatoria de reconstrucciones usando WPCE.")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al modelo WPCE entrenado.")
    parser.add_argument("--dataset_path", type=str, default="../dataset_labeled", help="Ruta al dataset etiquetado.")
    parser.add_argument("--input_dim", type=int, default=3, help="Dimensiones de entrada.")
    parser.add_argument("--embedding_dim", type=int, default=16, help="Dimensión del embedding.")
    parser.add_argument("--num_points", type=int, default=500, help="Número de puntos en las reconstrucciones.")
    parser.add_argument("--exclude_classes", type=int, nargs='*', default=[0, 1, 2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19], help="Clases a excluir del dataset.")
    parser.add_argument("--max_visualizations", type=int, default=10, help="Número máximo de visualizaciones por clase.")

    args = parser.parse_args()

    # Cargar el modelo
    model = load_model(args.model_path, args.input_dim, args.embedding_dim, args.num_points)

    # Visualizar reconstrucciones
    visualize_reconstruction(model, args.dataset_path, args.num_points, args.exclude_classes, args.max_visualizations)

if __name__ == "__main__":
    main()

