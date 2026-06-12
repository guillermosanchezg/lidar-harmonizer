import os
import numpy as np
import open3d as o3d
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import random
import json
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader  # Import necesario para DataLoader
import random
from collections import defaultdict

def iterate_dataset(dataset_path):
    """
    Itera sobre un dataset organizado en un árbol de directorios para cargar nubes de puntos, etiquetas, rutas y distancias.

    Args:
        dataset_path (str): Ruta al directorio raíz del dataset.

    Returns:
        tuple: Cuatro listas: nubes de puntos (data), etiquetas (labels), rutas de los archivos (paths) y distancias (gantry_distances).
    """
    data = []
    labels = []
    paths = []
    gantry_distances = []

    for gantry_dir in os.listdir(dataset_path):  # Iterar por gantry_distance
        gantry_path = os.path.join(dataset_path, gantry_dir)
        if os.path.isdir(gantry_path):
            for class_dir in os.listdir(gantry_path):  # Iterar por clases
                class_path = os.path.join(gantry_path, class_dir)
                if os.path.isdir(class_path):
                    for date_dir in os.listdir(class_path):  # Iterar por fechas
                        date_path = os.path.join(class_path, date_dir)
                        if os.path.isdir(date_path):
                            for hour_dir in os.listdir(date_path):  # Iterar por horas
                                hour_path = os.path.join(date_path, hour_dir)
                                if os.path.isdir(hour_path):
                                    for file in os.listdir(hour_path):  # Iterar por archivos
                                        if file.endswith(".pcd"):
                                            file_path = os.path.join(hour_path, file)

                                            # La etiqueta está en el nombre del directorio "class_dir"
                                            label = class_dir

                                            # La distancia está en el nivel superior a la clase
                                            gantry_distance = os.path.basename(gantry_path)

                                            # Cargar la nube de puntos
                                            points = load_point_cloud(file_path)

                                            # Almacenar la nube, etiqueta, ruta y distancia
                                            data.append(points)
                                            labels.append(label)
                                            paths.append(file_path)
                                            gantry_distances.append(gantry_distance)

    return data, labels, paths, gantry_distances

def load_point_cloud(file_path):
    """
    Carga una nube de puntos desde un archivo.

    Args:
        file_path (str): Ruta al archivo de la nube de puntos.

    Returns:
        np.ndarray: Nube de puntos cargada como un array NumPy.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    try:
        point_cloud = o3d.io.read_point_cloud(file_path)
        return np.asarray(point_cloud.points)
    except Exception as e:
        raise ValueError(f"Error al cargar la nube de puntos: {e}")


def save_point_cloud(file_path, points):
    """
    Guarda una nube de puntos en un archivo en formato PCD.

    Args:
        file_path (str): Ruta donde se guardará el archivo.
        points (np.ndarray): Nube de puntos a guardar.
    """
    try:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(file_path, point_cloud)
    except Exception as e:
        raise ValueError(f"Error al guardar la nube de puntos: {e}")


def normalize_point_cloud(points):
    """
    Normaliza una nube de puntos centrando y escalando.

    Args:
        points (np.ndarray): Nube de puntos original.

    Returns:
        np.ndarray: Nube de puntos normalizada.
    """
    center = np.mean(points, axis=0)
    points -= center
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance
    return points


def uniform_points(points, num_points):
    """
    Uniformiza el número de puntos en una nube de puntos.

    Args:
        points (np.ndarray): Nube de puntos original.
        num_points (int): Número deseado de puntos.

    Returns:
        np.ndarray: Nube de puntos ajustada al tamaño deseado.
    """
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        return points[indices]
    elif len(points) < num_points:
        padding = np.zeros((num_points - len(points), points.shape[1]))
        return np.vstack((points, padding))
    return points

def get_event_id(file_path):
    """
    Extrae un identificador único del evento a partir de la ruta del archivo.
    Asume la estructura: .../<gantry>/<class>/<date>/<hour>/<file.pcd>
    EXCLUYE 'gantry' para que todas las distancias del mismo vehículo 
    se agrupen en un único evento inseparable.
    """
    # Normalizar separadores
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Extraer las partes relevantes de la ruta (ignorar path_parts[-5] que es gantry)
    if len(path_parts) >= 5:
        cls = path_parts[-4]
        date = path_parts[-3]
        hour = path_parts[-2]
        # El ID único del evento es la combinación de clase, día y hora
        return f"{cls}_{date}_{hour}"
    else:
        # Fallback de seguridad si hay algún archivo inesperado
        return os.path.basename(os.path.dirname(file_path))

def balance_dataset(data, labels, paths, gantry_distances, max_instances=1000, percentage=0.7):
    """
    Balancea el dataset dividiendo en Base de Conocimiento (Train) y Consulta (Unseen).
    Agrupa los archivos por evento único de paso (agrupando todas las distancias gantry) 
    para evitar Data Leakage temporal.
    Garantiza estrictamente no superar el límite de max_instances.
    """
    print(f"\nAplicando balanceo: Max {max_instances} inst. (Estricto) | Ratio temporal {percentage*100}%")
    
    # Agrupar por clase y luego por identificador de evento
    grouped = defaultdict(lambda: defaultdict(list))
    
    for i, path in enumerate(paths):
        label = labels[i]
        event_id = get_event_id(path)
        grouped[label][event_id].append(i)

    bal_indices = []
    unseen_indices = []

    print("\n--- Distribución de Datos tras el Split Temporal ---")
    for class_id, time_folders in grouped.items():
        folders = list(time_folders.keys())
        random.shuffle(folders) # Barajar eventos

        total_pcds = sum(len(idxs) for idxs in time_folders.values())
        # El objetivo es el límite de porcentaje, sin pasarse NUNCA del max_instances
        target_train_size = min(max_instances, int(total_pcds * percentage))
        
        current_train = 0
        train_folders = set()

        # Rellenar Entrenamiento evaluando trayectoria completa (evento) por evento
        for folder in folders:
            folder_pcds = len(time_folders[folder])
            
            # Si el evento entero (todas las distancias) cabe sin pasarnos del objetivo, lo metemos
            if current_train + folder_pcds <= target_train_size:
                train_folders.add(folder)
                current_train += folder_pcds
                bal_indices.extend(time_folders[folder])
        
        # Todo lo que no entró al Entrenamiento, va a Consulta (Unseen)
        unseen_count = 0
        for folder in folders:
            if folder not in train_folders:
                unseen_count += len(time_folders[folder])
                unseen_indices.extend(time_folders[folder])

        print(f"Clase {class_id}: Total={total_pcds} | Train (Balanced)={current_train} | Unseen={unseen_count}")

    # Reconstruir listas para Knowledge/Train
    bal_data = [data[i] for i in bal_indices]
    bal_labels = [labels[i] for i in bal_indices]
    bal_paths = [paths[i] for i in bal_indices]
    bal_gantries = [gantry_distances[i] for i in bal_indices]

    # Reconstruir listas para Unseen/Consulta
    unseen_data = [data[i] for i in unseen_indices]
    unseen_labels = [labels[i] for i in unseen_indices]
    unseen_paths = [paths[i] for i in unseen_indices]
    unseen_gantries = [gantry_distances[i] for i in unseen_indices]

    print("-" * 50)
    print(f"-> Base de Conocimiento (Train): {len(bal_indices)} nubes")
    print(f"-> Set de Consulta (Unseen):     {len(unseen_indices)} nubes")
    print("-" * 50)

    return (bal_data, bal_labels, bal_paths, bal_gantries), (unseen_data, unseen_labels, unseen_paths, unseen_gantries)

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir, label_mapping=None):
    """
    Genera y guarda una matriz de confusión con etiquetas en inglés.
    """
    if label_mapping:
        idx_to_label = {v: k for k, v in label_mapping.items()}
        sorted_classes = sorted(classes, key=lambda idx: int(idx))
        display_labels = [idx_to_label[idx] if idx in idx_to_label else str(idx) for idx in sorted_classes]
    else:
        sorted_classes = sorted(classes, key=lambda x: int(x))
        display_labels = [str(c) for c in sorted_classes]

    cm = confusion_matrix(y_true, y_pred, labels=sorted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{model_name}.png")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Matriz de confusión guardada en {save_path}")
    plt.close()


def load_config(file_path):
    """
    Carga una configuración desde un archivo YAML.

    Args:
        file_path (str): Ruta al archivo YAML.

    Returns:
        dict: Configuración cargada.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error al cargar la configuración YAML: {e}")


def sample_random_points(points, num_samples):
    """
    Genera un subconjunto aleatorio de puntos.

    Args:
        points (np.ndarray): Nube de puntos original.
        num_samples (int): Número de puntos a seleccionar.

    Returns:
        np.ndarray: Subconjunto aleatorio de puntos.
    """
    if len(points) <= num_samples:
        return points
    indices = np.random.choice(len(points), num_samples, replace=False)
    return points[indices]


def compute_point_cloud_stats(points):
    """
    Calcula estadísticas básicas de una nube de puntos.

    Args:
        points (np.ndarray): Nube de puntos.

    Returns:
        dict: Estadísticas incluyendo centro, límites mínimos y máximos.
    """
    center = np.mean(points, axis=0)
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    return {
        "center": center,
        "min_bounds": min_bounds,
        "max_bounds": max_bounds,
    }


def convert_lidar_to_cartesian(ranges, angles):
    """
    Convierte coordenadas polares de un LIDAR a cartesianas.

    Args:
        ranges (np.ndarray): Distancias medidas por el LIDAR.
        angles (np.ndarray): Ángulos de las mediciones.

    Returns:
        np.ndarray: Coordenadas cartesianas.
    """
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.vstack((x, y)).T


def split_point_cloud_by_class(points, labels):
    """
    Divide una nube de puntos por clases o etiquetas.

    Args:
        points (np.ndarray): Nube de puntos.
        labels (np.ndarray): Etiquetas correspondientes.

    Returns:
        dict: Diccionario donde las claves son clases y los valores son nubes de puntos.
    """
    classes = np.unique(labels)
    split_points = {class_id: points[labels == class_id] for class_id in classes}
    return split_points

def save_training_statistics(stats, file_path):
    """
    Guarda las estadísticas del entrenamiento en un archivo JSON.

    Args:
        stats (dict): Diccionario con estadísticas del entrenamiento (pérdida, precisión, etc.).
        file_path (str): Ruta del archivo donde se guardarán las estadísticas.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Estadísticas guardadas en {file_path}")
    except Exception as e:
        raise ValueError(f"Error al guardar las estadísticas: {e}")

def generate_embeddings(model, dataset, device, batch_size=64):
    """
    Genera embeddings para todo el dataset usando el modelo entrenado.

    Args:
        model (torch.nn.Module): Modelo PointNet preentrenado.
        dataset (Dataset): Dataset con datos y etiquetas.
        device (torch.device): Dispositivo de computación (CPU/GPU).
        batch_size (int): Tamaño del batch.

    Returns:
        embeddings (np.array): Embeddings generados.
        
        labels (list): Etiquetas correspondientes.
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for points, label in dataloader:
            points = points.to(device)
            features = model.extract_features(points.transpose(2, 1))  # Extraer embeddings
            embeddings.append(features.cpu().numpy())
            labels.extend(label.numpy())

    embeddings = np.concatenate(embeddings, axis=0)  # Combinar todos los embeddings
    return embeddings, labels

# Aplicar T-SNE
def apply_tsne(embeddings, labels, perplexity=30):
    """
    Aplica T-SNE a los embeddings generados.

    Args:
        embeddings (np.array): Embeddings generados.
        labels (list): Etiquetas correspondientes.
        perplexity (int): Parámetro de T-SNE (controla el balance local/global).

    Returns:
        tsne_result (np.array): Embeddings en 2D.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    return tsne_result

def plot_tsne(tsne_result, labels, unique_labels, title="T-SNE Visualization", save_path=None):
    """
    Grafica los resultados de T-SNE con diferentes colores y formas para cada clase.
    Si se repiten colores, se asigna una nueva forma.

    Args:
        tsne_result (np.array): Coordenadas 2D de T-SNE.
        labels (list): Etiquetas correspondientes.
        unique_labels (list): Etiquetas únicas en el dataset.
        title (str): Título del gráfico.
        save_path (str): Ruta para guardar el gráfico (opcional).
    """
    plt.figure(figsize=(10, 8))

    # Definir colores y formas
    markers = ['^','x', 'o']
    colors = plt.cm.tab10.colors

    # Controlar asignación de colores y formas
    assigned_colors = {}
    assigned_markers = {}
    for idx, label in enumerate(unique_labels):
        color_idx = idx % len(colors)
        marker_idx = idx // len(colors)  # Cambiar forma si el color se repite
        assigned_colors[label] = colors[color_idx]
        assigned_markers[label] = markers[marker_idx % len(markers)]

    # Graficar cada clase con color y forma únicos
    for label in unique_labels:
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            label=f"Clase {label}",
            s=10,
            color=assigned_colors[label],
            marker=assigned_markers[label]
        )

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Imagen guardada en {save_path}")
    else:
        plt.show()
    plt.show()


def limit_instances_per_class(data, labels, max_instances):
    """
    Limita el número de instancias por clase en el dataset.

    Args:
        data (list): Lista de nubes de puntos.
        labels (list): Lista de etiquetas correspondientes.
        max_instances (int): Número máximo de instancias por clase.

    Returns:
        tuple: Datos y etiquetas limitados por clase.
    """
    class_instances = defaultdict(list)
    for point_cloud, label in zip(data, labels):
        class_instances[label].append(point_cloud)

    limited_data = []
    limited_labels = []
    for label, instances in class_instances.items():
        if len(instances) > max_instances:
            instances = random.sample(instances, max_instances)
        limited_data.extend(instances)
        limited_labels.extend([label] * len(instances))

    return limited_data, limited_labels

