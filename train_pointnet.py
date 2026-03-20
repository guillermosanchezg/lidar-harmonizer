import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import iterate_dataset, normalize_point_cloud, uniform_points, balance_dataset, plot_confusion_matrix, save_training_statistics
from pointnet import PointNetClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml
import time
from collections import Counter
import pandas as pd


# Cargar hiperparámetros desde params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
TRAINING_HYPERPARAMETERS = params["TRAINING_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]
SELECTED_CLASSES = params["SELECTED_CLASSES"]
CLASS_NAME_MAPPING = params["CLASS_NAME_MAPPING"]
CONFUSION_MATRIX_CONFIG = params["CONFUSION_MATRIX_CONFIG"]

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

# Early stopping
class EarlyStopping:
    def __init__(self, patience=TRAINING_HYPERPARAMETERS["early_stopping_patience"], delta=TRAINING_HYPERPARAMETERS["early_stopping_delta"]):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Entrenamiento del modelo
def train_pointnet(data, labels, gantry_distances, paths, unique_labels, label_to_idx, idx_to_label, num_classes, epochs, batch_size, learning_rate):
    # Obtener combinaciones de gantry_distance y etiquetas

    filtered_combined_labels = [(distance, label) for distance, label in zip(gantry_distances, labels)]

    # Verificar y filtrar combinaciones con menos de 2 instancias
    combined_counter = Counter(filtered_combined_labels)
    valid_combinations = [combo for combo, count in combined_counter.items() if count > 1]
    valid_indices = [i for i, combo in enumerate(filtered_combined_labels) if combo in valid_combinations]

    # Filtrar datos, etiquetas, rutas y distancias
    filtered_combined_labels = [filtered_combined_labels[i] for i in valid_indices]
    filtered_data = [data[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_paths = [paths[i] for i in valid_indices]
    filtered_gantry_distances = [gantry_distances[i] for i in valid_indices]

    # Dividir el dataset de manera estratificada por etiquetas
    train_indices, test_indices = train_test_split(
        range(len(filtered_labels)),
        test_size=0.2,
        stratify=filtered_labels,  # Estratificación solo por etiquetas
        random_state=42
    )

    # Crear datasets divididos
    train_data = [filtered_data[idx] for idx in train_indices]
    train_labels = [filtered_labels[idx] for idx in train_indices]
    train_paths = [filtered_paths[idx] for idx in train_indices]
    train_gantry_distances = [filtered_gantry_distances[idx] for idx in train_indices]

    test_data = [filtered_data[idx] for idx in test_indices]
    test_labels = [labels[idx] for idx in test_indices]
    test_paths = [paths[idx] for idx in test_indices]
    test_gantry_distances = [gantry_distances[idx] for idx in test_indices]

    distribution = Counter(zip(test_gantry_distances, test_labels))

    # Verificar distribución de etiquetas en el conjunto de prueba
    test_label_counts = Counter(test_labels)
    unique_labels_int = [int(label) for label in unique_labels]
    missing_classes = [label for label in set(filtered_labels) if label not in test_label_counts]
    if missing_classes:
        print(f"Advertencia: Las siguientes clases no están presentes en el conjunto de prueba: {missing_classes}")

    # Crear DataLoader
    train_loader = DataLoader(PointCloudDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PointCloudDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)

    # Inicializar el modelo
    model = PointNetClassifier(
    num_classes=num_classes,
    feature_vector_size=MODEL_HYPERPARAMETERS["feature_vector_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping()

    best_model_wts = None
    best_acc = 0

    # Declarar diccionario para almacenar estadísticas
    training_stats = {
        "epoch_stats": [],
        "best_accuracy": 0,
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward
            outputs = model(points.transpose(2, 1))  # Transponer para [B, 3, N]
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluar en el conjunto de prueba
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points.transpose(2, 1))
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        accuracy = 100. * correct / total
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "test_loss": test_loss / len(test_loader),
            "accuracy": accuracy,
        }
        training_stats["epoch_stats"].append(epoch_stats)
        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")

        # Comprobar early stopping
        early_stopping(test_loss / len(test_loader))
        if early_stopping.early_stop:
            print("Deteniendo entrenamiento temprano por early stopping.")
            break

        # Guardar el mejor modelo
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = model.state_dict()

    # Cargar el mejor modelo
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # Obtener timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{CONFUSION_MATRIX_CONFIG['model_name']}{timestamp}.pth"

    # Graficar matriz de confusión
    print("Generando matriz de confusión...")

    # Obtener etiquetas únicas y ordenarlas numéricamente
    unique_labels = sorted(set(all_labels) | set(all_preds))  # Índices únicos presentes en las predicciones y labels
    valid_classes = [label for label in unique_labels if label in idx_to_label]  # Filtrar solo índices válidos
    real_classes = sorted([int(idx_to_label[label]) for label in valid_classes])  # Etiquetas reales ordenadas
    all_possible_classes = sorted(idx_to_label.keys(), key=int)  # Ordenar las claves numéricamente

    # Llamar a la función con etiquetas reales ordenadas numéricamente
    plot_confusion_matrix(
        all_labels,
        all_preds,
        classes=all_possible_classes,  # Pasar etiquetas reales correctamente ordenadas
        model_name=model_name,
        save_dir=CONFUSION_MATRIX_CONFIG["save_dir"],
        label_mapping=label_to_idx
    )

    # Guardar rutas y etiquetas del conjunto de prueba en un archivo CSV
    conjuntos_dir = "./conjuntos_de_prueba"
    os.makedirs(conjuntos_dir, exist_ok=True)
    csv_name = os.path.splitext(os.path.basename(model_name))[0] + ".csv"
    csv_path = os.path.join(conjuntos_dir, csv_name)

    test_data_df = pd.DataFrame({
        "Path": test_paths,
        "Original Label": [idx_to_label[label] if label in idx_to_label else "Unknown" for label in test_labels],
        "Mapped Label": test_labels
    })

    test_data_df.to_csv(csv_path, index=False)
    print(f"Conjunto de prueba guardado en: {csv_path}")

    # Calcular F1-Score ponderado
    label_counts = Counter(all_labels)
    class_f1_scores = []
    weighted_f1_sum = 0
    total_instances = sum(label_counts.values())

    for label in unique_labels:
      tp = sum(1 for y_true, y_pred in zip(all_labels, all_preds) if y_true == label and y_pred == label)
      fp = sum(1 for y_true, y_pred in zip(all_labels, all_preds) if y_true != label and y_pred == label)
      fn = sum(1 for y_true, y_pred in zip(all_labels, all_preds) if y_true == label and y_pred != label)
  
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0
      f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  
      weighted_f1_sum += f1_score * label_counts[label]
      class_f1_scores.append({
          "Class": label,
          "F1-Score": f1_score,
          "Precision": precision,
          "Recall": recall
      })

    weighted_f1 = weighted_f1_sum / total_instances if total_instances > 0 else 0
    training_stats["weighted_f1"] = weighted_f1

    # Calcular métricas adicionales
    avg_f1 = np.mean([score["F1-Score"] for score in class_f1_scores])
    avg_precision = np.mean([score.get("Precision", 0) for score in class_f1_scores])
    avg_recall = np.mean([score.get("Recall", 0) for score in class_f1_scores])

    # Guardar las métricas en training_stats
    training_stats["average_f1"] = avg_f1
    training_stats["average_precision"] = avg_precision
    training_stats["average_recall"] = avg_recall
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    training_stats["class_distribution"] = {idx_to_label[int(label)]: count for label, count in label_counts.items()}

    # Guardar estadísticas del entrenamiento
    training_stats["best_accuracy"] = max(training_stats.get("best_accuracy", 0), best_acc)
    stats_dir = "./stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{model_name}_stats.json")
    save_training_statistics(training_stats, stats_path)

    # Guardar el modelo entrenado
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Modelo entrenado guardado como '{model_name}'.")

if __name__ == "__main__":
    # Cargar datos desde el dataset
    dataset_path = "../dataset_labeled"  # Ruta al dataset
    print("Cargando datos...")
    data, labels, paths, gantry_distances = iterate_dataset(dataset_path)


    # Filtrar solo las clases seleccionadas y extraer gantry_distance
    filtered_data = []
    filtered_labels = []
    filtered_paths = []
    filtered_gantry_distances = []

    for point_cloud, label, path, gantry_distance in zip(data, labels, paths, gantry_distances):
        if label in SELECTED_CLASSES:
            filtered_data.append(point_cloud)
            filtered_labels.append(label)
            filtered_paths.append(path)
            filtered_gantry_distances.append(gantry_distance)

    # Mapear etiquetas a índices
    unique_labels = list(set(filtered_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Imprimir mapeo de etiquetas a índices
    print("Mapeo de etiquetas a índices:")
    for label, idx in label_to_idx.items():
        print(f"Etiqueta original: {label}, índice asignado: {idx}")

    filtered_labels = [label_to_idx[label] for label in filtered_labels]

    # Balancear el dataset sin sobremuestreo
    max_instances = TRAINING_HYPERPARAMETERS["max_instances"]
    balanced_data, balanced_labels, balanced_paths, balanced_gantry_distances = balance_dataset(
        filtered_data,
        filtered_labels,
        paths=filtered_paths,
        gantry_distances=filtered_gantry_distances,
        max_instances=max_instances
    )

    print(f"Se encontraron {len(unique_labels)} clases: {unique_labels}.")
    
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_pointnet(
        balanced_data,
        balanced_labels,
        balanced_gantry_distances,
        balanced_paths,
        unique_labels,
        label_to_idx,
        idx_to_label,  
        num_classes=len(unique_labels),
        epochs=TRAINING_HYPERPARAMETERS["epochs"],
        batch_size=TRAINING_HYPERPARAMETERS["batch_size"],
        learning_rate=TRAINING_HYPERPARAMETERS["learning_rate"]
    )
