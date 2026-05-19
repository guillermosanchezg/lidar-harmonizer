import os
import argparse
import torch
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from models_pyg import DGCNNClassifier, PointNet2Classifier
from utils import iterate_dataset, load_point_cloud, normalize_point_cloud, uniform_points

# 1. Cargar configuración
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]
SELECTED_CLASSES = params["SELECTED_CLASSES"]
device = torch.device(DEVICE_CONFIG["device"])

class PointCloudDiskDataset(Dataset):
    def __init__(self, paths_list, labels_list, num_points=MODEL_HYPERPARAMETERS["num_points"]):
        self.paths = paths_list
        self.labels = labels_list
        self.num_points = num_points

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Carga dinámica desde disco para no saturar la RAM
        pc = load_point_cloud(self.paths[idx])
        pc = normalize_point_cloud(pc)
        pc = uniform_points(pc, self.num_points)
        
        return torch.tensor(pc, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.paths[idx]

def get_model(model_name, feature_size):
    # Detectar el modo de entrenamiento leyendo params.yaml
    training_mode = params.get('TRAINING_MODE', 'fine_grained')
    
    if training_mode == 'superclasses':
        num_classes = 6
    else:
        # Si es fine_grained, lee cuántas clases tienes seleccionadas
        num_classes = len(params.get('SELECTED_CLASSES', []))
    
    print(f"Cargando arquitectura para {num_classes} clases de salida ({training_mode})...")

    if "dgcnn" in model_name.lower():
        model = DGCNNClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    else:
        model = PointNet2Classifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    
    model_path = os.path.join("./models", model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encontrar y guardar PCDs mal clasificados")
    parser.add_argument("model_name", type=str, help="Nombre del modelo preentrenado en ./models")
    parser.add_argument("--output_file", type=str, default="errores_dataset.txt", help="Archivo txt de salida")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de lote para inferencia")
    args = parser.parse_args()

    feature_size = int(args.model_name.split("_")[1].replace("d", ""))
    
    print(f"Iniciando escáner con el modelo: {args.model_name}...")
    model = get_model(args.model_name, feature_size)

    print("Escaneando el dataset completo...")
    dataset_path = "../dataset_labeled"
    _, original_labels, paths, _ = iterate_dataset(dataset_path)

    # Cargar todos los mapeos desde YAML
    training_mode = params.get('TRAINING_MODE', 'fine_grained')
    superclass_map = params.get('SUPERCLASS_MAPPING', {})
    class_names = params.get('CLASS_NAME_MAPPING', {})
    superclass_names = params.get('SUPERCLASS_NAMES', {})
    
    filtered_paths = []
    filtered_labels = []
    
    # Pre-mapear el diccionario de etiquetas a índices contiguos si es fine_grained
    selected_sorted = sorted(SELECTED_CLASSES)
    label_to_idx = {label: idx for idx, label in enumerate(selected_sorted)}
    idx_to_original = {idx: label for label, idx in label_to_idx.items()}

    for path, label in zip(paths, original_labels):
        orig_label = int(label)
        if orig_label in SELECTED_CLASSES:
            if training_mode == 'superclasses':
                # Las etiquetas se mapean de (2, 3, 4...) a (0, 1, 2... hasta 5)
                mapped_label = superclass_map.get(orig_label, 5)
            else:
                # En fine grained usamos el índice contiguo (0 a 15)
                mapped_label = label_to_idx[orig_label]
                
            filtered_paths.append(path)
            filtered_labels.append(mapped_label)

    print(f"Total de archivos a evaluar: {len(filtered_paths)}")

    # Crear Dataset y DataLoader
    dataset = PointCloudDiskDataset(filtered_paths, filtered_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    misclassified_records = []

    print("Evaluando predicciones...")
    with torch.no_grad():
        for points, true_labels, batch_paths in tqdm(loader, desc="Inferencia"):
            points = points.to(device)
            true_labels = true_labels.to(device)
            
            b_size, n_points, _ = points.size()
            pos = points.view(-1, 3)
            batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_points)
            
            # Forward pass
            outputs = model(pos, batch_idx)
            _, preds = outputs.max(dim=1)
            
            # Comparar predicción vs realidad
            errors_mask = (preds != true_labels)
            
            # Extraer los datos de los que han fallado
            error_indices = torch.nonzero(errors_mask, as_tuple=True)[0]
            
            for idx in error_indices:
                path = batch_paths[idx]
                true_lbl_idx = true_labels[idx].item()
                pred_lbl_idx = preds[idx].item()
                
                if training_mode == 'superclasses':
                    true_name = superclass_names.get(true_lbl_idx, str(true_lbl_idx))
                    pred_name = superclass_names.get(pred_lbl_idx, str(pred_lbl_idx))
                else:
                    # En fine grained sacamos la etiqueta original desde el índice contiguo
                    orig_true_label = idx_to_original[true_lbl_idx]
                    orig_pred_label = idx_to_original[pred_lbl_idx]
                    true_name = f"{orig_true_label}_{class_names.get(orig_true_label, 'Unknown')}"
                    pred_name = f"{orig_pred_label}_{class_names.get(orig_pred_label, 'Unknown')}"
                
                # Guardamos la info estructurada (Path,True_Label,Predicted_Label)
                line = f"{path},{true_name},{pred_name}"
                misclassified_records.append(line)

    print(f"\nSe encontraron {len(misclassified_records)} errores de clasificación.")
    
    # Guardar a TXT
    with open(args.output_file, "w") as f:
        f.write("Path,True_Label,Predicted_Label\n")
        for record in misclassified_records:
            f.write(record + "\n")

    print(f"Lista de PCDs erróneos guardada en: {args.output_file}")