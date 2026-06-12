import os
import argparse
import json
import torch
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader, Dataset
from utils import (iterate_dataset, normalize_point_cloud, uniform_points,
                   balance_dataset, plot_confusion_matrix, save_training_statistics)
from models_pyg import DGCNNClassifier, PointNet2Classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import yaml
import time
from collections import Counter
import pandas as pd

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
TRAINING_HYPERPARAMETERS = params["TRAINING_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]
device = torch.device(DEVICE_CONFIG["device"])

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
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class EarlyStopping:
    def __init__(self, patience=TRAINING_HYPERPARAMETERS["early_stopping_patience"], delta=TRAINING_HYPERPARAMETERS["early_stopping_delta"]):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(data, labels, gantry_distances, paths, unique_labels, label_to_idx, idx_to_label, num_classes, epochs, batch_size, learning_rate, model_type, feature_size):
    filtered_combined_labels = [f"{distance}_{label}" for distance, label in zip(gantry_distances, labels)]
    
    combined_counter = Counter(filtered_combined_labels)
    valid_combinations = {combo for combo, count in combined_counter.items() if count > 1}
    valid_indices = [i for i, combo in enumerate(filtered_combined_labels) if combo in valid_combinations]

    filtered_data = [data[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_paths = [paths[i] for i in valid_indices]
    filtered_gantry_distances = [gantry_distances[i] for i in valid_indices]

    train_indices, test_indices = train_test_split(
        range(len(filtered_labels)), test_size=0.2, stratify=filtered_labels, random_state=42
    )

    train_data = [filtered_data[idx] for idx in train_indices]
    train_labels = [filtered_labels[idx] for idx in train_indices]
    train_paths = [filtered_paths[idx] for idx in train_indices]

    test_data = [filtered_data[idx] for idx in test_indices]
    test_labels = [filtered_labels[idx] for idx in test_indices]
    test_paths = [filtered_paths[idx] for idx in test_indices]

    train_loader = DataLoader(PointCloudDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PointCloudDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)

    print(f"\nInicializando modelo: {model_type.upper()} con vector de {feature_size} dimensiones y Joint Loss (WCE + Triplet)...")
    if model_type == "dgcnn":
        model = DGCNNClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    elif model_type == "pointnet2":
        model = PointNet2Classifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    else:
        raise ValueError(f"Modelo '{model_type}' no soportado.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # CÁLCULO DE PESOS PARA WEIGHTED CROSS-ENTROPY
    train_counter = Counter(train_labels)
    total_train_samples = len(train_labels)
    
    class_weights = torch.zeros(num_classes, dtype=torch.float32).to(device)
    for cls_idx in range(num_classes):
        if train_counter[cls_idx] > 0:
            class_weights[cls_idx] = total_train_samples / train_counter[cls_idx]
        else:
            class_weights[cls_idx] = 1.0
            
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"Pesos de Cross-Entropy por clase aplicados: {class_weights.cpu().numpy()}\n")

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    triplet_criterion = TripletMarginLoss(margin=1.0, p=2)
    alpha = 0.5 # Peso de la Triplet Loss respecto a la CE
    
    early_stopping = EarlyStopping()

    best_model_wts = None
    best_acc = 0
    training_stats = {"epoch_stats": [], "best_accuracy": 0}

    final_epoch = epochs

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_loss_ce_acum = 0
        train_loss_triplet_acum = 0
        
        for points, labels_batch in train_loader:
            points, labels_batch = points.to(device), labels_batch.to(device)
            b_size, n_points, _ = points.size()
            pos = points.view(-1, 3)
            batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_points)

            optimizer.zero_grad()
            
            embeddings = model.extract_features(pos, batch_idx)
            outputs = model(pos, batch_idx)
            
            # 1. Pérdida Weighted Cross-Entropy
            loss_ce = ce_criterion(outputs, labels_batch)
            
            # 2. Pérdida Triplet con (Semi) Hard Negative Mining en el batch
            anchor_idx, positive_idx, negative_idx = [], [], []
            
            with torch.no_grad():
                dist_matrix = torch.cdist(embeddings, embeddings, p=2)
            
            for i in range(b_size):
                lbl = labels_batch[i].item()
                
                pos_mask = (labels_batch == lbl)
                pos_mask[i] = False 
                neg_mask = (labels_batch != lbl)
                
                pos_candidates = torch.where(pos_mask)[0]
                neg_candidates = torch.where(neg_mask)[0]
                
                if len(pos_candidates) > 0 and len(neg_candidates) > 0:
                    dists_to_pos = dist_matrix[i][pos_candidates]
                    hardest_pos_idx = pos_candidates[torch.argmax(dists_to_pos)].item()
                    
                    dists_to_neg = dist_matrix[i][neg_candidates]
                    hardest_neg_idx = neg_candidates[torch.argmin(dists_to_neg)].item()
                    
                    anchor_idx.append(i)
                    positive_idx.append(hardest_pos_idx)
                    negative_idx.append(hardest_neg_idx)
                    
            loss_triplet = torch.tensor(0.0, device=device)
            if len(anchor_idx) > 0:
                loss_triplet = triplet_criterion(
                    embeddings[anchor_idx],
                    embeddings[positive_idx],
                    embeddings[negative_idx]
                )
            
            loss = loss_ce + (alpha * loss_triplet)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_ce_acum += loss_ce.item()
            train_loss_triplet_acum += loss_triplet.item()

        # Validación
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for points, labels_batch in test_loader:
                points, labels_batch = points.to(device), labels_batch.to(device)
                b_size, n_points, _ = points.size()
                pos = points.view(-1, 3)
                batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_points)

                outputs = model(pos, batch_idx) 
                loss = ce_criterion(outputs, labels_batch) 
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels_batch.size(0)
                correct += predicted.eq(labels_batch).sum().item()

                all_labels.extend(labels_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        accuracy = 100. * correct / total

        epoch_stat = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "test_loss": test_loss / len(test_loader),
            "accuracy": accuracy
        }
        training_stats["epoch_stats"].append(epoch_stat)

        print(f"Epoch [{epoch + 1}/{epochs}] Total Train Loss: {train_loss / len(train_loader):.4f} "
              f"(CE: {train_loss_ce_acum/len(train_loader):.4f}, Trip: {train_loss_triplet_acum/len(train_loader):.4f}), "
              f"Test Loss (WCE): {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = model.state_dict()

        early_stopping(test_loss / len(test_loader), epoch + 1)
        if early_stopping.early_stop:
            final_epoch = early_stopping.stopped_epoch
            print(f"\n[!] Deteniendo entrenamiento temprano (Early Stopping) en el Epoch {final_epoch}.")
            print(f"[!] Mejor Accuracy validada guardada: {best_acc:.2f}%")
            break

    if not early_stopping.early_stop:
        print(f"\n[!] Entrenamiento finalizado tras alcanzar los {epochs} epochs máximos.")
        print(f"[!] Mejor Accuracy validada guardada: {best_acc:.2f}%")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Nombres dinámicos
    arch_name = "dgcnn" if "DGCNN" in str(type(model)) else "pointnet2"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{arch_name}_{feature_size}d_{timestamp}.pth"

    # ==========================================
    # NUEVO: REPORTE DE MÉTRICAS DETALLADO
    # ==========================================
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACIÓN FINAL (EVALUACIÓN TEST)")
    print("="*50)
    
    # Calcular Precisión, Recall y F1 para cada clase individual
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(idx_to_label.keys()), zero_division=0
    )

    class_f1_scores = []
    class_metrics_dict = {}

    for i, class_idx in enumerate(idx_to_label.keys()):
        p = precision_arr[i]
        r = recall_arr[i]
        f1 = f1_arr[i]
        supp = support_arr[i]
        
        class_name = idx_to_label[class_idx]
        
        class_metrics_dict[class_name] = {
            "Precision": p,
            "Recall": r,
            "F1-Score": f1,
            "Support": int(supp)
        }
        class_f1_scores.append({"Class": class_name, "F1-Score": f1, "Precision": p, "Recall": r})
        
        print(f"Clase: {class_name:<20} | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | Instancias: {supp}")

    # Reporte global de Scikit-Learn
    report = classification_report(
        all_labels, all_preds, target_names=[idx_to_label[i] for i in sorted(idx_to_label.keys())], zero_division=0
    )
    print("\nResumen Global:")
    print(report)
    print("="*50 + "\n")

    # Guardar métricas extra en el JSON
    training_stats["final_epoch_stopped"] = final_epoch
    training_stats["class_detailed_metrics"] = class_metrics_dict
    
    weighted_f1 = sum([metrics["F1-Score"] * metrics["Support"] for metrics in class_metrics_dict.values()]) / sum(support_arr) if sum(support_arr) > 0 else 0
    training_stats["weighted_f1"] = weighted_f1
    training_stats["average_f1"] = np.mean([score["F1-Score"] for score in class_f1_scores])
    training_stats["average_precision"] = np.mean([score["Precision"] for score in class_f1_scores])
    training_stats["average_recall"] = np.mean([score["Recall"] for score in class_f1_scores])

    # ==========================================
    # GUARDADOS DE ARCHIVOS Y MATRICES
    # ==========================================
    print(f"Generando matriz de confusión para {model_name}...")
    all_possible_classes = sorted(idx_to_label.keys(), key=int)
    matrices_dir = "./visualizations/matrices"
    os.makedirs(matrices_dir, exist_ok=True)
    plot_confusion_matrix(
        all_labels, all_preds,
        classes=all_possible_classes,
        model_name=model_name,
        save_dir=matrices_dir,
        label_mapping=label_to_idx
    )

    conjuntos_dir = "./data/test_sets/"
    os.makedirs(conjuntos_dir, exist_ok=True)
    csv_path = os.path.join(conjuntos_dir, f"{arch_name}_{feature_size}d_{timestamp}.csv")
    test_data_df = pd.DataFrame({
        "Path": test_paths,
        "Original Label": [idx_to_label[label] if label in idx_to_label else "Unknown" for label in test_labels],
        "Mapped Label": test_labels
    })
    test_data_df.to_csv(csv_path, index=False)

    # ----------------------------------------------------
    # AÑADIDO: GUARDAR CONFIGURACIÓN Y PARÁMETROS EN EL JSON
    # ----------------------------------------------------
    training_stats["class_mapping"] = {idx_to_label[int(k)]: int(k) for k in idx_to_label.keys()}
    training_stats["class_distribution"] = {str(idx_to_label[int(label)]): count for label, count in Counter(all_labels).items()}
    training_stats["best_accuracy"] = best_acc
    
    # Insertar hiperparámetros
    training_stats["model_hyperparameters"] = MODEL_HYPERPARAMETERS
    training_stats["training_hyperparameters"] = TRAINING_HYPERPARAMETERS
    training_stats["training_mode"] = params.get("TRAINING_MODE", "fine_grained")

    stats_dir = "./outputs/stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{model_name}_stats.json")
    save_training_statistics(training_stats, stats_path)

    models_dir = "./outputs/models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Modelo entrenado guardado como {model_name}.")
    
    knowledge_dir = "./data/knowledge_sets/"
    os.makedirs(knowledge_dir, exist_ok=True)
    knowledge_csv = os.path.join(knowledge_dir, f"knowledge_{model_name.replace('.pth', '.csv')}")
    
    pd.DataFrame({
        "Path": paths,           
        "Mapped Label": labels   
    }).to_csv(knowledge_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos de PyTorch Geometric")
    parser.add_argument("--model_type", type=str, default="dgcnn", choices=["dgcnn", "pointnet2"])
    args = parser.parse_args()

    dataset_path = "../dataset_labeled"
    print("Cargando datos...")
    data, raw_labels, paths, gantry_distances = iterate_dataset(dataset_path)

    filtered_data = []
    filtered_labels = []
    filtered_paths = []
    filtered_gantry_distances = []
    
    TRAINING_MODE = params.get("TRAINING_MODE", "fine_grained")
    print(f"Modo de entrenamiento detectado: {TRAINING_MODE.upper()}")

    if TRAINING_MODE == "superclass":
        mapper = {str(k): int(v) for k, v in params["SUPERCLASS_MAPPING"].items()}
        names = {int(k): str(v) for k, v in params["SUPERCLASS_NAMES"].items()}
        for point_cloud, label_str, path, gantry in zip(data, raw_labels, paths, gantry_distances):
            if str(label_str) in mapper:
                net_idx = mapper[str(label_str)]
                filtered_data.append(point_cloud)
                filtered_labels.append(net_idx)
                filtered_paths.append(path)
                filtered_gantry_distances.append(gantry)
        label_to_idx = mapper
        idx_to_label = names
        num_classes = len(set(mapper.values()))
    else:
        valid_classes = [str(c) for c in params["SELECTED_CLASSES"]]
        names = {str(k): str(v) for k, v in params["CLASS_NAME_MAPPING"].items()}
        raw_filtered_labels = []
        for point_cloud, label_str, path, gantry in zip(data, raw_labels, paths, gantry_distances):
            if str(label_str) in valid_classes:
                filtered_data.append(point_cloud)
                raw_filtered_labels.append(str(label_str))
                filtered_paths.append(path)
                filtered_gantry_distances.append(gantry)
        unique_original_classes = sorted(list(set(raw_filtered_labels)), key=lambda x: int(x) if x.isdigit() else x)
        label_to_idx = {orig_label: net_idx for net_idx, orig_label in enumerate(unique_original_classes)}
        filtered_labels = [label_to_idx[lbl] for lbl in raw_filtered_labels]
        idx_to_label = {net_idx: names.get(orig_label, orig_label) for orig_label, net_idx in label_to_idx.items()}
        num_classes = len(unique_original_classes)

    print(f"Filtrado completado: {len(filtered_data)} nubes.")
    max_instances = TRAINING_HYPERPARAMETERS["max_instances"]
    
    (balanced_data, balanced_labels, balanced_paths, balanced_gantry_distances), \
    (unseen_data, unseen_labels, unseen_paths, unseen_gantries) = balance_dataset(
        filtered_data, filtered_labels, filtered_paths, filtered_gantry_distances, 
        max_instances=max_instances, percentage=0.7
    )

    unseen_dir = "./data/unseen_sets/"
    os.makedirs(unseen_dir, exist_ok=True)
    unseen_csv_path = os.path.join(unseen_dir, "unseen_dataset.csv")
    
    pd.DataFrame({
        "Path": unseen_paths,
        "Mapped Label": unseen_labels
    }).to_csv(unseen_csv_path, index=False)
    print(f"\n[!] Conjunto de Consulta (Unseen Data) guardado en: {unseen_csv_path}")

    train_model(
        balanced_data, balanced_labels, balanced_gantry_distances, balanced_paths,
        list(idx_to_label.keys()), label_to_idx, idx_to_label, num_classes,
        epochs=TRAINING_HYPERPARAMETERS["epochs"],
        batch_size=TRAINING_HYPERPARAMETERS["batch_size"],
        learning_rate=TRAINING_HYPERPARAMETERS["learning_rate"],
        model_type=args.model_type,
        feature_size=MODEL_HYPERPARAMETERS["feature_vector_size"]
    )