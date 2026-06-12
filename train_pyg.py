import os
import argparse
import json
import torch
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader, Dataset
from utils import (iterate_dataset, normalize_point_cloud, uniform_points,
                   load_or_create_split, subsample_knowledge, set_global_seed,
                   plot_confusion_matrix, save_training_statistics)
from models_pyg import DGCNNClassifier, PointNet2Classifier, PointMLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import yaml
import time
from collections import Counter
import pandas as pd

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS    = params["MODEL_HYPERPARAMETERS"]
TRAINING_HYPERPARAMETERS = params["TRAINING_HYPERPARAMETERS"]
DEVICE_CONFIG            = params["DEVICE_CONFIG"]
device = torch.device(DEVICE_CONFIG["device"])

SEED = params.get("seed", 42)


class PointCloudDataset(Dataset):
    def __init__(self, data, labels, num_points=MODEL_HYPERPARAMETERS["num_points"]):
        self.data      = data
        self.labels    = labels
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc    = normalize_point_cloud(self.data[idx])
        pc    = uniform_points(pc, self.num_points)
        label = self.labels[idx]
        return torch.tensor(pc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class EarlyStopping:
    def __init__(self,
                 patience=TRAINING_HYPERPARAMETERS["early_stopping_patience"],
                 delta=TRAINING_HYPERPARAMETERS["early_stopping_delta"]):
        self.patience    = patience
        self.delta       = delta
        self.best_loss   = None
        self.counter     = 0
        self.early_stop  = False
        self.stopped_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop    = True
                self.stopped_epoch = epoch
        else:
            self.best_loss = val_loss
            self.counter   = 0


def train_model(data, labels, gantry_distances, paths,
                unique_labels, label_to_idx, idx_to_label, num_classes,
                epochs, batch_size, learning_rate, model_type, feature_size,
                triplet_alpha):

    filtered_combined_labels = [f"{d}_{l}" for d, l in zip(gantry_distances, labels)]
    combined_counter  = Counter(filtered_combined_labels)
    valid_combinations = {c for c, n in combined_counter.items() if n > 1}
    valid_indices     = [i for i, c in enumerate(filtered_combined_labels) if c in valid_combinations]

    filtered_data    = [data[i]             for i in valid_indices]
    filtered_labels  = [labels[i]           for i in valid_indices]
    filtered_paths   = [paths[i]            for i in valid_indices]
    filtered_gantry  = [gantry_distances[i] for i in valid_indices]

    train_indices, test_indices = train_test_split(
        range(len(filtered_labels)), test_size=0.2,
        stratify=filtered_labels, random_state=SEED
    )

    train_data   = [filtered_data[i]   for i in train_indices]
    train_labels = [filtered_labels[i] for i in train_indices]
    train_paths  = [filtered_paths[i]  for i in train_indices]
    test_data    = [filtered_data[i]   for i in test_indices]
    test_labels  = [filtered_labels[i] for i in test_indices]
    test_paths   = [filtered_paths[i]  for i in test_indices]

    train_loader = DataLoader(PointCloudDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(PointCloudDataset(test_data,  test_labels),  batch_size=batch_size, shuffle=False)

    print(f"\nInicializando modelo: {model_type.upper()} | dim={feature_size} | alpha={triplet_alpha}")
    if model_type == "dgcnn":
        model = DGCNNClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    elif model_type == "pointnet2":
        model = PointNet2Classifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    elif model_type == "pointmlp":
        model = PointMLPClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    else:
        raise ValueError(f"Modelo '{model_type}' no soportado.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_counter      = Counter(train_labels)
    total_train        = len(train_labels)
    class_weights      = torch.zeros(num_classes, dtype=torch.float32).to(device)
    for cls_idx in range(num_classes):
        class_weights[cls_idx] = (total_train / train_counter[cls_idx]
                                  if train_counter[cls_idx] > 0 else 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"Pesos WCE: {class_weights.cpu().numpy()}\n")

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    if triplet_alpha > 0:
        triplet_criterion = TripletMarginLoss(margin=1.0, p=2)

    early_stopping = EarlyStopping()
    best_model_wts = None
    best_acc       = 0
    training_stats = {"epoch_stats": [], "best_accuracy": 0}
    final_epoch    = epochs

    for epoch in range(epochs):
        model.train()
        train_loss          = 0
        train_loss_ce_acum  = 0
        train_loss_trip_acum = 0

        for points, labels_batch in train_loader:
            points, labels_batch = points.to(device), labels_batch.to(device)
            b_size, n_points, _ = points.size()
            pos       = points.view(-1, 3)
            batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_points)

            optimizer.zero_grad()
            embeddings = model.extract_features(pos, batch_idx)
            outputs    = model(pos, batch_idx)

            loss_ce = ce_criterion(outputs, labels_batch)

            if triplet_alpha > 0:
                anchor_idx, positive_idx, negative_idx = [], [], []
                with torch.no_grad():
                    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

                for i in range(b_size):
                    lbl = labels_batch[i].item()
                    pos_mask = (labels_batch == lbl)
                    pos_mask[i] = False
                    neg_mask = (labels_batch != lbl)
                    pos_cands = torch.where(pos_mask)[0]
                    neg_cands = torch.where(neg_mask)[0]
                    if len(pos_cands) > 0 and len(neg_cands) > 0:
                        hardest_pos = pos_cands[torch.argmax(dist_matrix[i][pos_cands])].item()
                        hardest_neg = neg_cands[torch.argmin(dist_matrix[i][neg_cands])].item()
                        anchor_idx.append(i)
                        positive_idx.append(hardest_pos)
                        negative_idx.append(hardest_neg)

                loss_triplet = torch.tensor(0.0, device=device)
                if len(anchor_idx) > 0:
                    loss_triplet = triplet_criterion(
                        embeddings[anchor_idx],
                        embeddings[positive_idx],
                        embeddings[negative_idx]
                    )
                loss = loss_ce + triplet_alpha * loss_triplet
                train_loss_trip_acum += loss_triplet.item()
            else:
                loss         = loss_ce
                loss_triplet = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

            train_loss         += loss.item()
            train_loss_ce_acum += loss_ce.item()

        # Validación
        model.eval()
        test_loss = 0
        correct   = 0
        total     = 0
        all_labels_ep = []
        all_preds_ep  = []

        with torch.no_grad():
            for points, labels_batch in test_loader:
                points, labels_batch = points.to(device), labels_batch.to(device)
                b_size, n_points, _ = points.size()
                pos       = points.view(-1, 3)
                batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_points)

                outputs = model(pos, batch_idx)
                loss    = ce_criterion(outputs, labels_batch)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total   += labels_batch.size(0)
                correct += predicted.eq(labels_batch).sum().item()
                all_labels_ep.extend(labels_batch.cpu().numpy())
                all_preds_ep.extend(predicted.cpu().numpy())

        accuracy = 100. * correct / total
        training_stats["epoch_stats"].append({
            "epoch":      epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "test_loss":  test_loss  / len(test_loader),
            "accuracy":   accuracy
        })

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train: {train_loss/len(train_loader):.4f} "
              f"(CE:{train_loss_ce_acum/len(train_loader):.4f} "
              f"Trip:{train_loss_trip_acum/len(train_loader):.4f}) | "
              f"Val:{test_loss/len(test_loader):.4f} | Acc:{accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc       = accuracy
            best_model_wts = model.state_dict()

        early_stopping(test_loss / len(test_loader), epoch + 1)
        if early_stopping.early_stop:
            final_epoch = early_stopping.stopped_epoch
            print(f"\n[!] Early Stopping en epoch {final_epoch}. Mejor Acc: {best_acc:.2f}%")
            break

    if not early_stopping.early_stop:
        print(f"\n[!] Entrenamiento completado ({epochs} epochs). Mejor Acc: {best_acc:.2f}%")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # ==========================================
    # NOMBRE DEL MODELO (incluye alpha para no sobrescribir entre configs)
    # ==========================================
    timestamp  = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_type}_{feature_size}d_a{triplet_alpha:.1f}_{timestamp}.pth"

    # Reporte final
    print("\n" + "=" * 50)
    print("REPORTE DE CLASIFICACIÓN FINAL")
    print("=" * 50)

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        all_labels_ep, all_preds_ep, labels=list(idx_to_label.keys()), zero_division=0
    )

    class_f1_scores  = []
    class_metrics_dict = {}

    for i, class_idx in enumerate(idx_to_label.keys()):
        p, r, f1, supp = precision_arr[i], recall_arr[i], f1_arr[i], support_arr[i]
        class_name = idx_to_label[class_idx]
        class_metrics_dict[class_name] = {
            "Precision": p, "Recall": r, "F1-Score": f1, "Support": int(supp)
        }
        class_f1_scores.append({"Class": class_name, "F1-Score": f1, "Precision": p, "Recall": r})
        print(f"Clase: {class_name:<20} | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | Inst: {supp}")

    report = classification_report(
        all_labels_ep, all_preds_ep,
        target_names=[idx_to_label[i] for i in sorted(idx_to_label.keys())],
        zero_division=0
    )
    print(f"\nResumen:\n{report}")
    print("=" * 50 + "\n")

    training_stats["final_epoch_stopped"] = final_epoch
    training_stats["class_detailed_metrics"] = class_metrics_dict
    training_stats["triplet_alpha"] = triplet_alpha

    total_support = sum(support_arr)
    training_stats["weighted_f1"] = (
        sum(m["F1-Score"] * m["Support"] for m in class_metrics_dict.values()) / total_support
        if total_support > 0 else 0
    )
    training_stats["average_f1"]        = np.mean([s["F1-Score"]  for s in class_f1_scores])
    training_stats["average_precision"]  = np.mean([s["Precision"] for s in class_f1_scores])
    training_stats["average_recall"]     = np.mean([s["Recall"]    for s in class_f1_scores])

    # Matriz de confusión
    print(f"Generando matriz de confusión para {model_name}...")
    matrices_dir = "./visualizations/matrices"
    os.makedirs(matrices_dir, exist_ok=True)
    plot_confusion_matrix(
        all_labels_ep, all_preds_ep,
        classes=sorted(idx_to_label.keys(), key=int),
        model_name=model_name,
        save_dir=matrices_dir,
        label_mapping=label_to_idx
    )

    # CSV del test set
    conjuntos_dir = "./data/test_sets/"
    os.makedirs(conjuntos_dir, exist_ok=True)
    pd.DataFrame({
        "Path":           test_paths,
        "Original Label": [idx_to_label[l] if l in idx_to_label else "Unknown" for l in test_labels],
        "Mapped Label":   test_labels
    }).to_csv(os.path.join(conjuntos_dir, model_name.replace(".pth", ".csv")), index=False)

    # JSON de estadísticas
    training_stats["class_mapping"]           = {idx_to_label[int(k)]: int(k) for k in idx_to_label}
    training_stats["class_distribution"]      = {str(idx_to_label[int(l)]): cnt for l, cnt in Counter(all_labels_ep).items()}
    training_stats["best_accuracy"]           = best_acc
    training_stats["model_hyperparameters"]   = MODEL_HYPERPARAMETERS
    training_stats["training_hyperparameters"] = TRAINING_HYPERPARAMETERS
    training_stats["training_mode"]           = params.get("TRAINING_MODE", "fine_grained")
    training_stats["seed"]                    = SEED

    stats_dir = "./outputs/stats"
    os.makedirs(stats_dir, exist_ok=True)
    save_training_statistics(training_stats, os.path.join(stats_dir, f"{model_name}_stats.json"))

    # Modelo
    models_dir = "./outputs/models"
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, model_name))
    print(f"Modelo guardado: {model_name}")

    # Knowledge CSV (rutas usadas para este modelo)
    knowledge_dir = "./data/knowledge_sets/"
    os.makedirs(knowledge_dir, exist_ok=True)
    pd.DataFrame({
        "Path":        paths,
        "Mapped Label": labels
    }).to_csv(os.path.join(knowledge_dir, f"knowledge_{model_name.replace('.pth', '.csv')}"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos PyG")
    parser.add_argument("--model_type",    type=str, default="pointnet2",
                        choices=["dgcnn", "pointnet2", "pointmlp"])
    parser.add_argument("--triplet_alpha", type=float, default=None,
                        help="Peso de la triplet loss (sobreescribe params.yaml). "
                             "0 = CE-only puro (sin mining).")
    args = parser.parse_args()

    # Alpha: CLI > params.yaml > default 0.5
    triplet_alpha = (args.triplet_alpha
                     if args.triplet_alpha is not None
                     else TRAINING_HYPERPARAMETERS.get("triplet_alpha", 0.5))

    # Semilla global
    set_global_seed(SEED)

    dataset_path = "../dataset_labeled"
    print("Cargando datos...")
    data, raw_labels, paths, gantry_distances = iterate_dataset(dataset_path)

    filtered_data, filtered_labels, filtered_paths, filtered_gantry = [], [], [], []

    TRAINING_MODE = params.get("TRAINING_MODE", "fine_grained")
    print(f"Modo: {TRAINING_MODE.upper()}")

    if TRAINING_MODE == "superclass":
        mapper = {str(k): int(v) for k, v in params["SUPERCLASS_MAPPING"].items()}
        names  = {int(k): str(v) for k, v in params["SUPERCLASS_NAMES"].items()}
        for pc, lbl, path, gantry in zip(data, raw_labels, paths, gantry_distances):
            if str(lbl) in mapper:
                filtered_data.append(pc)
                filtered_labels.append(mapper[str(lbl)])
                filtered_paths.append(path)
                filtered_gantry.append(gantry)
        label_to_idx = mapper
        idx_to_label = names
        num_classes  = len(set(mapper.values()))
    else:
        valid_classes = [str(c) for c in params["SELECTED_CLASSES"]]
        names = {str(k): str(v) for k, v in params["CLASS_NAME_MAPPING"].items()}
        raw_filtered_labels = []
        for pc, lbl, path, gantry in zip(data, raw_labels, paths, gantry_distances):
            if str(lbl) in valid_classes:
                filtered_data.append(pc)
                raw_filtered_labels.append(str(lbl))
                filtered_paths.append(path)
                filtered_gantry.append(gantry)
        unique_original = sorted(set(raw_filtered_labels), key=lambda x: int(x) if x.isdigit() else x)
        label_to_idx    = {orig: net for net, orig in enumerate(unique_original)}
        filtered_labels = [label_to_idx[l] for l in raw_filtered_labels]
        idx_to_label    = {net: names.get(orig, orig) for orig, net in label_to_idx.items()}
        num_classes     = len(unique_original)

    print(f"Filtrado: {len(filtered_data)} nubes.")
    max_instances = TRAINING_HYPERPARAMETERS["max_instances"]

    # ---------- Split persistido ----------
    knowledge_indices, unseen_indices = load_or_create_split(
        filtered_data, filtered_labels, filtered_paths, filtered_gantry,
        seed=SEED, percentage=0.7,
        split_path="./data/splits/split.csv"
    )

    # Submuestrear Knowledge a max_instances (Unseen siempre intacto)
    knowledge_indices = subsample_knowledge(knowledge_indices, filtered_labels, max_instances, seed=SEED)

    balanced_data     = [filtered_data[i]   for i in knowledge_indices]
    balanced_labels   = [filtered_labels[i] for i in knowledge_indices]
    balanced_paths    = [filtered_paths[i]  for i in knowledge_indices]
    balanced_gantry   = [filtered_gantry[i] for i in knowledge_indices]

    unseen_data    = [filtered_data[i]   for i in unseen_indices]
    unseen_labels  = [filtered_labels[i] for i in unseen_indices]
    unseen_paths   = [filtered_paths[i]  for i in unseen_indices]
    unseen_gantries = [filtered_gantry[i] for i in unseen_indices]

    # Guardar Unseen CSV (siempre el mismo)
    unseen_dir = "./data/unseen_sets/"
    os.makedirs(unseen_dir, exist_ok=True)
    pd.DataFrame({"Path": unseen_paths, "Mapped Label": unseen_labels}).to_csv(
        os.path.join(unseen_dir, "unseen_dataset.csv"), index=False
    )
    print(f"[!] Unseen guardado: {len(unseen_paths)} muestras")

    train_model(
        balanced_data, balanced_labels, balanced_gantry, balanced_paths,
        list(idx_to_label.keys()), label_to_idx, idx_to_label, num_classes,
        epochs=TRAINING_HYPERPARAMETERS["epochs"],
        batch_size=TRAINING_HYPERPARAMETERS["batch_size"],
        learning_rate=TRAINING_HYPERPARAMETERS["learning_rate"],
        model_type=args.model_type,
        feature_size=MODEL_HYPERPARAMETERS["feature_vector_size"],
        triplet_alpha=triplet_alpha,
    )
