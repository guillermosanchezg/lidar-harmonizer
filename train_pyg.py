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
                triplet_alpha, mode_short="sc"):

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
    # NOMBRE DEL MODELO — incluye alpha y modo para no sobrescribir configs
    # formato: {type}_{dim}d_a{alpha}_{mode}_{timestamp}.pth
    # ==========================================
    timestamp  = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_type}_{feature_size}d_a{triplet_alpha:.1f}_{mode_short}_{timestamp}.pth"

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
    # Mezclar params.yaml con los valores reales usados (max_instances y batch_size
    # pueden venir de CLI o de un default por modelo)
    effective_hp = dict(TRAINING_HYPERPARAMETERS)
    effective_hp["max_instances"] = max_instances
    effective_hp["batch_size"]    = batch_size
    training_stats["training_hyperparameters"] = effective_hp
    training_stats["training_mode"]           = params.get("TRAINING_MODE", "fine_grained")
    training_stats["mode_short"]              = mode_short
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
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Máximo de instancias por clase en Knowledge (sobreescribe params.yaml).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size de entrenamiento (sobreescribe params.yaml). "
                             "PointMLP necesita un valor menor por su k-NN denso.")
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

    TRAINING_MODE = params.get("TRAINING_MODE", "fine_grained")
    mode_short    = "sc" if TRAINING_MODE == "superclass" else "fg"
    print(f"Modo: {TRAINING_MODE.upper()} ({mode_short})")

    # -----------------------------------------------------------------
    # 1. Filtrado con clases ORIGINALES (igual para ambos modos)
    #    SUPERCLASS_MAPPING y SELECTED_CLASSES cubren exactamente las
    #    mismas clases en nuestro dataset.
    # -----------------------------------------------------------------
    valid_classes_for_split = set(str(k) for k in params["SUPERCLASS_MAPPING"].keys())

    filt_data, filt_orig_labels, filt_paths, filt_gantry = [], [], [], []
    for pc, lbl, path, gantry in zip(data, raw_labels, paths, gantry_distances):
        if str(lbl) in valid_classes_for_split:
            filt_data.append(pc)
            filt_orig_labels.append(str(lbl))   # clase ORIGINAL: "2", "13", etc.
            filt_paths.append(path)
            filt_gantry.append(gantry)

    print(f"Filtrado: {len(filt_data)} nubes (clases originales: {sorted(set(filt_orig_labels), key=int)})")

    # -----------------------------------------------------------------
    # 2. Mapeo de clases según TRAINING_MODE
    # -----------------------------------------------------------------
    if TRAINING_MODE == "superclass":
        mapper       = {str(k): int(v) for k, v in params["SUPERCLASS_MAPPING"].items()}
        names        = {int(k): str(v) for k, v in params["SUPERCLASS_NAMES"].items()}
        label_to_idx = mapper           # str orig → int superclass
        idx_to_label = names            # int superclass → name
        num_classes  = len(set(mapper.values()))
    else:  # fine_grained
        eng_names    = {str(k): str(v) for k, v in params.get("ENGLISH_CLASS_NAMES", params["CLASS_NAME_MAPPING"]).items()}
        unique_orig  = sorted(set(filt_orig_labels), key=int)
        label_to_idx = {orig: net for net, orig in enumerate(unique_orig)}  # str orig → int fg
        idx_to_label = {net: eng_names.get(orig, orig) for orig, net in label_to_idx.items()}
        num_classes  = len(unique_orig)

    # -----------------------------------------------------------------
    # 3. Split persistido (siempre con clases ORIGINALES)
    # -----------------------------------------------------------------
    max_instances = (args.max_instances
                     if args.max_instances is not None
                     else TRAINING_HYPERPARAMETERS["max_instances"])

    # Batch size: CLI > default por modelo > params.yaml.
    # PointMLP usa k-NN denso (k=32) y materializa varias copias [E, out_ch]
    # por stage, por lo que necesita un batch menor para no agotar la VRAM.
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.model_type == "pointmlp":
        batch_size = 16
    else:
        batch_size = TRAINING_HYPERPARAMETERS["batch_size"]
    print(f"Batch size efectivo: {batch_size} (modelo: {args.model_type})")

    knowledge_indices, unseen_indices = load_or_create_split(
        filt_data, filt_orig_labels, filt_paths, filt_gantry,
        seed=SEED, percentage=0.7,
        split_path="./data/splits/split.csv"
    )

    # Filtrar índices cuya clase original no está en el mapeo actual
    # (en la práctica nunca ocurre, pero es defensivo)
    knowledge_indices = [i for i in knowledge_indices if filt_orig_labels[i] in label_to_idx]
    unseen_indices    = [i for i in unseen_indices    if filt_orig_labels[i] in label_to_idx]

    # Construir array de etiquetas mapeadas para subsample_knowledge
    mapped_labels_full = [label_to_idx.get(filt_orig_labels[i], -1)
                          for i in range(len(filt_orig_labels))]

    # Submuestrear Knowledge a max_instances por clase mapeada (Unseen intacto)
    knowledge_indices = subsample_knowledge(
        knowledge_indices, mapped_labels_full, max_instances, seed=SEED
    )

    balanced_data   = [filt_data[i]    for i in knowledge_indices]
    balanced_labels = [mapped_labels_full[i] for i in knowledge_indices]
    balanced_paths  = [filt_paths[i]   for i in knowledge_indices]
    balanced_gantry = [filt_gantry[i]  for i in knowledge_indices]

    unseen_data     = [filt_data[i]    for i in unseen_indices]
    unseen_labels   = [mapped_labels_full[i] for i in unseen_indices]
    unseen_paths    = [filt_paths[i]   for i in unseen_indices]
    unseen_gantries = [filt_gantry[i]  for i in unseen_indices]

    # -----------------------------------------------------------------
    # 4. Guardar Unseen CSV (por modo, para no mezclar etiquetas sc/fg)
    # -----------------------------------------------------------------
    unseen_dir = "./data/unseen_sets/"
    os.makedirs(unseen_dir, exist_ok=True)
    unseen_csv_name = f"unseen_{mode_short}.csv"
    pd.DataFrame({"Path": unseen_paths, "Mapped Label": unseen_labels}).to_csv(
        os.path.join(unseen_dir, unseen_csv_name), index=False
    )
    print(f"[!] Unseen ({mode_short}) guardado: {len(unseen_paths)} muestras → {unseen_csv_name}")

    train_model(
        balanced_data, balanced_labels, balanced_gantry, balanced_paths,
        list(idx_to_label.keys()), label_to_idx, idx_to_label, num_classes,
        epochs=TRAINING_HYPERPARAMETERS["epochs"],
        batch_size=batch_size,
        learning_rate=TRAINING_HYPERPARAMETERS["learning_rate"],
        model_type=args.model_type,
        feature_size=MODEL_HYPERPARAMETERS["feature_vector_size"],
        triplet_alpha=triplet_alpha,
        mode_short=mode_short,
    )
