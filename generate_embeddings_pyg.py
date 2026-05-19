import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml
import open3d as o3d

from models_pyg import DGCNNClassifier, PointNet2Classifier
from utils import normalize_point_cloud, uniform_points

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

MODEL_HYPERPARAMETERS = params["MODEL_HYPERPARAMETERS"]
DEVICE_CONFIG = params["DEVICE_CONFIG"]
device = torch.device(DEVICE_CONFIG["device"])


class PointCloudDataset(Dataset):
    def __init__(self, data, labels, paths, num_points=MODEL_HYPERPARAMETERS["num_points"]):
        self.data = data
        self.labels = labels
        self.paths = paths
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc = self.data[idx]
        pc = normalize_point_cloud(pc)
        pc = uniform_points(pc, self.num_points)
        return (
            torch.tensor(pc, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.paths[idx]
        )


def get_model(model_name, feature_size):
    model_path = os.path.join("./outputs/models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el modelo en {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    if "fc3.weight" not in state_dict:
        raise KeyError("No se ha encontrado 'fc3.weight' en el state_dict para inferir num_classes.")

    num_classes = state_dict["fc3.weight"].shape[0]
    print(f"Detectadas {num_classes} clases en los pesos del modelo.")

    if "dgcnn" in model_name.lower():
        model = DGCNNClassifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    else:
        model = PointNet2Classifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_point_clouds_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Path" not in df.columns or "Mapped Label" not in df.columns:
        raise ValueError(f"El CSV {csv_path} debe contener las columnas 'Path' y 'Mapped Label'.")

    paths = df["Path"].tolist()
    labels = df["Mapped Label"].astype(int).tolist()

    data = []
    valid_labels = []
    valid_paths = []

    print(f"Cargando nubes de puntos desde {csv_path} ...")
    for path, label in zip(paths, labels):
        if not os.path.exists(path):
            print(f"[WARN] No existe el archivo: {path}. Se ignora.")
            continue

        pc = np.asarray(o3d.io.read_point_cloud(path).points)
        if pc.size == 0:
            print(f"[WARN] Nube vacía en: {path}. Se ignora.")
            continue

        data.append(pc)
        valid_labels.append(label)
        valid_paths.append(path)

    print(f"  -> Nubes válidas cargadas: {len(data)}")
    return data, valid_labels, valid_paths


def generate_embeddings_for_csv(model, input_csv, output_csv, batch_size=32):
    data, labels, paths = load_point_clouds_from_csv(input_csv)

    if len(data) == 0:
        raise ValueError(f"No se han podido cargar nubes válidas desde {input_csv}")

    dataset = PointCloudDataset(data, labels, paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []
    all_paths = []

    print(f"Generando embeddings para {input_csv} ...")
    with torch.no_grad():
        for points, labels_batch, paths_batch in loader:
            points = points.to(device)
            b_size, n_points, _ = points.size()
            pos = points.view(-1, 3)
            batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_points)

            emb = model.extract_features(pos, batch_idx)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy().tolist())
            all_paths.extend(list(paths_batch))

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    emb_columns = [f"emb_{i}" for i in range(all_embeddings.shape[1])]
    emb_df = pd.DataFrame(all_embeddings, columns=emb_columns)
    emb_df["Label"] = all_labels
    emb_df["Path"] = all_paths

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    emb_df.to_csv(output_csv, index=False)
    print(f"Embeddings guardados en {output_csv}")

    return all_embeddings, all_labels


def generate_tsne(all_embeddings, all_labels, idx_to_label_name, output_png, title):
    print("Calculando TSNE...")
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)

    for class_idx in unique_labels:
        indices = [i for i, x in enumerate(all_labels) if x == class_idx]
        class_name = idx_to_label_name.get(class_idx, f"Clase {class_idx}")
        plt.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=class_name,
            alpha=0.7
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"Gráfico TSNE guardado en {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Nombre del modelo, ej. pointnet2_1024d_xxx.pth")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size para extraer embeddings")
    parser.add_argument("--tsne", action="store_true", help="Generar TSNE del knowledge set")
    args = parser.parse_args()

    feature_size = int(args.model_name.split("_")[1].replace("d", ""))

    stats_path = f"./outputs/stats/{args.model_name}_stats.json"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Falta el archivo de estadísticas: {stats_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    class_mapping = stats.get("class_mapping", {})
    idx_to_label_name = {int(v): str(k) for k, v in class_mapping.items()}

    print(f"Cargando modelo {args.model_name}...")
    model = get_model(args.model_name, feature_size)

    knowledge_csv = os.path.join(
        "./data/knowledge_sets",
        f"knowledge_{args.model_name.replace('.pth', '.csv')}"
    )
    unseen_csv = os.path.join(
        "./data/unseen_sets",
        "unseen_dataset.csv"
    )

    out_dir = "./outputs/embeddings"
    os.makedirs(out_dir, exist_ok=True)

    knowledge_out_csv = os.path.join(
        out_dir,
        f"embeddings_knowledge_{args.model_name.replace('.pth', '.csv')}"
    )
    unseen_out_csv = os.path.join(
        out_dir,
        f"embeddings_unseen_{args.model_name.replace('.pth', '.csv')}"
    )

    knowledge_embeddings, knowledge_labels = generate_embeddings_for_csv(
        model=model,
        input_csv=knowledge_csv,
        output_csv=knowledge_out_csv,
        batch_size=args.batch_size
    )

    unseen_embeddings, unseen_labels = generate_embeddings_for_csv(
        model=model,
        input_csv=unseen_csv,
        output_csv=unseen_out_csv,
        batch_size=args.batch_size
    )

    if args.tsne:
        tsne_dir = "./visualizations/tsne"
        os.makedirs(tsne_dir, exist_ok=True)

        tsne_path = os.path.join(
            tsne_dir,
            f"tsne_knowledge_{args.model_name.replace('.pth', '.png')}"
        )

        generate_tsne(
            all_embeddings=knowledge_embeddings,
            all_labels=knowledge_labels,
            idx_to_label_name=idx_to_label_name,
            output_png=tsne_path,
            title=f"T-SNE Latent Space (Knowledge)\n{args.model_name}"
        )