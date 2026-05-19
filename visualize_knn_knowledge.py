import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import yaml
from scipy.spatial.distance import cdist

# Cargar mapeos desde params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SUPERCLASS_NAMES = {int(k): str(v) for k, v in params.get("SUPERCLASS_NAMES", {}).items()}
ORDERED_CLASSES = sorted(list(SUPERCLASS_NAMES.keys()))

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=ORDERED_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[SUPERCLASS_NAMES[c] for c in ORDERED_CLASSES])
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
    plt.title(f"Matriz de Confusión KNN - ENTRENAMIENTO (K=10, Eq. Ignora a sí mismo)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ Matriz de confusión del Entrenamiento guardada en: {save_path}")

def plot_tsne(embeddings, labels, save_path):
    print("Calculando TSNE del conjunto de entrenamiento...")
    
    # En el train hay ~4000 nubes, TSNE puede procesarlas todas
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    
    for class_idx in unique_labels:
        indices = np.where(labels == class_idx)[0]
        class_name = SUPERCLASS_NAMES.get(class_idx, f"Superclase {class_idx}")
        plt.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=class_name,
            alpha=0.8,
            s=20
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Superclases")
    plt.title("T-SNE del Espacio Latente (Knowledge Set / Train)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ T-SNE del Entrenamiento guardado en: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Nombre del modelo (ej. dgcnn_1024d_xxx.pth)")
    args = parser.parse_args()

    # Cargar embeddings del knowledge set
    csv_emb = f"./outputs/embeddings/embeddings_knowledge_{args.model_name.replace('.pth', '.csv')}"
    out_dir = "./visualizations/knn_eval"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_emb):
        raise FileNotFoundError(f"No se encuentra el archivo de embeddings: {csv_emb}")

    print(f"Cargando embeddings de Entrenamiento (Knowledge) desde {csv_emb}...")
    df_emb = pd.read_csv(csv_emb)
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    
    embeddings = df_emb[emb_cols].values
    y_true = df_emb["Label"].values
    
    # Calcular K-NN manual excluyendo el punto en sí mismo (diagonales = 0 en cdist)
    print("Calculando distancias K-NN internas (evaluando el Train contra sí mismo)...")
    dist_matrix = cdist(embeddings, embeddings, metric='euclidean')
    
    # Para ignorar al propio punto, ponemos la diagonal a infinito
    np.fill_diagonal(dist_matrix, np.inf)

    K = 10
    y_pred = []

    for i in range(len(embeddings)):
        # Obtener los índices de los K vecinos más cercanos (excluyendo sí mismo)
        nearest_idx = np.argsort(dist_matrix[i])[:K]
        dists = dist_matrix[i][nearest_idx]
        neighbors_lbls = y_true[nearest_idx]

        # Votación Ponderada por Inversa de Distancia al Cuadrado
        weights = {}
        for d, n in zip(dists, neighbors_lbls):
            d = max(d, 1e-6)
            w = 1.0 / (d ** 2)
            weights[n] = weights.get(n, 0) + w
            
        pred_lbl = max(weights, key=weights.get) if weights else 5
        y_pred.append(pred_lbl)

    # Dibujar la matriz
    cm_path = os.path.join(out_dir, f"cm_knn_k10_train_internal_{args.model_name}.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # Dibujar T-SNE
    tsne_path = os.path.join(out_dir, f"tsne_train_{args.model_name}.png")
    plot_tsne(embeddings, y_true, tsne_path)

if __name__ == "__main__":
    main()

