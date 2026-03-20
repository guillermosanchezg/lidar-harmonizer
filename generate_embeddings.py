import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wpce import WPCE
from utils import load_pcd, iterate_dataset

# Configurar dispositivo para PyTorch (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Generar embeddings con el modelo
def generate_embeddings(model, data):
    embeddings = []
    with torch.no_grad():
        for cloud in data:
            cloud_tensor = torch.tensor(cloud, dtype=torch.float32).to(device)
            embedding = model.encoder(cloud_tensor)  # Solo devuelve el embedding
            embeddings.append(embedding.cpu().numpy())
    return np.array(embeddings)

# Cargar modelo preentrenado
def load_trained_model(model_path, input_dim, embedding_dim):
    model = WPCE(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


# Guardar los embeddings en un archivo CSV
def save_embeddings_to_csv(file_path, embeddings, labels):
    import csv
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Label"] + [f"Dim{i}" for i in range(embeddings.shape[1])])
        for label, embedding in zip(labels, embeddings):
            writer.writerow([label] + list(embedding))
    print(f"Embeddings guardados en {file_path}")

# Visualizar embeddings usando PCA o t-SNE
def visualize_embeddings(embeddings, labels, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Método no válido. Usa 'pca' o 'tsne'.")

    # Crear gráfica
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7
    )
    plt.colorbar(scatter, label="Etiquetas")
    plt.title(f"Visualización del espacio embebido ({method.upper()})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Configuración
    MODEL_PATH = "wpce_model.pth"  # Ruta al modelo preentrenado
    DATASET_PATH = "../dataset_labeled"
    EMBEDDINGS_CSV = "embeddings.csv"
    INPUT_DIM = 3
    EMBEDDING_DIM = 16

    # Cargar datos
    print("Cargando datos del dataset...")
    data, labels = iterate_dataset(DATASET_PATH)
    print(f"Se cargaron {len(data)} nubes de puntos.")

    # Cargar modelo preentrenado
    print("Cargando modelo WPCE...")
    wpce_model = load_trained_model(MODEL_PATH, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)

    # Generar embeddings
    print("Generando embeddings...")
    embeddings = generate_embeddings(wpce_model, data)

    # Guardar embeddings en un archivo CSV
    print("Guardando embeddings...")
    save_embeddings_to_csv(EMBEDDINGS_CSV, embeddings, labels)

    # Visualizar espacio embebido
    print("Visualizando espacio embebido...")
    visualize_embeddings(embeddings, labels, method="tsne")  # Cambia a "pca" si prefieres PCA
    print("Proceso completado.")

