import os
import torch
import torch.nn as nn
import numpy as np
from wpce import WPCE, initialize_weights
from utils import iterate_dataset, split_dataset
from geomloss import SamplesLoss  # Para la distancia de Sinkhorn

# Configurar dispositivo para PyTorch (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def normalize_point_cloud(cloud):
    """
    Normaliza una nube de puntos para centrarla y escalarla.
    """
    cloud = np.array(cloud)
    cloud -= cloud.mean(axis=0)  # Centrar en el origen
    scale = np.linalg.norm(cloud, axis=1).max()  # Escalar al rango unitario
    cloud /= scale
    return cloud

def filter_dataset(data, labels, include_classes):
    """
    Filtra el dataset incluyendo solo las clases especificadas.

    Args:
        data (list): Lista de nubes de puntos.
        labels (list): Lista de etiquetas correspondientes.
        include_classes (list): Lista de clases a incluir.

    Returns:
        tuple: Datos y etiquetas filtrados.
    """
    filtered_data = [cloud for cloud, label in zip(data, labels) if int(label) in include_classes]
    filtered_labels = [label for label in labels if int(label) in include_classes]
    return filtered_data, filtered_labels



def preprocess_point_cloud(cloud, num_points=1500):
    """
    Preprocesa una nube de puntos para normalizarla y ajustarla a un tamaño fijo.
    """
    cloud = normalize_point_cloud(cloud)
    if len(cloud.shape) == 2 and cloud.shape[1] == 3:
        if cloud.shape[0] > num_points:
            indices = np.random.choice(cloud.shape[0], num_points, replace=False)
            cloud = cloud[indices]
        elif cloud.shape[0] < num_points:
            padding = np.zeros((num_points - cloud.shape[0], 3))
            cloud = np.vstack((cloud, padding))
        return cloud
    else:
        raise ValueError(f"Forma inválida de la nube de puntos: {cloud.shape}")

def chamfer_distance(x, y):
    """
    Calcula la distancia de Chamfer entre dos nubes de puntos.
    """
    x_expanded = x.unsqueeze(2)  # [batch_size, num_points, 1, 3]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, num_points, 3]
    dist = (x_expanded - y_expanded).norm(dim=-1)  # [batch_size, num_points, num_points]

    dist_x_to_y = dist.min(dim=2)[0].mean()  # Distancia de x a y
    dist_y_to_x = dist.min(dim=1)[0].mean()  # Distancia de y a x

    return dist_x_to_y + dist_y_to_x

def sinkhorn_distance(x, y, sinkhorn_eps=0.001, sinkhorn_iters=100):
    """
    Calcula la distancia de Sinkhorn entre dos nubes de puntos.
    """
    loss = SamplesLoss("sinkhorn", p=2, blur=sinkhorn_eps, scaling=0.5, debias=True)
    return loss(x, y)

def evaluate_model(model, test_data, sinkhorn_eps=0.001, sinkhorn_iters=100):
    model.eval()
    chamfer_losses = []
    sinkhorn_losses = []

    with torch.no_grad():
        for cloud in test_data:
            cloud = cloud.unsqueeze(0)  # [1, num_points, 3]
            embedding, reconstruction = model(cloud)

            # Chamfer Distance
            chamfer = chamfer_distance(cloud, reconstruction)
            chamfer_losses.append(chamfer.item())

            # Sinkhorn Distance
            sinkhorn = sinkhorn_distance(cloud, reconstruction, sinkhorn_eps, sinkhorn_iters)
            sinkhorn_losses.append(sinkhorn.item())

    return np.mean(chamfer_losses), np.mean(sinkhorn_losses)

def train_wpce_model(data, labels, input_dim, embedding_dim, num_points, epochs=100, batch_size=16, learning_rate=0.001):

    # Filtrar el dataset excluyendo las clases 0, 1 y 2
    data, labels = filter_dataset(data, labels, include_classes=[4,5])

    # Dividir el dataset
    train_data, test_data, _, _ = split_dataset(data, labels, test_size=0.2)

    # Crear modelo WPCE
    model = WPCE(input_dim=input_dim, embedding_dim=embedding_dim, num_points=num_points).to(device)
    initialize_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Crear datasets en tensores
    train_tensor = [
        torch.tensor(preprocess_point_cloud(cloud, num_points), dtype=torch.float32).to(device)
        for cloud in train_data
    ]
    test_tensor = [
        torch.tensor(preprocess_point_cloud(cloud, num_points), dtype=torch.float32).to(device)
        for cloud in test_data
    ]

    # Entrenamiento
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for i in range(0, len(train_tensor), batch_size):
            batch_data = train_tensor[i:i + batch_size]
            batch_data = torch.stack(batch_data)

            # Forward pass
            optimizer.zero_grad()
            embeddings, reconstructions = model(batch_data)

            # Calcular la pérdida
            loss = criterion(reconstructions, batch_data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluar en el conjunto de prueba
        chamfer_loss, sinkhorn_loss = evaluate_model(model, test_tensor)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Chamfer: {chamfer_loss:.4f}, Sinkhorn: {sinkhorn_loss:.4f}")

    # Guardar modelo entrenado
    torch.save(model.state_dict(), "wpce_model_4_5.pth")
    print("Modelo entrenado y guardado como 'wpce_model_pointNet.pth'.")

if __name__ == "__main__":
    # Configuración
    DATASET_PATH = "../dataset_labeled"
    INPUT_DIM = 3
    EMBEDDING_DIM = 16
    NUM_POINTS = 500  # Número fijo de puntos en las reconstrucciones

    print("Cargando datos del dataset...")
    data, labels = iterate_dataset(DATASET_PATH)

    print("Iniciando entrenamiento del modelo WPCE...")
    train_wpce_model(data, labels, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, num_points=NUM_POINTS, epochs=100, batch_size=16, learning_rate=0.001)

