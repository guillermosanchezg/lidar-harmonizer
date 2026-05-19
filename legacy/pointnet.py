import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

def load_params(filepath="params.yaml"):
    """
    Carga los parámetros desde un archivo YAML.
    """
    with open(filepath, "r") as file:
        params = yaml.safe_load(file)
    return params


class STN3d(nn.Module):
    """
    Spatial Transformer Network para alinear nubes de puntos en 3D.
    """
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Reshape al espacio de transformación 3x3
        x = x.view(batch_size, 3, 3)
        identity = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + identity  # Agregar identidad para inicialización cercana a una matriz de rotación
        return x


class PointNetFeatureExtractor(nn.Module):
    """
    Extractor de características con max-pooling global.
    """
    def __init__(self, feature_vector_size):
        super(PointNetFeatureExtractor, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, feature_vector_size, 1)  # Usa el parámetro dinámico

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feature_vector_size)  # Usa el parámetro dinámico
        self.relu = nn.ReLU()

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)  # Cambiar a [B, N, 3] para multiplicación matricial
        x = torch.bmm(x, trans)  # Aplicar transformación
        x = x.transpose(2, 1)  # Regresar a [B, 3, N]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = torch.max(x, 2, keepdim=False)[0]  # [B, feature_vector_size]
        return x


class PointNetClassifier(nn.Module):
    """
    Clasificador basado en PointNet.
    """
    def __init__(self, num_classes, feature_vector_size):
        super(PointNetClassifier, self).__init__()
        self.feature_extractor = PointNetFeatureExtractor(feature_vector_size)

        self.fc1 = nn.Linear(feature_vector_size, 512)  # Usa el parámetro dinámico
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Usa el parámetro dinámico

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_extractor(x)  # Extraer características
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Predicción final
        return F.log_softmax(x, dim=1)
    
    def extract_features(self, x):
        return self.feature_extractor(x)

def initialize_weights(model):
    """
    Inicialización personalizada para pesos de convoluciones y capas lineales.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


if __name__ == "__main__":
    # Cargar parámetros desde params.yaml
    params = load_params("params.yaml")
    num_points = params["MODEL_HYPERPARAMETERS"]["num_points"]
    feature_vector_size = params["MODEL_HYPERPARAMETERS"]["feature_vector_size"]
    num_classes = params["MODEL_HYPERPARAMETERS"]["num_classes"]

    batch_size = 32  # Puedes ajustarlo también dinámicamente si es necesario

    # Inicializar el modelo con los parámetros dinámicos
    model = PointNetClassifier(num_classes=num_classes, feature_vector_size=feature_vector_size)
    initialize_weights(model)  # Inicializar pesos personalizados
    print(model)

    # Crear un lote de nubes de puntos aleatorias con el tamaño especificado
    input_data = torch.rand(batch_size, 3, num_points)  # Tamaño dinámico de num_points
    output = model(input_data)
    print("Output shape:", output.shape)

