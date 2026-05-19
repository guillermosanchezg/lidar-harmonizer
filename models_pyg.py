import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, PointNetConv, fps, radius

# ==========================================
# 1. DGCNN (Dynamic Graph CNN)
# ==========================================
class DGCNNClassifier(nn.Module):
    def __init__(self, num_classes, feature_vector_size=512, k=20):
        super(DGCNNClassifier, self).__init__()
        self.k = k
        
        # Capas EdgeConv
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, 'max')
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, 'max')
        
        # Esta capa lineal mapea la concatenación al TAMAÑO PARAMETRIZADO
        self.lin1 = nn.Linear(128 + 64, feature_vector_size)
        
        # Clasificador (mismas dimensiones que tu paper)
        self.fc1 = nn.Linear(feature_vector_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, pos, batch):
        """Devuelve tu embedding de 'feature_vector_size' dimensiones"""
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        
        x = torch.cat([x1, x2], dim=1)
        x = F.leaky_relu(self.lin1(x), 0.2)
        
        # Pooling global para colapsar la nube en un solo vector
        x = global_max_pool(x, batch)  # [Batch_size, feature_vector_size]
        return x

    def forward(self, pos, batch):
        x = self.extract_features(pos, batch)
        
        # Cabecera de clasificación
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

# ==========================================
# 2. POINTNET++ (Jerárquico)
# ==========================================
class SAModule(torch.nn.Module):
    """Módulo de Abstracción de Conjuntos para PointNet++"""
    def __init__(self, ratio, r, nn_module):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn_module, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    """Módulo Global Final para PointNet++"""
    def __init__(self, nn_module):
        super().__init__()
        self.nn = nn_module

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        return x

class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes, feature_vector_size=512):
        super(PointNet2Classifier, self).__init__()
        
        # Tres capas jerárquicas reduciendo puntos y aumentando canales
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        
        # El módulo global escupe el vector del tamaño parametrizado
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, feature_vector_size]))
        
        # Clasificador
        self.fc1 = nn.Linear(feature_vector_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, pos, batch):
        """Devuelve el embedding parametrizado (ej. 512D, 1024D)"""
        x = None # No hay features iniciales, solo coordenadas (pos)
        x, pos_1, batch_1 = self.sa1_module(x, pos, batch)
        x, pos_2, batch_2 = self.sa2_module(x, pos_1, batch_1)
        x = self.sa3_module(x, pos_2, batch_2) # [Batch_size, feature_vector_size]
        return x

    def forward(self, pos, batch):
        x = self.extract_features(pos, batch)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

