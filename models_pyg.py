import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MLP, DynamicEdgeConv, global_max_pool,
                                 global_mean_pool, PointNetConv, fps, radius, knn)

# ==========================================
# 1. DGCNN (Dynamic Graph CNN)
# ==========================================
class DGCNNClassifier(nn.Module):
    def __init__(self, num_classes, feature_vector_size=512, k=20):
        super(DGCNNClassifier, self).__init__()
        self.k = k

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, 'max')
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, 'max')

        self.lin1 = nn.Linear(128 + 64, feature_vector_size)

        self.fc1 = nn.Linear(feature_vector_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, pos, batch):
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x  = torch.cat([x1, x2], dim=1)
        x  = F.leaky_relu(self.lin1(x), 0.2)
        x  = global_max_pool(x, batch)
        return x

    def forward(self, pos, batch):
        x = self.extract_features(pos, batch)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)


# ==========================================
# 2. POINTNET++ (Jerárquico)
# ==========================================
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn_module):
        super().__init__()
        self.ratio = ratio
        self.r     = r
        self.conv  = PointNetConv(nn_module, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx           = fps(pos, batch, ratio=self.ratio)
        row, col      = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index    = torch.stack([col, row], dim=0)
        x_dst         = None if x is None else x[idx]
        x             = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch    = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
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

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, feature_vector_size]))

        self.fc1 = nn.Linear(feature_vector_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, pos, batch):
        x = None
        x, pos_1, batch_1 = self.sa1_module(x, pos, batch)
        x, pos_2, batch_2 = self.sa2_module(x, pos_1, batch_1)
        x                 = self.sa3_module(x, pos_2, batch_2)
        return x

    def forward(self, pos, batch):
        x = self.extract_features(pos, batch)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)


# ==========================================
# 3. POINTMLP-elite  (Ma et al., ICLR 2022)
#    "Rethinking Network Design and Local Geometry in Point Cloud:
#     A Simple Residual MLP Framework"
#
# Implementación de la variante elite (canales reducidos) adaptada a:
#   - nubes de 2900 puntos
#   - feature_vector_size parametrizable (1024 por defecto)
# La principal contribución arquitectónica (Geometric Affine Module +
# estructura pre-MLP / agregación / post-MLP con residual) se preserva íntegra.
# Los módulos de grouping se implementan en PyTorch puro + PyG (fps, knn)
# para compatibilidad con el pipeline existente.
# ==========================================

def _mlp(dims: list) -> nn.Sequential:
    """MLP con BN y ReLU entre capas (sin activación en la última)."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PointMLPStage(nn.Module):
    """
    Un bloque jerárquico de PointMLP-elite.

    Pipeline por bloque:
      1. FPS downsampling  →  centros
      2. kNN grouping      →  grafo local
      3. Pre-MLP           →  proyección de (rel_pos ∥ features) → out_ch
      4. GAM               →  normalización afín aprendida por grupo
      5. Max aggregation   →  un vector por centro
      6. Post-MLP          →  refinamiento
      7. Residual          →  proyección de features de entrada + fusión
    """

    def __init__(self, in_ch: int, out_ch: int, ratio: float, k: int):
        super().__init__()
        self.ratio = ratio
        self.k     = k

        # Pre-MLP: (rel_xyz ∥ features) → out_ch
        self.pre_mlp = _mlp([3 + in_ch, out_ch, out_ch])
        self.pre_bn  = nn.BatchNorm1d(out_ch)
        self.pre_act = nn.ReLU(inplace=True)

        # Geometric Affine Module (GAM): α y β aprendibles, normalización por grupo
        self.gam_alpha = nn.Parameter(torch.ones(out_ch))
        self.gam_beta  = nn.Parameter(torch.zeros(out_ch))

        # Post-MLP
        self.post_mlp = _mlp([out_ch, out_ch, out_ch])
        self.post_bn  = nn.BatchNorm1d(out_ch)
        self.post_act = nn.ReLU(inplace=True)

        # Residual: proyecta features de entrada en los centros a out_ch
        self.residual = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor):
        """
        x:     [N, in_ch]   features de entrada
        pos:   [N, 3]       posiciones
        batch: [N]          índices de nube (0..B-1)

        Devuelve (feat_out, center_pos, center_batch).
        """
        # 1. FPS: elegir centros representativos
        fps_idx    = fps(pos, batch, ratio=self.ratio)
        center_pos = pos[fps_idx]        # [M, 3]
        center_bat = batch[fps_idx]      # [M]

        # 2. kNN: para cada centro, sus k vecinos en la nube original
        #    knn(x_src, x_dst, k) → (row=dst_idx, col=src_idx)
        row, col = knn(pos, center_pos, self.k, batch, center_bat)
        # row ∈ [0, M), col ∈ [0, N)

        # 3. Features de arista: (posición relativa ∥ features del vecino)
        rel_pos   = pos[col] - center_pos[row]   # [E, 3]
        src_feat  = x[col]                        # [E, in_ch]
        edge_feat = torch.cat([rel_pos, src_feat], dim=1)  # [E, 3+in_ch]

        # Pre-MLP
        edge_feat = self.pre_act(self.pre_bn(self.pre_mlp(edge_feat)))   # [E, out_ch]

        # 4. GAM — normalización afín aprendida dentro de cada grupo
        #    media y varianza se calculan por centro via scatter
        M          = center_pos.shape[0]
        feat_mean  = global_mean_pool(edge_feat, row)          # [M, out_ch]
        mean_exp   = feat_mean[row]                            # [E, out_ch]
        feat_diff  = edge_feat - mean_exp
        feat_var   = global_mean_pool(feat_diff ** 2, row)     # [M, out_ch]
        std_exp    = (feat_var[row] + 1e-5).sqrt()             # [E, out_ch]
        feat_norm  = feat_diff / std_exp
        edge_feat  = self.gam_alpha * feat_norm + self.gam_beta  # [E, out_ch]

        # 5. Max aggregation por centro
        #    global_max_pool trata `row` como índice de "grafo" (0..M-1)
        feat_agg = global_max_pool(edge_feat, row)             # [M, out_ch]

        # Post-MLP
        feat_agg = self.post_act(self.post_bn(self.post_mlp(feat_agg)))  # [M, out_ch]

        # 6. Conexión residual sobre las features en los centros FPS
        res      = self.residual(x[fps_idx])                   # [M, out_ch]
        feat_out = F.relu(feat_agg + res)                      # [M, out_ch]

        return feat_out, center_pos, center_bat


class PointMLPClassifier(nn.Module):
    """
    PointMLP-elite adaptado a nuestro pipeline.

    Arquitectura (4 etapas jerárquicas, canales elite):
      embedding:  3 → 32
      stage 1:   ratio=0.5, k=32,  32 →  64
      stage 2:   ratio=0.5, k=32,  64 → 128
      stage 3:   ratio=0.5, k=32, 128 → 256
      stage 4:   ratio=0.5, k=32, 256 → 512
      global_max_pool → 512D
      proj: 512 → feature_vector_size   (identidad si feature_vector_size=512)
      head: feature_vector_size → 512 → 256 → num_classes

    Con feature_vector_size=1024 (nuestro default), el embedding final es
    1024D, igual que PointNet++ y DGCNN, por lo que el pipeline de embeddings
    + kNN funciona sin cambios.

    Nota: se usa PointMLP-elite porque la variante full resultaría excesivamente
    pesada para nuestra GPU (canales x4 mayores); elite logra resultados
    comparables a <1% de diferencia en ModelNet40.
    """

    def __init__(self, num_classes: int, feature_vector_size: int = 1024,
                 k: int = 32):
        super(PointMLPClassifier, self).__init__()

        # Embedding inicial de coordenadas crudas
        self.embedding = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # 4 etapas jerárquicas (canales de PointMLP-elite escalados a 512D final)
        self.stage1 = PointMLPStage(in_ch=32,  out_ch=64,  ratio=0.5, k=k)
        self.stage2 = PointMLPStage(in_ch=64,  out_ch=128, ratio=0.5, k=k)
        self.stage3 = PointMLPStage(in_ch=128, out_ch=256, ratio=0.5, k=k)
        self.stage4 = PointMLPStage(in_ch=256, out_ch=512, ratio=0.5, k=k)

        # Proyección al espacio de embedding pedido
        if feature_vector_size != 512:
            self.proj = nn.Sequential(
                nn.Linear(512, feature_vector_size),
                nn.BatchNorm1d(feature_vector_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

        # Cabecera de clasificación (igual que PointNet2 / DGCNN)
        self.fc1 = nn.Linear(feature_vector_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, pos: torch.Tensor,
                          batch: torch.Tensor) -> torch.Tensor:
        """
        Extrae el embedding de dimensión feature_vector_size.
        Interfaz idéntica a PointNet2Classifier y DGCNNClassifier.
        """
        x = self.embedding(pos)                          # [N, 32]
        x, pos, batch = self.stage1(x, pos, batch)      # [N/2,  64]
        x, pos, batch = self.stage2(x, pos, batch)      # [N/4, 128]
        x, pos, batch = self.stage3(x, pos, batch)      # [N/8, 256]
        x, pos, batch = self.stage4(x, pos, batch)      # [N/16, 512]
        x = global_max_pool(x, batch)                   # [B, 512]
        x = self.proj(x)                                # [B, feature_vector_size]
        return x

    def forward(self, pos: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(pos, batch)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)
