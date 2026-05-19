import os
import shutil
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from utils import iterate_dataset, normalize_point_cloud, uniform_points
from models_pyg import PointNet2Classifier
from tqdm import tqdm

# =======================================================
# CONFIGURACIÓN
# =======================================================
MODEL_PATH = "./outputs/models/pointnet2_1024d_20260518-163837.pth"
DATASET_PATH = "../dataset_labeled"

LABEL_ORIGINAL_VAN = "13"
LABEL_TARGET_CAR = "2"
ORDEN_DISTANCIAS = ["-13.68", "-8.68", "-3.68", "3.68", "8.68", "13.68"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CARGA DE PARÁMETROS ---
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)
num_points = params["MODEL_HYPERPARAMETERS"]["num_points"]
feature_size = params["MODEL_HYPERPARAMETERS"]["feature_vector_size"]

# FORZAMOS las 6 clases reales del modelo que acabas de entrenar para evitar el error de mismatch.
# Esto asegura que el mapeo sea exacto al de la matriz de confusión.
valid_classes = ["2", "3", "5", "9", "13", "19"]
num_classes = len(valid_classes)

# Creamos el diccionario para saber qué posición de salida de la red corresponde a cada clase
class_to_idx = {cls: idx for idx, cls in enumerate(valid_classes)}

# Obtenemos qué índice dentro de la red significa "Coche" (clase "2")
# Al estar en la primera posición de la lista, será el índice 0.
TARGET_PREDICTION_IDX = class_to_idx[LABEL_TARGET_CAR] 

# =======================================================
# DATASET Y UTILIDADES
# =======================================================
class InferDataset(Dataset):
    def __init__(self, data, paths):
        self.data = data
        self.paths = paths
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        pc = normalize_point_cloud(self.data[idx])
        pc = uniform_points(pc, num_points)
        return torch.tensor(pc, dtype=torch.float32), self.paths[idx]

def extraer_id_vehiculo(path_pcd):
    partes = os.path.normpath(path_pcd).split(os.sep)
    if len(partes) >= 3:
        return os.sep.join(partes[-3:])
    return None

def buscar_y_mover_relacionados(id_vehiculo, nueva_clase, dataset_base):
    archivos_movidos = 0
    for distancia in ORDEN_DISTANCIAS:
        dir_distancia = os.path.join(dataset_base, distancia)
        if not os.path.exists(dir_distancia):
            continue
            
        for clase_actual in os.listdir(dir_distancia):
            ruta_origen = os.path.join(dir_distancia, clase_actual, id_vehiculo)
            
            if os.path.exists(ruta_origen):
                clase_str = str(nueva_clase)
                # Si ya está en la clase correcta, no hacemos nada
                if clase_actual == clase_str:
                    continue 
                    
                ruta_destino = os.path.join(dir_distancia, clase_str, id_vehiculo)
                os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
                try:
                    shutil.move(ruta_origen, ruta_destino)
                    archivos_movidos += 1
                except Exception as e:
                    print(f"  [ERROR] Fallo al mover a {distancia}: {e}")
    return archivos_movidos

# =======================================================
# EJECUCIÓN PRINCIPAL
# =======================================================
if __name__ == "__main__":
    print("1. Cargando Modelo PointNet++...")
    model = PointNet2Classifier(num_classes=num_classes, feature_vector_size=feature_size).to(device)
    # weights_only=True corrige el warning rojo de seguridad
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"\n2. Buscando archivos de la clase Original {LABEL_ORIGINAL_VAN} (Van) en disco...")
    all_data, all_raw_labels, all_paths, _ = iterate_dataset(DATASET_PATH)

    van_data, van_paths = [], []
    for d, l, p in zip(all_data, all_raw_labels, all_paths):
        # Filtramos estrictamente los que están en la carpeta 13
        if str(l) == LABEL_ORIGINAL_VAN:
            van_data.append(d)
            van_paths.append(p)

    print(f"   Se encontraron {len(van_data)} fragmentos de clase {LABEL_ORIGINAL_VAN}.")

    if len(van_data) == 0:
        print("No hay archivos que evaluar. Finalizando.")
        exit()

    dataset = InferDataset(van_data, van_paths)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    vehiculos_procesados = set()
    vehiculos_movidos = 0
    archivos_totales_movidos = 0

    print("\n3. Evaluando modelo directamente y reubicando en la clase 2...")

    with torch.no_grad():
        for points, paths_batch in tqdm(loader):
            points = points.to(device)
            b_size, n_pts, _ = points.size()
            pos = points.view(-1, 3)
            batch_idx = torch.arange(b_size, device=device).repeat_interleave(n_pts)
            
            # Inferencia directa
            outputs = model(pos, batch_idx)
            # Sacamos el índice de clase con mayor puntuación
            predictions = outputs.argmax(dim=1)
            
            for i in range(b_size):
                # Si el modelo predice "2" (índice interno correspondiente)
                pred_idx = predictions[i].item()
                
                if pred_idx == TARGET_PREDICTION_IDX:
                    path_pcd = paths_batch[i]
                    id_vehiculo = extraer_id_vehiculo(path_pcd)
                    
                    if id_vehiculo and id_vehiculo not in vehiculos_procesados:
                        vehiculos_procesados.add(id_vehiculo)
                        
                        # Lo movemos a la etiqueta real en disco "2" en todas sus distancias
                        movidos = buscar_y_mover_relacionados(id_vehiculo, LABEL_TARGET_CAR, DATASET_PATH)
                        if movidos > 0:
                            vehiculos_movidos += 1
                            archivos_totales_movidos += movidos

    print("\n=======================================================")
    print("  RESULTADOS DEL AUTO-REETIQUETADO DIRECTO")
    print("=======================================================")
    print(f" 🚐 Vehículos clase {LABEL_ORIGINAL_VAN} detectados como clase {LABEL_TARGET_CAR}: {vehiculos_movidos}")
    print(f" 📁 Total de archivos .pcd reubicados en disco: {archivos_totales_movidos}")
    print("=======================================================")