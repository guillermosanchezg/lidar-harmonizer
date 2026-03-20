#!/bin/bash

# Directorios y configuraciones
MODELS_DIR="./models"
CONJUNTOS_DIR="./conjuntos_de_prueba"
RESULTS_DIR="./results"
PARAMS_FILE="params.yaml"

# Lista de modelos (se asume que los nombres terminan en .pth)
MODELS=("pointnet_20250124-171819.pth" "pointnet_20250124-170053.pth" "pointnet_20250124-133352.pth" "pointnet_20250124-162626.pth")

# Crear directorio de resultados si no existe
mkdir -p "$RESULTS_DIR"

for MODEL in "${MODELS[@]}"
do
    # Obtener el nombre base del modelo (sin extensión)
    MODEL_NAME=$(basename "$MODEL" .pth)

    # Modificar feature_vector_size en params.yaml
    echo "Actualizando feature_vector_size para $MODEL_NAME en $PARAMS_FILE..."
    FEATURE_VECTOR_SIZE=$(python -c "import torch; model = torch.load('$MODELS_DIR/$MODEL', map_location='cpu'); print(model.get('meta', {}).get('feature_vector_size', 2048))"); print(model.get('meta', {}).get('feature_vector_size', 128))")
    
    # Actualizar params.yaml
    yq eval -i ".MODEL_HYPERPARAMETERS.feature_vector_size = $FEATURE_VECTOR_SIZE" "$PARAMS_FILE"".MODEL_HYPERPARAMETERS.feature_vector_size = $FEATURE_VECTOR_SIZE" "$PARAMS_FILE"

    # Ejecutar el script de Python
    echo "Ejecutando cálculo de vecinos para $MODEL_NAME..."
    python calculate_k_neighbors.py "$MODEL_NAME.pth" --k 5

    echo "Resultados guardados en $RESULTS_DIR/k_neighbors_${MODEL_NAME}.csv"

done

echo "Ejecución completada para todos los modelos."
