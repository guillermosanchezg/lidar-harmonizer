#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_TYPE="pointnet2"
MODE="fine_grained"
LATENT_DIM=1024

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_type) MODEL_TYPE="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --latent_dim) LATENT_DIM="$2"; shift ;;
        *) echo "Parámetro desconocido: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================================"
echo " INICIANDO PIPELINE DE INVESTIGACIÓN V2X"
echo " Arquitectura: $MODEL_TYPE"
echo " Modo: $MODE"         
echo " Espacio Latente: ${LATENT_DIM}D"
echo "========================================================"

sed -i "s/^MODEL_TYPE:.*/MODEL_TYPE: \"$MODEL_TYPE\"/" params.yaml 2>/dev/null || true
sed -i "s/^TRAINING_MODE:.*/TRAINING_MODE: \"$MODE\"/" params.yaml 2>/dev/null || true
sed -i "s/^feature_vector_size:.*/feature_vector_size: $LATENT_DIM/" params.yaml 2>/dev/null || true

echo "[1/5] Entrenando modelo..."
# Ya no necesitamos el > training_temp.log, dejamos que se imprima normal
python train_pyg.py --model_type $MODEL_TYPE

# CAPTURA ROBUSTA: Buscar el último archivo .pth creado del tipo correcto
# ls -t ordena por fecha de modificación (el más nuevo primero)
LATEST_MODEL=$(ls -t ./outputs/models/${MODEL_TYPE}_${LATENT_DIM}d_*.pth 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ ERROR: No se pudo encontrar el modelo generado en ./outputs/models/"
    exit 1
fi

# Extraer solo el nombre del archivo sin la ruta completa y sin el ".pth"
MODEL_ID=$(basename "$LATEST_MODEL" .pth)

echo "✅ Entrenamiento completado. MODEL_ID: $MODEL_ID"

echo "[2/5] Generando embeddings y t-SNE..."
python generate_embeddings.py ${MODEL_ID}.pth --tsne
echo "✅ Embeddings y t-SNE generados."

echo "[3/5] Calculando K-NN..."
python kvecinos_pyg.py ${MODEL_ID}.pth ./data/test_sets/${MODEL_ID}.csv --model_type $MODEL_TYPE --k 20
echo "✅ K-NN completado."

echo "[4/5] Evaluando métricas..."
python evaluate_knn.py ${MODEL_ID}.pth
echo "✅ Evaluación completada."

echo "========================================================"
echo " PIPELINE FINALIZADO"
echo "========================================================"
