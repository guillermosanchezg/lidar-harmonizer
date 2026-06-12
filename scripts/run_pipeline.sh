#!/bin/bash
# Pipeline completo de investigación V2X
# Uso: ./scripts/run_pipeline.sh [--model_type {pointnet2|dgcnn|pointmlp}]
#                                 [--mode {superclass|fine_grained}]
#                                 [--latent_dim 1024]
#                                 [--triplet_alpha 0.5]
#                                 [--max_instances 1000]
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- Defaults ----------
MODEL_TYPE="pointnet2"
MODE="superclass"
LATENT_DIM=1024
TRIPLET_ALPHA=""   # vacío → usa el valor de params.yaml
MAX_INSTANCES=""   # vacío → usa el valor de params.yaml

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_type)    MODEL_TYPE="$2";    shift ;;
        --mode)          MODE="$2";          shift ;;
        --latent_dim)    LATENT_DIM="$2";    shift ;;
        --triplet_alpha) TRIPLET_ALPHA="$2"; shift ;;
        --max_instances) MAX_INSTANCES="$2"; shift ;;
        *) echo "Parámetro desconocido: $1"; exit 1 ;;
    esac
    shift
done

# Construir flags opcionales
ALPHA_FLAG=""
if [ -n "$TRIPLET_ALPHA" ]; then
    ALPHA_FLAG="--triplet_alpha $TRIPLET_ALPHA"
fi
MAX_INST_FLAG=""
if [ -n "$MAX_INSTANCES" ]; then
    MAX_INST_FLAG="--max_instances $MAX_INSTANCES"
fi

# Sufijo de modo
if [ "$MODE" == "fine_grained" ]; then
    MODE_SHORT="fg"
else
    MODE_SHORT="sc"
fi

# Patrón de búsqueda del .pth: incluye modo para no confundir sc y fg
if [ -n "$TRIPLET_ALPHA" ]; then
    ALPHA_LABEL=$(printf "%.1f" "$TRIPLET_ALPHA")
    PTH_PATTERN="${MODEL_TYPE}_${LATENT_DIM}d_a${ALPHA_LABEL}_${MODE_SHORT}_*.pth"
else
    PTH_PATTERN="${MODEL_TYPE}_${LATENT_DIM}d_a*_${MODE_SHORT}_*.pth"
fi

echo "========================================================"
echo " PIPELINE DE INVESTIGACIÓN V2X"
echo " Arquitectura : $MODEL_TYPE"
echo " Modo         : $MODE"
echo " Espacio lat. : ${LATENT_DIM}D"
echo " Triplet alpha: ${TRIPLET_ALPHA:-<params.yaml>}"
echo " Max instances: ${MAX_INSTANCES:-<params.yaml>}"
echo "========================================================"

# Actualizar TRAINING_MODE en params.yaml
sed -i "s/^TRAINING_MODE:.*/TRAINING_MODE: \"$MODE\"/" params.yaml 2>/dev/null || true

# -------------------------------------------------------
# [0/6] Verificar split persistido
# -------------------------------------------------------
echo ""
echo "[0/6] Verificando split persistido..."
if [ -f "./data/splits/split.csv" ]; then
    echo "  ✅ Split existente en ./data/splits/split.csv (se reutilizará)"
else
    echo "  ℹ️  No existe split.csv — se generará automáticamente en [1/6]"
fi

# -------------------------------------------------------
# [1/6] Entrenamiento
# -------------------------------------------------------
echo ""
echo "[1/6] Entrenando modelo ($MODEL_TYPE, alpha=${TRIPLET_ALPHA:-yaml})..."
python train_pyg.py --model_type "$MODEL_TYPE" $ALPHA_FLAG $MAX_INST_FLAG

LATEST_MODEL=$(ls -t "./outputs/models/"${PTH_PATTERN} 2>/dev/null | head -1)
if [ -z "$LATEST_MODEL" ]; then
    echo "❌ ERROR: No se encontró .pth con patrón: ${PTH_PATTERN}"
    exit 1
fi
MODEL_ID=$(basename "$LATEST_MODEL" .pth)
echo "✅ Modelo: $MODEL_ID"

# -------------------------------------------------------
# [2/6] Embeddings
# -------------------------------------------------------
echo ""
echo "[2/6] Generando embeddings para $MODEL_ID..."
python generate_embeddings_pyg.py "${MODEL_ID}.pth" --tsne
echo "✅ Embeddings generados."

# -------------------------------------------------------
# [3/6] K-NN masivo
# -------------------------------------------------------
echo ""
echo "[3/6] Calculando K-NN (k=20)..."
python kvecinos_pyg.py "${MODEL_ID}.pth" --k 20
echo "✅ K-NN completado."

# -------------------------------------------------------
# [4/6] Evaluación frame-level
# -------------------------------------------------------
echo ""
echo "[4/6] Evaluando métricas frame-level (3 reglas: majority, inverse, squared)..."
python evaluate_knn.py "${MODEL_ID}.pth"
echo "✅ Evaluación frame completada."

# -------------------------------------------------------
# [5/6] Evaluación trayectoria
# -------------------------------------------------------
echo ""
echo "[5/6] Evaluando métricas de trayectoria (event_id corregido)..."
python evaluate_KNN_trajectory.py "${MODEL_ID}.pth"
echo "✅ Evaluación trayectoria completada."

# -------------------------------------------------------
# [6/6] Benchmark de latencia
# -------------------------------------------------------
echo ""
echo "[6/6] Benchmark de latencia (K=2..10)..."
python benchmark_k_latency.py "${MODEL_ID}.pth"
echo "✅ Benchmark completado."

echo ""
echo "========================================================"
echo " PIPELINE FINALIZADO"
echo " MODEL_ID : $MODEL_ID"
echo " Resultados: ./outputs/"
echo "========================================================"
