#!/bin/bash
# 環境設置
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0

# 目錄設置
MODEL_NAME="Deeplabv3+"
BASE_SAVE_DIR="experiments/${MODEL_NAME}"  # 基礎保存目錄
CHECKPOINTS_DIR="${BASE_SAVE_DIR}/checkpoints"  # checkpoint 保存目錄
VAL_RESULTS_DIR="${BASE_SAVE_DIR}/val_results"  # 驗證結果保存目錄
METRICS_PLOTS_DIR="${BASE_SAVE_DIR}/metrics_plots"  # 指標圖表保存目錄
LOGS_DIR="${BASE_SAVE_DIR}/logs"  # 日誌保存目錄

# 創建必要的目錄結構
echo "Creating directories..."
mkdir -p ${CHECKPOINTS_DIR} \
        ${VAL_RESULTS_DIR} \
        ${METRICS_PLOTS_DIR} \
        ${LOGS_DIR}

# 檢查目錄權限
check_directory() {
  if [ ! -w "$1" ]; then
      echo "Error: Directory $1 is not writable"
      exit 1
  fi
}

# 檢查需要的目錄
echo "Checking directory permissions..."
for dir in ${CHECKPOINTS_DIR} ${VAL_RESULTS_DIR} ${LOGS_DIR} ${METRICS_PLOTS_DIR}; do
  check_directory "$dir"
done

# 檢查數據目錄
if [ ! -d "./Final_Training_Dataset" ]; then
  echo "Error: Data directory not found"
  exit 1
fi

# 訓練配置
MODEL="deeplabv3plus_resnet50"
DATASET="binary"
LOSS_TYPE="IWce_loss"
OPTIMIZER="sgd"
LEARNING_RATE=0.001
BATCH_SIZE=128
VAL_BATCH_SIZE=4
CROP_SIZE=200
TOTAL_ITRS=30000
VAL_INTERVAL=500
PRINT_INTERVAL=500
GPU_ID=1
RANDOM_SEED=1
OUTPUT_STRIDE=16

# 保存設置
SAVE_VAL_RESULTS=false
SAVE_CONFIDENCE_MAP=false
CROP_VAL=false
ENABLE_VIS=false

# 運行訓練
echo "Starting Training on GPU ${GPU_ID}..."
CONFIG_FILE="${LOGS_DIR}/config_${LOSS_TYPE}_${MODEL}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "Training Configuration (GPU ${GPU_ID}):"
  echo "========================================"
  echo "Model: ${MODEL}"
  echo "Dataset: ${DATASET}"
  echo "Loss Type: ${LOSS_TYPE}"
  echo "Optimizer: ${OPTIMIZER}"
  echo "Learning Rate: ${LEARNING_RATE}"
  echo "Batch Size: ${BATCH_SIZE}"
  echo "Val Batch Size: ${VAL_BATCH_SIZE}"
  echo "Crop Size: ${CROP_SIZE}"
  echo "Total Iterations: ${TOTAL_ITRS}"
  echo "Validation Interval: ${VAL_INTERVAL}"
  echo "Print Interval: ${PRINT_INTERVAL}"
  echo "GPU: ${GPU_ID}"
  echo "Output Stride: ${OUTPUT_STRIDE}"
  echo "Random Seed: ${RANDOM_SEED}"
  echo "Number of Workers: ${NUM_WORKERS}"
  echo "Save Val Results: ${SAVE_VAL_RESULTS}"
  echo "Save Confidence Map: ${SAVE_CONFIDENCE_MAP}"
  echo "Crop Val: ${CROP_VAL}"
  echo "Enable Visualization: ${ENABLE_VIS}"
  echo "Checkpoints Directory: ${CHECKPOINTS_DIR}"
  echo "Results Directory: ${VAL_RESULTS_DIR}"
  echo "Metrics Plots Directory: ${METRICS_PLOTS_DIR}"
} > "${CONFIG_FILE}"

python main_gpu0.py \
      --model ${MODEL} \
      --dataset ${DATASET} \
      --loss_type ${LOSS_TYPE} \
      --optimizer ${OPTIMIZER} \
      --lr ${LEARNING_RATE} \
      --batch_size ${BATCH_SIZE} \
      --val_batch_size ${VAL_BATCH_SIZE} \
      --crop_size ${CROP_SIZE} \
      --total_itrs ${TOTAL_ITRS} \
      --gpu_id ${GPU_ID} \
      --output_stride ${OUTPUT_STRIDE} \
      --random_seed ${RANDOM_SEED} \
      --val_interval ${VAL_INTERVAL} \
      --print_interval ${PRINT_INTERVAL} \
      --data_root ./Deeplabv3Plus_datasets \
      --checkpoints_dir ${CHECKPOINTS_DIR} \
      --val_results_dir  ${VAL_RESULTS_DIR} \
      --metrics_plots_dir ${METRICS_PLOTS_DIR} \
      $([ "$SAVE_VAL_RESULTS" = true ] && echo "--save_val_results") \
      $([ "$SAVE_CONFIDENCE_MAP" = true ] && echo "--save_confidence_map") \
      $([ "$CROP_VAL" = true ] && echo "--crop_val") \
      $([ "$ENABLE_VIS" = true ] && echo "--enable_vis") \
      2>&1 | tee ${LOGS_DIR}/training_${LOSS_TYPE}_${MODEL}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log

# 清理函數
cleanup() {
  echo "Training process completed"
  exit 0
}

trap cleanup EXIT SIGINT SIGTERM
wait
echo "Training completed!"