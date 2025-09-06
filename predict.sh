#!/bin/bash
# 設定預設值
INPUT_PATH="C:/Users/alana/Dropbox/lab/Himawari_Projects/ISWM/calibration_data"
RESULTS_FOLDER="test_results"

# 修正: 使用 Bash 的變數替換來構建路徑，而不是 Python 的 f-string
OUTPUT_PATH="C:/Users/alana/Dropbox/lab/Himawari_Projects/ISWM/test_output/${RESULTS_FOLDER}"

MODEL="deeplabv3plus_resnet50"
DATASET="binary"
CKPT="C:/Users/alana/Dropbox/lab/Himawari_Projects/ISWM/checkpoints/best_deeplabv3plus_resnet50_binary_os16_weighted0.556.pth"
GPU_ID="0"
OUTPUT_STRIDE=16

# 允許動態調整參數
SAVE_CONFIDENCE=false  # 是否存儲信心圖
SAVE_BINARY=false  # 是否存儲二值遮罩
BINARY_THRESHOLD=127  # 不再直接使用，但仍保留為兼容性參數
PRED_THRESHOLD=0.5  # 預測前景的閾值

# 破碎區域機率範圍
MIN_BROKEN_PROB=0.2  # 最小前景機率
MAX_BROKEN_PROB=0.8  # 最大前景機率

# 內波處理相關參數
ENABLE_WAVE_PROCESSING=false  # 停用內波特殊處理
INTERNAL_WAVE_AREA_THRESHOLD=0.01
SYNTHETIC_BROKEN_PROB=0.8
SYNTHETIC_BROKEN_RATIO=0.05

# 檢查必要的路徑和文件是否存在
if [ ! -d "${INPUT_PATH}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_PATH}"
    exit 1
fi

if [ ! -f "${CKPT}" ]; then
    echo "Error: Checkpoint file does not exist: ${CKPT}"
    exit 1
fi

# 確保輸出目錄存在
mkdir -p "${OUTPUT_PATH}"

# 輸出配置信息
echo "Configuration:"
echo "Input path: ${INPUT_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CKPT}"
echo "GPU ID: ${GPU_ID}"
echo "Output stride: ${OUTPUT_STRIDE}"
echo "Save confidence maps: ${SAVE_CONFIDENCE}"
echo "Save binary masks: ${SAVE_BINARY}"
echo "Binary threshold: ${BINARY_THRESHOLD}"
echo "Prediction threshold: ${PRED_THRESHOLD}"
echo "Enable wave processing: ${ENABLE_WAVE_PROCESSING}"
echo "Internal wave area threshold: ${INTERNAL_WAVE_AREA_THRESHOLD}"
echo "Synthetic broken probability: ${SYNTHETIC_BROKEN_PROB}"
echo "Synthetic broken ratio: ${SYNTHETIC_BROKEN_RATIO}"

# 構建執行命令
CMD="python predict.py \
    --input ${INPUT_PATH} \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --ckpt ${CKPT} \
    --gpu_id ${GPU_ID} \
    --output_stride ${OUTPUT_STRIDE} \
    --save_val_results_to ${OUTPUT_PATH} \
    --pred_threshold ${PRED_THRESHOLD}"

# 動態添加可選參數
if [ "${SAVE_CONFIDENCE}" = true ]; then
    CMD+=" --save_confidence"
fi

if [ "${SAVE_BINARY}" = true ]; then
    CMD+=" --save_binary --binary_threshold ${BINARY_THRESHOLD}"
fi

if [ "${ENABLE_WAVE_PROCESSING}" = true ]; then
    CMD+=" --enable_wave_processing \
          --internal_wave_area_threshold ${INTERNAL_WAVE_AREA_THRESHOLD} \
          --synthetic_broken_prob ${SYNTHETIC_BROKEN_PROB} \
          --synthetic_broken_ratio ${SYNTHETIC_BROKEN_RATIO}"
fi

# 執行預測
echo "Executing command: ${CMD}"
eval ${CMD}

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo "Prediction completed successfully"
    echo "Results saved in: ${OUTPUT_PATH}"
else
    echo "Error occurred during prediction"
    exit 1
fi