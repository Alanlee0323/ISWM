import torch
import torch.quantization
import os
import sys
import argparse
import time
import logging
import traceback
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)

# 確保可以從 src 導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.network.modeling import deeplabv3plus_resnet50
    from src.metrics import StreamMetrics
except ImportError:
    print("Error: Could not import project-specific modules.")
    print("Please ensure the script is run from the project's root directory (ISWM).")
    sys.exit(1)

# --- 數據集定義 ---

class EvaluationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, num_images=0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if num_images > 0:
            self.image_files = self.image_files[:num_images]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # --- 組合對應的遮罩檔案路徑 ---
        # 將 "image.png" 分割成 ("image", ".png")
        base_name, extension = os.path.splitext(img_name)
        # 組合出 "image_mask.png"
        mask_name = f"{base_name}_mask{extension}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            # --- 載入圖片與遮罩 ---
            image = Image.open(img_path).convert('RGB')

            if os.path.exists(mask_path):
                # 載入遮罩並轉換為灰階模式 ('L')
                mask_pil = Image.open(mask_path).convert('L')
                mask = np.array(mask_pil)
                # 進行二值化：所有非 0 的像素都設為 1
                mask[mask > 0] = 1
            else:
                # 如果找不到遮罩，記錄警告並生成一個空白遮罩
                logger.warning(f"找不到遮罩，已為圖片 '{img_name}' 生成空白遮罩。")
                # image.size 是 (寬, 高)，numpy array 的形狀是 (高, 寬)，因此需要反轉
                mask = np.zeros(image.size[::-1], dtype=np.uint8)

            # --- 回傳預處理後的資料 ---
            return self.transform(image), torch.from_numpy(mask).long(), img_name, np.array(image)

        except (FileNotFoundError, UnidentifiedImageError) as e:
            # 優雅地處理檔案不存在或檔案損毀等錯誤
            logger.error(f"讀取 '{img_name}' 資料時發生錯誤: {e}")
            # 重新拋出例外，讓 DataLoader 的錯誤處理機制接手
            raise e

# --- 輔助函式 ---

def get_argparser():
    parser = argparse.ArgumentParser(description="FP32 vs. INT8 Model Evaluation Script")
    parser.add_argument("--fp32_ckpt", required=True, type=str, help="Path to the trained FP32 model checkpoint (.pth)")
    parser.add_argument("--eval_data_dir", required=True, type=str, help="Path to the validation data directory (should contain 'imgs' and 'masks' subfolders)")
    parser.add_argument("--num_images", type=int, default=0, help="Number of images to evaluate on. Default: 0 (all images)")
    parser.add_argument("--output_stride", type=int, default=16, help="Output stride for DeepLabV3+")
    parser.add_argument("--num_visualizations", type=int, default=20, help="Number of comparison images to generate. Default: 5")
    parser.add_argument("--results_dir", type=str, default="evaluation_results", help="Directory to save visualization images.")
    return parser

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_model_size(model_path):
    if not os.path.exists(model_path): return -1
    return os.path.getsize(model_path) / (1024 * 1024)

# --- 模型載入與量化 ---

def load_fp32_model(ckpt_path, num_classes, output_stride, device):
    model = deeplabv3plus_resnet50(num_classes=num_classes, output_stride=output_stride, pretrained_backbone=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def create_int8_model_from_fp32(fp32_model, device, eval_loader):
    """
    從 FP32 模型創建 INT8 量化模型，並加入模組融合與詳細的錯誤處理。
    """
    try:
        model_to_quantize = copy.deepcopy(fp32_model)
        model_to_quantize.eval()
        
        # --- 步驟 1: 融合模組 (使用正確的函式) ---
        # 對於訓練後量化(PTSQ)，應使用 fuse_modules
        print("Attempting to fuse modules...")
        # 修正: 使用 fuse_modules 而不是 fuse_modules_qat
        torch.quantization.fuse_modules(model_to_quantize.backbone, [['conv1', 'bn1', 'relu']], inplace=True)
        
        # 設置量化配置
        model_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        print(f"Using quantization backend: {torch.backends.quantized.engine}")
        
        print("Preparing model for quantization...")
        torch.quantization.prepare(model_to_quantize, inplace=True)
        
        # --- 步驟 2: 校準過程 ---
        print("Running calibration for INT8 model...")
        with torch.no_grad():
            for i, (img_batch, _, _, _) in enumerate(tqdm(eval_loader, desc="Calibrating")):
                if i >= 25:
                    break
                model_to_quantize(img_batch.to(device))
        
        # --- 步驟 3: 轉換為量化模型 ---
        print("Converting model to INT8...")
        int8_model = torch.quantization.convert(model_to_quantize, inplace=True)
        
        print("INT8 model created successfully.")
        return int8_model
        
    except Exception as e:
        # --- 步驟 4: 捕捉並印出詳細錯誤 (保留唯一且正確的 except 區塊) ---
        print(f"ERROR: Quantization failed: {e}")
        print("="*20 + " FULL TRACEBACK " + "="*20)
        traceback.print_exc()
        print("="*58)
        return None

def save_visual_comparison(original_img, gt_mask, fp32_pred, int8_pred, output_path, img_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Comparison for {img_name}", fontsize=16)
    
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(fp32_pred, cmap='gray')
    axes[1, 0].set_title("FP32 Prediction")
    axes[1, 0].axis('off')
    
    if int8_pred is not None:
        axes[1, 1].imshow(int8_pred, cmap='gray')
        axes[1, 1].set_title("INT8 Prediction")
    else:
        axes[1, 1].text(0.5, 0.5, 'INT8 Model\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("INT8 Prediction (Failed)")
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_comparison.png")
    plt.savefig(save_path)
    plt.close(fig)

# --- 主評估邏輯 ---

def main():
    """
    Main function to run the evaluation script.
    
    Parses arguments, loads datasets and models, performs quantization,
    evaluates both FP32 and INT8 models, and prints a comparison report.
    """
    print("--- Script execution started. ---")
    opts = get_argparser().parse_args()
    print(f"--- Parsed Arguments: {opts} ---")

    os.makedirs(opts.results_dir, exist_ok=True)

    device = torch.device('cpu')
    print(f"Evaluation device: {device}")

    # --- Data Loading ---
    transform = get_transform()
    image_dir = os.path.join(opts.eval_data_dir, 'imgs')
    mask_dir = os.path.join(opts.eval_data_dir, 'masks')

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print(f"Error: 'imgs' or 'masks' subfolder not found in {opts.eval_data_dir}")
        return

    eval_dataset = EvaluationDataset(image_dir, mask_dir, transform, opts.num_images)
    if not eval_dataset: 
        print(f"Error: No images found in {image_dir}")
        return
        
    # DataLoader for model calibration
    eval_loader_for_calib = DataLoader(eval_dataset, batch_size=4, shuffle=False)
    
    # --- Model Preparation ---
    print("Loading FP32 model...")
    fp32_model = load_fp32_model(opts.fp32_ckpt, 2, opts.output_stride, device)

    print("\nCreating INT8 model from FP32 model...")
    int8_model = create_int8_model_from_fp32(fp32_model, device, eval_loader_for_calib)
    
    # --- 【新增】保存 INT8 模型到硬碟 ---
    if int8_model:
        # 根據 FP32 檢查點的路徑，自動生成 INT8 模型的儲存路徑
        # 例如： "best.pth" -> "best_int8.pth"
        base, ext = os.path.splitext(opts.fp32_ckpt)
        int8_ckpt_path = f"{base}_int8{ext}"
    
        # 保存整個量化後的模型
        torch.save(int8_model, int8_ckpt_path)
        print(f"\n✅ INT8 model checkpoint saved successfully to: {int8_ckpt_path}")

    # --- Evaluation Setup ---
    fp32_size = get_model_size(opts.fp32_ckpt)
    int8_size_est = fp32_size / 4 if int8_model is not None else 0
    
    fp32_metrics = StreamMetrics(2)
    int8_metrics = StreamMetrics(2) if int8_model is not None else None
    fp32_times = []
    int8_times = []
    visualizations_saved = 0

    print(f"\nStarting evaluation on {len(eval_dataset)} images...")
    
    # DataLoader for evaluation (batch_size=1)
    eval_loader_single = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # --- Evaluation Loop ---
    with torch.no_grad():
        for input_tensor, gt_mask, img_name, original_image in tqdm(eval_loader_single, desc="Evaluating"):
            img_name = img_name[0]
            original_image = original_image.squeeze(0).numpy()
            input_tensor_cpu = input_tensor.to(device)

            # FP32 Evaluation
            start_time = time.perf_counter()
            fp32_output = fp32_model(input_tensor_cpu)
            fp32_times.append(time.perf_counter() - start_time)

            fp32_prob = torch.softmax(fp32_output, dim=1)
            fp32_prob_foreground = fp32_prob[:, 1, :, :]
            
            threshold = 0.5 
            fp32_pred_mask = (fp32_prob_foreground > threshold).long()
            fp32_metrics.update(gt_mask.numpy(), fp32_pred_mask.cpu().numpy())
    
            # INT8 Evaluation
            int8_pred_mask = None
            if int8_model is not None:
                try:
                    start_time = time.perf_counter()
                    int8_output = int8_model(input_tensor_cpu)
                    int8_times.append(time.perf_counter() - start_time)

                    int8_prob = torch.softmax(int8_output, dim=1)
                    int8_prob_foreground = int8_prob[:, 1, :, :]
                    int8_pred_mask = (int8_prob_foreground > threshold).long()
                    int8_metrics.update(gt_mask.numpy(), int8_pred_mask.cpu().numpy())
                except Exception as e:
                    logger.error(f"\nINT8 inference failed for {img_name}: {e}")
                    int8_model = None # Disable further INT8 attempts if one fails

            # Save visualization images for the first few samples
            if visualizations_saved < opts.num_visualizations:
                save_visual_comparison(
                    original_image, 
                    gt_mask.squeeze(0).numpy(),
                    fp32_pred_mask.squeeze(0).cpu().numpy(),
                    int8_pred_mask.squeeze(0).cpu().numpy() if int8_pred_mask is not None else None,
                    opts.results_dir, 
                    img_name
                )
                visualizations_saved += 1

    # --- Print Final Report ---
    print("\n\n" + "="*25 + " Quantization Evaluation Report " + "="*25)
    print(f"Evaluated on {len(eval_dataset)} images.\n")
    
    if int8_model is not None and len(int8_times) > 0:
        print(f"{'Metric':<30} | {'FP32 Model (on CPU)':<20} | {'INT8 Model (on CPU)':<20} | {'Change':<15}")
        print("-" * 90)
        
        # Ignore the first inference time as it may include overhead
        avg_fp32_time = np.mean(fp32_times[1:]) * 1000 if len(fp32_times) > 1 else np.mean(fp32_times) * 1000
        avg_int8_time = np.mean(int8_times[1:]) * 1000 if len(int8_times) > 1 else np.mean(int8_times) * 1000
        speedup = avg_fp32_time / avg_int8_time if avg_int8_time > 0 else float('inf')
        
        print(f"{'Avg. Inference Time (ms)':<30} | {avg_fp32_time:<20.2f} | {avg_int8_time:<20.2f} | {speedup:.2f}x Speedup")
        print(f"{'Model Size (MB)':<30} | {fp32_size:<20.2f} | {int8_size_est:<20.2f} (est.) | ~4x smaller")
        print("-" * 90)
        
        fp32_score = fp32_metrics.get_results()
        int8_score = int8_metrics.get_results()
        
        miou_fp32 = fp32_score.get('MIoU', 0.0)
        miou_int8 = int8_score.get('MIoU', 0.0)
        print(f"{'Mean IoU (mIoU)':<30} | {miou_fp32:<20.4f} | {miou_int8:<20.4f} | {miou_int8 - miou_fp32:+.4f}")

        fg_iou_fp32 = fp32_score.get('Foreground IoU', 0.0)
        fg_iou_int8 = int8_score.get('Foreground IoU', 0.0)
        print(f"{'Foreground IoU':<30} | {fg_iou_fp32:<20.4f} | {fg_iou_int8:<20.4f} | {fg_iou_int8 - fg_iou_fp32:+.4f}")

        fg_f1_fp32 = fp32_score.get('Foreground F1', 0.0)
        fg_f1_int8 = int8_score.get('Foreground F1', 0.0)
        print(f"{'Foreground F1':<30} | {fg_f1_fp32:<20.4f} | {fg_f1_int8:<20.4f} | {fg_f1_int8 - fg_f1_fp32:+.4f}")
    else:
        print("INT8 quantization was not successful. Showing FP32 results only:")
        print(f"{'Metric':<30} | {'FP32 Model (on CPU)':<20}")
        print("-" * 55)
        
        avg_fp32_time = np.mean(fp32_times[1:]) * 1000 if len(fp32_times) > 1 else np.mean(fp32_times) * 1000
        print(f"{'Avg. Inference Time (ms)':<30} | {avg_fp32_time:<20.2f}")
        print(f"{'Model Size (MB)':<30} | {fp32_size:<20.2f}")
        print("-" * 55)
        
        fp32_score = fp32_metrics.get_results()
        
        miou_fp32 = fp32_score.get('MIoU', 0.0)
        fg_iou_fp32 = fp32_score.get('Foreground IoU', 0.0)
        fg_f1_fp32 = fp32_score.get('Foreground F1', 0.0)

        print(f"{'Mean IoU (mIoU)':<30} | {miou_fp32:<20.4f}")
        print(f"{'Foreground IoU':<30} | {fg_iou_fp32:<20.4f}")
        print(f"{'Foreground F1':<30} | {fg_f1_fp32:<20.4f}")
    
    print("-" * 90)
    print(f"Visualizations saved to: {opts.results_dir}")
    print("="*78)
    print("--- Script execution finished. ---")

if __name__ == '__main__':
    main()