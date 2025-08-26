import torch
import torch.quantization
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# Import project-specific modules
from src.network.modeling import deeplabv3plus_resnet50
from src.metrics.region_metrics import RegionMetrics
from src.metrics.front_tracking_metrics import FrontTrackingMetrics

# --- Helper Functions ---

def get_argparser():
    parser = argparse.ArgumentParser(description="FP32 vs. INT8 Model Evaluation Script")
    parser.add_argument("--fp32_ckpt", required=True, type=str, help="Path to the trained FP32 model checkpoint (.pth)")
    parser.add_argument("--int8_ckpt", required=True, type=str, help="Path to the quantized INT8 model state_dict (.pth)")
    parser.add_argument("--eval_data_dir", required=True, type=str, help="Path to the validation data directory (should contain 'images' and 'masks' subfolders)")
    parser.add_argument("--num_images", type=int, default=0, help="Number of images to evaluate on. Default: 0 (all images)")
    parser.add_argument("--output_stride", type=int, default=16, help="Output stride for DeepLabV3+")
    return parser

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_model_size(model_path):
    if not os.path.exists(model_path):
        return -1
    return os.path.getsize(model_path) / (1024 * 1024) # Size in MB

# --- Model Loading ---

def load_fp32_model(ckpt_path, num_classes, output_stride, device):
    model = deeplabv3plus_resnet50(num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def load_int8_model(ckpt_path, num_classes, output_stride, device):
    model_fp32 = deeplabv3plus_resnet50(num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False)
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model_prepared = torch.quantization.prepare(model_fp32)
    model_int8 = torch.quantization.convert(model_prepared)
    model_int8.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_int8.to(device)
    model_int8.eval()
    return model_int8

# --- Main Evaluation Logic ---

def main():
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")

    # Load models
    print("Loading FP32 model...")
    fp32_model = load_fp32_model(opts.fp32_ckpt, 2, opts.output_stride, device)
    print("Loading INT8 model...")
    int8_model = load_int8_model(opts.int8_ckpt, 2, opts.output_stride, device)

    # Get model sizes
    fp32_size = get_model_size(opts.fp32_ckpt)
    int8_size = get_model_size(opts.int8_ckpt)

    # Prepare dataset and metrics
    transform = get_transform()
    image_dir = os.path.join(opts.eval_data_dir, 'imgs')
    mask_dir = os.path.join(opts.eval_data_dir, 'masks')
    
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print(f"Error: 'images' or 'masks' subfolder not found in {opts.eval_data_dir}")
        return

    image_files = sorted(os.listdir(image_dir))
    if opts.num_images > 0:
        image_files = image_files[:opts.num_images]
    
    # Initialize metric calculators
    fp32_region_metrics = RegionMetrics()
    fp32_front_metrics = FrontTrackingMetrics()
    int8_region_metrics = RegionMetrics()
    int8_front_metrics = FrontTrackingMetrics()
    
    fp32_times = []
    int8_times = []

    print(f"\nStarting evaluation on {len(image_files)} images...")
    with torch.no_grad():
        for img_name in tqdm(image_files, desc="Evaluating"):
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            if not os.path.exists(mask_path):
                continue

            image = Image.open(img_path).convert('RGB')
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask[gt_mask > 0] = 1 # Ensure mask is binary
            
            input_tensor = transform(image).unsqueeze(0).to(device)

            # --- FP32 Evaluation ---
            start_time = time.perf_counter()
            fp32_output = fp32_model(input_tensor)
            fp32_times.append(time.perf_counter() - start_time)
            fp32_pred_mask = torch.argmax(fp32_output.squeeze(), dim=0).cpu().numpy()
            fp32_region_metrics.update(fp32_pred_mask, gt_mask)
            fp32_front_metrics.update(fp32_pred_mask, gt_mask)

            # --- INT8 Evaluation ---
            start_time = time.perf_counter()
            int8_output = int8_model(input_tensor)
            int8_times.append(time.perf_counter() - start_time)
            int8_pred_mask = torch.argmax(int8_output.squeeze(), dim=0).cpu().numpy()
            int8_region_metrics.update(int8_pred_mask, gt_mask)
            int8_front_metrics.update(int8_pred_mask, gt_mask)

    # --- Print Report ---
    print("\n--- Quantization Evaluation Report ---")
    print(f"Evaluated on {len(image_files)} images.\n")
    
    print(f"{ 'Metric':<30} | { 'FP32 Model':<20} | { 'INT8 Model':<20} | { 'Change':<15}")
    print("-" * 90)
    
    # Performance Metrics
    avg_fp32_time = np.mean(fp32_times[1:]) * 1000
    avg_int8_time = np.mean(int8_times[1:]) * 1000
    speedup = avg_fp32_time / avg_int8_time if avg_int8_time > 0 else float('inf')
    print(f"{ 'Avg. Inference Time (ms)':<30} | {avg_fp32_time:<20.2f} | {avg_int8_time:<20.2f} | {speedup:.2f}x Speedup")
    print(f"{ 'Model Size (MB)':<30} | {fp32_size:<20.2f} | {int8_size:<20.2f} |")
    print("-" * 90)

    # Accuracy Metrics
    fp32_region_score = fp32_region_metrics.get_mean_score()
    int8_region_score = int8_region_metrics.get_mean_score()
    print(f"{ 'Region Score (Final)':<30} | {fp32_region_score:<20.4f} | {int8_region_score:<20.4f} | {int8_region_score - fp32_region_score:+.4f}")

    fp32_front_error = fp32_front_metrics.get_mean_error()
    int8_front_error = int8_front_metrics.get_mean_error()
    print(f"{ 'Front Tracking Error':<30} | {fp32_front_error:<20.4f} | {int8_front_error:<20.4f} | {int8_front_error - fp32_front_error:+.4f}")
    print("-" * 90)

if __name__ == '__main__':
    main()