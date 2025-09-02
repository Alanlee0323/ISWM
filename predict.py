

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn as nn
import argparse
import cv2  # 導入OpenCV庫用於繪製線條
from src.network.modeling import deeplabv3plus_resnet50
from src.datasets import BinarySegmentation
from scipy import ndimage
from skimage import measure, morphology
import os

def get_argparser():
    parser = argparse.ArgumentParser()
    
    # Dataset Options
    parser.add_argument("--input", type=str, required=True,
                      help="Path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='binary',
                      choices=['binary'], help='Name of dataset')
    
    # Model Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                      help='Model name')
    parser.add_argument("--ckpt", default=None, type=str,
                      help="Path to trained model")
    parser.add_argument("--gpu_id", type=str, default='0',
                      help="GPU ID")
    parser.add_argument("--save_val_results_to", default=None,
                      help="Directory to save segmentation results")
    
    # Additional Parameters
    parser.add_argument("--output_stride", type=int, default=16,
                      help='Output stride for DeepLabV3+ (8 or 16)')
    
    # Confidence Map & Binary Mask Options
    parser.add_argument("--save_confidence", action='store_true',
                      help="Save confidence maps")
    parser.add_argument("--save_binary", action='store_true',
                      help="Save binary masks")
    parser.add_argument("--binary_threshold", type=int, default=200,
                      help="Threshold for binarizing confidence map")
    parser.add_argument("--pred_threshold", type=float, default=0.5,
                      help="Threshold for predicting foreground (default: 0.5)")
    
    # 新增: 內波處理選項
    parser.add_argument("--internal_wave_area_threshold", type=float, default=0.01,
                      help="Minimum foreground area ratio to consider image having internal waves")
    parser.add_argument("--synthetic_broken_prob", type=float, default=0.8,
                      help="Probability to generate synthetic broken areas for no-wave images")
    parser.add_argument("--synthetic_broken_ratio", type=float, default=0.05,
                      help="Ratio of image area to generate as synthetic broken areas")
    parser.add_argument("--enable_wave_processing", action='store_true',
                      help="Enable internal wave specific processing")
    
    parser.add_argument("--min_broken_prob", type=float, default=0.2,
                      help="Minimum foreground probability to consider pixel as broken area (default: 0.2)")
    parser.add_argument("--max_broken_prob", type=float, default=0.7,
                      help="Maximum foreground probability to consider pixel as broken area (default: 0.7)")
    
    return parser

def get_model(model_name, num_classes, output_stride=8):
    if model_name == 'deeplabv3plus_resnet50':
        model = deeplabv3plus_resnet50(
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=True
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def load_model(model, ckpt_path, device):
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        print(f"Model loaded from {ckpt_path}")
    else:
        print("[!] No checkpoint found")
        model = nn.DataParallel(model)
        model.to(device)
    return model

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def has_internal_wave(pred_mask, area_threshold=0.01):
    """
    檢測圖像是否含有內波，基於前景區域占整體的比例
    
    參數:
    - pred_mask: 二值預測遮罩 (0表示背景, 255表示前景)
    - area_threshold: 前景佔比閾值，超過此閾值視為有內波
    
    返回:
    - bool: 是否含有內波
    """
    # 確保處理的是numpy陣列
    if isinstance(pred_mask, Image.Image):
        pred_mask = np.array(pred_mask)
    
    # 如果是彩色圖，轉為灰度圖
    if pred_mask.ndim == 3 and pred_mask.shape[2] == 3:
        # 如果是RGB格式的預測圖，提取白色部分（通常是前景）
        foreground = np.all(pred_mask == [255, 255, 255], axis=2)
    else:
        # 如果是灰度圖，直接二值化
        foreground = pred_mask > 127
    
    # 計算前景面積佔比
    foreground_ratio = np.sum(foreground) / foreground.size
    
    return foreground_ratio > area_threshold

def generate_synthetic_broken_areas(image_shape, style='linear', ratio=0.05):
    """
    為無內波圖片生成人工破碎區域
    
    參數:
    - image_shape: 圖像形狀 (高, 寬)
    - style: 生成風格 ('random_structures', 'linear', 'blob')
    - ratio: 破碎區域佔整體面積的比例
    
    返回:
    - broken_mask: 二值化的破碎區域遮罩 (255表示破碎區域，0表示完好區域)
    """
    height, width = image_shape[:2]
    total_pixels = height * width
    target_broken_pixels = int(total_pixels * ratio)
    
    # 初始化空白遮罩
    broken_mask = np.zeros((height, width), dtype=np.uint8)
    
    if style == 'random_structures':
        # 生成2-5個隨機形狀
        num_structures = random.randint(2, 5)
        pixels_per_structure = target_broken_pixels // num_structures
        
        for _ in range(num_structures):
            # 隨機中心點
            center_y = random.randint(0, height - 1)
            center_x = random.randint(0, width - 1)
            
            # 隨機尺寸 (長軸和短軸)
            major_axis = random.randint(10, int(min(height, width) * 0.3))
            minor_axis = random.randint(5, major_axis)
            
            # 隨機角度
            angle = random.uniform(0, 180)
            
            # 創建橢圓形區域
            y, x = np.ogrid[:height, :width]
            # 旋轉坐標系統
            cos_angle = np.cos(np.radians(angle))
            sin_angle = np.sin(np.radians(angle))
            xc = x - center_x
            yc = y - center_y
            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle
            
            # 橢圓方程式
            ellipse = ((xct**2) / (major_axis**2) + (yct**2) / (minor_axis**2)) <= 1
            
            # 應用到遮罩
            broken_mask[ellipse] = 255
    
    elif style == 'linear':
        # 垂直線性結構 (模擬垂直內波的形狀)
        num_lines = random.randint(1, 3)
        thickness = random.randint(3, 15)
        
        for _ in range(num_lines):
            # 垂直線的起點和終點 (固定x坐標，調整y坐標)
            x_level = random.randint(int(width * 0.3), int(width * 0.7))
            start_y = random.randint(0, int(height * 0.3))
            end_y = random.randint(int(height * 0.7), height - 1)
            
            # 生成垂直曲線 (用B樣條或簡單的正弦波)
            y_points = np.linspace(start_y, end_y, 100)
            amplitude = random.uniform(5, 20)
            frequency = random.uniform(0.1, 0.5)
            x_points = x_level + amplitude * np.sin(frequency * np.pi * np.linspace(0, 1, 100))
            
            # 繪製曲線
            for i in range(len(y_points) - 1):
                x1, y1 = int(x_points[i]), int(y_points[i])
                x2, y2 = int(x_points[i+1]), int(y_points[i+1])
                if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                    cv2.line(broken_mask, (x1, y1), (x2, y2), 255, thickness)
                    
    # 確保破碎區域總面積達到目標
    current_broken_pixels = np.sum(broken_mask > 0)
    if current_broken_pixels < target_broken_pixels:
        remaining = target_broken_pixels - current_broken_pixels
        # 隨機添加一些小噪點
        y_indices = np.random.randint(0, height, size=remaining)
        x_indices = np.random.randint(0, width, size=remaining)
        broken_mask[y_indices, x_indices] = 255
    
    return broken_mask

def binarize_confidence_map(confidence_map, threshold, pred_mask=None, enable_wave_processing=False, 
                            internal_wave_area_threshold=0.01, synthetic_prob=0.8, synthetic_ratio=0.05, 
                            force_style=None, min_prob=0.2, max_prob=0.7):
    """
    將信心圖（前景機率圖）轉換為二值化的待修復區域遮罩
    只標記前景機率在 min_prob 到 max_prob 之間的區域為待修復區域
    
    參數:
    - confidence_map: 前景機率圖 (0-255，值越高表示前景機率越高)
    - threshold: 原本的二值化閾值 (現在不直接使用)
    - min_prob: 最小前景機率閾值 (0-1)，低於此值不標記為破碎區域
    - max_prob: 最大前景機率閾值 (0-1)，高於此值不標記為破碎區域
    """
    confidence_np = np.array(confidence_map)
    
    # 將像素值範圍從 0-255 縮放到 0-1
    confidence_prob = confidence_np / 255.0
    
    # 只標記前景機率在 min_prob 到 max_prob 之間的區域為待修復（白色255）
    binary_mask = np.zeros_like(confidence_np, dtype=np.uint8)
    binary_mask[(confidence_prob >= min_prob) & (confidence_prob <= max_prob)] = 255
    
    # 其餘內波處理部分不變
    if enable_wave_processing and pred_mask is not None:
        has_wave = has_internal_wave(pred_mask, area_threshold=internal_wave_area_threshold)
        
        if not has_wave:
            if random.random() < synthetic_prob:
                if force_style is not None:
                    style = force_style
                else:
                    style = random.choice(['random_structures', 'blob', 'linear'])
                
                print(f"生成人工破碎區域，使用風格: {style}")
                synthetic_broken = generate_synthetic_broken_areas(
                    confidence_np.shape, 
                    style=style, 
                    ratio=synthetic_ratio
                )
                
                binary_mask = np.maximum(binary_mask, synthetic_broken)
    
    return Image.fromarray(binary_mask)

def predict_mask(model, image_path, transform, device, decode_fn, threshold=0.2):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.softmax(logits, dim=1)
        
        # 獲取前景的機率
        prob_foreground = prob[:, 1, :, :]  # [1, H, W]
        
        # 輸出前景機率的統計信息，協助調試
        prob_foreground_np = prob_foreground.cpu().numpy()
        print(f"前景機率統計: 最小值={prob_foreground_np.min():.4f}, 最大值={prob_foreground_np.max():.4f}, 平均值={prob_foreground_np.mean():.4f}")
        print(f"低於預測閾值 {threshold} 的像素比例: {(prob_foreground_np < threshold).mean():.2%}")
        
        # 根據閾值生成預測遮罩（前景機率 > threshold 的為前景）
        pred = (prob_foreground > threshold).float()  # [1, H, W]
        
        # 使用前景機率作為信心度圖
        confidence = prob_foreground.unsqueeze(1)  # [1, 1, H, W]
    
    pred_mask = pred.squeeze(0).cpu().numpy()  # [H, W]
    pred_mask = decode_fn(pred_mask).astype('uint8')  # 轉為彩色標註圖
    
    if pred_mask.ndim == 4:
        pred_mask = pred_mask.squeeze(0)  # [1, H, W, 3] -> [H, W, 3]
    
    # 前景機率圖（高機率為白色，低機率為黑色）
    confidence_map = confidence.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
    confidence_map = (confidence_map * 255).astype(np.uint8)  # 轉換為 uint8 (0~255)
    
    return Image.fromarray(pred_mask), Image.fromarray(confidence_map)

def process_images(input_base_path, output_path, model, transform, device, decode_fn, save_confidence, save_binary, 
                  binary_threshold, pred_threshold=0.2, enable_wave_processing=False, internal_wave_area_threshold=0.01, 
                  synthetic_broken_prob=0.8, synthetic_broken_ratio=0.05, force_style='linear', 
                  min_prob=0.2, max_prob=0.7):
    """
    處理圖像並生成預測結果
    
    參數:
    - input_base_path: 輸入圖像目錄路徑
    - output_path: 輸出結果目錄路徑
    - model: 預測模型
    - transform: 圖像轉換函數
    - device: 計算設備
    - decode_fn: 解碼函數
    - save_confidence: 是否保存信心度圖
    - save_binary: 是否保存二值遮罩
    - binary_threshold: 二值化閾值
    - pred_threshold: 預測閾值
    - enable_wave_processing: 是否啟用內波特殊處理
    - internal_wave_area_threshold: 判斷含內波的面積閾值
    - synthetic_broken_prob: 為無內波圖生成人工破碎區域的機率
    - synthetic_broken_ratio: 人工破碎區域佔整體面積的比例
    - force_style: 強制使用的風格，默認為'linear'
    """
    os.makedirs(output_path, exist_ok=True)
    subdirs = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]
    
    print(f"\n找到 {len(subdirs)} 個子資料夾")
    total_images = sum(len([f for f in os.listdir(os.path.join(input_base_path, d))
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]) for d in subdirs)
    print(f"總共發現 {total_images} 張圖片")
    
    with tqdm(total=total_images, desc="總進度") as pbar:
        for subdir in subdirs:
            subdir_path = os.path.join(input_base_path, subdir)
            print(f"\n處理資料夾: {subdir}")
            
            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            print(f"在 {subdir} 中發現 {len(image_files)} 張圖片")
            
            subdir_output = os.path.join(output_path, subdir)
            os.makedirs(subdir_output, exist_ok=True)
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(subdir_path, img_file)
                    base_name = os.path.splitext(img_file)[0]

                    pred_mask, confidence_map = predict_mask(model, img_path, transform, device, decode_fn, threshold=pred_threshold)

                    pred_mask.save(os.path.join(subdir_output, f'{base_name}_predict.png'))
                    
                    if save_confidence:
                        confidence_map.save(os.path.join(subdir_output, f'{base_name}_confidence.png'))
                    
                    if save_binary:
                        binary_mask = binarize_confidence_map(
                            confidence_map, 
                            binary_threshold, 
                            pred_mask=pred_mask,
                            enable_wave_processing=enable_wave_processing,
                            internal_wave_area_threshold=internal_wave_area_threshold,
                            synthetic_prob=synthetic_broken_prob,
                            synthetic_ratio=synthetic_broken_ratio,
                            force_style=force_style,  # 強制使用指定風格
                            min_prob=min_prob,
                            max_prob=max_prob
                        )
                        binary_mask.save(os.path.join(subdir_output, f'{base_name}_binary_mask.png'))
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"\n處理 {img_path} 時發生錯誤: {str(e)}")
            
            print(f"完成資料夾 {subdir} 的處理")
    
    print(f"\n預測完成，結果保存在: {output_path}")

def main():
    opts = get_argparser().parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if opts.dataset.lower() == 'binary':
        num_classes = 2
        decode_fn = BinarySegmentation.decode_target
    
    model = get_model(opts.model, num_classes, opts.output_stride)
    model = load_model(model, opts.ckpt, device)
    model.eval()
    
    transform = get_transform()
    
    # 注意: 添加了新的參數
    process_images(
        opts.input,
        opts.save_val_results_to,
        model,
        transform,
        device,
        decode_fn,
        opts.save_confidence,
        opts.save_binary,
        opts.binary_threshold,
        pred_threshold=opts.pred_threshold,
        enable_wave_processing=opts.enable_wave_processing,
        internal_wave_area_threshold=opts.internal_wave_area_threshold,
        synthetic_broken_prob=opts.synthetic_broken_prob,
        synthetic_broken_ratio=opts.synthetic_broken_ratio,
        min_prob=opts.min_broken_prob,
        max_prob=opts.max_broken_prob
    )

if __name__ == '__main__':
    main()
