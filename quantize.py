

import torch
import torch.quantization
import os
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import copy

# Reuse model definition from existing scripts
from src.network.modeling import deeplabv3plus_resnet50

def get_argparser():
    parser = argparse.ArgumentParser(description="Post-Training Quantization Script")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the trained FP32 model checkpoint (.pth)")
    parser.add_argument("--calibration_data", required=True, type=str, help="Path to the directory containing calibration images")
    parser.add_argument("--output_model", required=True, type=str, help="Path to save the quantized INT8 model")
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50', help='Model name')
    parser.add_argument("--dataset", type=str, default='binary', choices=['binary'], help='Name of dataset')
    parser.add_argument("--output_stride", type=int, default=16, help='Output stride for DeepLabV3+')
    return parser

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    opts = get_argparser().parse_args()
    # Quantization is performed on the CPU
    device = torch.device('cpu')

    # 1. Load the FP32 model
    print("Loading FP32 model...")
    if opts.dataset.lower() == 'binary':
        num_classes = 2
    
    # Create model structure
    model_fp32 = deeplabv3plus_resnet50(num_classes=num_classes, output_stride=opts.output_stride, pretrained_backbone=False)
    
    # Load trained weights
    print(f"Loading checkpoint from {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=device)
    # Handle potential 'module.' prefix from DataParallel
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
    model_fp32.load_state_dict(new_state_dict)
    model_fp32.eval()
    print("FP32 model loaded successfully.")

    # It's crucial to use a deepcopy to avoid modifying the original model
    model_to_quantize = copy.deepcopy(model_fp32)

    # 2. Prepare the model for quantization
    print("\nPreparing model for quantization...")
    # For your target Jetson device (ARM architecture), 'qnnpack' is the correct backend.
    qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model_to_quantize.qconfig = qconfig
    
    # Insert observers to collect statistics
    print("Inserting observers...")
    model_prepared = torch.quantization.prepare(model_to_quantize)
    print("Observers inserted.")

    # 3. Calibrate the model
    print("\nStarting calibration...")
    transform = get_transform()
    calibration_files = [os.path.join(opts.calibration_data, f) for f in os.listdir(opts.calibration_data) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not calibration_files:
        raise ValueError(f"No images found in the calibration directory: {opts.calibration_data}")

    print(f"Found {len(calibration_files)} images for calibration.")
    
    with torch.no_grad():
        for img_path in tqdm(calibration_files, desc="Calibrating"):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            model_prepared(img_tensor)
    print("Calibration finished.")

    # 4. Convert the model to INT8
    print("\nConverting model to INT8...")
    model_int8 = torch.quantization.convert(model_prepared)
    print("Model converted to INT8 successfully.")

    # 5. Save the quantized model
    print(f"\nSaving quantized model to {opts.output_model}...")
    # Standard torch.save is sufficient for quantized models
    torch.save(model_int8.state_dict(), opts.output_model)
    print("Quantized model saved successfully.")

if __name__ == '__main__':
    main()
