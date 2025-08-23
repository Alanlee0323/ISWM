
import torch
import os
import argparse
from network.modeling import deeplabv3plus_resnet50

def get_argparser():
    """
    新增一個函數來處理 ONNX 導出的參數。
    """
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    
    parser.add_argument("--ckpt", type=str, required=True,
                      help="Path to the PyTorch model checkpoint (.pth file)")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Path to save the output ONNX file")
    parser.add_argument('--model', type=str, default='deeplabv3plus_resnet50',
                      help='Model name')
    parser.add_argument("--num_classes", type=int, default=2,
                      help="Number of classes in the model")
    parser.add_argument("--output_stride", type=int, default=16,
                      help='Output stride for DeepLabV3+ (8 or 16)')
    parser.add_argument("--input_height", type=int, default=513,
                      help="The height of the dummy input tensor for ONNX export.")
    parser.add_argument("--input_width", type=int, default=513,
                      help="The width of the dummy input tensor for ONNX export.")

    return parser

def main():
    """
    主函數，執行模型載入、轉換和儲存。
    """
    opts = get_argparser().parse_args()
    
    # 1. 設備選擇
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 建立模型架構
    print(f"Loading model: {opts.model}")
    model = deeplabv3plus_resnet50(
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
        pretrained_backbone=False  # 在這裡設為 False，因為我們要載入自己的權重
    )

    # 3. 載入訓練好的權重
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found at {opts.ckpt}")
        
    print(f"Loading checkpoint from: {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=device)
    
    # 處理 'module.' 前綴 (通常在 DataParallel 訓練後出現)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    
    # 4. 設定為評估模式
    model.eval()
    print("Model set to evaluation mode.")

    # 5. 建立一個 dummy (虛擬) 輸入
    dummy_input = torch.randn(1, 3, opts.input_height, opts.input_width, device=device)
    print(f"Created a dummy input tensor of shape: {dummy_input.shape}")

    # 6. 導出為 ONNX
    print(f"Exporting model to ONNX format at: {opts.output_file}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            opts.output_file,
            verbose=False,
            input_names=['input'],   # 輸入層的名稱
            output_names=['output'], # 輸出層的名稱
            opset_version=11,        # ONNX 的版本
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'}, # 讓 batch, height, width 可以是動態的
                'output': {0: 'batch_size', 2: 'height', 3: 'width'} # 讓輸出的維度也跟著動態調整
            }
        )
        print("\nONNX model exported successfully!")
        print(f"You can now use the file: {opts.output_file}")

    except Exception as e:
        print(f"\nAn error occurred during ONNX export: {e}")

if __name__ == '__main__':
    main()
