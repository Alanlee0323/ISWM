import torch
import os
import argparse
from src.network.modeling import deeplabv3plus_resnet50

def get_argparser():
    """
    新增一個函數來處理 ONNX 導出的參數。
    """
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format for TensorRT")
    
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
    parser.add_argument("--input_height", type=int, default=200,
                        help="The height of the dummy input tensor for ONNX export.")
    parser.add_argument("--input_width", type=int, default=200,
                        help="The width of the dummy input tensor for ONNX export.")

    return parser

def main():
    """
    主函數，執行模型載入、轉換和儲存。
    """
    opts = get_argparser().parse_args()
    
    # 1. 設備選擇 (建議: 強制使用 CPU 導出以獲得最佳相容性)
    device = torch.device('cpu')
    print(f"Using device for export: {device}")

    # 2. 建立模型架構
    print(f"Loading model: {opts.model}")
    model = deeplabv3plus_resnet50(
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
        pretrained_backbone=False # 載入自己的權重時，這裡設為 False 即可
    )

    # 3. 載入訓練好的權重
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found at {opts.ckpt}")
        
    print(f"Loading checkpoint from: {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=device)
    
    # 處理 'module.' 前綴
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(state_dict)
    
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
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            # 【建議】啟用常數摺疊，對 TensorRT 最佳化有益
            do_constant_folding=True,
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print("\n✅ ONNX model exported successfully!")
        print(f"You can now use the file: {opts.output_file}")

    except Exception as e:
        print(f"\n❌ An error occurred during ONNX export: {e}")

if __name__ == '__main__':
    main()