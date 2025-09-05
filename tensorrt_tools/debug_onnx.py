import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
import argparse

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Debugging Script")
    parser.add_argument("--onnx_model", required=True, type=str, help="Path to the ONNX model file.")
    parser.add_argument("--image", required=True, type=str, help="Path to a single image for testing.")
    opts = parser.parse_args()

    print(f"Loading ONNX model: {opts.onnx_model}")
    # 設置 ONNX Runtime 的 session，指定使用 CUDA
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(opts.onnx_model, providers=providers)
    
    input_name = session.get_inputs()[0].name
    print(f"Model Input Name: {input_name}")

    print(f"Loading and preprocessing image: {opts.image}")
    transform = get_transform()
    img = Image.open(opts.image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    input_np = img_tensor.cpu().numpy()

    print(f"Running inference with ONNX Runtime...")
    result = session.run(None, {input_name: input_np})
    
    # 取得原始輸出 (logits)
    logits_np = result[0]
    
    # 執行 Softmax 來取得機率
    # 手動在 NumPy 中實現 Softmax
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    pred_foreground_prob = probs[:, 1, :, :]

    # 印出除錯資訊
    max_prob = np.max(pred_foreground_prob)
    min_prob = np.min(pred_foreground_prob)
    mean_prob = np.mean(pred_foreground_prob)
    
    print("\n" + "="*20 + " ONNX DEBUG REPORT " + "="*20)
    print(f"[ONNX DEBUG] For image '{opts.image}':")
    print(f"[ONNX DEBUG] Max Prob={max_prob:.6f}, Min Prob={min_prob:.6f}, Mean Prob={mean_prob:.6f}")
    print("="*63)


if __name__ == '__main__':
    main()