#!/usr/bin/env python3
"""
TensorRT vs ONNX 快速診斷腳本
用於對比 ONNX 和 TensorRT 模型的輸出差異
"""

import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
from PIL import Image
import torchvision.transforms as T

class TensorRTInfer:
    def __init__(self, engine_path, input_shape, output_shape):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # 🔥 關鍵修復：明確設定輸入形狀
        input_name = self.engine.get_binding_name(0)
        self.context.set_binding_shape(0, input_shape)
        
        # 驗證形狀設定是否成功
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all input dimensions specified")
        
        # 重新計算輸出形狀（因為可能是動態的）
        output_shape_actual = self.context.get_binding_shape(1)
        print(f"🔧 TensorRT 引擎信息:")
        print(f"   輸入形狀: {input_shape}")
        print(f"   期望輸出形狀: {output_shape}")
        print(f"   實際輸出形狀: {output_shape_actual}")
        
        # 分配記憶體
        input_volume = trt.volume(input_shape)
        output_volume = trt.volume(output_shape_actual)
        
        self.h_input = cuda.pagelocked_empty(input_volume, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_volume, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        
        # 更新實際輸出形狀
        self.output_shape = output_shape_actual

    def infer(self, image_tensor):
        """
        執行 TensorRT 推論
        
        Args:
            image_tensor: PyTorch tensor, shape=(1, 3, H, W)
        
        Returns:
            numpy array: 推論結果
        """
        input_np = image_tensor.cpu().numpy().astype(np.float32)
        
        # 驗證輸入形狀
        if input_np.shape != self.input_shape:
            raise ValueError(f"輸入形狀不匹配: 期望 {self.input_shape}, 實際 {input_np.shape}")
        
        print(f"🔍 [TRT] 輸入統計: min={input_np.min():.6f}, max={input_np.max():.6f}, mean={input_np.mean():.6f}")
        
        # 複製輸入數據
        np.copyto(self.h_input, input_np.ravel())
        
        # 執行推論
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT 推論執行失敗")
            
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        result = self.h_output.reshape(self.output_shape)
        print(f"🔍 [TRT] 輸出統計: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")
        
        return result
    
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def compare_models(onnx_path, engine_path, test_input_type="zeros", test_image_path=None):
    """
    對比 ONNX 和 TensorRT 模型的輸出
    
    Args:
        onnx_path: ONNX 模型路徑
        engine_path: TensorRT 引擎路徑
        test_input_type: 測試輸入類型 ("zeros", "ones", "random", "image")
        test_image_path: 如果 test_input_type="image"，指定圖片路徑
    """
    
    print("="*60)
    print("🔍 TensorRT vs ONNX 診斷開始")
    print("="*60)
    
    # 1. 準備測試輸入
    if test_input_type == "zeros":
        test_input = torch.zeros(1, 3, 200, 200)
        print("📝 使用全零張量作為測試輸入")
    elif test_input_type == "ones":
        test_input = torch.ones(1, 3, 200, 200)
        print("📝 使用全一張量作為測試輸入")
    elif test_input_type == "random":
        test_input = torch.randn(1, 3, 200, 200)
        print("📝 使用隨機張量作為測試輸入")
    elif test_input_type == "image" and test_image_path:
        transform = get_transform()
        image = Image.open(test_image_path).convert('RGB').resize((200, 200))
        test_input = transform(image).unsqueeze(0)
        print(f"📝 使用真實圖片作為測試輸入: {test_image_path}")
    else:
        raise ValueError("Invalid test_input_type or missing test_image_path")
    
    print(f"📊 輸入張量統計: shape={test_input.shape}, min={test_input.min():.6f}, max={test_input.max():.6f}")
    
    # 2. ONNX 推論
    print("\n🔄 執行 ONNX 推論...")
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(None, {input_name: test_input.numpy()})
        onnx_logits = onnx_output[0]
        print(f"✅ ONNX 推論成功")
        print(f"📊 ONNX 原始輸出 (logits): shape={onnx_logits.shape}")
        print(f"   Channel 0 stats: min={onnx_logits[0,0].min():.6f}, max={onnx_logits[0,0].max():.6f}, mean={onnx_logits[0,0].mean():.6f}")
        print(f"   Channel 1 stats: min={onnx_logits[0,1].min():.6f}, max={onnx_logits[0,1].max():.6f}, mean={onnx_logits[0,1].mean():.6f}")
        
        # 計算 softmax
        exp_logits = np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True))
        onnx_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        onnx_fg_prob = onnx_probs[0, 1]
        print(f"📊 ONNX 前景機率: min={onnx_fg_prob.min():.6f}, max={onnx_fg_prob.max():.6f}, mean={onnx_fg_prob.mean():.6f}")
        
    except Exception as e:
        print(f"❌ ONNX 推論失敗: {e}")
        return
    
    # 3. TensorRT 推論
    print("\n🔄 執行 TensorRT 推論...")
    try:
        trt_model = TensorRTInfer(engine_path, (1, 3, 200, 200), (1, 2, 200, 200))
        trt_output = trt_model.infer(test_input)
        print(f"✅ TensorRT 推論成功")
        print(f"📊 TensorRT 原始輸出 (logits): shape={trt_output.shape}")
        print(f"   Channel 0 stats: min={trt_output[0,0].min():.6f}, max={trt_output[0,0].max():.6f}, mean={trt_output[0,0].mean():.6f}")
        print(f"   Channel 1 stats: min={trt_output[0,1].min():.6f}, max={trt_output[0,1].max():.6f}, mean={trt_output[0,1].mean():.6f}")
        
        # 計算 softmax
        exp_logits_trt = np.exp(trt_output - np.max(trt_output, axis=1, keepdims=True))
        trt_probs = exp_logits_trt / np.sum(exp_logits_trt, axis=1, keepdims=True)
        trt_fg_prob = trt_probs[0, 1]
        print(f"📊 TensorRT 前景機率: min={trt_fg_prob.min():.6f}, max={trt_fg_prob.max():.6f}, mean={trt_fg_prob.mean():.6f}")
        
    except Exception as e:
        print(f"❌ TensorRT 推論失敗: {e}")
        return
    
    # 4. 對比結果
    print("\n📋 對比結果")
    print("="*60)
    
    # 對比原始 logits
    logits_diff = np.abs(onnx_logits - trt_output)
    print(f"🔍 原始 logits 差異:")
    print(f"   最大絕對差異: {logits_diff.max():.8f}")
    print(f"   平均絕對差異: {logits_diff.mean():.8f}")
    print(f"   差異標準差: {logits_diff.std():.8f}")
    
    # 對比前景機率
    prob_diff = np.abs(onnx_fg_prob - trt_fg_prob)
    print(f"🔍 前景機率差異:")
    print(f"   最大絕對差異: {prob_diff.max():.8f}")
    print(f"   平均絕對差異: {prob_diff.mean():.8f}")
    
    # 判斷是否正常
    if logits_diff.max() < 1e-5:
        print("✅ 結果一致性: 極好 (差異 < 1e-5)")
    elif logits_diff.max() < 1e-3:
        print("⚠️  結果一致性: 可接受 (差異 < 1e-3)")
    elif logits_diff.max() < 1e-1:
        print("🚨 結果一致性: 有問題 (差異 < 1e-1)")
    else:
        print("💥 結果一致性: 嚴重錯誤 (差異 >= 1e-1)")
        
    # 5. 診斷建議
    print("\n💡 診斷建議:")
    if logits_diff.max() > 1e-3:
        print("   ⚠️  檢測到顯著差異，建議:")
        print("   1. 確認 TensorRT 引擎是否使用 FP16 (可能導致精度損失)")
        print("   2. 檢查輸入預處理是否完全一致")
        print("   3. 驗證 ONNX 導出時的 opset_version")
        print("   4. 嘗試重新構建 TensorRT 引擎 (使用 FP32)")
    else:
        print("   ✅ 模型轉換正確，差異在可接受範圍內")
        print("   如果仍有預測問題，請檢查:")
        print("   1. 閾值設定 (threshold)")
        print("   2. 後處理邏輯")
        print("   3. 評估指標計算")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="TensorRT vs ONNX 快速診斷工具")
    parser.add_argument("--onnx", required=True, help="ONNX 模型路徑")
    parser.add_argument("--engine", required=True, help="TensorRT 引擎路徑")
    parser.add_argument("--test_type", default="zeros", choices=["zeros", "ones", "random", "image"], 
                       help="測試輸入類型")
    parser.add_argument("--test_image", help="如果 test_type=image，指定測試圖片路徑")
    
    opts = parser.parse_args()
    
    compare_models(opts.onnx, opts.engine, opts.test_type, opts.test_image)

if __name__ == "__main__":
    main()