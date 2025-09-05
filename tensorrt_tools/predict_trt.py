import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import argparse
import time
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
import sys

# --- 解決 GDK 圖形介面錯誤 ---
import matplotlib
matplotlib.use('Agg')
# --------------------------------

# --- 修正 Import 路徑 ---
# 假設您的 StreamMetrics 類別在您專案根目錄下的 metrics.py 中
# 如果不是，請修改成正確的路徑，例如 from datasets.metrics import StreamMetrics
try:
    from metrics import StreamMetrics
except ImportError:
    print("="*50)
    print("錯誤：找不到 'StreamMetrics' 類別！")
    print("請確認您的專案根目錄 (ISWM) 下有一個 metrics.py 檔案，")
    print("或者修改 evaluate_trt.py 中的 import 路徑。")
    print("="*50)
    exit()
# --------------------------------

logger = logging.getLogger(__name__)

# --- 直接將 EvaluationDataset 類別複製進來，不再依賴外部檔案 ---
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
        base_name, extension = os.path.splitext(img_name)
        mask_name = f"{base_name}_mask{extension}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            image = Image.open(img_path).convert('RGB')
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert('L')
                mask = np.array(mask_pil)
                mask[mask > 0] = 1
            else:
                logger.warning(f"Mask not found for {img_name}, creating an empty mask.")
                mask = np.zeros(image.size[::-1], dtype=np.uint8)
            return self.transform(image), torch.from_numpy(mask).long(), img_name, np.array(image)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error(f"Error reading data for {img_name}: {e}")
            raise e

# --- TensorRT 推論器類別 (最終版) ---
class TensorRTInfer:
    def __init__(self, engine_path, input_shape, output_shape):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # 關鍵修復：明確設定輸入形狀
        self.context.set_input_shape(self.engine.get_tensor_name(0), input_shape)
        
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

def get_model_size_mb(path):
    if not os.path.exists(path): return 0
    return os.path.getsize(path) / (1024 * 1024)

# --- 主函式 ---
def main():
    parser = argparse.ArgumentParser(description="TensorRT Model Evaluation Script")
    parser.add_argument("--engine", required=True, type=str, help="Path to the TensorRT engine file.")
    parser.add_argument("--eval_data_dir", required=True, type=str, help="Path to the validation data directory (should contain 'imgs' and 'masks' subfolders)")
    parser.add_argument("--num_images", type=int, default=0, help="Number of images to evaluate on. Default: 0 (all images)")
    parser.add_argument("--pred_threshold", type=float, default=0.5, help="Threshold for predicting foreground.")
    opts = parser.parse_args()

    # 在執行推論前，先將專案根目錄加入 Python 路徑
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}")

    print(f"Loading TensorRT engine from: {opts.engine}")
    input_shape = (1, 3, 200, 200)
    output_shape = (1, 2, 200, 200)
    trt_model = TensorRTInfer(
    "checkpoints/deeplabv3plus_resnet50_fp32_fixed.engine", 
    (1, 3, 200, 200), 
    (1, 2, 200, 200)
)
    print("Engine loaded successfully.")

    transform = get_transform()
    image_dir = os.path.join(opts.eval_data_dir, 'imgs')
    mask_dir = os.path.join(opts.eval_data_dir, 'masks')
    eval_dataset = EvaluationDataset(image_dir, mask_dir, transform, opts.num_images)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    print(f"Found {len(eval_dataset)} images for evaluation.")

    metrics = StreamMetrics(2)
    inference_times = []

    for img_tensor, gt_mask, _, _ in tqdm(eval_loader, desc="Evaluating with TensorRT"):
        start_time = time.perf_counter()
        logits_np = trt_model.infer(img_tensor)
        inference_times.append(time.perf_counter() - start_time)
        
        logits = torch.from_numpy(logits_np).to(device)
        probs = torch.softmax(logits, dim=1)
        pred_foreground_prob = probs[:, 1, :, :]
        # --- 在這裡插入以下除錯程式碼 ---
        max_prob = torch.max(pred_foreground_prob).item()
        min_prob = torch.min(pred_foreground_prob).item()
        mean_prob = torch.mean(pred_foreground_prob).item()
        print(f"\n[DEBUG] Image Stats: Max Prob={max_prob:.6f}, Min Prob={min_prob:.6f}, Mean Prob={mean_prob:.6f}")
# --- 插入結束 ---

        pred_mask = (pred_foreground_prob > opts.pred_threshold).long()
        
        metrics.update(gt_mask.numpy(), pred_mask.cpu().numpy())

    engine_size = get_model_size_mb(opts.engine)
    scores = metrics.get_results()
    avg_latency_ms = np.mean(inference_times[1:]) * 1000 if len(inference_times) > 1 else np.mean(inference_times) * 1000
    throughput_qps = 1000 / avg_latency_ms if avg_latency_ms > 0 else float('inf')

    print("\n\n" + "="*25 + " TensorRT Evaluation Report " + "="*25)
    print(f"Evaluated on {len(eval_dataset)} images.\n")
    print(f"{'Metric':<30} | {'Value':<20}")
    print("-" * 55)
    print(f"{'Engine File Size (MB)':<30} | {engine_size:<20.2f}")
    print(f"{'Avg. Latency (ms)':<30} | {avg_latency_ms:<20.2f}")
    print(f"{'Throughput (images/sec)':<30} | {throughput_qps:<20.2f}")
    print("-" * 55)
    print(f"{'Mean IoU (mIoU)':<30} | {scores.get('MIoU', 0.0):<20.4f}")
    print(f"{'Foreground IoU':<30} | {scores.get('Foreground IoU', 0.0):<20.4f}")
    print(f"{'Foreground F1':<30} | {scores.get('Foreground F1', 0.0):<20.4f}")
    print("=" * 77)

if __name__ == '__main__':
    main()