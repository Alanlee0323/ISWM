from tqdm import tqdm
import network
import utils
import time
import mlflow
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import random
import argparse
import numpy as np
import seaborn as sns
import sys
from torch.utils import data
from datasets import BinarySegmentation
from utils import ext_transforms as et
from metrics import StreamMetrics
import torch
import torch.nn as nn
from datetime import datetime
from PIL import Image
import shutil
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F

OPTIMIZER_CONFIGS = {
    'adamw': {
        'lr': 1e-4,
        'weight_decay': 0.01,
        'backbone_weight_decay': 0.02,  # backbone使用更大的權重衰減
        'scheduler': 'cosine',
        'warmup_epochs': 0.1,  # 10%預熱
        'min_lr_factor': 0.01  # 最小學習率因子
    },


    'adam': {
        'lr': 1e-4,
        'scheduler': 'onecycle',
        'pct_start': 0.3
    },

    'sgd': {
        'lr': 1e-5,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'scheduler': 'poly',
        'power': 0.9
    }
}

def setup_mlflow(opts):
    """設置 MLflow 的集中管理"""
    # 設置 MLflow 追蹤 URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # 生成更具描述性的實驗名稱
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"DeepLabV3Plus_{opts.model}_{opts.loss_type}_{opts.optimizer}_os{opts.output_stride}_{timestamp}"
    
    try:
        # 檢查實驗是否已存在
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # 創建新實驗並獲取 ID
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # 明確設置當前實驗
        mlflow.set_experiment(experiment_name)
        
        print(f"\nMLflow 設置完成:")
        print(f"- 實驗名稱: {experiment_name}")
        print(f"- 實驗 ID: {experiment_id}")
        print(f"- 追蹤 URI: {mlflow.get_tracking_uri()}")
        
        return experiment_name, experiment_id  # 返回兩個值
        
    except Exception as e:
        print(f"Error setting up MLflow: {str(e)}")
        raise e

def print_weights_config(weights):
    print("\nMetrics Weights Configuration:")
    print("=" * 40)
    for metric, weight in weights.items():
        print(f"{metric:.<30} {weight:.3f}")
    print("=" * 40 + "\n")

class MetricsLogger:
    def __init__(self, save_dir, weights=None):
        self.metrics = {
            'train_loss': [],
            'val_miou': [],
            'val_foreground_iou': [],
            'val_foreground_f1': [],
            'val_temporal_consistency': [],
            'val_front_tracking_error': [],
            'val_region_continuity': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': []
        }
        
        # 默認權重
        self.weights = weights if weights is not None else {
            "MIoU": 0.05,
            "Foreground IoU": 0.25,
            "Foreground F1": 0.25,
            "Front Tracking Error": 0.25,
            "Temporal Consistency": 0.10,
            "Region Continuity": 0.10
        }
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, metric_name, value):
        """更新指標值"""
        if metric_name in self.metrics:
            if isinstance(value, (np.ndarray, torch.Tensor)):
                value = float(value)  # 確保是標量
            self.metrics[metric_name].append(value)

    def get_weighted_score(self):
        """計算加權總分"""
        latest_metrics = self.get_latest_metrics()
        weighted_score = 0.0
        print("\nCalculating weighted score:")

        base_metrics = [
            ('Foreground IoU', 'val_foreground_iou'),
            ('Foreground F1', 'val_foreground_f1'),
            ('Region Continuity', 'val_region_continuity')
        ]
        
        # 處理正向指標
        for metric_name, key in base_metrics:
            if latest_metrics[key] is not None:
                value = float(latest_metrics[key])
                if not np.isnan(value):
                    contribution = self.weights[metric_name] * value
                    weighted_score += contribution
                    print(f"{metric_name}: {value:.4f} * {self.weights[metric_name]:.4f} = {contribution:.4f}")
    
        # 前緣追蹤誤差特殊處理
        if latest_metrics['val_front_tracking_error'] is not None:
            error = float(latest_metrics['val_front_tracking_error'])
            max_error = 10.0
            error_score = max(0, 1 - (error / max_error))
            contribution = abs(self.weights['Front Tracking Error']) * error_score
            weighted_score += contribution
            print(f"Front Tracking Error: {error:.4f} -> {error_score:.4f} * {abs(self.weights['Front Tracking Error']):.4f} = {contribution:.4f}")
    
        # 時序一致性特殊處理
        temporal_key = 'val_temporal_consistency'
        if latest_metrics[temporal_key] is not None and not np.isnan(latest_metrics[temporal_key]):
            value = float(latest_metrics[temporal_key])
            contribution = self.weights['Temporal Consistency'] * value
            weighted_score += contribution
            print(f"Temporal Consistency: {value:.4f} * {self.weights['Temporal Consistency']:.4f} = {contribution:.4f}")
    
        print(f"Final weighted score: {weighted_score:.4f}")
        return weighted_score
    
    def save_plots(self):
        """生成並保存所有圖表，包括權重貢獻"""
        # 原有的圖表保存邏輯
        self._save_training_loss_plot()
        self._save_validation_metrics_plot()
        self._save_learning_rate_plot()
        

    def _save_training_loss_plot(self):
        """保存訓練損失圖"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'training_loss.png'))
        plt.close()

    def _save_validation_metrics_plot(self):
        """保存驗證指標圖"""
        plt.figure(figsize=(12, 6))
        metrics_to_plot = [
            ('val_miou', 'MIoU'),
            ('val_foreground_iou', 'Foreground IoU'),
            ('val_foreground_f1', 'Foreground F1'),
            ('val_temporal_consistency', 'Temporal Consistency'),
            ('val_front_tracking_error', 'Front Tracking Error'),
            ('val_region_continuity', 'Region Continuity'),
            ('val_precision', 'Precision'),
            ('val_recall', 'Recall')
        ]
        
        for metric_name, label in metrics_to_plot:
            if self.metrics[metric_name]:
                weight = self.weights.get(label, 0.0)  # 使用 get 方法防止 KeyError
                if weight != 0.0:  # 只顯示有權重的指標的權重
                    plt.plot(self.metrics[metric_name], 
                            label=f'{label} (w={weight:.2f})')
                else:
                    plt.plot(self.metrics[metric_name], label=label)
        
        plt.title('Validation Metrics Over Time (with weights)')
        plt.xlabel('Validation Steps')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'validation_metrics.png'))
        plt.close()

    def _save_learning_rate_plot(self):
        if not self.metrics['learning_rate']:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['learning_rate'])
        plt.title('Learning Rate Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'learning_rate.png'))
        plt.close()


    def get_latest_metrics(self):
        """獲取最新的指標值"""
        return {
            metric_name: values[-1] if values else None 
            for metric_name, values in self.metrics.items()
        }

    def get_best_metrics(self):
        """獲取最佳指標值"""
        best_metrics = {
            'best_miou': max(self.metrics['val_miou']) if self.metrics['val_miou'] else None,  # 新增 MIoU
            'best_foreground_iou': max(self.metrics['val_foreground_iou']) if self.metrics['val_foreground_iou'] else None,
            'best_foreground_f1': max(self.metrics['val_foreground_f1']) if self.metrics['val_foreground_f1'] else None,
            'best_temporal_consistency': max(self.metrics['val_temporal_consistency']) if self.metrics['val_temporal_consistency'] else None,
            'best_front_tracking_error': min(self.metrics['val_front_tracking_error']) if self.metrics['val_front_tracking_error'] else None,
            'best_region_continuity': max(self.metrics['val_region_continuity']) if self.metrics['val_region_continuity'] else None,
            'best_precision': max(self.metrics['val_precision']) if self.metrics['val_precision'] else None,
            'best_recall': max(self.metrics['val_recall']) if self.metrics['val_recall'] else None,
            'best_weighted_score': self.get_weighted_score()
        }
        return best_metrics

    def save_confusion_matrix(self, confusion_matrix, iteration):
        """保存混淆矩陣"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, 
                    annot=True, 
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=['Background', 'Wave'],
                    yticklabels=['Background', 'Wave'])
        plt.title(f'Confusion Matrix (Iteration {iteration})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_{iteration}.png'))
        plt.close()


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='binary',
                        choices=['binary'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--optimizer",type=str, default='adamw',choices=['sgd', 'adam', 'adamw'], help="Type of optimizer to use")
    
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./Original_results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['ce_loss', 'IWce_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=500,
                        help="epoch interval for eval (default: 500)")

    #val option
    parser.add_argument("--checkpoints_dir", type=str, default='checkpoints',
                        help="directory for saving checkpoints")
    parser.add_argument("--val_results_dir", type=str, default='val_results',
                        help="directory for saving validation results")
    parser.add_argument("--metrics_plots_dir", type=str, default='metrics_plots',
                        help="directory for saving metrics plots")
    parser.add_argument("--save_confidence_map", action='store_true', default=False,
                        help="save confidence maps along with predictions")
    parser.add_argument("--sequence_length", type=int, default=7,
                        help="Sequence length for temporal metrics (default: 7)")
    
    #控制特徵圖可視化
    parser.add_argument('--save_feature_maps', action='store_true', default=False
                        , help='save feature maps visualization')
    parser.add_argument('--feature_maps_dir', type=str, help='path to save feature maps')
    
    parser.add_argument("--training_stage", type=str, default='spatial',
                    choices=['spatial', 'temporal_p1', 'temporal_p2', 'temporal_p3', 'temporal_p4'],
                    help="Training stage")

    
    return parser

def get_dataset(opts):
    # 定義轉換
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    # 原來的非時序數據集
    train_dst = BinarySegmentation(
        root=opts.data_root,
        split='train',
        transform=train_transform
    )
    val_dst = BinarySegmentation(
           root=opts.data_root,
        split='val',
        transform=val_transform
    )
    
    return train_dst, val_dst

"""
Black pixel (Background) : 0
White pixel (Internal wave) : 1
"""
def calculate_class_weights(loader):
    black_pixels = 0
    white_pixels = 0
    print("Calculating class weights...")
    
    for batch in tqdm(loader, desc="Calculating class weights"):
        if isinstance(batch, dict):
            labels = batch['mask']
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            _, labels = batch
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
            
        black_pixels += (labels == 0).sum().item()
        white_pixels += (labels == 1).sum().item()
    
    weight_black = 1.0
    weight_white = np.sqrt(black_pixels/white_pixels)
    
    print(f"Pixel distribution - Black: {black_pixels}, White: {white_pixels}")
    print(f"Class weights - Black: {weight_black}, White: {weight_white}")
    
    return torch.FloatTensor([weight_black, weight_white])

def setup_model(opts):
    """根據選項設置模型"""
    model = network.modeling.deeplabv3plus_resnet50(
        num_classes=opts.num_classes,
        output_stride=opts.output_stride
    )
    
    return model

def setup_optimizer(model, opts):
    """根據訓練階段和選擇的優化器類型設置優化器和學習率"""
    # 將模型所有參數傳入
    param_groups = model.parameters()
    
    if opts.optimizer == 'sgd':
        return torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=opts.weight_decay,
            nesterov=True
        )
    elif opts.optimizer == 'adam':
        return torch.optim.Adam(
            param_groups,
            weight_decay=opts.weight_decay
        )
    elif opts.optimizer == 'adamw':
        return torch.optim.AdamW(
            param_groups,
            weight_decay=opts.weight_decay
        )
    else:
        raise ValueError(f'Unsupported optimizer: {opts.optimizer}')
    
def setup_scheduler(optimizer, opts):
    """根據訓練階段設置學習率調度器"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opts.total_itrs,
        eta_min=opts.lr * 0.01
    )
    
def setup_criterion(opts, class_weights):
    """設置損失函數"""
    if opts.loss_type == 'ce_loss':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'IWce_loss':
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean')
    
def save_validation_results(model, val_loader, device, opts, cur_itrs, weighted_score, saved_images_count=0):
    is_temporal = getattr(opts, 'use_temporal', False)
    results_dir = os.path.join(opts.val_results_dir, f'best_model_iter_{cur_itrs}_score_{weighted_score:.4f}')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Saving validation results")):
            if is_temporal:
                if isinstance(batch, dict):
                    images = batch['images'].to(device)  # [B, T, C, H, W]
                    labels = batch['mask'].to(device)    # [B, H, W]
                    # 提取第一個樣本的最後一幀，保留通道維度
                    original_image = images[0, -1, :, :, :].cpu()  # [C, H, W]
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    original_image = images[0].cpu()  # [C, H, W]
                else:
                    raise ValueError(f"時序模型期望字典或 (images, labels) 格式，但收到 {type(batch)}")
            else:
                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    raise ValueError(f"單幀模型期望 (images, labels) 元組格式，但收到 {type(batch)}")
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                original_image = images[0].cpu()  # [C, H, W]

            original_image = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(original_image)
            original_image = (original_image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            logits = model(images)
            prob = F.softmax(logits, dim=1)
            confidence, pred = torch.max(prob, dim=1, keepdim=True)
            targets = labels.cpu().numpy()
            
            try:
                unique_id = f"{saved_images_count:04d}"
                Image.fromarray(original_image).save(os.path.join(results_dir, f'{unique_id}_original.png'))
                target = targets[0]
                pred_0 = pred[0].squeeze(0).cpu().numpy()
                conf_0 = confidence[0].squeeze(0).cpu().numpy()
                target_rgb = decode_target(target, opts.num_classes)
                pred_rgb = decode_target(pred_0, opts.num_classes)
                Image.fromarray(target_rgb).save(os.path.join(results_dir, f'{unique_id}_target.png'))
                Image.fromarray(pred_rgb).save(os.path.join(results_dir, f'{unique_id}_pred.png'))
                plt.figure(figsize=(10, 10))
                plt.imshow(original_image)
                plt.axis('off')
                plt.imshow(pred_rgb, alpha=0.7)
                plt.savefig(os.path.join(results_dir, f'{unique_id}_overlay.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
                if getattr(opts, 'save_confidence_map', False):
                    conf_vis = (conf_0 * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(conf_vis).save(os.path.join(results_dir, f'{unique_id}_confidence.png'))
                saved_images_count += 1
            except Exception as e:
                print(f"保存驗證圖像 {unique_id} 時出錯: {str(e)}")
                continue
    
    print(f"為最佳模型保存了 {saved_images_count} 張驗證圖像")
    return saved_images_count

def save_best_model(model, optimizer, scheduler, opts, val_score, weighted_score, cur_itrs, best_score):
    """保存最佳模型的checkpoint"""
    print(f"\nAttempting to save best model checkpoint:")
    print(f"Weighted score: {weighted_score}")
    print(f"Current iteration: {cur_itrs}")
    
    save_dir = opts.checkpoints_dir
    print(f"Save directory: {save_dir}")
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # 檢查目錄是否可寫
        if not os.access(save_dir, os.W_OK):
            print(f"Error: Directory {save_dir} is not writable")
            return None
            
        # 清理舊的checkpoint文件
        for old_file in os.listdir(save_dir):
            if old_file.startswith('best_') and old_file.endswith('.pth'):
                old_file_path = os.path.join(save_dir, old_file)
                try:
                    os.remove(old_file_path)
                    print(f"Removed old checkpoint: {old_file}")
                except Exception as e:
                    print(f"Warning: Could not remove old checkpoint {old_file}: {str(e)}")
        
        # 生成新的checkpoint文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, 
            f'best_{opts.model}_{opts.dataset}_os{opts.output_stride}_weighted{weighted_score:.3f}.pth')
        
        # 如果模型是DataParallel，獲取內部模型
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # 檢查硬碟空間
        import shutil
        total, used, free = shutil.disk_usage(os.path.dirname(save_dir))
        if free < 1e9:  # 如果剩餘空間小於1GB
            print(f"Warning: Low disk space. Only {free/1e9:.2f}GB available")
        
        # 準備要保存的數據
        checkpoint = {
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_score": val_score,
            "weighted_score": weighted_score,
            "cur_itrs": cur_itrs,
            "best_score": best_score,
            "save_time": timestamp,
            "model_config": {
            "model_name": opts.model,
            "dataset": opts.dataset,
            "output_stride": opts.output_stride,
            "num_classes": opts.num_classes,
            }
        }
        
        # 使用臨時文件保存，避免中斷時文件損壞
        temp_path = save_path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, save_path)  # 原子操作替換文件
        
        # 驗證文件是否保存成功
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"\nBest model checkpoint saved successfully:")
            print(f"Path: {save_path}")
            print(f"Size: {file_size/1e6:.2f}MB")
            print(f"Iteration: {cur_itrs}")
            print(f"Weighted score: {weighted_score:.4f}")
            return save_path
        else:
            print(f"Error: Checkpoint file not found after saving")
            return None
            
    except Exception as e:
        print(f"\nError saving checkpoint:")
        print(f"Error message: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Attempted path: {save_path}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def decode_target(target, num_classes=2):
    """將mask轉換為RGB圖像用於可視化"""
    target = np.array(target)
    rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    if num_classes == 2:  # binary
        rgb[target == 1] = [255, 255, 255]  # 前景為白色
        # 背景保持黑色
    return rgb

def validate_and_save(model, optimizer, scheduler, val_loader, device, metrics_logger, cur_itrs, opts, best_score, metric_weights):
    print("\n" + "="*50)
    print(f"Starting validation at iteration {cur_itrs}")
    print(f"Current best score: {best_score}")
    print("="*50)
    
    model.eval()
    # 這裡初始化時序評估器
    metrics = StreamMetrics(opts.num_classes, sequence_length=opts.sequence_length)
    
    # 用來存儲所有驗證樣本的信息：(timestamp, pred, gt)
    all_samples = []
    global_idx = 0

    with torch.no_grad():

        # 遍歷整個驗證集，收集所有預測和真值
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            if isinstance(batch, dict):  # 如果已經是時序格式，直接更新
                images = batch['images'].to(device)
                labels = batch['mask'].to(device)
                logits = model(images)  # [B, num_classes, H, W]
                prob = F.softmax(logits, dim=1)
                confidence, pred = torch.max(prob, dim=1, keepdim=True)
                preds = logits.max(1)[1].cpu().numpy()
                gts = labels.cpu().numpy()
                # 假設當前batch的樣本數量為B
                for i in range(gts.shape[0]):
                    # 取文件名作為時間戳，這裡假設 val_loader.dataset.images 是文件名列表
                    timestamp = val_loader.dataset.images[global_idx + i]
                    all_samples.append((timestamp, preds[i], gts[i]))
                global_idx += gts.shape[0]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:  # BinarySegmentation 格式
                images, labels = batch
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                logits = model(images)
                prob = F.softmax(logits, dim=1)
                confidence, pred = torch.max(prob, dim=1, keepdim=True)
                preds = logits.max(1)[1].cpu().numpy()
                gts = labels.cpu().numpy()
                for i in range(gts.shape[0]):
                    timestamp = val_loader.dataset.images[global_idx + i]
                    all_samples.append((timestamp, preds[i], gts[i]))
                global_idx += gts.shape[0]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
    
    # 如果樣本數量不足，則不更新時序評估（或返回默認值）
    if len(all_samples) < opts.sequence_length:
        print("Not enough samples for sequence evaluation.")
    else:
        # 如果文件名並非已排序，則根據時間戳排序；假設文件名中含有時間信息
        all_samples.sort(key=lambda x: x[0])
        # 使用 sliding window 遍歷所有樣本
        sequence_scores = []
        for i in range(len(all_samples) - opts.sequence_length + 1):
            window = all_samples[i: i + opts.sequence_length]
            window_preds = np.stack([item[1] for item in window])
            window_gts = np.stack([item[2] for item in window])
            # 對這個完整序列進行更新（metrics.update 將累計該序列的時序評分）
            metrics.update(window_gts, window_preds, sequence_data=True)
            # 如果需要，也可以把每個窗口的分數保存下來（例如 metrics.get_latest_score()）
            sequence_scores.append(metrics.temporal_evaluator.get_latest_score())
    
    # 最終結果
    val_score = metrics.get_results()
    metrics_logger.update('val_miou', val_score['MIoU'])
    metrics_logger.update('val_foreground_iou', val_score['Foreground IoU'])
    metrics_logger.update('val_foreground_f1', val_score['Foreground F1'])
    metrics_logger.update('val_temporal_consistency', val_score.get('Temporal Consistency', 0))
    metrics_logger.update('val_front_tracking_error', val_score.get('Front Tracking Error', 0))
    metrics_logger.update('val_region_continuity', val_score.get('Region Continuity', 0))
    metrics_logger.update('val_precision', val_score['Precision'])
    metrics_logger.update('val_recall', val_score['Recall'])
    
    weighted_score = metrics_logger.get_weighted_score()
    is_best = is_best_score(val_score, best_score, metric_weights)

    # 以下部分保持不變，用於打印、保存模型、保存圖表等
    print("\nValidation Results:")
    print("Current Scores:")
    for key, value in val_score.items():
        if isinstance(value, (float, int, np.float32, np.float64)):
            print(f"{key}: {value:.4f}")
    
    print("\nBest Scores:")
    for key, value in best_score.items():
        if isinstance(value, (float, int, np.float32, np.float64)):
            print(f"{key}: {value:.4f}")
    
    print(f"\nWeighted score: {weighted_score:.4f}")
    print(f"Is this the best score? {is_best}")
    
    if is_best:
        print("\n" + "="*50)
        print("New best model found! Saving model and validation results...")
        print("Previous best score:", best_score)
        print("New best score:", val_score)
        print("="*50)
        
        best_score = update_best_score(val_score)
        best_model_path = save_best_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            opts=opts,
            val_score=val_score,
            weighted_score=weighted_score,
            cur_itrs=cur_itrs,
            best_score=best_score
        )
        
        if opts.save_val_results and best_model_path:
            save_validation_results(
                model, val_loader, device, opts, 
                cur_itrs, weighted_score, saved_images_count=0
            )
    
    metrics_logger.save_confusion_matrix(metrics.confusion_matrix, cur_itrs)
    
    print("\n" + "="*50)
    print("Validation complete")
    print("="*50)
    
    return val_score, best_score, is_best

def initialize_best_score():
    """初始化最佳分數"""
    return {
        'MIoU': -float('inf'),
        'Foreground IoU': -float('inf'),
        'Foreground F1': -float('inf'),
        'Temporal Consistency': -float('inf'),
        'Front Tracking Error': float('inf'),  # 誤差指標初始化為正無窮
        'Region Continuity': -float('inf'),
        'Precision': -float('inf'),
        'Recall': -float('inf')
    }

def is_best_score(current_score, best_score, weights):
    """計算加權分數並比較"""
    if best_score is None:  # 第一次比較
        return True
        
    current_total = 0
    best_total = 0
    
    # 使用傳入的權重而不是寫死的值
    for metric in ['MIoU', 'Foreground IoU', 'Foreground F1', 'Temporal Consistency', 'Region Continuity']:
        if metric in weights and weights[metric] > 0:
            current_val = float(current_score[metric])
            best_val = float(best_score.get(metric, 0.0))
            weight = weights[metric]
            if not np.isnan(current_val):
                current_total += weight * current_val
                best_total += weight * best_val
    
    # 處理前緣追蹤誤差 - 使用傳入的權重
    if 'Front Tracking Error' in current_score:
        max_error = 10.0
        current_error = float(current_score['Front Tracking Error'])
        best_error = float(best_score.get('Front Tracking Error', max_error))
        
        # 轉換為[0,1]範圍的分數
        current_error_score = max(0, 1 - (current_error / max_error))
        best_error_score = max(0, 1 - (best_error / max_error))
        
        # 使用絕對值，因為Front Tracking Error的權重可能是負數
        weight = abs(weights.get('Front Tracking Error', 0.03))
        current_total += weight * current_error_score
        best_total += weight * best_error_score
    
    print(f"\nScore comparison:")
    print(f"Current total score: {current_total:.4f}")
    print(f"Best total score: {best_total:.4f}")
    
    return current_total > best_total

def update_best_score(val_score):
    """更新最佳分數，確保所有值都是有效的數值"""
    print("\nUpdating best score:")  # 添加調試信息
    
    best_score = {}
    for metric in ['MIoU', 'Foreground IoU', 'Foreground F1', 'Region Continuity']:
        if metric in val_score and not np.isnan(val_score[metric]):
            best_score[metric] = float(val_score[metric])
            print(f"{metric}: {best_score[metric]:.4f}")
        else:
            best_score[metric] = 0.0
            print(f"{metric}: 0.0 (default)")
    
    # 特殊處理前緣追蹤誤差
    if 'Front Tracking Error' in val_score:
        error = float(val_score['Front Tracking Error'])
        if not np.isnan(error):
            best_score['Front Tracking Error'] = error
            print(f"Front Tracking Error: {error:.4f}")
        else:
            best_score['Front Tracking Error'] = 10.0
            print(f"Front Tracking Error: 10.0 (default)")
    
    # 特殊處理時序一致性
    if 'Temporal Consistency' in val_score:
        value = val_score['Temporal Consistency']
        if value is not None and not np.isnan(value):
            best_score['Temporal Consistency'] = float(value)
            print(f"Temporal Consistency: {value:.4f}")
        else:
            best_score['Temporal Consistency'] = 0.0
            print(f"Temporal Consistency: 0.0 (default)")
    
    # 其他指標
    for metric in ['Precision', 'Recall']:
        if metric in val_score and not np.isnan(val_score[metric]):
            best_score[metric] = float(val_score[metric])
            print(f"{metric}: {best_score[metric]:.4f}")
    
    return best_score

def main():
    # 定義權重
    def get_metric_weights():
        return {
            "MIoU": 0.05,
            "Foreground IoU": 0.25,
            "Foreground F1": 0.25,
            "Front Tracking Error": 0.25,
            "Temporal Consistency": 0.10,
            "Region Continuity": 0.10
        }
        
    opts = get_argparser().parse_args()
    opts.num_classes = 2

    metric_weights = get_metric_weights()
    
    print("\nDirectory settings:")
    print(f"Checkpoints dir: {opts.checkpoints_dir}")
    print(f"Val results dir: {opts.val_results_dir}")
    print(f"Metrics plots dir: {opts.metrics_plots_dir}")
    print(f"Save feature maps: {opts.save_feature_maps}")
    print(f"Feature maps directory: {opts.feature_maps_dir}")

    for directory in [opts.checkpoints_dir, opts.val_results_dir, opts.metrics_plots_dir]:
        os.makedirs(directory, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise RuntimeError(f"Directory {directory} is not writable")
    
    if opts.save_feature_maps:
        if opts.feature_maps_dir is None:
            opts.feature_maps_dir = os.path.join(opts.val_results_dir, 'feature_maps')
        os.makedirs(opts.feature_maps_dir, exist_ok=True)

    # 設置 MLflow
    experiment_name, experiment_id = setup_mlflow(opts)  # 接收兩個返回值

    with mlflow.start_run(experiment_id=experiment_id):
        
        # 設置設備
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 記錄所有參數，包括權重設置
        mlflow.log_params({
            # 權重參數
            **{f"weight_{k.lower().replace(' ', '_')}": v 
               for k, v in metric_weights.items()},
            
            # 模型配置
            "model": opts.model,
            "dataset": opts.dataset,
            "num_classes": opts.num_classes,
            "output_stride": opts.output_stride,
            "separable_conv": opts.separable_conv,
            
            # 訓練配置
            "loss_type": opts.loss_type,
            "learning_rate": opts.lr,
            "batch_size": opts.batch_size,
            "val_batch_size": opts.val_batch_size,
            "crop_size": opts.crop_size,
            "total_iterations": opts.total_itrs,
            "optimizer": opts.optimizer,
            "weight_decay": opts.weight_decay,
            "momentum": 0.9,
            "step_size": opts.step_size,
            
            # 驗證配置
            "val_interval": opts.val_interval,
            "print_interval": opts.print_interval,
            "save_val_results": opts.save_val_results,
            "crop_val": opts.crop_val,
            
            # 系統配置
            "device": str(device),
            "gpu_id": opts.gpu_id,
            "random_seed": opts.random_seed,
            "num_workers": 4,
            
            # 環境信息
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "experiment_time": time.strftime("%Y%m%d-%H%M%S"),
            
            # 數據路徑
            "data_root": opts.data_root,
            
            # 繼續訓練配置
            "continue_training": opts.continue_training,
            "checkpoint_path": opts.ckpt if opts.ckpt else "None",

            "save_feature_maps": opts.save_feature_maps,
            "feature_maps_dir": opts.feature_maps_dir if opts.save_feature_maps else "None",
        })

        # 設置指標記錄器
        metrics_logger = MetricsLogger(opts.metrics_plots_dir, weights=metric_weights)
        
        # 設置隨機種子
        torch.manual_seed(opts.random_seed)
        np.random.seed(opts.random_seed)
        random.seed(opts.random_seed)

        print_weights_config(metric_weights)

        # 設置數據加載器
        train_dst, val_dst = get_dataset(opts)
        print(f"數據集類型 - 訓練: {type(train_dst).__name__}, 驗證: {type(val_dst).__name__}")
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
            drop_last=True)
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=4)

        # 計算並記錄類別權重
        class_weights = calculate_class_weights(train_loader)
        class_weights = class_weights.to(device)
        mlflow.log_params({
            "class_weight_background": float(class_weights[0]),
            "class_weight_wave": float(class_weights[1])
        })

        # 從checkpoint恢復訓練
        best_score = initialize_best_score()
        cur_itrs = 0
        cur_epochs = 0

        # 設置模型、優化器和損失函數
        model = setup_model(opts)
        model = nn.DataParallel(model)
        # 載入預訓練權重（如果有）
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            print(f"\nLoading checkpoint: {opts.ckpt}")
            try:
                checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
                model.module.load_state_dict(checkpoint["model_state"], strict=False)
                state_dict = checkpoint["model_state"]
                # 處理 state dict
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]  # 去掉 'module.' 前綴
                    else:
                        name = k
                    new_state_dict[name] = v
            
                # 載入權重
                ret = model.load_state_dict(new_state_dict, strict=False)
                print(f"Model restored from {opts.ckpt}")
                print(f"Missing keys: {len(ret.missing_keys)}")
                print(f"Unexpected keys: {len(ret.unexpected_keys)}")
            
                # 如果是繼續訓練，恢復其他狀態
                if opts.continue_training:
                    cur_itrs = checkpoint["cur_itrs"]
                    best_score = checkpoint.get('best_score', initialize_best_score())
                    cur_epochs = cur_itrs // len(train_loader)
                    print(f"Training resumed from iteration {cur_itrs}")
                    print(f"Best score restored: {best_score}")
            
                del checkpoint  # 釋放內存
            
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                return
        
        model.to(device)
        optimizer = setup_optimizer(model, opts)  # 先設置優化器        
        scheduler = setup_scheduler(optimizer, opts)
        criterion = setup_criterion(opts, class_weights)

        # 如果是繼續訓練，恢復優化器和調度器的狀態
        if opts.ckpt is not None and opts.continue_training:
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        # 主訓練循環
        print(f"\nStarting training from iteration {cur_itrs}")
        pbar = tqdm(total=opts.total_itrs, initial=cur_itrs, 
                    desc="Training Progress")
    
        try:
            while cur_itrs < opts.total_itrs:
                model.train()
                cur_epochs += 1
                interval_loss = 0

                for batch in train_loader:
                    cur_itrs += 1
                    pbar.update(1)

                    # 根據訓練階段和數據格式處理輸入
                    if isinstance(batch, dict):  # TemporalSegmentation 格式
                        images = batch['images'].to(device)  # [B,T,C,H,W]
                        labels = batch['mask'].to(device)    # [B,H,W]
                    elif isinstance(batch, (list, tuple)) and len(batch) == 2:  # BinarySegmentation 格式
                        images, labels = batch
                        images = images.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.long)
                    else:
                        raise ValueError(f"Unexpected batch format: {type(batch)}")

                    # 前向傳播
                    logits = model(images)  # 只回傳 logits
                    loss = criterion(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    np_loss = loss.detach().cpu().numpy()
                    interval_loss += np_loss

                    metrics_logger.update('train_loss', np_loss)
                    metrics_logger.update('learning_rate', optimizer.param_groups[0]['lr'])

                    if cur_itrs % opts.print_interval == 0:
                        avg_loss = interval_loss / opts.print_interval
                        print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={avg_loss:.6f}")
                        mlflow.log_metrics({
                            'train_loss': avg_loss,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'epoch': cur_epochs
                        }, step=cur_itrs)
                        interval_loss = 0.0

                    if cur_itrs % opts.val_interval == 0:
                        print(f"\nValidation at iteration {cur_itrs}")
                        val_score, best_score, is_best = validate_and_save(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            val_loader=val_loader,
                            device=device,
                            metrics_logger=metrics_logger,
                            cur_itrs=cur_itrs,
                            opts=opts,
                            best_score=best_score,
                            metric_weights=metric_weights
                        )
                        # 生成特徵圖可視化
                        if opts.save_feature_maps:
                            print("Generating feature maps visualization...")
                            # 統一使用 getattr 來獲取原始模型
                            model_to_vis = getattr(model, 'module', model)
                            # model_to_vis.visualize_features_for_sequences(device)
                        print("Validation complete")

                        weighted_score = metrics_logger.get_weighted_score()
                        mlflow.log_metric('weighted_score', weighted_score, step=cur_itrs)
                        
                        mlflow.log_metrics({
                            'val_miou': val_score['MIoU'],
                            'val_foreground_iou': val_score['Foreground IoU'],
                            'val_foreground_f1': val_score['Foreground F1'],
                            'val_temporal_consistency': val_score.get('Temporal Consistency', 0),
                            'val_front_tracking_error': val_score.get('Front Tracking Error', 0),
                            'val_region_continuity': val_score.get('Region Continuity', 0),
                            'val_precision': val_score['Precision'],
                            'val_recall': val_score['Recall']
                        }, step=cur_itrs)

                    scheduler.step()

                    if cur_itrs >= opts.total_itrs:
                        break

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            raise e
        finally:
            pbar.close()
            # 保存最終結果
            metrics_logger.save_plots()
            mlflow.log_metrics(metrics_logger.get_best_metrics())
            mlflow.log_param("status", "completed")

if __name__ == '__main__':
    main()