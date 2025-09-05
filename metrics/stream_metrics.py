import numpy as np
from .base import MetricsBase
from .region_metrics import RegionMetrics
from .temporal_metrics import TemporalMetrics
from .front_tracking_metrics import FrontTrackingMetrics

class StreamMetrics(MetricsBase):
    """串流評估指標系統"""
    
    def __init__(self, n_classes, sequence_length=7, temporal_stride=1, threshold=0.005):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.FOREGROUND_CLASS = 1
        self.best_score = {'weighted_score': 0.0}  # 加權分數
        
        # 初始化評估器
        self.temporal_evaluator = TemporalMetrics(
            sequence_length=sequence_length,
            threshold=threshold
        )
        self.region_evaluator = RegionMetrics()
        self.front_tracking_evaluator = FrontTrackingMetrics()

    def _fast_hist(self, label_true, label_pred):
        """快速計算混淆矩陣"""
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def _calculate_foreground_metrics(self, hist):
        """計算前景指標"""
        true_positives = hist[self.FOREGROUND_CLASS, self.FOREGROUND_CLASS]
        false_positives = hist[:, self.FOREGROUND_CLASS].sum() - true_positives
        false_negatives = hist[self.FOREGROUND_CLASS, :].sum() - true_positives
        true_negatives = hist.sum() - (true_positives + false_positives + false_negatives)

        # 添加調試資訊
        print(f"\nConfusion Matrix Components:")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"True Negatives: {true_negatives}")
        print(f"Total Pixels: {hist.sum()}")

        # 計算指標
        eps = 1e-7  # 避免除零
        foreground_iou = true_positives / (true_positives + false_positives + false_negatives + eps)
        precision = true_positives / (true_positives + false_positives + eps)
        recall = true_positives / (true_positives + false_negatives + eps)
        f1_score = 2 * precision * recall / (precision + recall + eps)
        
        # 計算平均IoU (MIoU)
        # 對於二分類問題，MIoU是背景IoU和前景IoU的平均
        background_tp = hist[0, 0]
        background_fp = hist[:, 0].sum() - background_tp
        background_fn = hist[0, :].sum() - background_tp
        background_iou = background_tp / (background_tp + background_fp + background_fn + eps)
        miou = (background_iou + foreground_iou) / 2.0

        return miou, foreground_iou, precision, recall, f1_score
    
    def _calculate_weighted_score(self, results):
        """計算加權分數，整合所有指標，使用指定的權重"""
        # 設定各指標權重 - 按照指定的權重設置
        weights = {
            "MIoU": 0.05,
            "Foreground IoU": 0.25,
            "Foreground F1": 0.25,
            "Front Tracking Error": 0.25,  # 注意：這是錯誤值，所以需要逆轉
            "Temporal Consistency": 0.10,
            "Region Continuity": 0.10
        }
        
        # 獲取各指標值
        miou = results["MIoU"]
        f_iou = results["Foreground IoU"]
        f1 = results["Foreground F1"]
        temp_consist = results["Temporal Consistency"]
        region_cont = results["Region Continuity"]
        
        # 前緣追蹤誤差需要歸一化和反轉
        # 假設前緣追蹤誤差的理想範圍是 0-10 像素
        max_fte = 10.0
        front_error = results["Front Tracking Error"]
        norm_front_error = 1.0 - min(front_error / max_fte, 1.0)  # 反轉和歸一化
        
        # 計算加權分數
        weighted_score = (
            weights["MIoU"] * miou +
            weights["Foreground IoU"] * f_iou +
            weights["Foreground F1"] * f1 +
            weights["Front Tracking Error"] * norm_front_error +
            weights["Temporal Consistency"] * temp_consist +
            weights["Region Continuity"] * region_cont
        )
        
        return weighted_score
    
    def update(self, label_trues, label_preds, sequence_data=True):
        """更新所有指標"""
        if sequence_data:
            # 更新時序評估器
            self.temporal_evaluator.update(label_preds, label_trues)
        
            # 對最後一幀進行區域和前緣評估
            self.region_evaluator.update(label_preds[-1], label_trues[-1])
            self.front_tracking_evaluator.update(label_preds[-1], label_trues[-1])
        
            # 更新混淆矩陣
            hist = self._fast_hist(label_trues[-1].flatten(), 
                             label_preds[-1].flatten())
        else:
            # 單幀評估
            self.region_evaluator.update(label_preds, label_trues)
            self.front_tracking_evaluator.update(label_preds, label_trues)
            hist = self._fast_hist(label_trues.flatten(), 
                                label_preds.flatten())
    
        self.confusion_matrix += hist
        
        # 更新當前結果並檢查是否是最佳分數
        current_results = self.get_results(update_best=False)
        current_weighted_score = self._calculate_weighted_score(current_results)
        
        # 如果當前分數更高，則更新最佳分數
        if current_weighted_score > self.best_score['weighted_score']:
            self.best_score['weighted_score'] = current_weighted_score
            self.best_score.update({
                'miou': current_results["MIoU"],
                'foreground_iou': current_results["Foreground IoU"],
                'foreground_f1': current_results["Foreground F1"],
                'temporal_consistency': current_results["Temporal Consistency"],
                'front_tracking_error': current_results["Front Tracking Error"],
                'region_continuity': current_results["Region Continuity"],
            })

    def get_results(self, update_best=True):
        """獲取所有評估結果"""
        # 計算空間指標
        miou, foreground_iou, precision, recall, f1_score = self._calculate_foreground_metrics(
            self.confusion_matrix
        )
        
        # 獲取各評估器的結果
        temporal_consistency = self.temporal_evaluator.get_mean_score()
        front_tracking_error = self.front_tracking_evaluator.get_mean_error()
        region_continuity = self.region_evaluator.get_mean_score()
        
        # 建立結果字典
        results = {
            "MIoU": miou,
            "Foreground IoU": foreground_iou,
            "Foreground F1": f1_score,
            "Temporal Consistency": temporal_consistency,
            "Front Tracking Error": front_tracking_error,
            "Region Continuity": region_continuity,
            "Precision": precision,
            "Recall": recall
        }
        
        # 獲取詳細的時序評估指標
        if hasattr(self.temporal_evaluator, 'get_detailed_statistics'):
            temporal_stats = self.temporal_evaluator.get_detailed_statistics()
            results.update({
                "Transition Accuracy": temporal_stats['mean_transition'],
                "Stability Score": temporal_stats['mean_stability'],
                "Motion Consistency": temporal_stats['mean_motion'],
                "Wave Segment Score": temporal_stats['mean_wave_segment']
            })
        
        # 獲取區域評估的詳細指標
        if hasattr(self.region_evaluator, 'get_statistics'):
            region_stats = self.region_evaluator.get_statistics()
            if 'valid_ratio' in region_stats:
                results["Region Valid Ratio"] = region_stats['valid_ratio']
        
        # 如果需要更新最佳分數
        if update_best:
            current_weighted_score = self._calculate_weighted_score(results)
            if current_weighted_score > self.best_score['weighted_score']:
                self.best_score['weighted_score'] = current_weighted_score
        
        # 添加最佳分數
        results["Best Score"] = self.best_score['weighted_score']
        
        return results

    def reset(self):
        """重置所有指標"""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.temporal_evaluator.reset()
        self.region_evaluator.reset()
        self.front_tracking_evaluator.reset()