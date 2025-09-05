import numpy as np
from .utils.mask_utils import MaskUtils
from .utils.decorators import safe_process
from .base import MetricsBase

class FrontTrackingMetrics:
    """前緣追蹤誤差評估器"""
    
    def __init__(self):
        self.max_distance_threshold = None
        self.tracking_errors = []
        
    def set_max_distance_threshold(self, image_width):
        """設置最大距離閾值"""
        self.max_distance_threshold = image_width * 0.1
        
    @safe_process(error_value=float('inf'))
    def calculate_error(self, pred, gt):
        """計算改進後的前緣追蹤誤差"""
        # 確保 max_distance_threshold 被初始化
        if self.max_distance_threshold is None:
            self.set_max_distance_threshold(pred.shape[1])
            
        pred = MaskUtils.preprocess_mask(pred)
        gt = MaskUtils.preprocess_mask(gt)
    
        pred_fronts = MaskUtils.find_front_positions(pred)
        gt_fronts = MaskUtils.find_front_positions(gt)
    
        # 特殊情況處理
        # 當真值有前緣但預測沒有時 (漏檢)
        if gt_fronts and not pred_fronts:
            return self.max_distance_threshold * 2.0  # 給予2τ的懲罰
            
        # 當真值沒有前緣但預測有時 (誤檢)
        if not gt_fronts and pred_fronts:
            return self.max_distance_threshold * 1.5  # 給予1.5τ的懲罰
            
        # 當都沒有前緣時
        if not gt_fronts and not pred_fronts:
            return 0.0  # 這是完美預測，給予零誤差
        
        # 計算從預測前緣到真值前緣的距離 (預測點的準確性)
        pred_to_gt_error = 0
        pred_valid_points = 0
        pred_weights_sum = 0
        
        for p_y, p_x in pred_fronts:
            min_dist = float('inf')
            min_dist_x = float('inf')
            
            for g_y, g_x in gt_fronts:
                dist = np.sqrt((p_y - g_y)**2 + (p_x - g_x)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_x = abs(p_x - g_x)

            # 只有距離小於閾值的點才被視為有效點
            if min_dist < self.max_distance_threshold:
                # 使用水平距離計算權重，避免除零問題
                weight = 1.0 / (min_dist_x + 1e-6)  # 加入極小值防止除零
                pred_to_gt_error += min_dist * weight
                pred_weights_sum += weight
                pred_valid_points += 1
        
        # 計算從真值前緣到預測前緣的距離 (真值的覆蓋率)
        gt_to_pred_error = 0
        gt_valid_points = 0
        gt_weights_sum = 0
        
        for g_y, g_x in gt_fronts:
            min_dist = float('inf')
            min_dist_x = float('inf')
            
            for p_y, p_x in pred_fronts:
                dist = np.sqrt((g_y - p_y)**2 + (g_x - p_x)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_x = abs(g_x - p_x)

            # 只有距離小於閾值的點才被視為有效點
            if min_dist < self.max_distance_threshold:
                weight = 1.0 / (min_dist_x + 1e-6)
                gt_to_pred_error += min_dist * weight
                gt_weights_sum += weight
                gt_valid_points += 1
        
        # 特殊情況處理：所有預測點與真實點之間的距離均超過閾值
        if pred_valid_points == 0 or gt_valid_points == 0:
            return self.max_distance_threshold * 2.0  # 給予2τ的懲罰
            
        # 計算加權平均誤差
        pred_avg_error = pred_to_gt_error / pred_weights_sum if pred_weights_sum > 0 else float('inf')
        gt_avg_error = gt_to_pred_error / gt_weights_sum if gt_weights_sum > 0 else float('inf')
        
        # 計算真值前緣覆蓋率
        gt_coverage = gt_valid_points / len(gt_fronts)
        
        # 結合誤差和覆蓋率
        # 使用最大值策略確保兩個方向的誤差都被考慮
        max_error = max(pred_avg_error, gt_avg_error)
        
        # 加入覆蓋率懲罰：覆蓋率越低，懲罰越高
        coverage_penalty = (1.0 - gt_coverage) * self.max_distance_threshold
        
        # 最終誤差 = 距離誤差 + 覆蓋率懲罰
        final_error = max_error + coverage_penalty * 0.5  # 可調整覆蓋率懲罰的權重
        
        return final_error

    def update(self, pred, gt):
        """更新前緣追蹤誤差"""
        # 確保在第一次更新時初始化 max_distance_threshold
        if self.max_distance_threshold is None:
            self.set_max_distance_threshold(pred.shape[1])
            
        error = self.calculate_error(pred, gt)
        if error is not None:
            self.tracking_errors.append(error)
        return error
    
    def get_mean_error(self):
        """計算平均誤差"""
        valid_errors = [x for x in self.tracking_errors if x is not None and not np.isinf(x)]
        if not valid_errors:
            if self.max_distance_threshold is not None:
                return self.max_distance_threshold * 2.0  # 當沒有有效評估時返回最大懲罰
            return float('inf')  # 如果 max_distance_threshold 還未設置
        return np.mean(valid_errors)
    
    def reset(self):
        """重置評估器"""
        self.tracking_errors = []
        # 不重置 max_distance_threshold，因為圖像尺寸通常不會改變