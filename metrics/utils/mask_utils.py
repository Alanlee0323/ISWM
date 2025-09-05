import numpy as np
import torch
import cv2

class MaskUtils:
    @staticmethod
    def preprocess_mask(mask):
        """統一的遮罩預處理邏輯"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if len(mask.shape) == 3:
            mask = mask[-1]
        
        mask = (mask > 0).astype(np.uint8)
        
        try:
            # 形態學操作清理雜訊
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
            # 計算連通區域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
            if num_labels > 1:  # 如果有區域（排除背景）
                # 計算所有區域的面積
                areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
                min_valid_area = mask.size * 0.001
            
                # 找出所有"有效"的區域（面積大於閾值）
                valid_regions = areas >= min_valid_area
                valid_labels = np.where(valid_regions)[0] + 1  # +1因為stats[0]是背景
            
                if len(valid_labels) > 0:
                    # 取最大的區域
                    largest_label = valid_labels[np.argmax(areas[valid_labels-1])]
                    base_mask = (labels == largest_label).astype(np.uint8)
                
                    # 如果有多個有效區域，根據區域數量降低權重
                    if len(valid_labels) > 1:
                        weight = max(0.4, 1.0 - 0.2 * (len(valid_labels) - 1))
                        return base_mask * weight
                
                    return base_mask
                else:
                    return np.zeros_like(mask)
                
        except Exception as e:
            print(f"Error in preprocess_mask: {e}")
            return np.zeros_like(mask)
    
        return mask

    @staticmethod
    def find_front_positions(mask):
        """統一的前緣檢測邏輯"""
        # 使用預處理過的遮罩
        mask = MaskUtils.preprocess_mask(mask)
        
        # 如果沒有有效區域，返回空列表
        if not np.any(mask):
            return []
            
        # 掃描並找出最大區域的前緣點
        h, w = mask.shape
        front_positions = []
    
        for i in range(h):
            # 找出該行的白色像素
            white_pixels = np.where(mask[i] == 1)[0]
            if len(white_pixels) > 0:
                # 只取該行最左邊的點（前緣）
                front_positions.append((i, white_pixels[0]))
    
        return front_positions

    @staticmethod
    def calculate_motion(curr_pred, prev_pred):
        """計算運動特徵分數"""
        # 獲取前緣位置
        curr_fronts = MaskUtils.find_front_positions(curr_pred)
        prev_fronts = MaskUtils.find_front_positions(prev_pred)
    
        # 如果任一幀沒有有效的前緣點，返回0分
        if not curr_fronts or not prev_fronts:
            return 0.0
    
        # 計算前緣的平均位置變化
        curr_y = np.mean([y for y, x in curr_fronts])
        curr_x = np.mean([x for y, x in curr_fronts])
        prev_y = np.mean([y for y, x in prev_fronts])
        prev_x = np.mean([x for y, x in prev_fronts])
    
        # 計算前緣整體移動距離
        distance = np.sqrt((curr_y - prev_y)**2 + (curr_x - prev_x)**2)
    
        # 使用圖像高度的10%作為合理移動距離的閾值
        max_reasonable_movement = curr_pred.shape[0] * 0.1
    
        # 計算分數（距離越接近合理範圍，分數越高）
        return 1.0 / (1.0 + distance / max_reasonable_movement)

    @staticmethod
    def calculate_stability(curr_pred, prev_pred):
        """計算穩定性分數"""
        curr_pred = MaskUtils.preprocess_mask(curr_pred)
        prev_pred = MaskUtils.preprocess_mask(prev_pred)
    
        # 設定水平和垂直搜索窗口大小
        window_size = int(curr_pred.shape[1] * 0.1)  # 水平方向10%寬度

        stability_scores = []
    
        # 逐行分析
        for i in range(curr_pred.shape[0]):
            # 找出當前行中的前景點（內波）
            curr_pixels = np.where(curr_pred[i] == 1)[0]
        
            if len(curr_pixels) > 0:
                # 取該行最左邊的點（內波前緣）
                curr_front = curr_pixels[0]
            
                # 在前一幀的鄰近區域搜索
                search_start = max(0, curr_front - window_size)
                search_end = min(curr_pred.shape[1], curr_front + window_size)
            
                # 在搜索範圍內找前一幀的前景點
                prev_pixels = np.where(prev_pred[i, search_start:search_end] == 1)[0]
                if len(prev_pixels) > 0:
                    # 計算前緣點的距離差異
                    front_diff = abs(curr_front - (prev_pixels[0] + search_start))
                    # 距離越近，穩定性越高
                    stability_scores.append(1.0 / (1.0 + front_diff / window_size))
    
        # 回傳所有行的平均穩定性分數
        return np.mean(stability_scores) if stability_scores else 0.0

    @staticmethod
    def check_wave_presence(mask, threshold=0.005):
        """檢查是否存在內波"""
        mask = MaskUtils.preprocess_mask(mask)
        ratio = np.sum(mask) / mask.size
        return ratio >= threshold