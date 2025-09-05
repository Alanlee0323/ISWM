import numpy as np
import cv2
from scipy.ndimage import label, generate_binary_structure
from .utils.decorators import safe_process

def repair_small_gaps(mask):
    """修復小的破損區域"""
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)
    closed = cv2.erode(dilated, kernel, iterations=2)
    return closed

class RegionMetrics:
    def __init__(self):
        self.valid_scores = []      
        self.total_cases = 0        
        self.invalid_cases = 0      
        self.min_area_threshold = 50  

    def _calculate_fragmentation_score(self, regions):
        """計算分割分數"""
        if not regions:
            return 0.0
            
        sorted_regions = sorted(regions, key=lambda x: x['area'], reverse=True)
        total_area = sum(r['area'] for r in regions)
        area_ratios = [r['area']/total_area for r in sorted_regions]
        
        fragmentation_score = area_ratios[0]  # Am
        
        if len(regions) > 1:
            penalty = sum(ratio * (i+1)/len(regions) 
                        for i, ratio in enumerate(area_ratios[1:]))
            fragmentation_score -= penalty * 0.5
            
        return max(0.0, min(1.0, fragmentation_score))

    @safe_process(error_value={'fragmentation_score': 0.0, 
                              'similarity_score': 0.0, 
                              'final_score': 0.0, 
                              'num_regions': 0})
    def _calculate_shape_metrics(self, pred):
        """計算形態指標"""
        s = generate_binary_structure(2, 2)
        labeled_array, num_features = label(pred, structure=s)

        valid_regions = []
        total_valid_area = 0
        
        for label_id in range(1, num_features + 1):
            region_mask = (labeled_array == label_id).astype(np.uint8)
            area = np.sum(region_mask)
            
            if area >= self.min_area_threshold:
                valid_regions.append({
                    'area': area,
                    'label_id': label_id
                })
                total_valid_area += area

        if not valid_regions:
            return {
                'fragmentation_score': 0.0,
                'similarity_score': 0.0,
                'final_score': 0.0,
                'num_regions': 0
            }

        return {
            'fragmentation_score': float(self._calculate_fragmentation_score(valid_regions)),
            'num_regions': len(valid_regions)
        }

    def calculate_region_metrics(self, pred, gt):
        """計算區域評估指標"""
        pred = (pred > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)
        
        if np.sum(pred) == 0 and np.sum(gt) == 0:
            return None
        
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return None
        
        # 只對預測結果進行預處理，修復小的破損區域
        pred = repair_small_gaps(pred)

        # 計算相似度
        intersection = np.logical_and(pred, gt)
        union = np.logical_or(pred, gt)
        similarity_score = intersection.sum() / union.sum()
        
        # 計算形態指標
        shape_metrics = self._calculate_shape_metrics(pred)
        
        # 整合評分
        metrics = {
            'fragmentation_score': shape_metrics['fragmentation_score'],
            'similarity_score': float(similarity_score),
            'num_regions': shape_metrics['num_regions']
        }
        
        # 最終分數計算
        weights = {
            'fragmentation': 0.7,  # 分割程度權重
            'similarity': 0.3     # 相似度權重
        }
        
        final_score = (
            weights['fragmentation'] * metrics['fragmentation_score'] +
            weights['similarity'] * metrics['similarity_score']
        )
        metrics['final_score'] = float(final_score)
        
        return metrics

    def update(self, pred, gt):
        """更新評估指標"""
        self.total_cases += 1
        metrics = self.calculate_region_metrics(pred, gt)
        
        if metrics is not None:
            self.valid_scores.append(metrics['final_score'])
        else:
            self.invalid_cases += 1
            
        return metrics

    def get_mean_score(self):
        """獲取平均分數"""
        if not self.valid_scores:
            return 0.0
        return np.mean(self.valid_scores)
    
    def get_statistics(self):
        """獲取統計結果"""
        if not self.valid_scores:
            return {
                'mean_score': None,
                'total_cases': self.total_cases,
                'valid_cases': len(self.valid_scores),
                'invalid_cases': self.invalid_cases,
                'valid_ratio': 0.0
            }
            
        return {
            'mean_score': np.mean(self.valid_scores),
            'total_cases': self.total_cases,
            'valid_cases': len(self.valid_scores),
            'invalid_cases': self.invalid_cases,
            'valid_ratio': len(self.valid_scores) / self.total_cases
        }

    def reset(self):
        """重置評估器"""
        self.valid_scores = []
        self.total_cases = 0
        self.invalid_cases = 0