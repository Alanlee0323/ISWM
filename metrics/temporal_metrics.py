import numpy as np
from .utils.mask_utils import MaskUtils
from .utils.decorators import safe_process

class TemporalMetrics:
    """時序評估指標"""
    
    def __init__(self, sequence_length=7, threshold=0.005):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.sequence_predictions = []
        self.sequence_groundtruth = []
        self.temporal_scores = []
        
        # 新增詳細指標追蹤
        self.transition_scores = []
        self.stability_scores = []
        self.motion_scores = []
        self.wave_segment_scores = []

    def _evaluate_transitions(self, gt_has_wave, pred_has_wave):
        """評估轉換點的準確性"""
        gt_transitions = np.diff(gt_has_wave).astype(int)
        pred_transitions = np.diff(pred_has_wave).astype(int)
        
        if not np.any(gt_transitions):
            score = 1.0 if not np.any(pred_transitions) else 0.0
            self.transition_scores.append(score)
            return score
            
        gt_trans_idx = np.where(gt_transitions)[0]
        pred_trans_idx = np.where(pred_transitions)[0]
        
        if len(pred_trans_idx) != len(gt_trans_idx):
            self.transition_scores.append(0.0)
            return 0.0
            
        timing_errors = np.abs(gt_trans_idx - pred_trans_idx)
        score = 1.0 / (1.0 + np.mean(timing_errors))
        self.transition_scores.append(score)
        return score

    def _evaluate_wave_sequence(self, pred_sequence, gt_sequence):
        """評估有內波序列"""
        stability_values = []
        motion_values = []
        
        for t in range(1, len(pred_sequence)):
            stability = MaskUtils.calculate_stability(
                pred_sequence[t],
                pred_sequence[t-1]
            )
            stability_values.append(stability)
            
            motion = MaskUtils.calculate_motion(
                pred_sequence[t],
                pred_sequence[t-1]
            )
            motion_values.append(motion)
        
        # 更新詳細指標
        mean_stability = np.mean(stability_values) if stability_values else 0.0
        mean_motion = np.mean(motion_values) if motion_values else 0.0
        self.stability_scores.append(mean_stability)
        self.motion_scores.append(mean_motion)
        
        return np.mean([0.5 * s + 0.5 * m for s, m in zip(stability_values, motion_values)]) if stability_values else 0.0

    def _evaluate_no_wave_sequence(self, pred_has_wave):
        """評估無內波序列"""
        error_ratio = sum(pred_has_wave) / len(pred_has_wave)
        return 1.0 - error_ratio

    @safe_process(error_value=0.0)
    def _evaluate_wave_segments(self, pred_sequence, gt_sequence, pred_has_wave, gt_has_wave):
        """評估波段區間"""
        wave_scores = []
        for t in range(1, len(pred_sequence)):
            if gt_has_wave[t]:
                # 評估預測序列的時序連續性
                pred_stability = MaskUtils.calculate_stability(
                    pred_sequence[t],
                    pred_sequence[t-1]
                )
            
                # 評估與真實值的匹配度
                match_score = MaskUtils.calculate_stability(
                    pred_sequence[t],
                    gt_sequence[t]
                )
            
                # 綜合分數
                score = 0.5 * pred_stability + 0.5 * match_score
                wave_scores.append(score)
        
        segment_score = np.mean(wave_scores) if wave_scores else 0.0
        self.wave_segment_scores.append(segment_score)
        return segment_score

    def _evaluate_mixed_sequence(self, pred_sequence, gt_sequence, pred_has_wave, gt_has_wave):
        """評估混合序列"""
        transition_accuracy = self._evaluate_transitions(gt_has_wave, pred_has_wave)
        wave_segments_score = self._evaluate_wave_segments(
            pred_sequence, gt_sequence, pred_has_wave, gt_has_wave
        )
        # 使用與 check_Temp.py 相同的權重
        return 0.6 * transition_accuracy + 0.4 * wave_segments_score

    @safe_process(error_value=0.0)
    def _calculate_sequence_temporal_consistency(self, pred_sequence, gt_sequence):
        """評估序列的時序一致性"""
        gt_has_wave = [MaskUtils.check_wave_presence(frame, self.threshold) 
                      for frame in gt_sequence]
        pred_has_wave = [MaskUtils.check_wave_presence(frame, self.threshold) 
                        for frame in pred_sequence]
        
        if not any(gt_has_wave):
            return self._evaluate_no_wave_sequence(pred_has_wave)
        elif all(gt_has_wave):
            return self._evaluate_wave_sequence(pred_sequence, gt_sequence)
        else:
            return self._evaluate_mixed_sequence(
                pred_sequence, gt_sequence, pred_has_wave, gt_has_wave
            )

    def update(self, pred, gt):
        """更新序列並評估"""
        # 預處理輸入遮罩
        if len(pred.shape) > 2:
            pred = MaskUtils.preprocess_mask(pred)
            
        if len(gt.shape) > 2:
            gt = MaskUtils.preprocess_mask(gt)
        
        self.sequence_predictions.append(pred)
        self.sequence_groundtruth.append(gt)
        
        score = None
        
        if len(self.sequence_predictions) == self.sequence_length:
            score = self._calculate_sequence_temporal_consistency(
                self.sequence_predictions,
                self.sequence_groundtruth
            )
            self.temporal_scores.append(score)
            
            # 移除最舊的幀，保持固定長度
            self.sequence_predictions = self.sequence_predictions[1:]
            self.sequence_groundtruth = self.sequence_groundtruth[1:]
            
        return score
    
    def get_latest_score(self):
        """獲取最新的評分"""
        return self.temporal_scores[-1] if self.temporal_scores else 0.0

    def get_mean_score(self):
        """獲取平均評分"""
        if not self.temporal_scores:
            return 0.0
        return np.mean(self.temporal_scores)
    
    def get_detailed_statistics(self):
        """獲取詳細的統計指標"""
        return {
            'mean_score': self.get_mean_score(),
            'mean_transition': np.mean(self.transition_scores) if self.transition_scores else 0.0,
            'mean_stability': np.mean(self.stability_scores) if self.stability_scores else 0.0,
            'mean_motion': np.mean(self.motion_scores) if self.motion_scores else 0.0,
            'mean_wave_segment': np.mean(self.wave_segment_scores) if self.wave_segment_scores else 0.0,
            'score_count': len(self.temporal_scores)
        }
    
    def reset(self):
        """重置評估器"""
        self.sequence_predictions = []
        self.sequence_groundtruth = []
        self.temporal_scores = []
        self.transition_scores = []
        self.stability_scores = []
        self.motion_scores = []
        self.wave_segment_scores = []