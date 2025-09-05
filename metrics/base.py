from abc import ABC, abstractmethod

class MetricsBase(ABC):
    """度量指標的基礎類別"""
    
    @abstractmethod
    def update(self, gt, pred):
        """更新指標
        
        Args:
            gt: Ground truth mask
            pred: Predicted mask
        """
        raise NotImplementedError()

    @abstractmethod
    def get_results(self):
        """獲取評估結果
        
        Returns:
            dict: 包含各項指標的字典
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """重置所有指標"""
        raise NotImplementedError()
        
    def to_str(self, metrics):
        """將指標轉換為字符串形式
        
        Args:
            metrics (dict): 指標字典
            
        Returns:
            str: 格式化的指標字符串
        """
        string = "\n"
        for k, v in metrics.items():
            string += f"{k}: {v:.4f}\n"
        return string