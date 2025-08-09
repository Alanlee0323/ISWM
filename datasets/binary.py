import json
import os
import cv2
from collections import namedtuple
from pathlib import Path
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from utils import ext_transforms as et
from torchvision import transforms 
from torch.nn import functional as F
from datetime import datetime

class BinarySegmentation(data.Dataset):
    """
    Custom dataset for semantic segmentation
    """
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (string): 根目錄路徑
            split (string): 'train' or 'val'
            transform (callable, optional): 轉換操作
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(self.root, self.split, 'imgs')
        self.masks_dir = os.path.join(self.root, self.split, 'masks')
        
        # 添加路徑檢查
        print(f"檢查圖片目錄路徑: {self.images_dir}")
        print(f"檢查遮罩目錄路徑: {self.masks_dir}")

        # 檢查目錄是否存在
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"圖片目錄不存在: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"遮罩目錄不存在: {self.masks_dir}")

        # 獲取所有圖片文件名
        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))
        
        print(f"找到的圖片數量: {len(self.images)}")
        print(f"找到的遮罩數量: {len(self.masks)}")

        # 打印前幾個文件名來檢查
        print("\n前5個圖片文件:")
        print('\n'.join(self.images[:5]))
        print("\n前5個遮罩文件:")
        print('\n'.join(self.masks[:5]))
        
        # 過濾掉隱藏文件
        self.images = [f for f in self.images if not f.startswith('.')]
        self.masks = [f for f in self.masks if not f.startswith('.')]
        
        print(f"\n過濾後的圖片數量: {len(self.images)}")
        print(f"過濾後的遮罩數量: {len(self.masks)}")

        # 確保圖片和遮罩數量相同
        assert len(self.images) == len(self.masks), \
            f"圖片數量({len(self.images)})和遮罩數量({len(self.masks)})不相等"

    def __getitem__(self, index):
        """
        Args:
            index (int): 索引
        Returns:
            tuple: (image, mask)
        """
        img_path = os.path.join(self.images_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.masks[index])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 假設遮罩是單通道的
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 確保遮罩值為 0 或 1（二分類情況）
        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
        
        return image, mask.long()

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """
        將遮罩轉換為可視化的 RGB 圖像
        Args:
            mask (numpy.ndarray): 形狀為 (H, W) 的遮罩
        Returns:
            numpy.ndarray: 形狀為 (H, W, 3) 的 RGB 圖像
        """
        mask = mask.astype(np.uint8)  # 將遮罩轉換為 8 位無符號整數
        colormap = np.array([[0, 0, 0],    # 索引 0 對應黑色 (背景)
                     [255, 255, 255]])  # 索引 1 對應白色 (前景)
        return colormap[mask]  # 使用遮罩值作為索引來獲取顏色

class TemporalBinarySegmentation(data.Dataset):
    """時序語義分割數據集，支援跨天和缺失檢查"""
    def __init__(self, root, split='train', transform=None, sequence_length=5, max_gap_minutes=20, max_missing_frames=2):
        super(TemporalBinarySegmentation, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.sequence_length = sequence_length
        self.max_gap_minutes = max_gap_minutes  # 最大允許時間間隔（分鐘）
        self.max_missing_frames = max_missing_frames  # 最大允許缺失幀數
        
        self.images_dir = os.path.join(self.root, self.split, 'imgs')
        self.masks_dir = os.path.join(self.root, self.split, 'masks')
        
        # 檢查目錄
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"圖片目錄不存在: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"遮罩目錄不存在: {self.masks_dir}")
        
        # 獲取所有圖片文件名並按時間順序排序
        self.images = sorted([f for f in os.listdir(self.images_dir) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(self.masks_dir) if not f.startswith('.')])
        
        # 假設圖片和遮罩文件名一致，檢查數量是否匹配
        if len(self.images) != len(self.masks):
            raise ValueError(f"圖片數 ({len(self.images)}) 和遮罩數 ({len(self.masks)}) 不匹配")
        
        # 創建時序序列
        self.sequences = self._create_sequences()
    
    def _parse_timestamp(self, filename):
        """從文件名中解析時間戳，例如 '201905070850.png' -> datetime"""
        timestamp_str = filename.split('.')[0]  # 移除 .png
        return datetime.strptime(timestamp_str, '%Y%m%d%H%M')
    
    def _create_sequences(self):
        """創建時序序列，檢查跨天和時間間隔"""
        sequences = []
        i = 0
        
        while i <= len(self.images) - self.sequence_length:
            # 獲取當前序列的圖片和遮罩
            img_seq = self.images[i:i + self.sequence_length]
            mask = self.masks[i + self.sequence_length - 1]
            
            # 解析時間戳
            timestamps = [self._parse_timestamp(img) for img in img_seq]
            dates = [ts.date() for ts in timestamps]
            
            # 檢查是否跨天
            if len(set(dates)) > 1:  # 如果日期不唯一，則跨天
                i = i + self.sequence_length - 1  # 跳到下一個日期的第一幀
                continue
            
            # 檢查時間間隔和缺失
            valid_sequence = True
            missing_count = 0
            for j in range(1, len(timestamps)):
                time_diff = (timestamps[j] - timestamps[j-1]).total_seconds() / 60  # 轉換為分鐘
                if time_diff > self.max_gap_minutes:
                    valid_sequence = False
                    break
                elif time_diff > 10:  # 假設正常間隔是 10 分鐘，超過即視為缺失
                    missing_count += 1
                    if missing_count > self.max_missing_frames:
                        valid_sequence = False
                        break
            
            if valid_sequence:
                sequences.append((img_seq, mask))
                i += 1  # 正常移動到下一幀
            else:
                # 如果序列無效，跳到下一個可能的起點
                i += self.sequence_length - (j - 1)
        
        print(f"Generated {len(sequences)} valid sequences")
        return sequences
    
    def __getitem__(self, index):
        """返回一個時序序列和對應的遮罩，以字典格式"""
        img_seq, mask_name = self.sequences[index]
        
        # 加載圖片序列
        images = []
        for img_name in img_seq:
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            images.append(image)
        
        # 加載遮罩
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert('L')
        
        # 應用轉換
        if self.transform is not None:
            transformed_images = []
            for image in images:
                result = self.transform(image, mask)
                # 假設 transform 接受單張圖像並返回張量
                transformed_image = self.transform(image, mask)[0]  # 取元組的第一個元素
                transformed_images.append(transformed_image)
            image_tensor = torch.stack(transformed_images)  # [T, C, H, W]
            # 提取最後一幀的轉換後遮罩
            mask_tensor = self.transform(images[-1], mask)[1]  # 取元組的第二個元素
        else:
            image_tensor = torch.stack([transforms.ToTensor()(img) for img in images])  # [T, C, H, W]
            mask_tensor = transforms.ToTensor()(mask)
    
        # 確保遮罩值為 0 或 1
        mask_tensor = torch.where(mask_tensor > 0, torch.ones_like(mask_tensor), torch.zeros_like(mask_tensor))
    
        # 返回字典格式
        return {'images': image_tensor, 'mask': mask_tensor.long()}
    
    def __len__(self):
        return len(self.sequences)
    
class FeatureVisDataset(data.Dataset):
    def __init__(self, transform=None):
        """
        用於特徵可視化的固定測試集，包含兩組時序序列
        Args:
            transform (callable, optional): 圖片預處理轉換
        """
        # 基礎路徑
        base_path = "./20241209_Training_dataset/train"   

        # 第一組序列 (2019/06/06)
        self.sequence1_imgs = [
            f"{base_path}/imgs/201906060320.png",
            f"{base_path}/imgs/201906060330.png",
            f"{base_path}/imgs/201906060340.png",
            f"{base_path}/imgs/201906060350.png"
        ]
        self.sequence1_masks = [
            f"{base_path}/masks/201906060320_mask.png",
            f"{base_path}/masks/201906060330_mask.png",
            f"{base_path}/masks/201906060340_mask.png",
            f"{base_path}/masks/201906060350_mask.png"
        ]
        
        # 第二組序列 (2019/06/08)
        self.sequence2_imgs = [
            f"{base_path}/imgs/201906080700.png",
            f"{base_path}/imgs/201906080710.png",
            f"{base_path}/imgs/201906080720.png",
            f"{base_path}/imgs/201906080730.png"
        ]
        self.sequence2_masks = [
            f"{base_path}/masks/201906080700_mask.png",
            f"{base_path}/masks/201906080710_mask.png",
            f"{base_path}/masks/201906080720_mask.png",
            f"{base_path}/masks/201906080730_mask.png"
        ]
        
        # 組合所有序列
        self.sequences = [
            (self.sequence1_imgs, self.sequence1_masks, "sequence1"),
            (self.sequence2_imgs, self.sequence2_masks, "sequence2")
        ]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __len__(self):
        return len(self.sequences)  # 返回2（兩個序列）

    def __getitem__(self, idx):
        # 根據idx選擇序列
        if idx == 0:
            images = self.sequence1_imgs
            masks = self.sequence1_masks
            seq_name = "sequence1"
        else:
            images = self.sequence2_imgs
            masks = self.sequence2_masks
            seq_name = "sequence2"

        # 讀取序列中的所有圖片
        seq_images = []
        seq_masks = []
        for img_path, mask_path in zip(images, masks):
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # 使用標準的 transforms
            image = self.transform(image)
            mask = torch.from_numpy(np.array(mask))
            
            seq_images.append(image)
            seq_masks.append(mask)
            
        # 堆疊時序圖片
        seq_images = torch.stack(seq_images, dim=0)  # [T, C, H, W]
    
        return seq_images, seq_masks, seq_name, images