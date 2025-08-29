#!/usr/bin/env python3
"""
检查数据集大小和训练步数
"""

from tutorial_dataset import MyDataset
from torch.utils.data import DataLoader

def check_dataset_info():
    """检查数据集信息"""
    
    # 创建数据集
    dataset = MyDataset()
    print(f"数据集大小: {len(dataset)} 张图像")
    
    # 创建DataLoader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 计算每个epoch的步数
    steps_per_epoch = len(dataloader)
    print(f"Batch size: {batch_size}")
    print(f"每个epoch的步数: {steps_per_epoch}")
    
    # 计算图像记录频率
    logger_freq = 100
    images_per_epoch = steps_per_epoch // logger_freq
    print(f"图像记录频率: 每{logger_freq}步")
    print(f"每个epoch记录的图像数量: {images_per_epoch}")
    
    # 建议
    if steps_per_epoch < logger_freq:
        print(f"⚠️  警告: 每个epoch的步数({steps_per_epoch})少于图像记录频率({logger_freq})")
        print(f"建议将logger_freq设置为: {max(1, steps_per_epoch // 4)}")
    else:
        print(f"✅ 图像记录频率设置合理")
    
    return len(dataset), steps_per_epoch

if __name__ == "__main__":
    check_dataset_info()
