import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from PIL import Image
import json

to_tensor = transforms.ToTensor()  # HWC [0,255] → CHW [0,1]

class MyDataset(Dataset):
    def __init__(self,
                 source_dir="/workspace/new_data/spatial_ediffsr/tvt/train/2021_new_bicubic8m_clean_png",
                 target_dir="/workspace/new_data/spatial_ediffsr/tvt/train/2021_new_clean_png",
                 prompt_file=""):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.data = []

        # ===== 1. 如果有 prompt.json 就加载，否则直接从文件夹匹配 =====
        if os.path.exists(prompt_file) and os.path.isfile(prompt_file):
            with open(prompt_file, 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            self.data = self._load_images_from_folders()

    def _load_images_from_folders(self):
     
        source_files = [f for f in os.listdir(self.source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        target_files = [f for f in os.listdir(self.target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        data = []
        for source_file in source_files:
            if source_file in target_files:
                data.append({
                    "source": source_file,
                    "target": source_file,
                    "prompt": ""  # 如果你后续完全不用 text，可以忽略
                })
        return data

    def _upsample_4x(self, image):
        
        h, w = image.shape[:2]
        new_h, new_w = h * 4, w * 4
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        # ===== 2. 读取图像 =====
        source = cv2.imread(os.path.join(self.source_dir, source_filename))
        target = cv2.imread(os.path.join(self.target_dir, target_filename))

        if source is None or target is None:
            raise ValueError(f"Failed to load: source={source_filename}, target={target_filename}")

        # ===== 3. 先对 source 上采样到 256×256 =====
        source = self._upsample_4x(source)

        # ===== 4. BGR → RGB =====
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # ===== 5. Normalization =====
        # 给 ControlNet 用的 hint (0~1)，即上采样后的 source
        source = source.astype(np.float32) / 255.0  

        # 给 UNet 用的 target (-1~1)，即 GT
        target = (target.astype(np.float32) / 127.5) - 1.0  

        # ===== 6. 返回 =====
        return dict(
            jpg=target,              # 目标 GT (float32, HWC, [-1,1])
            #txt=prompt,              # 如果现在不用，可以留空字符串
            hint=source,             # 上采样后的 source (float32, HWC, [0,1])
            cond_image=(source * 255.0).astype(np.uint8)  # 改为 numpy 数组 (uint8, HWC)，避免 PIL 导致 collate 报错
        )
