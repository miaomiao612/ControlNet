import torch
from PIL import Image
import numpy as np
from ldm.modules.encoders.modules import FrozenDINOv3Embedder  # 路径按你项目改

# 随便做一个假图片
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
pil_img = Image.fromarray(img)

# 实例化 DINOv3
encoder = FrozenDINOv3Embedder(
    backbone_ckpt="/workspace/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    repo_dir="/workspace/dinov3",
    model_name="vit_large",
    patch_size=16,
    out_dim=768,
    device="cuda"
)

# 前向
with torch.no_grad():
    feats = encoder.encode([pil_img])
print("最终 DINOv3 features shape:", feats.shape)