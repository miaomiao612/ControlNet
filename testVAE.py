import torch
from PIL import Image
import numpy as np
import cv2
from einops import rearrange

from cldm.model import create_model, load_state_dict

CKPT_PATH = "./checkpoints/last.ckpt"

# ===== 检查 checkpoint 里是否包含 VAE 权重 =====
print(f"检查 {CKPT_PATH} 中的 first_stage_model 权重...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
keys = [k for k in ckpt["state_dict"].keys() if "first_stage_model" in k]
print(f"找到 {len(keys)} 个 first_stage_model 参数")
if len(keys) > 0:
    print("示例 keys:", keys[:10])
else:
    print("⚠️ 这个 checkpoint 里没有保存 VAE 权重 (first_stage_model)！")

# ===== 创建模型 =====
model = create_model("./models/cldm_v15.yaml").cuda().eval()
_state = load_state_dict(CKPT_PATH, location="cpu")
missing, unexpected = model.load_state_dict(_state, strict=False)
print(f"加载完成 (missing={len(missing)}, unexpected={len(unexpected)})")

# ===== 读取一张GT图 =====
img = cv2.imread("/workspace/new_data/spatial_ediffsr/tvt/test/2021_new_clean_png/11_30.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256,256))
img = (img.astype(np.float32) / 127.5) - 1.0   # [-1,1]
x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()

# ===== VAE encode->decode =====
z_dist = model.encode_first_stage(x)
z = z_dist.mode()   # 用均值而不是 sample，更稳定
print("Latent z shape:", z.shape, "min:", z.min().item(), "max:", z.max().item())

x_rec = model.decode_first_stage(z)
print("Decoded shape:", x_rec.shape, "min:", x_rec.min().item(), "max:", x_rec.max().item())

x_rec = (rearrange(x_rec, "b c h w -> b h w c").clamp(-1,1) + 1.0) / 2.0
out = (x_rec[0].detach().cpu().numpy() * 255).astype(np.uint8)

cv2.imwrite("./test_imgs/vae_recon.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("✅ 保存VAE重建结果到 ./test_imgs/vae_recon.png")
